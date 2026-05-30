"""PPO training entry point with stage-aware reward dispatch for PatternBloom."""

import hydra
import ray
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import (
    _default_compute_score_format,
    _default_compute_score_answer_f1,
    _default_compute_score_answer_em,
    _default_compute_score_format_answer,
    _compute_stage1_reward,
    _compute_stage2_reward,
    _compute_coverage_score,
    _compute_num_graph_blocks,
)

from agent.tool import ToolEnv, PathReasoningToolEnv
from agent.tool.tools import _default_tools


class RewardManager:
    """Dispatch reward computation based on the configured training stage.

    Stage 0: format-and-answer baseline.
    Stage 1: Information-Density Reward.
    Stage 2: Pattern-Augmented Reward.
    """

    def __init__(self, tokenizer, num_examine, reward_stage=0):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_stage = reward_stage
        self.global_step = 0
        self._train_reward_ref = None

    def set_train_ref(self, train_rm):
        self._train_reward_ref = train_rm

    @property
    def effective_step(self):
        if self._train_reward_ref is not None:
            return self._train_reward_ref.global_step
        return self.global_step

    def __call__(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        answer_lst_f1, answer_lst_em, format_lst = [], [], []
        result_lst, coverage_lst, graph_blocks_lst = [], [], []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length].long()

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id
            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]

            if self.reward_stage == 1:
                score = _compute_stage1_reward(data_source, sequences_str, ground_truth)
            elif self.reward_stage == 2:
                score = _compute_stage2_reward(data_source, sequences_str, ground_truth)
            else:
                score = _default_compute_score_format_answer(
                    data_source, sequences_str, ground_truth
                )

            answer_lst_f1.append(_default_compute_score_answer_f1(data_source, sequences_str, ground_truth))
            answer_lst_em.append(_default_compute_score_answer_em(data_source, sequences_str, ground_truth))
            format_lst.append(_default_compute_score_format(data_source, sequences_str))
            coverage_lst.append(_compute_coverage_score(data_source, sequences_str, ground_truth))
            graph_blocks_lst.append(_compute_num_graph_blocks(data_source, sequences_str))
            result_lst.append(sequences_str)

            reward_tensor[i, valid_response_length - 1] = score

            already_print_data_sources.setdefault(data_source, 0)
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return reward_tensor, answer_lst_f1, answer_lst_em, format_lst, result_lst, coverage_lst, graph_blocks_lst


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)
def main_task(config, compute_score=None):
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_to_local

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    tool_env_name = config.tool.env
    tools = _default_tools(tool_env_name)

    if tool_env_name == "pathwise":
        env = PathReasoningToolEnv(tools=tools, max_turns=config.tool.max_turns)
    else:
        env = ToolEnv(tools=tools, max_turns=config.tool.max_turns)

    reward_stage = 0
    if hasattr(config, "reward") and hasattr(config.reward, "stage"):
        reward_stage = config.reward.stage
    else:
        reward_stage = getattr(config, "reward_stage", 0)

    print(f"Tool environment: {tool_env_name}; reward stage: {reward_stage}")

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, reward_stage=reward_stage)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, reward_stage=reward_stage)
    val_reward_fn.set_train_ref(reward_fn)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        env=env,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
