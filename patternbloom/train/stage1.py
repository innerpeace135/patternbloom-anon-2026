"""Stage I trainer entry point: policy gradient with Information-Density Reward."""

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_idr.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    os.environ["PATTERNBLOOM_STAGE"] = "1"
    os.environ["PATTERNBLOOM_ORACLE_ENDPOINT"] = cfg.oracle.endpoint
    os.environ["PATTERNBLOOM_IDR_TEMPERATURE"] = str(cfg.idr.temperature)
    os.environ["PATTERNBLOOM_RETRIEVAL_ENDPOINT"] = cfg.retrieval.endpoint
    os.environ["PATTERNBLOOM_RETRIEVAL_TOP_K"] = str(cfg.retrieval.top_k)
    os.environ["PATTERNBLOOM_PROMPT_TEMPLATE"] = cfg.data.prompt_template

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    verl_args = [
        "verl.trainer.main_ppo",
        f"data.train_files={cfg.data.train_path}",
        f"data.val_files={cfg.data.val_path}",
        f"actor_rollout_ref.model.path={cfg.policy.model_name_or_path}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.policy.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.n={cfg.policy.rollout_n}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.train.gpu_memory_utilization}",
        f"data.max_prompt_length={cfg.policy.max_prompt_len}",
        f"data.max_response_length={cfg.policy.max_response_len}",
        f"actor_rollout_ref.actor.optim.lr={cfg.train.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.train.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg.train.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.clip_ratio={cfg.train.clip_ratio}",
        f"algorithm.kl_ctrl.kl_coef={cfg.train.kl_coef}",
        f"data.train_batch_size={cfg.train.global_batch_size}",
        f"trainer.total_epochs={cfg.train.total_epochs}",
        f"trainer.save_freq={cfg.train.checkpoint_interval}",
        f"trainer.default_local_dir={cfg.train.checkpoint_dir}",
        f"trainer.project_name=patternbloom",
        f"trainer.experiment_name=stage1_idr",
        f"reward_model.reward_manager=patternbloom_idr",
        f"trainer.tool.env=search",
        f"trainer.tool.endpoint={cfg.retrieval.endpoint}",
        f"trainer.seed={cfg.seed}",
    ]

    sys.argv = verl_args
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
