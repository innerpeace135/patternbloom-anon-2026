"""Multi-turn evaluation: rollout PatternBloom agent on a test split and score it."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import yaml

from patternbloom.agent.prompts import (
    build_stage1_user_message,
    build_stage2_user_message,
)
from patternbloom.gpm.memory import retrieve_hint_text
from patternbloom.gpm.retrieval import PatternRetriever
from patternbloom.reward.par import extract_answer
from patternbloom.utils.text import (
    contain_accuracy,
    exact_match,
    normalize_gold,
    token_f1,
)


_QUERY_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_evidence(
    client: httpx.Client, endpoint: str, queries: List[str], top_k: int
) -> List[List[Dict[str, Any]]]:
    if not queries:
        return []
    response = client.post(
        f"{endpoint.rstrip('/')}/search",
        json={"queries": queries, "top_k": top_k},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("results", [[] for _ in queries])


def format_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "<knowledge>(no results)</knowledge>"
    lines = []
    for i, item in enumerate(evidence[:10], 1):
        text = item.get("contents") or item.get("text") or str(item)
        lines.append(f"[{i}] {text}")
    return "<knowledge>\n" + "\n".join(lines) + "\n</knowledge>"


def build_initial_prompt(
    question: str,
    retriever: Optional[PatternRetriever],
    top_k: int,
) -> str:
    if retriever is None or len(retriever) == 0:
        return build_stage1_user_message(question)
    from patternbloom.gpm.signature import extract_signature
    sig_emb = retriever._embed(extract_signature(question))
    hint = retrieve_hint_text(retriever.gpm, sig_emb, top_k=top_k)
    return build_stage2_user_message(question, hint)


def run_turn(
    engine,
    sampling_params,
    conversation_text: str,
) -> str:
    outputs = engine.generate([conversation_text], sampling_params)
    return outputs[0].outputs[0].text


def rollout_one(
    engine,
    sampling_params,
    http_client: httpx.Client,
    question: str,
    retriever: Optional[PatternRetriever],
    retrieval_endpoint: str,
    retrieval_top_k: int,
    gpm_top_k: int,
    max_turns: int,
) -> Dict[str, Any]:
    user_msg = build_initial_prompt(question, retriever, gpm_top_k)
    history = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    )

    full_assistant_text = ""
    turns = 0
    for _ in range(max_turns):
        turns += 1
        generation = run_turn(engine, sampling_params, history)
        full_assistant_text += generation
        history += generation

        if _ANSWER_RE.search(generation):
            break

        query_match = _QUERY_RE.search(generation)
        if not query_match:
            break

        query = query_match.group(1).strip()
        try:
            results = fetch_evidence(
                http_client, retrieval_endpoint, [query], retrieval_top_k
            )
            evidence_block = format_evidence(results[0] if results else [])
        except Exception as e:
            evidence_block = f"<knowledge>(retrieval error: {e})</knowledge>"

        history += (
            f"<|im_end|>\n<|im_start|>user\n{evidence_block}<|im_end|>"
            f"\n<|im_start|>assistant\n"
        )

    final_conversation = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{full_assistant_text}<|im_end|>"
    )
    prediction = extract_answer(final_conversation) or ""

    return {
        "question": question,
        "prediction": prediction,
        "trajectory": full_assistant_text,
        "turns": turns,
    }


def compute_metrics(prediction: str, golds: List[str]) -> Dict[str, float]:
    if not prediction or not golds:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "contain_accuracy": 0.0,
        }
    em = max(exact_match(prediction, g) for g in golds)
    f1 = max(token_f1(prediction, g) for g in golds)
    ca = max(contain_accuracy(prediction, g) for g in golds)
    return {
        "exact_match": em,
        "token_f1": f1,
        "contain_accuracy": ca,
    }


def main():
    parser = argparse.ArgumentParser(description="PatternBloom evaluation")
    parser.add_argument(
        "--config", default="configs/eval.yaml", help="YAML config path"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["eval"]
    retr_cfg = cfg["retrieval"]
    gpm_cfg = cfg.get("gpm", {})
    inf_cfg = cfg.get("inference", {})

    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    test_df = pd.read_parquet(eval_cfg["test_path"])
    if eval_cfg.get("num_samples"):
        test_df = test_df.head(int(eval_cfg["num_samples"]))

    retriever: Optional[PatternRetriever] = None
    if gpm_cfg.get("memory_path") and Path(gpm_cfg["memory_path"]).exists():
        retriever = PatternRetriever(
            memory_path=gpm_cfg["memory_path"],
            use_encoder=True,
        )

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=eval_cfg["checkpoint_dir"],
        tensor_parallel_size=inf_cfg.get("tensor_model_parallel_size", 1),
        gpu_memory_utilization=inf_cfg.get("gpu_memory_utilization", 0.6),
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=inf_cfg.get("temperature", 0.0),
        max_tokens=inf_cfg.get("max_response_len", 4096),
        stop=["</answer>", "</query>"],
    )

    http_client = httpx.Client()

    total = len(test_df)
    em_sum = f1_sum = ca_sum = turns_sum = 0.0
    start = time.time()

    with open(pred_path, "w") as f_out:
        for i, row in test_df.iterrows():
            question = row["question"]
            golds = normalize_gold(row.get("golden_answers", []))

            result = rollout_one(
                engine=engine,
                sampling_params=sampling_params,
                http_client=http_client,
                question=question,
                retriever=retriever,
                retrieval_endpoint=retr_cfg["endpoint"],
                retrieval_top_k=retr_cfg.get("top_k", 5),
                gpm_top_k=gpm_cfg.get("top_k", 3),
                max_turns=inf_cfg.get("max_turns", 6),
            )

            scores = compute_metrics(result["prediction"], golds)
            em_sum += scores["exact_match"]
            f1_sum += scores["token_f1"]
            ca_sum += scores["contain_accuracy"]
            turns_sum += result["turns"]

            f_out.write(
                json.dumps(
                    {
                        "question": question,
                        "gold_answers": golds,
                        "prediction": result["prediction"],
                        "trajectory": result["trajectory"],
                        "turns": result["turns"],
                        **scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(
                    f"[{i + 1}/{total}] elapsed={elapsed:.0f}s "
                    f"em={em_sum / (i + 1):.4f} "
                    f"f1={f1_sum / (i + 1):.4f}"
                )

    aggregated = {
        "exact_match": em_sum / max(total, 1),
        "token_f1": f1_sum / max(total, 1),
        "contain_accuracy": ca_sum / max(total, 1),
        "avg_turns": turns_sum / max(total, 1),
        "num_samples": total,
    }
    with open(metrics_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Metrics saved to {metrics_path}: {aggregated}")

    http_client.close()


if __name__ == "__main__":
    main()
