"""Information-Density Reward (IDR) for stage-1 graph construction."""

from __future__ import annotations

import logging
import math
import re
from typing import List, Optional

from patternbloom.reward.format_gate import format_score
from patternbloom.reward.oracle_client import OracleClient, serialize_graph
from patternbloom.utils.text import normalize_gold


# Placeholders from the prompt template must be filtered to avoid contaminating
# the reward with template-only matches when the model emits no real graphs.
_TEMPLATE_PLACEHOLDER_SUBJECTS = {"entitya", "entityb", "entity_name", "next_entity"}
_TEMPLATE_PLACEHOLDER_RELATIONS = {"relation", "rel"}

MAX_GRAPH_BLOCKS = 10
MAX_TRIPLES_PER_BLOCK = 20

_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL
)
_GRAPH_RE = re.compile(r"<graph>(.*?)</graph>", re.DOTALL)

log = logging.getLogger("patternbloom.idr")


def _is_template_placeholder(triple: List[str]) -> bool:
    if len(triple) != 3:
        return True
    s, r, o = (p.strip().lower() for p in triple)
    return (
        s in _TEMPLATE_PLACEHOLDER_SUBJECTS
        or o in _TEMPLATE_PLACEHOLDER_SUBJECTS
        or r in _TEMPLATE_PLACEHOLDER_RELATIONS
    )


def _parse_pipe_triples(raw: str) -> List[List[str]]:
    triples = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(parts)
    return triples


def extract_evidence_triples(
    solution_str: str, max_blocks: int = MAX_GRAPH_BLOCKS
) -> List[List[str]]:
    """Collect triples only from assistant turns, capped to prevent reward hacking."""
    if not solution_str:
        return []
    triples: List[List[str]] = []
    cap = max_blocks * MAX_TRIPLES_PER_BLOCK
    for block in _ASSISTANT_BLOCK_RE.findall(solution_str):
        for raw in _GRAPH_RE.findall(block):
            if len(triples) >= cap:
                return triples
            for t in _parse_pipe_triples(raw)[:MAX_TRIPLES_PER_BLOCK]:
                if _is_template_placeholder(t):
                    continue
                triples.append(t)
                if len(triples) >= cap:
                    return triples
    return triples


def compute_idr_reward(
    solution_str: str,
    ground_truth,
    question: str,
    oracle_client: OracleClient,
    temp: float = 1.0,
    use_cache: bool = True,
) -> float:
    """IDR reward in [-1, 1] combining format gate and oracle information gain."""
    if solution_str is None or ground_truth is None or not question:
        return 0.0

    if temp < 0.1:
        raise ValueError(f"temp must be >= 0.1 for numerical stability, got {temp}")

    try:
        fmt = min(format_score(solution_str), 1.0)
        if fmt < 1.0:
            return -1.0 + fmt

        gold_list = normalize_gold(ground_truth)
        gold = gold_list[0] if gold_list else ""
        if not gold:
            return -1.0 + fmt

        triples = extract_evidence_triples(solution_str)
        n_triples = len(triples)

        if n_triples == 0:
            delta_logp = 0.0
        else:
            graph_text = serialize_graph(triples)
            delta_logp = oracle_client.delta_logp(
                question, graph_text, gold, use_cache=use_cache
            )

        denominator = max(n_triples, 1) * temp
        scaled = delta_logp / denominator
        scaled = max(-50.0, min(50.0, scaled))
        kernel = 1.0 / (1.0 + math.exp(-scaled))

        reward = -1.0 + fmt + kernel
        return max(-1.0, min(1.0, reward))
    except ValueError:
        raise
    except Exception as e:
        log.error(f"compute_idr_reward error (propagating): {e}")
        raise


def compute_evidence_mi(
    solution_str: str,
    ground_truth,
    question: str,
    oracle_client: OracleClient,
    use_cache: bool = True,
) -> float:
    """Information gain without size penalty or sigmoid (for logging only)."""
    if solution_str is None or ground_truth is None or not question:
        return 0.0
    try:
        gold_list = normalize_gold(ground_truth)
        gold = gold_list[0] if gold_list else ""
        if not gold:
            return 0.0
        triples = extract_evidence_triples(solution_str)
        if not triples:
            return 0.0
        graph_text = serialize_graph(triples)
        return oracle_client.delta_logp(
            question, graph_text, gold, use_cache=use_cache
        )
    except Exception:
        return 0.0
