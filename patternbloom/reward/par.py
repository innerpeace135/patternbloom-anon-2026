"""Pattern-Augmented Reward (PAR) for stage-2 multi-turn reasoning."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from patternbloom.gpm.memory import GraphPatternMemory, pattern_consistency
from patternbloom.reward.format_gate import format_score
from patternbloom.reward.idr import extract_evidence_triples
from patternbloom.utils.text import (
    normalize_answer,
    normalize_gold,
    token_f1,
)


_USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_user_question(solution_str: str) -> Optional[str]:
    if not solution_str:
        return None
    matches = _USER_BLOCK_RE.findall(solution_str)
    return matches[0].strip() if matches else None


def extract_answer(solution_str: str) -> Optional[str]:
    blocks = _ASSISTANT_BLOCK_RE.findall(solution_str)
    if not blocks:
        return None
    match = _ANSWER_RE.search(blocks[-1])
    return match.group(1).strip() if match else None


def _max_f1(answer: str, gt_list: List[str]) -> float:
    na = normalize_answer(answer)
    best = 0.0
    for g in gt_list:
        if normalize_answer(g) == na:
            return 1.0
        best = max(best, token_f1(answer, g))
    return best


def compute_par_reward(
    solution_str: str,
    ground_truth,
    question: str,
    gpm: GraphPatternMemory,
    sig_embedding: np.ndarray,
    lambda_p: float = 0.2,
    min_sim: float = 0.5,
) -> float:
    """PAR reward in [-1, 1] blending answer F1 with pattern consistency."""
    if solution_str is None or ground_truth is None:
        return -1.0
    if not (0.0 <= lambda_p <= 1.0):
        raise ValueError(f"lambda_p must be in [0, 1], got {lambda_p}")

    try:
        fmt = min(format_score(solution_str), 1.0)
        if fmt < 1.0:
            return -1.0 + fmt

        answer = extract_answer(solution_str)
        gt_list = normalize_gold(ground_truth)
        if not answer or not gt_list:
            f1 = 0.0
        else:
            f1 = _max_f1(answer, gt_list)

        pat_c = 0.0
        if lambda_p > 0 and len(gpm) > 0:
            triples = extract_evidence_triples(solution_str)
            if triples:
                hits = gpm.search(sig_embedding, top_k=1, min_sim=min_sim)
                if hits:
                    best_pattern, _ = hits[0]
                    pat_c = pattern_consistency(triples, best_pattern)

        quality = (1.0 - lambda_p) * f1 + lambda_p * pat_c
        reward = -1.0 + fmt + quality
        return max(-1.0, min(1.0, reward))
    except Exception:
        raise


def compute_pattern_consistency_logging(
    solution_str: str,
    gpm: GraphPatternMemory,
    sig_embedding: np.ndarray,
    min_sim: float = 0.5,
) -> float:
    """Pattern consistency alone, for logging."""
    if solution_str is None:
        return 0.0
    try:
        triples = extract_evidence_triples(solution_str)
        if not triples or len(gpm) == 0:
            return 0.0
        hits = gpm.search(sig_embedding, top_k=1, min_sim=min_sim)
        if not hits:
            return 0.0
        best_pattern, _ = hits[0]
        return pattern_consistency(triples, best_pattern)
    except Exception:
        return 0.0
