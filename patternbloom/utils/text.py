"""Text normalization and token-level metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contain_accuracy(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(gold) in normalize_answer(prediction) else 0.0


def max_over_golds(prediction: str, golds: Iterable[str], metric) -> float:
    best = 0.0
    for g in golds:
        score = metric(prediction, str(g))
        if score > best:
            best = score
        if best >= 1.0:
            break
    return best


def normalize_gold(ground_truth) -> List[str]:
    if hasattr(ground_truth, "tolist"):
        gt = ground_truth.tolist()
    elif isinstance(ground_truth, str):
        gt = [ground_truth]
    else:
        gt = list(ground_truth)
    return [str(g) for g in gt]
