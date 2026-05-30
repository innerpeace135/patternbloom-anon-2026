"""Format gate for multi-turn trajectories with think/query/graph/answer blocks."""

from __future__ import annotations

import re
from typing import List


_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_QUERY_RE = re.compile(r"<query>.*?</query>", re.DOTALL)
_GRAPH_RE = re.compile(r"<graph>.*?</graph>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def _extract_assistant_blocks(trajectory_text: str) -> List[str]:
    return _ASSISTANT_BLOCK_RE.findall(trajectory_text)


def format_score(trajectory_text: str) -> float:
    """Continuous format score in [0, 1] for partial-credit reward shaping."""
    if not trajectory_text:
        return 0.0
    blocks = _extract_assistant_blocks(trajectory_text)
    if not blocks:
        return 0.0

    score = 0.0
    for block in blocks[:-1]:
        has_think = _THINK_RE.search(block)
        has_action = _QUERY_RE.search(block) or _GRAPH_RE.search(block)
        if has_think and has_action:
            score += 0.5

    last = blocks[-1]
    if _THINK_RE.search(last) and _ANSWER_RE.search(last):
        score += 0.5

    return min(score, 1.0)


def format_gate(trajectory_text: str) -> int:
    """Return 1 if every assistant turn has well-formed blocks, else 0."""
    if not trajectory_text:
        return 0
    blocks = _extract_assistant_blocks(trajectory_text)
    if not blocks:
        return 0

    for block in blocks[:-1]:
        if not _THINK_RE.search(block):
            return 0
        if not (_QUERY_RE.search(block) or _GRAPH_RE.search(block)):
            return 0

    last = blocks[-1]
    if not (_THINK_RE.search(last) and _ANSWER_RE.search(last)):
        return 0

    return 1
