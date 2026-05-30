"""Multi-turn agent rollout against the retrieval API.

This module is consumed by the evaluation script and by the bundled trainer
under verl_fork/. Stage I and Stage II training delegate the actual rollout
loop to the verl agent worker for performance; this module provides a
reference single-process implementation suitable for evaluation.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
QUERY_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL)
GRAPH_RE = re.compile(r"<graph>(.*?)</graph>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
KNOWLEDGE_RE = re.compile(r"<knowledge>(.*?)</knowledge>", re.DOTALL)


@dataclass
class RolloutTurn:
    role: str
    text: str


@dataclass
class RolloutResult:
    question: str
    turns: List[RolloutTurn] = field(default_factory=list)
    answer: str = ""
    num_turns: int = 0

    def transcript(self) -> str:
        return "\n".join(f"<|{t.role}|>{t.text}" for t in self.turns)


def _format_knowledge(hits: Sequence[dict]) -> str:
    parts = []
    for hit in hits:
        title = hit.get("title", "")
        content = hit.get("content", "")
        parts.append(f"Title: {title}\nContent: {content}".strip())
    return "\n\n".join(parts)


def run_rollout(
    question: str,
    generate_fn: Callable[[List[RolloutTurn]], str],
    retrieve_fn: Callable[[str], List[dict]],
    system_prompt: str,
    pattern_prompt: Optional[str] = None,
    max_turns: int = 6,
) -> RolloutResult:
    result = RolloutResult(question=question)
    turns: List[RolloutTurn] = [RolloutTurn(role="system", text=system_prompt)]
    if pattern_prompt:
        turns.append(RolloutTurn(role="system", text=f"<knowledge>\n{pattern_prompt}\n</knowledge>"))
    turns.append(RolloutTurn(role="user", text=question))

    for step in range(max_turns):
        response_text = generate_fn(turns)
        turns.append(RolloutTurn(role="assistant", text=response_text))
        result.num_turns = step + 1

        answer_match = ANSWER_RE.search(response_text)
        if answer_match:
            result.answer = answer_match.group(1).strip()
            break

        query_match = QUERY_RE.search(response_text)
        if not query_match:
            break

        retrieved = retrieve_fn(query_match.group(1).strip())
        knowledge_text = _format_knowledge(retrieved)
        turns.append(RolloutTurn(role="user", text=f"<knowledge>\n{knowledge_text}\n</knowledge>"))

    result.turns = turns
    return result
