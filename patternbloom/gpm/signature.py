"""Rule-based question signature extraction with entity-type suffix."""

from __future__ import annotations

import re
from typing import List, Set

from patternbloom.gpm.memory import extract_signature_base, infer_entity_type


_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")

_SKIP_SINGLE_WORDS: Set[str] = {
    "Who", "What", "When", "Where", "Why", "How", "Which",
    "Was", "Were", "Is", "Are", "Am", "Did", "Do", "Does",
    "Has", "Have", "Had",
    "Can", "Could", "Should", "Would", "May", "Might", "Will",
    "The", "A", "An",
    "In", "On", "At", "To", "For", "With", "By", "From", "Of",
    "And", "Or", "But", "If",
    "Compare", "Compares", "Name",
    "This", "That", "These", "Those",
    "Both",
}

_STRIPPABLE_LEADING: Set[str] = {"The", "A", "An", "Who", "What", "When", "Where"}

_MEANINGFUL_TYPES: Set[str] = {"ENTITY", "YEAR", "DATE", "NUMBER"}


def extract_question_entities(question: str) -> List[str]:
    if not question:
        return []
    entities: List[str] = []
    seen: Set[str] = set()

    def _add(e: str) -> None:
        e = e.strip()
        if e and e not in seen:
            entities.append(e)
            seen.add(e)

    for match in _QUOTED_RE.finditer(question):
        _add(match.group(1) or match.group(2) or "")

    for y in _YEAR_RE.findall(question):
        _add(y)

    for match in _PROPER_NOUN_RE.finditer(question):
        phrase = match.group(1).strip()
        if not phrase:
            continue
        words = phrase.split()

        if len(words) == 1 and words[0] in _SKIP_SINGLE_WORDS:
            continue

        if words[0] in _STRIPPABLE_LEADING and len(words) > 1:
            phrase = " ".join(words[1:])
            words = phrase.split()

        if len(phrase) < 2:
            continue
        if len(words) == 1 and words[0] in _SKIP_SINGLE_WORDS:
            continue

        _add(phrase)

    return entities


def extract_signature(question: str) -> str:
    """Combine base structural signature with sorted entity-type suffix."""
    base = extract_signature_base(question)

    entities = extract_question_entities(question)
    raw_types = [infer_entity_type(e) for e in entities]
    meaningful_types = sorted({t for t in raw_types if t in _MEANINGFUL_TYPES})

    suffix = "+".join(meaningful_types) if meaningful_types else "UNK"
    return f"{base}[{suffix}]"
