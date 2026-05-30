"""Graph Pattern Memory: type abstraction, GraphPattern, and FAISS-backed store."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


_YEAR_RE = re.compile(r"^\d{4}$")
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")
_DATE_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_BOOL_SET = {"true", "false", "yes", "no"}


def infer_entity_type(text: str) -> str:
    """One of: EMPTY, YEAR, NUMBER, DATE, BOOLEAN, ENTITY."""
    t = (text or "").strip()
    if not t:
        return "EMPTY"
    if _YEAR_RE.match(t):
        return "YEAR"
    if _DATE_RE.match(t):
        return "DATE"
    if _NUMBER_RE.match(t):
        return "NUMBER"
    if t.lower() in _BOOL_SET:
        return "BOOLEAN"
    return "ENTITY"


def abstract_triple(triple: List[str]) -> Tuple[str, str, str]:
    if len(triple) != 3:
        return ("EMPTY", "", "EMPTY")
    subj, rel, obj = triple
    return (
        infer_entity_type(subj),
        (rel or "").lower().strip(),
        infer_entity_type(obj),
    )


def abstract_triples(triples: List[List[str]]) -> List[Tuple[str, str, str]]:
    return [abstract_triple(t) for t in triples if t and len(t) == 3]


def extract_signature_base(question: str) -> str:
    """Coarse structural bucket from question phrasing."""
    q = (question or "").lower().strip()
    if not q:
        return "unknown"

    compare_kw = re.search(
        r"\b(compar|earlier|later|first|last|before|after|older|younger|"
        r"bigger|smaller|higher|lower|more|less|greater)\b",
        q,
    )
    if compare_kw:
        if re.search(r"born|birth", q):
            return "compare(by[birth_year])"
        if re.search(r"direct|film|movie", q):
            return "compare(by[director])"
        if re.search(r"year|date|when", q):
            return "compare(by[date])"
        return "compare(by[attr])"

    if re.match(r"^(is|was|were|did|does|do|has|have|had)\b", q):
        return "yesno(relation)"

    if re.search(r"\bhow (many|much)\b", q):
        return "count(entity, relation)"

    has_and = " and " in q

    if re.search(r"\bwho (is|was|are|were|did)\b", q):
        return "lookup_chain(person)" if has_and else "lookup(person)"
    if re.search(r"\bwhen (did|was|is|were|does)\b", q):
        return "lookup(time)"
    if re.search(r"\bwhere (is|was|are|were|did|does)\b", q):
        return "lookup(location)"
    if re.search(r"\bwhat (is|was|are|were|did|does)\b", q):
        return "lookup_chain(entity)" if has_and else "lookup(entity)"
    if re.search(r"\bwhich\b", q):
        return "lookup_chain(entity)" if has_and else "lookup(entity)"
    if re.search(r"\bwhy\b", q):
        return "lookup(reason)"
    if re.search(r"\bhow\b", q):
        return "lookup(method)"

    return "lookup(entity)"


@dataclass
class GraphPattern:
    """Type-abstracted reasoning pattern distilled from a successful trajectory."""

    pattern_id: str
    sig: str
    sig_embedding: np.ndarray
    path_patterns: List[List[Tuple[str, str, str]]]
    descriptor: str
    anchor_trajectory_ids: List[str] = field(default_factory=list)
    count: int = 1
    avg_f1: float = 0.0
    avg_idr: float = 0.0
    version: int = 1
    created_at: float = field(default_factory=time.time)

    def __setstate__(self, state):
        if "path_pattern" in state and "path_patterns" not in state:
            state["path_patterns"] = [state.pop("path_pattern")]
        self.__dict__.update(state)

    @property
    def path_pattern(self) -> List[Tuple[str, str, str]]:
        return self.path_patterns[0] if self.path_patterns else []

    def to_json_dict(self) -> Dict:
        d = asdict(self)
        d["sig_embedding"] = None
        d["path_patterns"] = [
            [list(t) for t in skel] for skel in self.path_patterns
        ]
        d["path_pattern"] = d["path_patterns"][0] if d["path_patterns"] else []
        return d


def _hash_pattern(sig: str) -> str:
    h = hashlib.sha1()
    h.update(sig.encode())
    return h.hexdigest()[:16]


def make_pattern(
    question: str,
    triples: List[List[str]],
    answer: str,
    sig_embedding: np.ndarray,
    idr: float = 0.0,
    f1: float = 0.0,
    trajectory_id: Optional[str] = None,
    sig: Optional[str] = None,
) -> GraphPattern:
    if sig is None:
        sig = extract_signature_base(question)
    path_pattern_single = abstract_triples(triples)
    pid = _hash_pattern(sig)
    descriptor = f"{sig} -> {answer[:80]}" if answer else sig
    return GraphPattern(
        pattern_id=pid,
        sig=sig,
        sig_embedding=sig_embedding.astype(np.float32),
        path_patterns=[path_pattern_single],
        descriptor=descriptor,
        anchor_trajectory_ids=[trajectory_id] if trajectory_id else [],
        count=1,
        avg_f1=float(f1),
        avg_idr=float(idr),
    )


def pattern_consistency(
    trajectory_triples: List[List[str]], pattern: GraphPattern
) -> float:
    """Pattern recall: |abstract(trajectory) intersect skeleton| / |skeleton|."""
    if not pattern.path_patterns:
        return 0.0
    tau_abs = set(abstract_triples(trajectory_triples))
    best = 0.0
    for skeleton in pattern.path_patterns:
        p_set = set(skeleton)
        if not p_set:
            continue
        c = len(tau_abs & p_set) / len(p_set)
        if c > best:
            best = c
    return best


class GraphPatternMemory:
    """FAISS-backed pattern store with add/merge/search/prune/snapshot."""

    def __init__(
        self,
        storage_dir: str,
        embedding_dim: int = 1024,
        merge_threshold: float = 0.85,
        max_size: int = 100_000,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.max_size = max_size

        self.patterns: List[GraphPattern] = []
        self._index: faiss.Index = faiss.IndexFlatIP(embedding_dim)
        self._id_to_idx: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.patterns)

    def __iter__(self):
        return iter(self.patterns)

    def add_or_merge(self, pattern: GraphPattern) -> Tuple[str, bool]:
        assert pattern.sig_embedding.shape == (self.embedding_dim,), (
            f"embedding shape {pattern.sig_embedding.shape} != ({self.embedding_dim},)"
        )
        emb = self._normalize(pattern.sig_embedding).reshape(1, -1)

        if len(self.patterns) > 0:
            sims, idxs = self._index.search(emb, 1)
            sim = float(sims[0][0])
            if sim >= self.merge_threshold:
                existing = self.patterns[int(idxs[0][0])]
                new_count = existing.count + 1
                existing.avg_f1 = (
                    existing.avg_f1 * existing.count + pattern.avg_f1
                ) / new_count
                existing.avg_idr = (
                    existing.avg_idr * existing.count + pattern.avg_idr
                ) / new_count
                existing.count = new_count
                existing.anchor_trajectory_ids.extend(pattern.anchor_trajectory_ids)
                existing.anchor_trajectory_ids = existing.anchor_trajectory_ids[-20:]
                existing_skels = {
                    tuple(map(tuple, s)) for s in existing.path_patterns
                }
                for new_skel in pattern.path_patterns:
                    new_key = tuple(map(tuple, new_skel))
                    if new_key not in existing_skels:
                        existing.path_patterns.append(new_skel)
                        existing_skels.add(new_key)
                return existing.pattern_id, False

        if len(self.patterns) >= self.max_size:
            raise ValueError(
                f"GPM at max_size {self.max_size}; call dedup_prune() first"
            )

        pattern.sig_embedding = emb[0].copy()
        self.patterns.append(pattern)
        self._index.add(emb)
        self._id_to_idx[pattern.pattern_id] = len(self.patterns) - 1
        return pattern.pattern_id, True

    def search(
        self,
        sig_embedding: np.ndarray,
        top_k: int = 3,
        min_sim: float = 0.5,
    ) -> List[Tuple[GraphPattern, float]]:
        if len(self.patterns) == 0:
            return []
        emb = self._normalize(sig_embedding).reshape(1, -1)
        k = min(top_k, len(self.patterns))
        sims, idxs = self._index.search(emb, k)
        out = []
        for sim, idx in zip(sims[0], idxs[0]):
            s = float(sim)
            if s < min_sim:
                break
            out.append((self.patterns[int(idx)], s))
        return out

    def dedup_prune(self, min_count: int = 2) -> int:
        before = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.count >= min_count]
        self._rebuild_index()
        return before - len(self.patterns)

    def _rebuild_index(self) -> None:
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._id_to_idx = {}
        if not self.patterns:
            return
        embs = np.stack(
            [self._normalize(p.sig_embedding) for p in self.patterns]
        ).astype(np.float32)
        self._index.add(embs)
        for i, p in enumerate(self.patterns):
            self._id_to_idx[p.pattern_id] = i

    def snapshot(self, version_tag: str) -> str:
        snap_path = self.storage_dir / f"gpm_{version_tag}.pkl"
        self.save(str(snap_path))
        return str(snap_path)

    def rollback(self, version_tag: str) -> None:
        snap_path = self.storage_dir / f"gpm_{version_tag}.pkl"
        if not snap_path.exists():
            raise FileNotFoundError(f"No snapshot at {snap_path}")
        self.load(str(snap_path))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "patterns": self.patterns,
                    "embedding_dim": self.embedding_dim,
                    "merge_threshold": self.merge_threshold,
                    "max_size": self.max_size,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.patterns = state["patterns"]
        self.embedding_dim = state["embedding_dim"]
        self.merge_threshold = state["merge_threshold"]
        self.max_size = state["max_size"]
        self._rebuild_index()

    @classmethod
    def from_file(
        cls, path: str, storage_dir: Optional[str] = None
    ) -> "GraphPatternMemory":
        with open(path, "rb") as f:
            state = pickle.load(f)
        gpm = cls(
            storage_dir=storage_dir or str(Path(path).parent),
            embedding_dim=state["embedding_dim"],
            merge_threshold=state["merge_threshold"],
            max_size=state["max_size"],
        )
        gpm.patterns = state["patterns"]
        gpm._rebuild_index()
        return gpm

    def export_json(self, path: str) -> None:
        data = [p.to_json_dict() for p in self.patterns]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        emb = emb.astype(np.float32)
        n = np.linalg.norm(emb) + 1e-10
        return emb / n


GENERIC_MULTIHOP_HINT = (
    "# Hint (no matching prior pattern; generic multi-hop guidance):\n"
    "Break the question into sub-questions; resolve each via <query>entity</query>, "
    "store intermediate facts as <graph>A | rel | B</graph>, and compose the final answer."
)


def retrieve_hint_text(
    gpm: GraphPatternMemory,
    sig_embedding: np.ndarray,
    top_k: int = 3,
    min_sim: float = 0.5,
    cold_start_fallback: bool = True,
) -> str:
    if len(gpm) == 0:
        return GENERIC_MULTIHOP_HINT if cold_start_fallback else ""

    results = gpm.search(sig_embedding, top_k=top_k, min_sim=min_sim)
    if not results:
        return GENERIC_MULTIHOP_HINT if cold_start_fallback else ""

    lines = ["# Retrieved reasoning patterns (prior knowledge):"]
    for i, (p, sim) in enumerate(results, 1):
        lines.append(f"## Pattern {i} (sim={sim:.2f}, used {p.count} times)")
        lines.append(f"Signature: {p.sig}")
        n_skels = len(p.path_patterns)
        for j, skel in enumerate(p.path_patterns[:3]):
            label = (
                f"Path skeleton variant {j + 1}/{min(n_skels, 3)}:"
                if n_skels > 1
                else "Path skeleton:"
            )
            lines.append(label)
            for t in skel[:8]:
                lines.append(f"  ({t[0]}, {t[1]}, {t[2]})")
            if len(skel) > 8:
                lines.append(f"  ... ({len(skel) - 8} more triples)")
        if n_skels > 3:
            lines.append(f"  ... ({n_skels - 3} more skeleton variants)")
        lines.append("")
    return "\n".join(lines)
