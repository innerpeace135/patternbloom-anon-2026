"""Top-k pattern retrieval via FAISS cosine similarity on signature embeddings."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from patternbloom.gpm.encoder import encode_signature
from patternbloom.gpm.memory import GraphPattern, GraphPatternMemory
from patternbloom.gpm.signature import extract_signature


class PatternRetriever:
    """Loads a serialized GPM and retrieves top-k patterns for a question."""

    def __init__(
        self,
        memory_path: str,
        embedding_dim: int = 1024,
        use_encoder: bool = True,
    ):
        path = Path(memory_path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {memory_path}")

        if path.suffix in {".pkl", ".pickle"}:
            self.gpm = GraphPatternMemory.from_file(str(path))
        elif path.suffix == ".json":
            self.gpm = self._load_from_json(str(path), embedding_dim)
        else:
            raise ValueError(
                f"Unsupported memory file extension: {path.suffix}"
            )

        self.use_encoder = use_encoder
        self.embedding_dim = self.gpm.embedding_dim

    def _load_from_json(
        self, path: str, embedding_dim: int
    ) -> GraphPatternMemory:
        with open(path) as f:
            records = json.load(f)
        gpm = GraphPatternMemory(
            storage_dir=str(Path(path).parent),
            embedding_dim=embedding_dim,
        )
        from patternbloom.gpm.distill import hash_signature_embedding
        for r in records:
            emb = (
                np.asarray(r["sig_embedding"], dtype=np.float32)
                if r.get("sig_embedding") is not None
                else hash_signature_embedding(r["sig"], embedding_dim)
            )
            skel = [
                [tuple(t) for t in skel]
                for skel in r.get("path_patterns", [])
            ]
            pattern = GraphPattern(
                pattern_id=r["pattern_id"],
                sig=r["sig"],
                sig_embedding=emb,
                path_patterns=skel,
                descriptor=r.get("descriptor", ""),
                anchor_trajectory_ids=r.get("anchor_trajectory_ids", []),
                count=r.get("count", 1),
                avg_f1=r.get("avg_f1", 0.0),
                avg_idr=r.get("avg_idr", 0.0),
                version=r.get("version", 1),
            )
            gpm.add_or_merge(pattern)
        return gpm

    def _embed(self, sig: str) -> np.ndarray:
        if self.use_encoder:
            return encode_signature(sig, self.embedding_dim)
        from patternbloom.gpm.distill import hash_signature_embedding
        return hash_signature_embedding(sig, self.embedding_dim)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_sim: float = 0.5,
    ) -> List[Tuple[GraphPattern, float]]:
        sig = extract_signature(query)
        sig_emb = self._embed(sig)
        return self.gpm.search(sig_emb, top_k=top_k, min_sim=min_sim)

    def retrieve_patterns(
        self,
        query: str,
        top_k: int = 3,
        min_sim: float = 0.5,
    ) -> List[GraphPattern]:
        return [p for p, _ in self.retrieve(query, top_k=top_k, min_sim=min_sim)]

    def __len__(self) -> int:
        return len(self.gpm)
