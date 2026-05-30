"""Build an initial Graph Pattern Memory from saved trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from patternbloom.gpm.memory import GraphPatternMemory, make_pattern
from patternbloom.gpm.signature import extract_signature


def hash_signature_embedding(sig: str, dim: int = 1024) -> np.ndarray:
    """Deterministic pseudo-random unit vector keyed by sig string."""
    h = hashlib.sha256(sig.encode()).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-10)).astype(np.float32)


def iter_trajectories(path: str) -> Iterable[Dict]:
    p = Path(path)
    if p.is_dir():
        files = sorted(list(p.glob("*.jsonl")) + list(p.glob("*.json")))
    else:
        files = [p]

    for f in files:
        if not f.exists():
            continue
        with open(f) as fh:
            if f.suffix == ".jsonl":
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            elif f.suffix == ".json":
                try:
                    data = json.load(fh)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    if "results" in data and isinstance(data["results"], list):
                        yield from data["results"]
                    elif "trajectories" in data and isinstance(
                        data["trajectories"], list
                    ):
                        yield from data["trajectories"]


def build_gpm(
    trajectory_path: str,
    output_path: str,
    idr_threshold: float = 0.85,
    min_triples: int = 2,
    max_steps: int = 6,
    embedding_dim: int = 1024,
    merge_threshold: float = 0.85,
    max_size: int = 100_000,
    use_encoder: bool = False,
    verbose: bool = True,
) -> GraphPatternMemory:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    gpm = GraphPatternMemory(
        storage_dir=str(output_path_obj.parent),
        embedding_dim=embedding_dim,
        merge_threshold=merge_threshold,
        max_size=max_size,
    )

    if use_encoder:
        from patternbloom.gpm.encoder import encode_signature
        embed_fn = encode_signature
    else:
        embed_fn = hash_signature_embedding

    stats = {
        "total": 0,
        "filter_idr": 0,
        "filter_triples": 0,
        "filter_steps": 0,
        "added": 0,
        "merged": 0,
        "gold_fallback": 0,
    }

    for i, traj in enumerate(iter_trajectories(trajectory_path), start=1):
        stats["total"] += 1

        q = traj.get("question", "")
        triples = traj.get("triples", [])
        if "gold_answer" in traj and traj["gold_answer"]:
            gold = traj["gold_answer"]
        else:
            stats["gold_fallback"] += 1
            gold = traj.get("answer", "")
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        idr = float(traj.get("idr_score", 0.0))
        f1 = float(traj.get("f1_score", 0.0))
        steps = int(traj.get("steps", 0))
        traj_id = traj.get("trajectory_id") or f"traj_{i}"

        if idr < idr_threshold:
            stats["filter_idr"] += 1
            continue
        if len(triples) < min_triples:
            stats["filter_triples"] += 1
            continue
        if steps > max_steps:
            stats["filter_steps"] += 1
            continue

        sig = extract_signature(q)
        sig_emb = embed_fn(sig, embedding_dim)
        pattern = make_pattern(
            question=q,
            triples=triples,
            answer=str(gold),
            sig_embedding=sig_emb,
            idr=idr,
            f1=f1,
            trajectory_id=traj_id,
            sig=sig,
        )
        _, is_new = gpm.add_or_merge(pattern)
        if is_new:
            stats["added"] += 1
        else:
            stats["merged"] += 1

        if verbose and stats["total"] % 500 == 0:
            print(
                f"  processed {stats['total']} trajectories, "
                f"GPM size = {len(gpm)}"
            )

    gpm.save(output_path)

    if verbose:
        print("=" * 60)
        print(f"GPM build summary - output: {output_path}")
        print(f"  Total trajectories:     {stats['total']}")
        print(f"  Filtered (IDR):         {stats['filter_idr']}")
        print(f"  Filtered (triples):     {stats['filter_triples']}")
        print(f"  Filtered (steps):       {stats['filter_steps']}")
        print(f"  Patterns added:         {stats['added']}")
        print(f"  Patterns merged:        {stats['merged']}")
        print(f"  Final GPM size:         {len(gpm)}")
        if stats["gold_fallback"] > 0:
            pct = 100.0 * stats["gold_fallback"] / max(stats["total"], 1)
            print(
                f"  WARN: gold_answer missing in {stats['gold_fallback']}/"
                f"{stats['total']} ({pct:.1f}%) trajectories"
            )
        print("=" * 60)

    return gpm


def main():
    parser = argparse.ArgumentParser(
        description="Build initial GPM from stage-1 trajectories"
    )
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--idr_threshold", type=float, default=0.85)
    parser.add_argument("--min_triples", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--merge_threshold", type=float, default=0.85)
    parser.add_argument("--max_size", type=int, default=100_000)
    parser.add_argument(
        "--use_encoder",
        action="store_true",
        help="Use BGE encoder for sig embeddings instead of hash-based",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build_gpm(
        trajectory_path=args.trajectories,
        output_path=args.output,
        idr_threshold=args.idr_threshold,
        min_triples=args.min_triples,
        max_steps=args.max_steps,
        embedding_dim=args.embedding_dim,
        merge_threshold=args.merge_threshold,
        max_size=args.max_size,
        use_encoder=args.use_encoder,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
