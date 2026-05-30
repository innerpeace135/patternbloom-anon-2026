"""Pattern coverage metric: trajectory recall against a pattern skeleton."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from patternbloom.gpm.memory import GraphPattern, abstract_triples


Skeleton = Iterable[Tuple[str, str, str]]


def pattern_coverage(
    trajectory_triples: List[List[str]],
    pattern_skeleton: Skeleton,
) -> float:
    """Coverage = |abstract(trajectory) intersect skeleton| / |skeleton|."""
    skel_set = {tuple(t) for t in pattern_skeleton}
    if not skel_set:
        return 0.0
    traj_set = set(abstract_triples(trajectory_triples))
    return len(traj_set & skel_set) / len(skel_set)


def best_pattern_coverage(
    trajectory_triples: List[List[str]],
    pattern: GraphPattern,
) -> float:
    """Max coverage over all skeletons stored in a multi-skeleton pattern."""
    if not pattern.path_patterns:
        return 0.0
    traj_set = set(abstract_triples(trajectory_triples))
    best = 0.0
    for skel in pattern.path_patterns:
        s = set(skel)
        if not s:
            continue
        c = len(traj_set & s) / len(s)
        if c > best:
            best = c
    return best
