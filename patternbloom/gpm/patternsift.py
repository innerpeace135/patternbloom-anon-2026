"""Inference-time GPM self-evolution with a three-level safeguard gate."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from patternbloom.gpm.memory import (
    GraphPattern,
    GraphPatternMemory,
    make_pattern,
)
from patternbloom.gpm.signature import extract_signature
from patternbloom.reward.oracle_client import OracleClient, serialize_graph


@dataclass
class PatternSiftDecision:
    trajectory_id: str
    question: str
    idr_self_eval: float
    steps: int
    triples_count: int
    sig: str
    l1_pass: bool
    l2_pass: bool
    l3_pass: bool
    final_added: bool
    pattern_id: Optional[str] = None
    is_new: bool = False
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafeguardConfig:
    l1_threshold: float = 0.85
    l1_bump_threshold: float = 0.90
    l1_bump_after_fails: int = 5
    min_triples: int = 2
    max_steps: int = 6

    l2_enabled: bool = True
    l2_holdout_per_batch: int = 10
    l2_success_threshold: float = 0.6

    l3_enabled: bool = False
    l3_audit_interval: int = 500
    l3_sample_size: int = 50
    l3_error_tolerance: float = 0.10
    l3_judge_model: str = ""
    l3_judge_url: str = ""
    l3_judge_api_key_env: str = "OPENAI_API_KEY"


class PatternSift:
    """Three-level admission gate for new patterns from completed trajectories."""

    def __init__(
        self,
        gpm: GraphPatternMemory,
        oracle: OracleClient,
        safeguard: Optional[SafeguardConfig] = None,
        sig_embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        audit_log_path: Optional[str] = None,
    ):
        self.gpm = gpm
        self.oracle = oracle
        self.sg = safeguard or SafeguardConfig()
        self.sig_embedding_fn = sig_embedding_fn
        self.audit_log_path = audit_log_path

        self._fail_count_by_sig: Dict[str, int] = {}
        self._new_patterns_since_last_audit: List[str] = []
        self._decisions: List[PatternSiftDecision] = []

        self._current_l1_threshold = self.sg.l1_threshold

        self._last_audit_version: Optional[str] = None
        self._l3_audits_run: int = 0
        self._l3_rollbacks: int = 0

    def try_add(
        self,
        question: str,
        triples: List[List[str]],
        agent_answer: str,
        steps: int,
        trajectory_id: Optional[str] = None,
        holdout_queries: Optional[List[Tuple[str, str]]] = None,
    ) -> PatternSiftDecision:
        tid = trajectory_id or f"ps_{int(time.time() * 1000)}"
        sig = extract_signature(question)

        decision = PatternSiftDecision(
            trajectory_id=tid,
            question=question,
            idr_self_eval=0.0,
            steps=steps,
            triples_count=len(triples),
            sig=sig,
            l1_pass=False,
            l2_pass=False,
            l3_pass=False,
            final_added=False,
        )

        if len(triples) < self.sg.min_triples:
            decision.reason = f"L1 fail: triples<{self.sg.min_triples}"
            self._bump_fail_counter(sig)
            self._record(decision)
            return decision
        if steps > self.sg.max_steps:
            decision.reason = f"L1 fail: steps>{self.sg.max_steps}"
            self._bump_fail_counter(sig)
            self._record(decision)
            return decision

        idr = self._idr_self_eval(question, triples, agent_answer)
        decision.idr_self_eval = idr

        if idr < self._current_l1_threshold:
            decision.reason = (
                f"L1 fail: IDR={idr:.3f} < {self._current_l1_threshold}"
            )
            self._bump_fail_counter(sig)
            self._record(decision)
            return decision

        decision.l1_pass = True
        self._reset_fail_counter(sig)

        if self.sg.l2_enabled and holdout_queries:
            l2_success_rate = self._l2_cross_validate(
                triples, sig, holdout_queries
            )
            if l2_success_rate < self.sg.l2_success_threshold:
                decision.reason = (
                    f"L2 fail: CV success={l2_success_rate:.2f} "
                    f"< {self.sg.l2_success_threshold}"
                )
                self._record(decision)
                return decision
            decision.l2_pass = True
        else:
            decision.l2_pass = True

        decision.l3_pass = True

        sig_emb = self._get_sig_embedding(sig)
        pattern = make_pattern(
            question=question,
            triples=triples,
            answer=agent_answer,
            sig_embedding=sig_emb,
            idr=idr,
            f1=0.0,
            trajectory_id=tid,
        )
        pid, is_new = self.gpm.add_or_merge(pattern)
        decision.pattern_id = pid
        decision.is_new = is_new
        decision.final_added = True
        if is_new:
            self._new_patterns_since_last_audit.append(pid)
        decision.reason = "admitted"
        self._record(decision)

        if (
            self.sg.l3_enabled
            and len(self._new_patterns_since_last_audit)
            >= self.sg.l3_audit_interval
        ):
            try:
                self._l3_audit_batch()
            except Exception:
                pass

        return decision

    def _bump_fail_counter(self, sig: str) -> None:
        self._fail_count_by_sig[sig] = self._fail_count_by_sig.get(sig, 0) + 1
        if self._fail_count_by_sig[sig] >= self.sg.l1_bump_after_fails:
            new_thresh = min(
                self.sg.l1_bump_threshold,
                self._current_l1_threshold + 0.05,
            )
            if new_thresh > self._current_l1_threshold:
                self._current_l1_threshold = new_thresh
            self._fail_count_by_sig[sig] = 0

    def _reset_fail_counter(self, sig: str) -> None:
        self._fail_count_by_sig.pop(sig, None)

    def _idr_self_eval(
        self,
        question: str,
        triples: List[List[str]],
        agent_answer: str,
    ) -> float:
        if not triples or not agent_answer:
            return 0.0
        try:
            graph_text = serialize_graph(triples)
            dlp = self.oracle.delta_logp(
                question, graph_text, agent_answer, use_cache=True
            )
            denom = max(len(triples), 1)
            return 1.0 / (1.0 + math.exp(-dlp / denom))
        except Exception:
            return 0.0

    def _l2_cross_validate(
        self,
        candidate_triples: List[List[str]],
        sig: str,
        holdout_queries: List[Tuple[str, str]],
    ) -> float:
        if not holdout_queries:
            return 1.0
        n = min(len(holdout_queries), self.sg.l2_holdout_per_batch)
        graph_text = serialize_graph(candidate_triples)
        helpful = 0
        for q, gold in holdout_queries[:n]:
            try:
                dlp = self.oracle.delta_logp(
                    q, graph_text, gold, use_cache=True
                )
                if dlp > 0:
                    helpful += 1
            except Exception:
                continue
        return helpful / max(n, 1)

    def _l3_audit_batch(self) -> Dict:
        if not self.sg.l3_enabled:
            return {"skipped": "l3_disabled"}

        audit_tag = f"audit_{int(time.time())}"
        try:
            self.checkpoint_gpm(audit_tag)
        except Exception as e:
            return {"error": f"pre-audit checkpoint failed: {e}"}

        window = self._new_patterns_since_last_audit
        n_sample = min(self.sg.l3_sample_size, len(window))
        if n_sample == 0:
            self._new_patterns_since_last_audit.clear()
            return {"skipped": "empty_window"}
        sampled_pids = random.sample(window, n_sample)

        pid_to_pattern = {p.pattern_id: p for p in self.gpm.patterns}
        sampled_patterns = [
            pid_to_pattern[pid]
            for pid in sampled_pids
            if pid in pid_to_pattern
        ]
        if not sampled_patterns:
            return {"error": "sampled pids not found in GPM"}

        api_key = os.environ.get(self.sg.l3_judge_api_key_env, "").strip()
        if not api_key:
            self._new_patterns_since_last_audit.clear()
            return {"skipped": "no_api_key", "audit_tag": audit_tag}

        error_count, total = self._judge_patterns(
            sampled_patterns, api_key=api_key
        )
        error_rate = error_count / max(total, 1)
        rolled_back = False
        if error_rate > self.sg.l3_error_tolerance:
            try:
                if self._last_audit_version is not None:
                    self.rollback_gpm(self._last_audit_version)
                    rolled_back = True
                    self._l3_rollbacks += 1
            except Exception:
                pass

        if not rolled_back:
            self._last_audit_version = audit_tag
        self._new_patterns_since_last_audit.clear()
        self._l3_audits_run += 1
        return {
            "audit_tag": audit_tag,
            "sampled": total,
            "errors": error_count,
            "error_rate": error_rate,
            "tolerance": self.sg.l3_error_tolerance,
            "rolled_back": rolled_back,
        }

    def _judge_patterns(
        self, patterns: List[GraphPattern], api_key: str
    ) -> Tuple[int, int]:
        import requests

        system_prompt = (
            "You are an expert in multi-hop question answering. Given a "
            "question signature and a type-abstracted reasoning skeleton, "
            "judge whether the skeleton structurally makes sense for "
            "answering questions of that signature. Answer ONLY 'YES' or 'NO' "
            "(no explanation). YES = skeleton is a sensible template; "
            "NO = skeleton is nonsensical, wrong-typed, or doesn't fit sig."
        )

        errors = 0
        total = 0
        for p in patterns:
            skel = p.path_patterns[0] if p.path_patterns else []
            skel_str = ", ".join(
                f"({t[0]}, {t[1]}, {t[2]})" for t in skel[:8]
            )
            user_msg = (
                f"Signature: {p.sig}\n"
                f"Skeleton: {skel_str}\n"
                f"Descriptor: {p.descriptor[:160]}\n\n"
                f"Judgment (YES/NO):"
            )
            payload = {
                "model": self.sg.l3_judge_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 4,
            }
            try:
                r = requests.post(
                    f"{self.sg.l3_judge_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()
                verdict = (
                    data["choices"][0]["message"]["content"].strip().upper()
                )
                total += 1
                if verdict.startswith("NO"):
                    errors += 1
            except Exception:
                continue

        return errors, total

    def _get_sig_embedding(self, sig: str) -> np.ndarray:
        if self.sig_embedding_fn is not None:
            return self.sig_embedding_fn(sig)
        from patternbloom.gpm.distill import hash_signature_embedding
        return hash_signature_embedding(sig, dim=self.gpm.embedding_dim)

    def _record(self, decision: PatternSiftDecision) -> None:
        self._decisions.append(decision)
        if self.audit_log_path:
            try:
                import json
                with open(self.audit_log_path, "a") as f:
                    d = {
                        "trajectory_id": decision.trajectory_id,
                        "question": decision.question[:200],
                        "idr": round(decision.idr_self_eval, 4),
                        "steps": decision.steps,
                        "triples": decision.triples_count,
                        "sig": decision.sig,
                        "l1": decision.l1_pass,
                        "l2": decision.l2_pass,
                        "l3": decision.l3_pass,
                        "added": decision.final_added,
                        "pid": decision.pattern_id,
                        "is_new": decision.is_new,
                        "reason": decision.reason,
                        "ts": decision.timestamp,
                    }
                    f.write(json.dumps(d) + "\n")
            except Exception:
                pass

    def stats(self) -> Dict:
        total = len(self._decisions)
        added = sum(1 for d in self._decisions if d.final_added)
        new = sum(1 for d in self._decisions if d.is_new)
        return {
            "total_attempts": total,
            "admitted": added,
            "new_patterns": new,
            "merged_patterns": added - new,
            "reject_rate": (total - added) / max(total, 1),
            "current_l1_threshold": self._current_l1_threshold,
            "fail_counts_by_sig": dict(self._fail_count_by_sig),
            "l3_enabled": self.sg.l3_enabled,
            "l3_audits_run": self._l3_audits_run,
            "l3_rollbacks": self._l3_rollbacks,
            "l3_pending_batch": len(self._new_patterns_since_last_audit),
        }

    def checkpoint_gpm(self, tag: Optional[str] = None) -> str:
        tag = tag or f"v{int(time.time())}"
        return self.gpm.snapshot(tag)

    def rollback_gpm(self, tag: str) -> None:
        self.gpm.rollback(tag)
