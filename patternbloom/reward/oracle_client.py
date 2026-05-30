"""HTTP client for the frozen oracle scorer used by the information-density reward."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import diskcache
import requests


_TOKENIZER = None
_TOKENIZER_LOCK = threading.Lock()


def _get_tokenizer(model_name: str):
    global _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        return _TOKENIZER


def serialize_graph(triples: List[List[str]]) -> str:
    """Canonical text form for triples fed into the oracle prompt."""
    if not triples:
        return ""
    lines = []
    for t in triples:
        if len(t) == 3 and all(t):
            lines.append(f"- ({t[0]}) --[{t[1]}]--> ({t[2]})")
    return "\n".join(lines)


def build_prompt(question: str, graph_text: Optional[str] = None) -> str:
    """Stable prompt format. Changing this invalidates cached baselines."""
    ctx = graph_text.strip() if graph_text else "(no context)"
    return f"Question: {question.strip()}\nContext: {ctx}\nAnswer:"


@dataclass
class OracleConfig:
    url: str = "http://localhost:8100"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir: str = "./cache/oracle_baseline"
    timeout: float = 30.0
    max_retries: int = 3
    temperature: float = 0.0
    _max_tokens: int = 0


class OracleClient:
    """Thread-safe scorer; one instance per worker, diskcache shared via FS locks."""

    def __init__(self, config: Optional[OracleConfig] = None, **kwargs):
        if config is None:
            config = OracleConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                setattr(config, k, v)
        self.config = config

        os.makedirs(config.cache_dir, exist_ok=True)
        self._baseline_cache = diskcache.Cache(
            config.cache_dir, size_limit=2 * 1024 ** 3
        )

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        self._call_count = 0
        self._cache_hits = 0
        self._lock = threading.Lock()

    def logp(
        self,
        question: str,
        graph_text: Optional[str],
        gold_answer: str,
        use_cache: bool = True,
    ) -> float:
        if graph_text in (None, ""):
            return self._baseline_logp(question, gold_answer, use_cache=use_cache)
        return self._logp_uncached(question, graph_text, gold_answer)

    def delta_logp(
        self,
        question: str,
        graph_text: str,
        gold_answer: str,
        use_cache: bool = True,
    ) -> float:
        l_g = self.logp(question, graph_text, gold_answer, use_cache=use_cache)
        l_0 = self.logp(question, None, gold_answer, use_cache=use_cache)
        return l_g - l_0

    def logp_batch(
        self,
        items: List[Tuple[str, Optional[str], str]],
        use_cache: bool = True,
    ) -> List[float]:
        results: List[Optional[float]] = [None] * len(items)
        pending: List[Tuple[int, str, str, Optional[str]]] = []

        for i, (q, g, a) in enumerate(items):
            if g in (None, "") and use_cache:
                key = self._baseline_key(q, a)
                cached = self._baseline_cache.get(key)
                if cached is not None:
                    with self._lock:
                        self._cache_hits += 1
                    results[i] = cached
                    continue
                pending.append((i, build_prompt(q, None), a, key))
            else:
                pending.append((i, build_prompt(q, g), a, None))

        if pending:
            prompts = [p + " " + a for (_, p, a, _) in pending]
            prompt_prefixes = [p for (_, p, _, _) in pending]
            answers = [a for (_, _, a, _) in pending]
            logps = self._batch_score(prompts, prompt_prefixes, answers)
            with self._lock:
                self._call_count += len(prompts)
            for (i, _, _, cache_key), lp in zip(pending, logps):
                results[i] = lp
                if cache_key is not None:
                    self._baseline_cache[cache_key] = lp

        return [r if r is not None else 0.0 for r in results]

    def stats(self) -> dict:
        with self._lock:
            total = self._call_count + self._cache_hits
            hit_rate = self._cache_hits / max(total, 1)
        return {
            "calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self._baseline_cache),
        }

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.config.url}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _baseline_key(self, question: str, gold_answer: str) -> str:
        h = hashlib.sha1()
        h.update(self.config.model_name.encode())
        h.update(b"\x00")
        h.update(question.encode())
        h.update(b"\x00")
        h.update(gold_answer.encode())
        return h.hexdigest()

    def _baseline_logp(
        self, question: str, gold_answer: str, use_cache: bool = True
    ) -> float:
        if use_cache:
            key = self._baseline_key(question, gold_answer)
            cached = self._baseline_cache.get(key)
            if cached is not None:
                with self._lock:
                    self._cache_hits += 1
                return cached

        lp = self._logp_uncached(question, None, gold_answer)
        if use_cache:
            self._baseline_cache[self._baseline_key(question, gold_answer)] = lp
        return lp

    def _logp_uncached(
        self, question: str, graph_text: Optional[str], gold_answer: str
    ) -> float:
        prompt_prefix = build_prompt(question, graph_text)
        full_prompt = prompt_prefix + " " + gold_answer
        lps = self._batch_score([full_prompt], [prompt_prefix], [gold_answer])
        with self._lock:
            self._call_count += 1
        return lps[0]

    def _batch_score(
        self,
        full_prompts: List[str],
        prompt_prefixes: List[str],
        gold_answers: List[str],
    ) -> List[float]:
        tokenizer = _get_tokenizer(self.config.model_name)

        # BPE attaches the leading space to the first answer token, so we count
        # tokens for `" " + answer` to slice the answer logprobs precisely.
        ans_prefixed = [" " + ans for ans in gold_answers]
        encs = tokenizer(ans_prefixed, add_special_tokens=False, padding=False)
        ans_token_counts: List[int] = [
            max(1, len(ids)) for ids in encs["input_ids"]
        ]

        payload = {
            "model": self.config.model_name,
            "prompt": full_prompts,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": self.config.temperature,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                r = self._session.post(
                    f"{self.config.url}/v1/completions",
                    json=payload,
                    timeout=self.config.timeout,
                )
                r.raise_for_status()
                data = r.json()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        if last_exc is not None:
            raise RuntimeError(
                f"Oracle call failed after {self.config.max_retries} retries: {last_exc}"
            ) from last_exc

        choices = data["choices"]
        out: List[float] = []
        for choice, n_ans in zip(choices, ans_token_counts):
            token_logprobs = choice["logprobs"]["token_logprobs"]
            answer_slice = token_logprobs[-n_ans:]
            ans_lps = [lp for lp in answer_slice if lp is not None]
            out.append(float(sum(ans_lps)))
        return out

    def close(self):
        self._baseline_cache.close()
        self._session.close()
