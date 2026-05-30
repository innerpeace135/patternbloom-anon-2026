"""BGE-large-en-v1.5 encoder for signature strings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np


_MODEL = None
_TOKENIZER = None
_DEVICE = None

_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_EMBEDDING_DIM = 1024

_CACHE: Dict[str, np.ndarray] = {}


def _lazy_load():
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = os.environ.get(
        "SIG_ENCODER_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME).to(device)
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    _DEVICE = device
    return model, tokenizer, device


def encode_signature(sig: str, dim: int = 1024) -> np.ndarray:
    """L2-normalized embedding of a signature string. Cached in-process."""
    if dim != _EMBEDDING_DIM:
        raise ValueError(
            f"{_MODEL_NAME} dim is {_EMBEDDING_DIM}, got {dim}"
        )

    if sig in _CACHE:
        return _CACHE[sig]

    import torch

    model, tokenizer, device = _lazy_load()
    with torch.no_grad():
        inputs = tokenizer(
            [sig],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        arr = emb.cpu().numpy().astype(np.float32)[0]

    _CACHE[sig] = arr
    return arr


def save_cache(path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(_CACHE.keys())
    arrays = {f"emb_{i}": _CACHE[k] for i, k in enumerate(keys)}
    meta = {"sigs": np.array(keys, dtype=object)}
    np.savez(path_obj, **arrays, **meta)


def load_cache(path: str) -> int:
    data = np.load(path, allow_pickle=True)
    sigs = data["sigs"].tolist()
    n = 0
    for i, sig in enumerate(sigs):
        _CACHE[sig] = data[f"emb_{i}"].astype(np.float32)
        n += 1
    return n


def cache_size() -> int:
    return len(_CACHE)
