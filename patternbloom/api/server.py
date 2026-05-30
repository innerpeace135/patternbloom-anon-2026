"""Retrieval API server backed by a FAISS hypergraph index."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path
from typing import List

import faiss

faiss.omp_set_num_threads(1)
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from omegaconf import OmegaConf
from FlagEmbedding import FlagAutoModel


class SearchRequest(BaseModel):
    queries: List[str]
    top_k: int = 5


class SearchResponse(BaseModel):
    results: List[List[dict]]


class AnchorRecallRequest(BaseModel):
    queries: List[str]
    anchor_entities: List[List[str]]
    top_k: int = 10


def _load_index(index_dir: Path):
    chunks_path = index_dir / "hyperedges.jsonl"
    faiss_path = index_dir / "faiss.index"
    embeddings_path = index_dir / "embeddings.npy"
    missing = [p for p in [chunks_path, faiss_path, embeddings_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Hypergraph index incomplete. Missing: " + ", ".join(str(p) for p in missing)
        )
    chunks = []
    with chunks_path.open() as f:
        for line in f:
            chunks.append(json.loads(line))
    index = faiss.read_index(str(faiss_path))
    embeddings = np.load(str(embeddings_path))
    return chunks, index, embeddings


def build_app(config_path: str) -> FastAPI:
    cfg = OmegaConf.load(config_path)
    index_dir = Path(cfg.index.hypergraph_dir)
    chunks, index, embeddings = _load_index(index_dir)
    encoder = FlagAutoModel.from_finetuned(
        cfg.retriever.encoder_name,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        devices=cfg.retriever.device,
    )
    default_top_k = int(cfg.retriever.top_k)
    anchor_top_k = int(cfg.retriever.anchor_recall_top_k)

    app = FastAPI(title="PatternBloom Retrieval API")

    @app.get("/health")
    def health():
        return {"status": "ok", "num_hyperedges": len(chunks)}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        k = req.top_k or default_top_k
        query_embeddings = encoder.encode_queries(req.queries)
        scores, indices = index.search(query_embeddings, k)
        results = []
        for row_scores, row_indices in zip(scores, indices):
            hits = []
            for s, i in zip(row_scores.tolist(), row_indices.tolist()):
                if i < 0:
                    continue
                hit = dict(chunks[i])
                hit["score"] = float(s)
                hits.append(hit)
            results.append(hits)
        return SearchResponse(results=results)

    @app.post("/anchor_recall", response_model=SearchResponse)
    def anchor_recall(req: AnchorRecallRequest):
        k = req.top_k or anchor_top_k
        query_embeddings = encoder.encode_queries(req.queries)
        scores, indices = index.search(query_embeddings, k * 2)
        results = []
        for row_scores, row_indices, anchors in zip(scores, indices, req.anchor_entities):
            anchor_set = {a.lower() for a in anchors}
            hits = []
            for s, i in zip(row_scores.tolist(), row_indices.tolist()):
                if i < 0:
                    continue
                chunk = chunks[i]
                entities = {e.lower() for e in chunk.get("entities", [])}
                bonus = len(anchor_set & entities)
                hit = dict(chunk)
                hit["score"] = float(s) + 0.1 * bonus
                hits.append(hit)
            hits.sort(key=lambda h: h["score"], reverse=True)
            results.append(hits[:k])
        return SearchResponse(results=results)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/api.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    host = args.host or cfg.server.host
    port = args.port or int(cfg.server.port)
    app = build_app(args.config)
    uvicorn.run(app, host=host, port=port, workers=int(cfg.server.workers))


if __name__ == "__main__":
    main()
