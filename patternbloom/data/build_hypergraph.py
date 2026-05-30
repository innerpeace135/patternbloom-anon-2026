"""Build the hypergraph index from a Wikipedia corpus."""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import OmegaConf
from datasets import load_dataset
from FlagEmbedding import FlagAutoModel
from openai import OpenAI
import faiss


EXTRACTOR_PROMPT = (
    "Extract n-ary facts as JSON from the passage below. Each fact is an "
    "object with fields 'entities' (list of canonical entity strings) and "
    "'relation' (short relation name). Return a JSON array; if no facts are "
    "extractable return []. Passage:\n\n{passage}\n"
)


def chunk_passage(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
    return chunks


def extract_hyperedges(client: OpenAI, model: str, passage: str, temperature: float):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": EXTRACTOR_PROMPT.format(passage=passage)}],
        temperature=temperature,
    )
    text = response.choices[0].message.content or "[]"
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        facts = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(facts, list):
        return []
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hypergraph.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    output_dir = Path(cfg.index.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.jsonl"
    hyperedges_path = output_dir / "hyperedges.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    faiss_path = output_dir / "faiss.index"

    extractor = OpenAI(api_key=cfg.extractor.api_key, base_url=cfg.extractor.endpoint)

    chunk_counter = 0
    edge_records = []
    with chunks_path.open("w") as chunk_out, hyperedges_path.open("w") as edge_out:
        for split in cfg.corpus.splits:
            dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", split.lower(), split="train")
            for row in dataset:
                title = row.get("title", "")
                context = row.get("context", "") or row.get("passage", "") or ""
                if not context:
                    continue
                for chunk_text in chunk_passage(context, cfg.corpus.chunk_size, cfg.corpus.chunk_overlap):
                    chunk_id = f"{split}-{chunk_counter}"
                    chunk_counter += 1
                    chunk_out.write(json.dumps({"id": chunk_id, "title": title, "text": chunk_text}) + "\n")
                    facts = extract_hyperedges(
                        extractor, cfg.extractor.model_name, chunk_text, cfg.extractor.temperature
                    )
                    for fact in facts:
                        entities = fact.get("entities", []) if isinstance(fact, dict) else []
                        relation = fact.get("relation", "") if isinstance(fact, dict) else ""
                        if not entities:
                            continue
                        record = {
                            "chunk_id": chunk_id,
                            "title": title,
                            "entities": entities,
                            "relation": relation,
                            "content": chunk_text,
                        }
                        edge_records.append(record)
                        edge_out.write(json.dumps(record) + "\n")

    if not edge_records:
        raise RuntimeError("No hyperedges extracted; check extractor endpoint and prompt.")

    encoder = FlagAutoModel.from_finetuned(
        cfg.encoder.model_name,
        query_instruction_for_retrieval="",
        devices=cfg.encoder.device,
    )
    contents = [rec["content"] for rec in edge_records]
    embeddings = encoder.encode_corpus(contents, batch_size=int(cfg.encoder.batch_size))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if bool(cfg.encoder.normalize):
        faiss.normalize_L2(embeddings)
    np.save(str(embeddings_path), embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if cfg.index.metric == "cosine" else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_path))

    print(f"Wrote {chunk_counter} chunks and {len(edge_records)} hyperedges to {output_dir}.")


if __name__ == "__main__":
    main()
