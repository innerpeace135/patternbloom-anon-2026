"""Prepare PatternBloom train (HotpotQA + 2WikiMultiHopQA) and test (HotpotQA) splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import datasets
import pandas as pd


HF_REPO = "RUC-NLPIR/FlashRAG_datasets"
TRAIN_SUBSETS = ["hotpotqa", "2wikimultihopqa"]
TEST_SUBSET = "hotpotqa"


def _load_split(subset: str, split: str) -> datasets.Dataset:
    return datasets.load_dataset(HF_REPO, subset, split=split)


def _sample(
    ds: datasets.Dataset, n: int, seed: int
) -> datasets.Dataset:
    if len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _to_records(ds: datasets.Dataset, source: str) -> List[dict]:
    records = []
    for ex in ds:
        records.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "golden_answers": ex.get("golden_answers", []),
                "metadata": ex.get("metadata", {}),
                "source": source,
            }
        )
    return records


def prepare(
    output_dir: str,
    seed_train: int = 42,
    seed_test: int = 40,
    n_per_dataset: int = 7000,
    n_test: int = 1000,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_records: List[dict] = []
    for subset in TRAIN_SUBSETS:
        print(f"Loading {subset} train split...")
        ds = _load_split(subset, "train")
        ds = _sample(ds, n_per_dataset, seed_train)
        train_records.extend(_to_records(ds, subset))

    train_df = pd.DataFrame(train_records)
    train_df = train_df.sample(frac=1.0, random_state=seed_train).reset_index(
        drop=True
    )
    train_path = out / "train.parquet"
    train_df.to_parquet(train_path, index=False)
    print(f"Train: {len(train_df)} rows -> {train_path}")

    print(f"Loading {TEST_SUBSET} test split...")
    test_ds = _load_split(TEST_SUBSET, "test")
    test_ds = _sample(test_ds, n_test, seed_test)
    test_records = _to_records(test_ds, TEST_SUBSET)
    test_df = pd.DataFrame(test_records)
    test_path = out / "hotpotqa_test.parquet"
    test_df.to_parquet(test_path, index=False)
    print(f"Test: {len(test_df)} rows -> {test_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PatternBloom train/test splits from FlashRAG"
    )
    parser.add_argument("--seed_train", type=int, default=42)
    parser.add_argument("--seed_test", type=int, default=40)
    parser.add_argument("--n_per_dataset", type=int, default=7000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    prepare(
        output_dir=args.output_dir,
        seed_train=args.seed_train,
        seed_test=args.seed_test,
        n_per_dataset=args.n_per_dataset,
        n_test=args.n_test,
    )


if __name__ == "__main__":
    main()
