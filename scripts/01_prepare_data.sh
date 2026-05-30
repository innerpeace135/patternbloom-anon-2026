#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

# Activate conda env via shell hook so 'conda activate' works in non-interactive bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"

echo "[01_prepare_data] Building 14K balanced training subsample -> ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python -m patternbloom.data.prepare_dataset --output_dir "${OUTPUT_DIR}"

TRAIN_COUNT=$(find "${OUTPUT_DIR}" -name 'train*.parquet' -o -name 'train*.jsonl' 2>/dev/null | xargs -I{} wc -l {} 2>/dev/null | tail -1 | awk '{print $1}' || echo "unknown")
echo "[01_prepare_data] Done. Files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"
