#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

MODEL="${ORACLE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${ORACLE_PORT:-8100}"
GPU="${ORACLE_GPU:-0}"

echo "[04a_start_oracle] Launching frozen oracle service."
echo "[04a_start_oracle] Model: ${MODEL}"
echo "[04a_start_oracle] Oracle service listening on http://localhost:${PORT}"

exec python -m patternbloom.train.oracle_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --gpu "${GPU}"
