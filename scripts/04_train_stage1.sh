#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

require_endpoint() {
    local name="$1" url="$2"
    if ! curl --max-time 5 -fsS "$url" >/dev/null 2>&1; then
        echo "[04_train_stage1] ERROR: ${name} not reachable at ${url}" >&2
        echo "[04_train_stage1] Start it first (see scripts/03_start_api.sh, scripts/04a_start_oracle.sh)." >&2
        exit 1
    fi
}

API_URL="${API_URL:-http://localhost:8000/health}"
ORACLE_URL="${ORACLE_URL:-http://localhost:8100/health}"

require_endpoint "retrieval API" "${API_URL}"
require_endpoint "oracle service" "${ORACLE_URL}"

mkdir -p checkpoints/stage1
echo "[04_train_stage1] Stage I training started. Logs to checkpoints/stage1/log.txt."

python -m patternbloom.train.stage1 --config configs/stage1_idr.yaml \
    2>&1 | tee checkpoints/stage1/log.txt
