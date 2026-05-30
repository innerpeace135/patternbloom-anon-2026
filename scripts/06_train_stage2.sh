#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

require_endpoint() {
    local name="$1" url="$2"
    if ! curl --max-time 5 -fsS "$url" >/dev/null 2>&1; then
        echo "[06_train_stage2] ERROR: ${name} not reachable at ${url}" >&2
        exit 1
    fi
}

require_path() {
    local name="$1" path="$2"
    if [[ ! -e "$path" ]]; then
        echo "[06_train_stage2] ERROR: ${name} not found at ${path}" >&2
        exit 1
    fi
}

API_URL="${API_URL:-http://localhost:8000/health}"
ORACLE_URL="${ORACLE_URL:-http://localhost:8100/health}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/stage1/best}"
GPM_MEMORY="${GPM_MEMORY:-data/gpm/memory.json}"

require_endpoint "retrieval API"  "${API_URL}"
require_endpoint "oracle service" "${ORACLE_URL}"
require_path     "Stage I best checkpoint" "${STAGE1_CKPT}"
require_path     "GPM memory file"         "${GPM_MEMORY}"

mkdir -p checkpoints/stage2
echo "[06_train_stage2] Stage II training started. Logs to checkpoints/stage2/log.txt."

python -m patternbloom.train.stage2 --config configs/stage2_par.yaml \
    2>&1 | tee checkpoints/stage2/log.txt
