#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

require_endpoint() {
    local name="$1" url="$2"
    if ! curl --max-time 5 -fsS "$url" >/dev/null 2>&1; then
        echo "[07_evaluate] ERROR: ${name} not reachable at ${url}" >&2
        exit 1
    fi
}

require_path() {
    local name="$1" path="$2"
    if [[ ! -e "$path" ]]; then
        echo "[07_evaluate] ERROR: ${name} not found at ${path}" >&2
        exit 1
    fi
}

API_URL="${API_URL:-http://localhost:8000/health}"
STAGE2_CKPT="${STAGE2_CKPT:-checkpoints/stage2/best}"
GPM_MEMORY="${GPM_MEMORY:-data/gpm/memory.json}"
METRICS_FILE="${METRICS_FILE:-outputs/eval/metrics.json}"

require_endpoint "retrieval API" "${API_URL}"
require_path "Stage II best checkpoint" "${STAGE2_CKPT}"
require_path "GPM memory file"          "${GPM_MEMORY}"

mkdir -p "$(dirname "${METRICS_FILE}")"

python -m patternbloom.eval.run_eval --config configs/eval.yaml

if [[ -f "${METRICS_FILE}" ]]; then
    echo "[07_evaluate] Final metrics (${METRICS_FILE}):"
    cat "${METRICS_FILE}"
else
    echo "[07_evaluate] WARNING: expected metrics file not found at ${METRICS_FILE}" >&2
    exit 1
fi
