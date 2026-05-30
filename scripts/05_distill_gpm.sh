#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

require_path() {
    local name="$1" path="$2"
    if [[ ! -e "$path" ]]; then
        echo "[05_distill_gpm] ERROR: ${name} not found at ${path}" >&2
        exit 1
    fi
}

CKPT="${STAGE1_CKPT:-checkpoints/stage1/best}"
OUTPUT="${GPM_OUTPUT:-data/gpm/memory.json}"
IDR_THRESHOLD="${IDR_THRESHOLD:-0.85}"
MERGE_THRESHOLD="${MERGE_THRESHOLD:-0.85}"

require_path "Stage I best checkpoint" "${CKPT}"

mkdir -p "$(dirname "${OUTPUT}")"

python -m patternbloom.gpm.distill \
    --checkpoint "${CKPT}" \
    --output "${OUTPUT}" \
    --idr_threshold "${IDR_THRESHOLD}" \
    --merge_threshold "${MERGE_THRESHOLD}"

# Pattern count is reported by the distill module via stdout; surface it here too
PATTERN_COUNT=$(python -c "import json; print(len(json.load(open('${OUTPUT}'))))" 2>/dev/null || echo "?")
echo "[05_distill_gpm] Distilled ${PATTERN_COUNT} patterns to ${OUTPUT}"
