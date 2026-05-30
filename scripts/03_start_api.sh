#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

echo "[03_start_api] Starting retrieval API on port 8000. Press Ctrl-C to stop."

# Pass through any extra args (e.g. --port 8001 --host 0.0.0.0) to the server
exec python -m patternbloom.api.server --config configs/api.yaml "$@"
