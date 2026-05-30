#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate patternbloom

echo "[02_build_hypergraph] Building hypergraph index from scratch."
echo "[02_build_hypergraph] Expected runtime: 2-4 hours on one H100 GPU."
echo "[02_build_hypergraph] Starting in 5 seconds (Ctrl-C to abort)..."
sleep 5

python -m patternbloom.data.build_hypergraph --config configs/hypergraph.yaml

echo "[02_build_hypergraph] Hypergraph index build complete."
