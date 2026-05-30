# verl_fork

This directory contains the PatternBloom integration patches for the
`verl` reinforcement-learning trainer. The patches add stage-aware reward
dispatch (Stage I IDR, Stage II PAR) and wire the multi-turn agent
rollout to the retrieval API.

## Files

```
verl_fork/
├── README.md
└── verl/
    ├── trainer/
    │   └── main_ppo.py                       # PPO entry with stage-aware reward
    └── utils/
        └── reward_score/
            ├── __init__.py                   # Reward dispatch registry
            └── qa_em_and_format.py           # Answer EM/F1 and format reward
```

## Installation

The patches overlay an upstream verl installation. Install the upstream
package first, then copy these files into its package directory.

```bash
pip install verl
python - <<'PY'
import os, shutil, verl
target = os.path.dirname(verl.__file__)
src = os.path.join(os.path.dirname(__file__), "verl_fork", "verl")
for root, dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    dst = os.path.join(target, rel) if rel != "." else target
    os.makedirs(dst, exist_ok=True)
    for name in files:
        shutil.copy2(os.path.join(root, name), os.path.join(dst, name))
PY
```

You can also run the same overlay manually from the repo root:

```bash
SITE_VERL=$(python -c "import verl, os; print(os.path.dirname(verl.__file__))")
cp -r verl_fork/verl/* "$SITE_VERL/"
```

Re-run the overlay any time upstream verl is upgraded. The patches are
written against verl 0.2 and depend only on stable public interfaces
(`verl.trainer.ppo.ray_trainer`, `verl.DataProto`, `verl.utils`).

## Reward dispatch

`verl.utils.reward_score.__init__` exposes three reward entry points:

* `_default_compute_score_format_answer` — the baseline format-and-F1
  reward (`reward_stage=0`).
* `_compute_stage1_reward` — the Information-Density Reward
  (`reward_stage=1`); requires the oracle service to be running.
* `_compute_stage2_reward` — the Pattern-Augmented Reward
  (`reward_stage=2`); requires the Graph Pattern Memory and the
  retrieval API to be running.

Stage 1 and Stage 2 import the corresponding reward implementations from
the project package (`patternbloom.reward.idr` and
`patternbloom.reward.par`) via a thin adapter at runtime. The adapter is
installed alongside this overlay; see `verl/utils/reward_score/path_reward.py`
after installation for the import wiring.

## Agent rollout

The multi-turn rollout (think, query, graph, answer) is registered as a
verl tool environment through `agent.tool`. The PatternBloom package
ships its own `agent/` namespace which the patched `main_ppo.py`
imports; ensure the project is on the Python path
(`pip install -e .` from the repo root).
