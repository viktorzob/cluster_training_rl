"""
MVP 3b — Ablation: curriculum training WITHOUT cluster indicator.

Identical curriculum to MVP3 (C1→C3→C2 phases, same mixing ratios) but
obs = 24h demand only (no cluster one-hot, obs_dim=24).

This isolates what the cluster indicator contributes ON TOP OF curriculum.

The critical conflict: Phase 0 teaches bid-HIGH in hours 13-15 (C1).
Phase 1 teaches bid-LOW in hours 13-15 (C3). Without the indicator the
network cannot route differently — it must overwrite the C1 policy with
C3 or find a compromise that succeeds at neither.

With 20% C1 mixing in Phase 1 the network is pulled in both directions
simultaneously. Without the one-hot to gate behaviour, this is a direct
weight conflict: same input features, opposite target outputs.

Expected result: Phase 1 shows catastrophic forgetting of C1 profit or
failure to learn C3 (or both). Phase 2 profit also degraded.
Compare in TensorBoard: mvp3b < mvp2 ≈ mvp3 is the expected ordering.

Three-way ablation table:
  mvp2b  (no indicator, diluted)        — baseline failure
  mvp2   (indicator, diluted)           — partial success
  mvp3b  (curriculum, no indicator)     — curriculum helps but indicator needed
  mvp3   (curriculum + indicator)       — full success

Run:
    python scripts/mvp_3b_curriculum_no_feature.py
    tensorboard --logdir runs/
"""

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

from env.market import DayAheadMarketEnv, ACTION_DIM
from agents.td3 import TD3
from train.trainer import CumulativeTrainer

BASE_DEMAND_SHIFT  = -20.0
STEPS_PER_PHASE    = 300_000
OBS_DIM_NO_FEATURE = 24

phases = [
    {
        "cluster": 1,
        "steps":   STEPS_PER_PHASE,
        "mixing":  None,
    },
    {
        "cluster": 3,
        "steps":   STEPS_PER_PHASE,
        "mixing":  {1: 0.2, 3: 0.8},
    },
    {
        "cluster": 2,
        "steps":   STEPS_PER_PHASE,
        "mixing":  {1: 0.1, 3: 0.1, 2: 0.8},
    },
]

base_env_kwargs = {
    "demand_center_shift":       BASE_DEMAND_SHIFT,
    "noise_std_fraction":        0.05,
    "seed":                      99,
    "include_cluster_indicator": False,   # no one-hot in obs
}

agent = TD3(
    obs_dim    = OBS_DIM_NO_FEATURE,
    action_dim = ACTION_DIM,
    tensorboard_log        = f"runs/mvp3b/{RUN_ID}",
    normalize_observations = True,
    normalize_rewards      = True,
    total_training_steps   = len(phases) * STEPS_PER_PHASE,
)

trainer = CumulativeTrainer(
    agent           = agent,
    base_env_kwargs = base_env_kwargs,
    phases          = phases,
    checkpoint_dir  = "checkpoints/mvp3b",
    eval_episodes   = 300,
)

summaries = trainer.run()

print("\n=== MVP3b COMPLETE (curriculum, no cluster indicator) ===")
print("Expected: C1/C3 profit conflict → forgetting or compromise policy.")
print("Compare: mvp2b ≤ mvp3b < mvp2 < mvp3 is the ideal ordering.")
