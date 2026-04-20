"""
MVP 3 — Cumulative cluster training.

Phase 0: Train on Cluster 0 only          (infra-marginal)
Phase 1: Train on Cluster 1 only          (headroom, same weights)
Phase 2: Train on Cluster 2 only          (super-headroom, same weights)

Each phase uses a 10% mixing ratio of the previous cluster(s) to
guard against catastrophic forgetting.  The one-hot cluster indicator
in the observation acts as context — the policy routes behaviour
per cluster without interference.

Evaluation after each phase uses the FULL distribution so you can
watch all three cluster profit curves in TensorBoard simultaneously.

Cluster indicator reference:
  [1,0,0] → Cluster 0 (infra-marginal, demand < 300)
  [0,1,0] → Cluster 1 (headroom, demand >= 300, fringe online)
  [0,0,1] → Cluster 2 (super-headroom, fringe offline)

Run:
    python scripts/mvp_3_cumulative.py
    tensorboard --logdir runs/mvp3
"""

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM
from agents.td3 import TD3
from train.trainer import CumulativeTrainer

# ── Demand configuration ─────────────────────────────────────────────────────
# Slight negative shift so Cluster 1/2 events occur ~20-30% of the time
# in the full evaluation env — enough to evaluate both clusters but
# still a meaningful imbalance.
BASE_DEMAND_SHIFT = -40.0   # center=260 MW → agent zone reachable but not dominant

STEPS_PER_PHASE = 300_000

phases = [
    {
        "cluster": 0,
        "steps":   STEPS_PER_PHASE,
        "mixing":  None,                        # pure Cluster 0
    },
    {
        "cluster": 1,
        "steps":   STEPS_PER_PHASE,
        "mixing":  {0: 0.1, 1: 0.9},           # 90% Cluster 1, 10% Cluster 0
    },
    {
        "cluster": 2,
        "steps":   STEPS_PER_PHASE,
        "mixing":  {0: 0.1, 1: 0.1, 2: 0.8},  # 80% Cluster 2, 10% each previous
    },
]

base_env_kwargs = {
    "demand_center_shift": BASE_DEMAND_SHIFT,
    "noise_std_fraction":  0.05,
    "seed":                99,
}

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log        = f"runs/mvp3/{RUN_ID}",
    normalize_observations = True,
    normalize_rewards      = True,
    total_training_steps   = len(phases) * STEPS_PER_PHASE,
)

trainer = CumulativeTrainer(
    agent           = agent,
    base_env_kwargs = base_env_kwargs,
    phases          = phases,
    checkpoint_dir  = "checkpoints/mvp3",
    eval_episodes   = 300,
)

summaries = trainer.run()

print("\n=== CUMULATIVE TRAINING COMPLETE ===")
print("Compare mvp2 (naive rare) vs mvp3 (clustered) TensorBoard logs:")
print("  tensorboard --logdir runs/")
print("\nExpected in mvp3:")
print("  After Phase 0: cluster0 profit ≈ 0, cluster1/2 ≈ 0")
print("  After Phase 1: cluster0 ≈ 0, cluster1 profit > 0 (agent exploits headroom)")
print("  After Phase 2: all clusters show positive profit where opportunity exists")
