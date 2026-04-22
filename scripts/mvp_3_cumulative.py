
"""
MVP 3 — Cumulative cluster training.

Phase 0: Train on Cluster 1 only          (headroom: bid HIGH, earn margin)
Phase 1: Train on Cluster 2 (+ 20% C1)   (super-headroom: bid up to 150)
Phase 2: Train on Cluster 3 (+ 20% each) (redispatch: bid BELOW MC)

Starting from C1 (not C0) ensures the agent has a positive reward signal
from step 1. C0 (agent always idle) provides no gradient — it is only
useful as a benchmark comparison, not as a training phase.

Cluster indicator reference (one-hot in obs):
  [1,0,0,0] → Cluster 0 (infra-marginal, agent idle — appears only in eval)
  [0,1,0,0] → Cluster 1 (headroom, agent zone 250-300 MW, bid HIGH ≤ 100)
  [0,0,1,0] → Cluster 2 (fringe offline, bid HIGH ≤ 150 even on low-demand days)
  [0,0,0,1] → Cluster 3 (redispatch hours 13-15, bid BELOW MC for 160 €/MWh)

Key thesis: C2 looks like C0 from the demand profile alone (fringe offline days
can have low demand) — the one-hot cluster indicator is what lets the agent
route to bid-high vs bid-low vs idle, proving that cluster context is necessary.

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
# Eval env uses a moderate shift so all 4 clusters appear during evaluation.
# Training phases use force_cluster / mixing, so this mainly controls eval rarity.
BASE_DEMAND_SHIFT = -20.0   # center=280 MW → agent zone frequent, all clusters reachable

STEPS_PER_PHASE = 300_000

phases = [
    {
        "cluster": 1,
        "steps":   STEPS_PER_PHASE,
        "mixing":  None,                        # pure C1: bid HIGH in hours 13-15
    },
    {
        "cluster": 3,
        "steps":   STEPS_PER_PHASE,
        "mixing":  {1: 0.2, 3: 0.8},           # 80% C3, 20% C1 anti-forgetting
        # KEY TRANSITION: C3 requires bid LOW in hours 13-15 — opposite of C1.
        # Without cluster indicator the policy conflicts; with it the network routes
        # C1-onehot→bid-high and C3-onehot→bid-low simultaneously.
    },
    {
        "cluster": 2,
        "steps":   STEPS_PER_PHASE,
        "mixing":  {1: 0.1, 3: 0.1, 2: 0.8},  # 80% C2, maintain C1+C3 skills
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
print("  Phase 0 end: C1 profit HIGH, C2/C3 not yet learned")
print("  Phase 1 end: C1 maintained, C2 profit HIGH (bid up to 150)")
print("  Phase 2 end: C3 profit HIGH (redispatch), C1/C2 maintained")
print("  vs mvp2: all cluster profits LOW despite cluster indicator in obs")
