"""
MVP 2 — Rare headroom: ~5% of episodes contain meaningful Cluster 1/2 hours.

Demand center shifted well below 300 so agent-zone hours are very rare.
Even though the cluster one-hot IS in the observation (agent can distinguish
clusters), the signal is too weak for TD3 to learn the correct bid strategy.

This is the BASELINE FAILURE case for the paper:
  - Agent sees the C1/C2 one-hot indicator but almost never experiences it
  - Even a globally high bid would take too long to discover because
    the reward signal from rare agent-zone hours is diluted across episodes
    dominated by idle (demand < 250 MW) hours
  - Compare against mvp3: same observation space, vastly better cluster profits

Watch the cluster1 / cluster2 profit curves in TensorBoard —
they should stay near 0 while mvp3 converges to high cluster profits.

Run:
    python scripts/mvp_2_rare_10pct.py
    tensorboard --logdir runs/mvp2
"""

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM
from agents.td3 import TD3
from train.trainer import train_phase

TOTAL_STEPS  = 500_000
# demand_center_shift = -80 → center=220 MW, peaks ≈ 264 MW
# → ~5% of episodes have any agent-zone hours (very weak gradient signal)
DEMAND_SHIFT = -80.0

env = DayAheadMarketEnv(
    demand_center_shift = DEMAND_SHIFT,
    seed                = 1,
)

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log        = f"runs/mvp2/{RUN_ID}",
    normalize_observations = True,
    normalize_rewards      = True,
    total_training_steps   = TOTAL_STEPS,
)

eval_env = DayAheadMarketEnv(demand_center_shift=DEMAND_SHIFT)

summary = train_phase(
    agent          = agent,
    env            = env,
    total_steps    = TOTAL_STEPS,
    phase_name     = "mvp2_rare",
    checkpoint_dir = "checkpoints/mvp2",
    eval_env       = eval_env,
    eval_interval  = 50_000,
    eval_episodes  = 500,   # many episodes needed to capture rare-event stats
)

print("\nFinal summary:", summary)
print("\nExpected: cluster1/2 profit remains near 0 despite cluster indicator in obs.")
print("This failure motivates the cumulative cluster training in mvp_3.")
