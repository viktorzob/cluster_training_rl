"""
MVP 2 — Rare headroom: ~10% of episodes contain Cluster 1/2 events.

Demand is shifted slightly below 300 so most hours are infra-marginal.
This is the BASELINE FAILURE case: agent sees so few headroom events
that it never learns to exploit them, even though the profit opportunity
is large.

Watch the cluster1 / cluster2 profit curves in TensorBoard —
they should stay flat near 0 while cluster0 profit converges.

Run:
    python scripts/mvp_2_rare_10pct.py
    tensorboard --logdir runs/mvp2
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM
from agents.td3 import TD3
from train.trainer import train_phase

TOTAL_STEPS  = 500_000
# demand_center_shift = -20 → ~10-15% of hours above 300
DEMAND_SHIFT = -20.0

env = DayAheadMarketEnv(
    demand_center_shift = DEMAND_SHIFT,
    seed                = 1,
)

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log = "runs/mvp2",
    normalize_observations = True,
    normalize_rewards      = True,
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
    eval_episodes  = 500,   # more episodes to capture rare-event stats
)

print("\nFinal summary:", summary)
print("\nExpected: cluster1/2 profit remains near 0 despite large potential gain.")
print("This failure motivates the cumulative cluster training in mvp_3.")
