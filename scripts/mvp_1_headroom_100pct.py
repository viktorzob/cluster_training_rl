"""
MVP 1 — 100% headroom events (Cluster 1 only).

Every episode has demand above 300 MW; fringe clears at ~100 €/MWh.
Agent can always earn up to 50 €/MWh profit by setting price below 100.
Expected behaviour: agent converges to bidding just below 100 €/MWh.

Run:
    python scripts/mvp_1_headroom_100pct.py
    tensorboard --logdir runs/mvp1
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM
from agents.td3 import TD3
from train.trainer import train_phase

TOTAL_STEPS  = 300_000
DEMAND_SHIFT = +40.0   # shifts demand to ~340 MW mean → reliably above 300

env = DayAheadMarketEnv(
    demand_center_shift = DEMAND_SHIFT,
    force_cluster       = 1,
    seed                = 0,
)

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log = "runs/mvp1",
    normalize_observations = True,
    normalize_rewards      = True,
)

summary = train_phase(
    agent          = agent,
    env            = env,
    total_steps    = TOTAL_STEPS,
    phase_name     = "mvp1_headroom",
    checkpoint_dir = "checkpoints/mvp1",
    eval_env       = DayAheadMarketEnv(demand_center_shift=DEMAND_SHIFT, force_cluster=1),
    eval_interval  = 50_000,
    eval_episodes  = 200,
)

print("\nFinal summary:", summary)
print("\nExpected: mean_profit_overall ≈ 50 * 50 (per dispatched hour) × ~24 hours")
