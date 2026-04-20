"""
MVP 0 — Infra-marginal only (0% headroom events).

Demand is shifted well below the 300 MW threshold so the agent is always
a price-taker earning zero profit.  Expected behaviour: agent converges
to bidding around its marginal cost (action ≈ 0 from tanh).

Run:
    python scripts/mvp_0_infra_marginal.py
    tensorboard --logdir runs/mvp0
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM
from agents.td3 import TD3
from train.trainer import train_phase, evaluate

TOTAL_STEPS      = 300_000
DEMAND_SHIFT     = -60.0   # pushes demand to ~240 MW mean → always below 300

env = DayAheadMarketEnv(
    demand_center_shift = DEMAND_SHIFT,
    force_cluster       = 0,
    seed                = 42,
)

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log = "runs/mvp0",
    normalize_observations = True,
    normalize_rewards      = True,
)

summary = train_phase(
    agent          = agent,
    env            = env,
    total_steps    = TOTAL_STEPS,
    phase_name     = "mvp0_infra",
    checkpoint_dir = "checkpoints/mvp0",
    eval_env       = DayAheadMarketEnv(demand_center_shift=DEMAND_SHIFT, force_cluster=0),
    eval_interval  = 50_000,
    eval_episodes  = 200,
)

print("\nFinal summary:", summary)
print("\nExpected: mean_profit_overall ≈ 0  (agent earns MC, no headroom to exploit)")
