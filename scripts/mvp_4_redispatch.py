"""
MVP 4 — Redispatch arbitrage (Cluster 3 only, 100% frequency).

In hours 11, 12, 13: if demand is in agent zone (250-300 MW) AND agent
bids BELOW marginal cost (action < 0 from tanh), the grid operator pays
a redispatch price of 160 €/MWh for the agent's full 50 MW capacity.

This requires the OPPOSITE action to Cluster 1:
  Cluster 1 → bid HIGH (near 100) to capture headroom
  Cluster 3 → bid LOW (below 50 MC) in hours 11-13 to capture redispatch

Expected learning: agent discovers that for hours 11-13 on Cluster 3 days,
negative tanh actions (prices below MC) yield the maximum profit.

Watch TensorBoard:
  rollout/profit_cluster3  → should rise to ~(160-50)*50 * 3h = 16500 €/day

Run:
    python scripts/mvp_4_redispatch.py
    tensorboard --logdir runs/mvp4
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM, REDISPATCH_HOURS, REDISPATCH_PRICE, AGENT_MC, AGENT_CAPACITY
from agents.td3 import TD3
from train.trainer import train_phase

print(f"Redispatch hours: {REDISPATCH_HOURS}")
print(f"Redispatch price: {REDISPATCH_PRICE} €/MWh")
print(f"Max redispatch profit: {(REDISPATCH_PRICE - AGENT_MC) * AGENT_CAPACITY * len(REDISPATCH_HOURS):.0f} €/day")
print(f"OBS_DIM: {OBS_DIM}  (24h demand + {OBS_DIM - 24}-cluster indicator)")

TOTAL_STEPS  = 300_000
DEMAND_SHIFT = -20.0  # center=280 MW → hours 13-15 land at 293-299 MW (agent zone 250-300)

env = DayAheadMarketEnv(
    demand_center_shift = DEMAND_SHIFT,
    force_cluster       = 3,
    seed                = 10,
)

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log        = "runs/mvp4",
    normalize_observations = True,
    normalize_rewards      = True,
)

summary = train_phase(
    agent          = agent,
    env            = env,
    total_steps    = TOTAL_STEPS,
    phase_name     = "mvp4_redispatch",
    checkpoint_dir = "checkpoints/mvp4",
    eval_env       = DayAheadMarketEnv(demand_center_shift=DEMAND_SHIFT, force_cluster=3),
    eval_interval  = 50_000,
    eval_episodes  = 200,
)

print("\nFinal summary:", summary)
print(f"\nExpected: mean_profit_cluster3 ≈ {(REDISPATCH_PRICE - AGENT_MC) * AGENT_CAPACITY * len(REDISPATCH_HOURS):.0f} €/day")
print("If profit stays near 0: agent failed to discover that bidding below MC earns redispatch payment.")
