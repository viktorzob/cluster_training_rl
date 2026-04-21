"""
MVP 2 — Diluted rare events: 10% each of C1/C2/C3, 70% C0 (idle).

Control condition for the paper thesis: same training budget as MVP3,
same cluster one-hot in observation, same profit potential per cluster —
but rare clusters appear only 10% of the time (diluted in replay buffer).

Why this fails (the thesis):
  TD3 samples replay buffer uniformly. With 70% C0 (zero-reward) transitions,
  every gradient batch is ~70% uninformative. The critic's C1/C2/C3 Q-values
  receive weak, infrequent updates and never converge to the correct action.
  This is REPLAY BUFFER DILUTION — not data quantity.

  Compare MVP3: Phase 0 fills the buffer 100% with C1 → every gradient step
  is a C1 gradient → fast, clean convergence. Same budget, very different result.

Key controls (everything identical to MVP3 except training structure):
  - Same total steps (900k = 3 × 300k)
  - Same demand environment: shift=-20, C1/C2/C3 episodes use shift_map offsets
    so C1 center=280 MW (rich residuals, large profit potential)
  - Same observation: 24h demand + 4-dim cluster one-hot
  - Same TD3 hyperparameters

Expected result:
  mean_profit_c1 / c2 / c3 all remain near 0 despite cluster indicator
  being present and profit potential being large.

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

# Same total budget as MVP3 (3 × 300k phases)
TOTAL_STEPS  = 900_000

# Base shift -20: C1 force-generation uses shift_map[1]=0 → total=-20 → center=280 MW
# C0 force-generation uses shift_map[0]=-100 → total=-120 → center=180 MW (agent idle)
# C2 uses shift_map[2]=-30 → total=-50 → center=250 MW (fringe-shutdown conditions)
# C3 uses shift_map[3]=-20 → total=-40 → center=260 MW (redispatch conditions)
DEMAND_SHIFT = -20.0

# 10% each rare cluster — each rare cluster has same profit potential as in MVP3
RARE_MIX = {0: 0.70, 1: 0.10, 2: 0.10, 3: 0.10}

env = DayAheadMarketEnv(
    demand_center_shift  = DEMAND_SHIFT,
    cluster_mixing_ratio = RARE_MIX,
    seed                 = 1,
)

agent = TD3(
    obs_dim    = OBS_DIM,
    action_dim = ACTION_DIM,
    tensorboard_log        = f"runs/mvp2/{RUN_ID}",
    normalize_observations = True,
    normalize_rewards      = True,
    total_training_steps   = TOTAL_STEPS,
)

eval_env = DayAheadMarketEnv(
    demand_center_shift  = DEMAND_SHIFT,
    cluster_mixing_ratio = RARE_MIX,
)

summary = train_phase(
    agent          = agent,
    env            = env,
    total_steps    = TOTAL_STEPS,
    phase_name     = "mvp2_rare",
    checkpoint_dir = "checkpoints/mvp2",
    eval_env       = eval_env,
    eval_interval  = 50_000,
    eval_episodes  = 500,
)

print("\nFinal summary:", summary)
print(f"\nExpected: mean_profit_c1/c2/c3 all near 0 despite cluster indicator present.")
print("C0 rare mix: 10% each → ~90k C1 episodes in 900k steps (diluted in replay buffer).")
print("Compare MVP3: ~300k concentrated C1 episodes in Phase 0 → same budget, better result.")
