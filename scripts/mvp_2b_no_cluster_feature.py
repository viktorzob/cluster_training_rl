"""
MVP 2b — Ablation: same as MVP2 but WITHOUT cluster indicator in observation.

Identical setup to MVP2:
  - Same total steps (900k), same demand distribution, same cluster mix
  - Same TD3 hyperparameters, same random seed

Only difference: obs = 24h demand only (OBS_DIM=24, no cluster one-hot).
The agent cannot distinguish which cluster it is in.

This ablation answers the question:
  "Does the cluster indicator in MVP2 actually help, or is MVP2 succeeding
   because demand patterns alone are sufficient to identify the cluster?"

Expected result: similar or worse profit vs MVP2.
  - If similar → cluster indicator adds little; the demand profile alone
    encodes enough cluster context (thesis weakened)
  - If clearly worse → cluster indicator is valuable even in diluted training;
    and MVP3's advantage is structural (curriculum), not just the indicator
  - Compare all three in TensorBoard: mvp2b ≤ mvp2 << mvp3 is the ideal result

Run:
    python scripts/mvp_2b_no_cluster_feature.py
    tensorboard --logdir runs/
"""

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

from env.market import DayAheadMarketEnv, ACTION_DIM
from agents.td3 import TD3
from train.trainer import train_phase

TOTAL_STEPS  = 900_000
DEMAND_SHIFT = -20.0
RARE_MIX     = {0: 0.70, 1: 0.10, 2: 0.10, 3: 0.10}
OBS_DIM_NO_FEATURE = 24   # demand only, no cluster one-hot

env = DayAheadMarketEnv(
    demand_center_shift      = DEMAND_SHIFT,
    cluster_mixing_ratio     = RARE_MIX,
    seed                     = 1,
    include_cluster_indicator = False,
)

agent = TD3(
    obs_dim    = OBS_DIM_NO_FEATURE,
    action_dim = ACTION_DIM,
    tensorboard_log        = f"runs/mvp2b/{RUN_ID}",
    normalize_observations = True,
    normalize_rewards      = True,
    total_training_steps   = TOTAL_STEPS,
)

eval_env = DayAheadMarketEnv(
    demand_center_shift      = DEMAND_SHIFT,
    cluster_mixing_ratio     = RARE_MIX,
    include_cluster_indicator = False,
)

summary = train_phase(
    agent          = agent,
    env            = env,
    total_steps    = TOTAL_STEPS,
    phase_name     = "mvp2b_no_feature",
    checkpoint_dir = "checkpoints/mvp2b",
    eval_env       = eval_env,
    eval_interval  = 50_000,
    eval_episodes  = 500,
)

print("\nFinal summary:", summary)
print("\nCompare against mvp2 (same setup + cluster indicator) and mvp3 (curriculum).")
print("Key metric: mean_profit_c1 / c2 / c3 on the same test distribution.")
