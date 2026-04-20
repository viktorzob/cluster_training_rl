"""
Final benchmark script — three experimental dimensions:

1. RARITY SWEEP (base + one non-base cluster at varying frequency)
   Tests: at what rarity does the agent start failing to learn a cluster skill?
   Runs: Base(C0) + C1 at [100%, 50%, 20%, 10%, 5%, 1%]
         Base(C0) + C2 at [100%, 50%, 20%, 10%, 5%, 1%]
         Base(C0) + C3 at [100%, 50%, 20%, 10%, 5%, 1%]

2. ALL-CLUSTER MIX — three training conditions:
   A. No cluster indicator  (obs = 24h demand only, no one-hot)
   B. Cluster indicator, naive joint training
   C. Cumulative cluster training (C0→C1→C2→C3), then eval on mix

3. Results logged to TensorBoard under runs/benchmark/<experiment_name>

Usage:
    # Run a single rarity sweep for Cluster 1:
    python scripts/final_benchmark.py --mode rarity --target_cluster 1

    # Run all rarity sweeps:
    python scripts/final_benchmark.py --mode rarity --target_cluster all

    # Run all-cluster mix benchmark (all 3 conditions):
    python scripts/final_benchmark.py --mode mix

    # Run everything:
    python scripts/final_benchmark.py --mode all
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM, N_CLUSTERS
from agents.td3 import TD3
from train.trainer import train_phase, evaluate, CumulativeTrainer

# ── Experiment configuration ──────────────────────────────────────────────────

STEPS_RARITY   = 400_000    # steps per rarity-sweep run
STEPS_MIX      = 500_000    # steps for joint-mix runs
STEPS_CUMUL    = 300_000    # steps per phase in cumulative training
EVAL_EPISODES  = 300
BASE_SEED      = 42

RARITY_LEVELS  = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]   # fraction of non-base episodes

# Demand shifts: controls how often each non-base cluster naturally appears.
# force_cluster is used instead of shifts during rarity sweep.
BASE_DEMAND_SHIFT = -40.0   # full-env evaluation shift (center ≈ 260 MW)


# ── Helper: build a fresh agent ───────────────────────────────────────────────

def make_agent(
    run_name: str,
    use_cluster_feature: bool = True,
    seed: int = BASE_SEED,
) -> TD3:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = OBS_DIM if use_cluster_feature else ACTION_DIM  # 24h demand only when no feature
    return TD3(
        obs_dim         = obs_dim,
        action_dim      = ACTION_DIM,
        tensorboard_log = f"runs/benchmark/{run_name}",
        normalize_observations = True,
        normalize_rewards      = True,
    )


def make_env(
    cluster_mixing_ratio: dict | None = None,
    force_cluster: int | None = None,
    demand_shift: float = BASE_DEMAND_SHIFT,
    use_cluster_feature: bool = True,
    seed: int | None = None,
) -> DayAheadMarketEnv:
    env = DayAheadMarketEnv(
        demand_center_shift  = demand_shift,
        force_cluster        = force_cluster,
        cluster_mixing_ratio = cluster_mixing_ratio,
        seed                 = seed,
    )
    if not use_cluster_feature:
        # Monkey-patch obs to strip cluster indicator (last N_CLUSTERS dims)
        _orig_make_obs = env._make_obs.__func__

        @staticmethod
        def _stripped_obs(demand, cluster):
            full = _orig_make_obs(demand, cluster)
            return full[:24]   # drop cluster one-hot

        env._make_obs = _stripped_obs
        # Also fix observation_space shape
        import gymnasium as gym
        import numpy as np
        env.observation_space = gym.spaces.Box(
            low=np.zeros(24, dtype=np.float32),
            high=np.full(24, 4.0, dtype=np.float32),
        )
    return env


# ── 1. RARITY SWEEP ───────────────────────────────────────────────────────────

def run_rarity_sweep(target_clusters: list[int]):
    """
    For each target cluster and each rarity level, train an agent on
    Base(C0) + TargetCluster at the given rarity, then evaluate on both
    clusters separately.
    """
    print("\n" + "="*70)
    print("  RARITY SWEEP")
    print("="*70)

    all_results = {}

    for tc in target_clusters:
        for rarity in RARITY_LEVELS:
            run_name = f"rarity_c{tc}_p{int(rarity*100):03d}"
            print(f"\n  [{run_name}]  cluster={tc}  rarity={rarity:.0%}")

            mixing = {0: 1.0 - rarity, tc: rarity}
            train_env = make_env(cluster_mixing_ratio=mixing, seed=BASE_SEED)
            eval_env  = make_env(demand_shift=BASE_DEMAND_SHIFT)

            agent = make_agent(run_name, use_cluster_feature=True, seed=BASE_SEED)

            summary = train_phase(
                agent          = agent,
                env            = train_env,
                total_steps    = STEPS_RARITY,
                phase_name     = run_name,
                checkpoint_dir = f"checkpoints/benchmark/{run_name}",
                eval_env       = eval_env,
                eval_interval  = 50_000,
                eval_episodes  = EVAL_EPISODES,
            )
            all_results[(tc, rarity)] = summary
            _print_rarity_result(tc, rarity, summary)

    _summarise_rarity(all_results, target_clusters)
    return all_results


def _print_rarity_result(tc: int, rarity: float, summary: dict):
    base_profit  = summary.get("mean_profit_c0", float("nan"))
    skill_profit = summary.get(f"mean_profit_c{tc}", float("nan"))
    print(f"    base(C0) profit={base_profit:.0f}  cluster{tc} profit={skill_profit:.0f}")


def _summarise_rarity(results: dict, target_clusters: list[int]):
    print("\n" + "="*70)
    print("  RARITY SWEEP SUMMARY  (cluster profit at each rarity level)")
    print("="*70)
    for tc in target_clusters:
        print(f"\n  Cluster {tc}:")
        print(f"  {'Rarity':>8}  {'C0 profit':>12}  {'C{tc} profit':>12}".format(tc=tc))
        for rarity in RARITY_LEVELS:
            s = results.get((tc, rarity), {})
            c0 = s.get("mean_profit_c0", float("nan"))
            ct = s.get(f"mean_profit_c{tc}", float("nan"))
            print(f"  {rarity:>8.0%}  {c0:>12.0f}  {ct:>12.0f}")


# ── 2. ALL-CLUSTER MIX ────────────────────────────────────────────────────────

# Mixing ratio for the full environment (roughly equal representation of rare clusters)
FULL_MIX = {0: 0.60, 1: 0.20, 2: 0.10, 3: 0.10}


def run_mix_no_feature():
    """Condition A: joint training, NO cluster indicator."""
    run_name = "mix_no_feature"
    print(f"\n  [{run_name}]  joint training, NO cluster feature")
    agent = make_agent(run_name, use_cluster_feature=False, seed=BASE_SEED)
    train_env = make_env(cluster_mixing_ratio=FULL_MIX, use_cluster_feature=False, seed=BASE_SEED)
    eval_env  = make_env(cluster_mixing_ratio=FULL_MIX, use_cluster_feature=False)
    return train_phase(
        agent          = agent,
        env            = train_env,
        total_steps    = STEPS_MIX,
        phase_name     = run_name,
        checkpoint_dir = f"checkpoints/benchmark/{run_name}",
        eval_env       = eval_env,
        eval_interval  = 50_000,
        eval_episodes  = EVAL_EPISODES,
    )


def run_mix_with_feature():
    """Condition B: joint training WITH cluster indicator."""
    run_name = "mix_with_feature"
    print(f"\n  [{run_name}]  joint training, WITH cluster feature")
    agent     = make_agent(run_name, use_cluster_feature=True, seed=BASE_SEED)
    train_env = make_env(cluster_mixing_ratio=FULL_MIX, seed=BASE_SEED)
    eval_env  = make_env(cluster_mixing_ratio=FULL_MIX)
    return train_phase(
        agent          = agent,
        env            = train_env,
        total_steps    = STEPS_MIX,
        phase_name     = run_name,
        checkpoint_dir = f"checkpoints/benchmark/{run_name}",
        eval_env       = eval_env,
        eval_interval  = 50_000,
        eval_episodes  = EVAL_EPISODES,
    )


def run_mix_cumulative():
    """
    Condition C: cumulative cluster training (C0→C1→C2→C3),
    then evaluated on full mix WITHOUT further training.
    """
    run_name = "mix_cumulative"
    print(f"\n  [{run_name}]  cumulative training C0→C1→C2→C3")

    phases = [
        {"cluster": 0, "steps": STEPS_CUMUL, "mixing": None},
        {"cluster": 1, "steps": STEPS_CUMUL, "mixing": {0: 0.10, 1: 0.90}},
        {"cluster": 2, "steps": STEPS_CUMUL, "mixing": {0: 0.10, 1: 0.10, 2: 0.80}},
        {"cluster": 3, "steps": STEPS_CUMUL, "mixing": {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.70}},
    ]

    agent = make_agent(run_name, use_cluster_feature=True, seed=BASE_SEED)
    trainer = CumulativeTrainer(
        agent           = agent,
        base_env_kwargs = {"demand_center_shift": BASE_DEMAND_SHIFT, "seed": BASE_SEED},
        phases          = phases,
        checkpoint_dir  = f"checkpoints/benchmark/{run_name}",
        eval_episodes   = EVAL_EPISODES,
    )
    summaries = trainer.run()

    # Final evaluation on full mix
    eval_env = make_env(cluster_mixing_ratio=FULL_MIX)
    final = evaluate(agent, eval_env, EVAL_EPISODES * 2)
    print(f"\n  [{run_name}] FINAL EVAL on full mix: {final}")
    return final


# ── 3. Main entry point ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Final benchmark runner")
    p.add_argument("--mode", choices=["rarity", "mix", "all"], default="all",
                   help="Which experiments to run")
    p.add_argument("--target_cluster", default="all",
                   help="Which cluster for rarity sweep: 1, 2, 3, or 'all'")
    p.add_argument("--rarity_levels", nargs="+", type=float, default=None,
                   help="Override rarity levels, e.g. --rarity_levels 0.1 0.2 0.5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.rarity_levels:
        RARITY_LEVELS.clear()
        RARITY_LEVELS.extend(args.rarity_levels)

    target_clusters = (
        [1, 2, 3] if args.target_cluster == "all"
        else [int(args.target_cluster)]
    )

    rarity_results, mix_results = {}, {}

    if args.mode in ("rarity", "all"):
        rarity_results = run_rarity_sweep(target_clusters)

    if args.mode in ("mix", "all"):
        print("\n" + "="*70)
        print("  ALL-CLUSTER MIX BENCHMARK")
        print("="*70)
        print(f"  Mix ratio: {FULL_MIX}")

        ra = run_mix_no_feature()
        rb = run_mix_with_feature()
        rc = run_mix_cumulative()

        mix_results = {"no_feature": ra, "with_feature": rb, "cumulative": rc}

        print("\n" + "="*70)
        print("  MIX BENCHMARK RESULTS COMPARISON")
        print("="*70)
        for cond, res in mix_results.items():
            print(f"\n  {cond}:")
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.1f}")

    print("\nAll experiments complete.")
    print("View results: tensorboard --logdir runs/benchmark/")
