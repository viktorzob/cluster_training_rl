"""
Single-phase and cumulative (cluster-sequential) trainers.

Single-phase:
  train_phase(agent, env, total_steps)

Cumulative trainer:
  CumulativeTrainer runs Phase 0 → Phase 1 → Phase 2 sequentially,
  loading weights from the previous phase into the same agent.
  Optional mixing_ratio per phase adds minority-cluster episodes to
  mitigate catastrophic forgetting.
"""

import os
import numpy as np
from agents.td3 import TD3
from env.market import DayAheadMarketEnv, OBS_DIM, ACTION_DIM, N_CLUSTERS


def train_phase(
    agent: TD3,
    env: DayAheadMarketEnv,
    total_steps: int,
    phase_name: str = "phase",
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 50_000,
    eval_env: DayAheadMarketEnv | None = None,
    eval_interval: int = 10_000,
    eval_episodes: int = 100,
) -> dict:
    """
    Run one training phase.

    Returns summary dict with final eval metrics (if eval_env provided).
    """
    obs, _ = env.reset()
    steps  = 0
    ep_num = 0

    print(f"\n{'='*60}")
    print(f"  Training phase: {phase_name}  |  steps: {total_steps:,}")
    print(f"{'='*60}")

    while steps < total_steps:
        # Random warm-up (matches SB3 behaviour): sample uniformly from action
        # space before learning starts so the replay buffer contains diverse
        # price bids, not just near-MC actions from the untrained actor.
        if agent.total_steps < agent.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(obs, action, reward, next_obs, float(done))
        metrics = agent.train_step()
        if metrics:
            agent.log_train_metrics(metrics)

        agent.log_episode(reward, info)

        steps += 1
        ep_num += 1

        if steps % 2_000 == 0:
            cluster_str = f"cluster={info['cluster']} profit={info['total_profit']:.1f}"
            print(f"  [{phase_name}] step {steps:>7,} / {total_steps:,}  {cluster_str}")

        # Periodic evaluation
        if eval_env is not None and steps % eval_interval == 0:
            eval_metrics = evaluate(agent, eval_env, eval_episodes)
            _log_eval(agent, eval_metrics, phase_name)

        # Checkpoint
        if checkpoint_dir and steps % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"{phase_name}_step{steps}.pt")
            agent.save(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        obs, _ = env.reset()   # single-step episode: always reset

    # Final checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        agent.save(os.path.join(checkpoint_dir, f"{phase_name}_final.pt"))

    summary = {}
    if eval_env is not None:
        summary = evaluate(agent, eval_env, eval_episodes * 2)
        _log_eval(agent, summary, f"{phase_name}_final")
        print(f"\n  [{phase_name}] FINAL EVAL  {_fmt_eval(summary)}")

    return summary


def evaluate(
    agent: TD3,
    env: DayAheadMarketEnv,
    n_episodes: int = 200,
) -> dict:
    """
    Deterministic evaluation. Returns per-cluster and overall profit stats.
    """
    profits   = {c: [] for c in range(N_CLUSTERS)}
    all_profit = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        action = agent.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        c = info["cluster"]
        profits[c].append(info["total_profit"])
        all_profit.append(info["total_profit"])

    result = {
        "mean_profit_overall": float(np.mean(all_profit)),
        "std_profit_overall":  float(np.std(all_profit)),
    }
    for c, vals in profits.items():
        if vals:
            result[f"mean_profit_c{c}"] = float(np.mean(vals))
            result[f"n_episodes_c{c}"]  = len(vals)

    return result


def _log_eval(agent: TD3, metrics: dict, tag: str):
    if agent.writer:
        for k, v in metrics.items():
            agent.writer.add_scalar(f"eval/{tag}/{k}", v, agent.total_steps)


def _fmt_eval(m: dict) -> str:
    parts = [f"overall={m.get('mean_profit_overall', 0):.1f}"]
    for c in range(N_CLUSTERS):
        k = f"mean_profit_c{c}"
        if k in m:
            parts.append(f"c{c}={m[k]:.1f} (n={m.get(f'n_episodes_c{c}', 0)})")
    return "  ".join(parts)


# ── Cumulative trainer ────────────────────────────────────────────────────────

class CumulativeTrainer:
    """
    Trains a single TD3 agent sequentially across cluster phases.

    Phase schedule example:
        phases = [
            {"cluster": 0, "steps": 300_000, "mixing": None},
            {"cluster": 1, "steps": 300_000, "mixing": {0: 0.1, 1: 0.9}},
            {"cluster": 2, "steps": 300_000, "mixing": {0: 0.1, 1: 0.1, 2: 0.8}},
        ]

    After all phases, a final benchmark evaluation runs on a controlled
    10% rare-event mix — identical conditions to MVP2 — so performance
    can be compared directly.
    """

    def __init__(
        self,
        agent: TD3,
        base_env_kwargs: dict,
        phases: list[dict],
        checkpoint_dir: str = "checkpoints/cumulative",
        eval_episodes: int  = 200,
        # Benchmark eval: mirrors MVP2's 10% rare-event mix for direct comparison
        benchmark_rare_frac: float = 0.10,
    ):
        self.agent               = agent
        self.base_env_kwargs     = base_env_kwargs
        self.phases              = phases
        self.checkpoint_dir      = checkpoint_dir
        self.eval_episodes       = eval_episodes
        self.benchmark_rare_frac = benchmark_rare_frac

    def run(self) -> list[dict]:
        summaries = []
        for i, phase_cfg in enumerate(self.phases):
            cluster = phase_cfg["cluster"]
            steps   = phase_cfg["steps"]
            mixing  = phase_cfg.get("mixing", None)

            # Log phase boundary as a TensorBoard text marker
            if self.agent.writer:
                self.agent.writer.add_text(
                    "training/phase",
                    f"Phase {i} start — cluster {cluster}",
                    self.agent.total_steps,
                )

            # Build training env for this phase
            env_kwargs = dict(self.base_env_kwargs)
            if mixing is not None:
                env_kwargs["cluster_mixing_ratio"] = mixing
                env_kwargs.pop("force_cluster", None)
            else:
                env_kwargs["force_cluster"] = cluster
                env_kwargs.pop("cluster_mixing_ratio", None)

            train_env = DayAheadMarketEnv(**env_kwargs)

            # Eval env: full distribution (no forcing) — tracks all clusters
            eval_kwargs = dict(self.base_env_kwargs)
            eval_kwargs.pop("force_cluster", None)
            eval_kwargs.pop("cluster_mixing_ratio", None)
            eval_env = DayAheadMarketEnv(**eval_kwargs)

            phase_name = f"phase{i}_cluster{cluster}"
            summary = train_phase(
                agent              = self.agent,
                env                = train_env,
                total_steps        = steps,
                phase_name         = phase_name,
                checkpoint_dir     = self.checkpoint_dir,
                eval_env           = eval_env,
                eval_episodes      = self.eval_episodes,
            )
            summaries.append({"phase": i, "cluster": cluster, **summary})

        # ── Final benchmark: same 10% rare-event mix as MVP2 ─────────────────
        # This is the key comparison: cumulative-trained vs naive-trained agent
        # on identical test conditions.
        print("\n  Running final benchmark eval (10% rare-event mix = MVP2 conditions)...")
        non_base_clusters = [p["cluster"] for p in self.phases if p["cluster"] != 0]
        for tc in non_base_clusters:
            bench_mix = {0: 1.0 - self.benchmark_rare_frac, tc: self.benchmark_rare_frac}
            bench_kwargs = dict(self.base_env_kwargs)
            bench_kwargs["cluster_mixing_ratio"] = bench_mix
            bench_kwargs.pop("force_cluster", None)
            bench_env = DayAheadMarketEnv(**bench_kwargs)
            bench_result = evaluate(self.agent, bench_env, self.eval_episodes * 3)
            _log_eval(self.agent, bench_result, f"benchmark_10pct_vs_cluster{tc}")
            print(f"  Benchmark (C0 90% + C{tc} 10%): {_fmt_eval(bench_result)}")

        print("\n" + "="*60)
        print("  Cumulative training complete.")
        for s in summaries:
            print(f"  Phase {s['phase']} (cluster {s['cluster']}): {_fmt_eval(s)}")
        print("="*60)
        return summaries
