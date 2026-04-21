"""
Day-ahead electricity market environment.

One episode = one trading day.
Single step: agent submits 24 price bids simultaneously,
market clears all hours, episode ends.

Observation  : [demand_24h_normalised (24,), cluster_onehot (3,)] = shape (27,)
Action       : 24 price bids, raw tanh output scaled so 0 → agent MC
Reward       : total daily profit = sum_h max(0, (clearing_price_h - MC) * dispatch_h)

Clusters (one-hot [c0, c1, c2, c3]):
  [1,0,0,0] Cluster 0 — no headroom    : demand never in agent zone; agent idle or infra-marginal
  [0,1,0,0] Cluster 1 — headroom       : some hours 250 < demand ≤ 300; agent marginal, bid HIGH (up to 100)
  [0,0,1,0] Cluster 2 — super-headroom : fringe offline; agent marginal up to 150 regardless of demand
  [0,0,0,1] Cluster 3 — redispatch     : hours 11-13 in agent zone; bid BELOW MC to earn 160 €/MWh

Priority: Cluster 2 > Cluster 3 > Cluster 1 > Cluster 0
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.demand import generate_demand_profile, DEMAND_CENTER, _simulate_shutdown

# ── Market constants ─────────────────────────────────────────────────────────
BASELOAD_CAPACITY    = 250.0   # MW — always-on cheap generation (below agent MC)
FRINGE_THRESHOLD     = 300.0   # MW — above this, expensive fringe is needed and sets price
FRINGE_SHUTDOWN_HOURS = 4      # consecutive below-threshold hours to trigger shutdown
FRINGE_RECOVERY_HOURS = 4      # hours fringe stays offline after shutdown

AGENT_CAPACITY   = 50.0        # MW  (covers residual demand between baseload and fringe)
AGENT_MC         = 50.0        # €/MWh  (marginal cost, action=0 maps here)

# Merit order:
#   [0, BASELOAD_CAPACITY]     → baseload, always dispatched, price below AGENT_MC
#   [BASELOAD_CAPACITY, FRINGE_THRESHOLD] → agent zone: agent is marginal, sets own price
#   [FRINGE_THRESHOLD, ∞]      → expensive fringe at FRINGE_PRICE_BASE, sets clearing price

# Fringe price when online and demand >= FRINGE_THRESHOLD
FRINGE_PRICE_BASE      = 100.0   # €/MWh
FRINGE_PRICE_DEMAND_K  = 0.10    # €/MWh per MW above threshold (small dynamic component)

# Price caps per regime
HEADROOM_PRICE_CAP       = 100.0   # max agent can earn as marginal (Cluster 1)
SUPER_HEADROOM_PRICE_CAP = 150.0   # fringe offline, agent price setter (Cluster 2)

# Cluster 3 — redispatch arbitrage
# If agent bids BELOW MC in REDISPATCH_HOURS during an agent-zone hour,
# it receives a redispatch payment (simulates grid operator paying for downward flexibility).
# Optimal action is the OPPOSITE of Cluster 1: bid low, not high.
REDISPATCH_HOURS   = [13, 14, 15]   # post-lunch dip: demand reliably in 250-300 MW zone
REDISPATCH_PRICE   = 160.0          # €/MWh paid for full capacity when agent bids below MC

PRICE_FLOOR = AGENT_MC   # agent bids at least MC (action clipped)

N_CLUSTERS = 4
OBS_DIM    = 24 + N_CLUSTERS   # 24h demand + one-hot (now 28)
ACTION_DIM = 24                # one price bid per hour


class DayAheadMarketEnv(gym.Env):
    """
    Parameters
    ----------
    demand_center_shift : float
        Shifts demand profile up/down, controlling rarity of Cluster 1/2.
        0.0  → ~50% hours above threshold (balanced).
        -30  → headroom events rare.
        +30  → headroom events frequent.
    force_cluster : int | None
        If set (0/1/2), only episodes of that cluster are generated.
        Used during per-cluster training phases.
    cluster_mixing_ratio : dict | None
        e.g. {0: 0.2, 1: 0.8} — sample cluster proportionally.
        Overrides force_cluster if provided.
    noise_std_fraction : float
        Per-hour demand noise as fraction of local demand.
    seed : int | None
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_center_shift: float = 0.0,
        force_cluster: int | None = None,
        cluster_mixing_ratio: dict | None = None,
        noise_std_fraction: float = 0.05,
        seed: int | None = None,
        include_cluster_indicator: bool = True,
    ):
        super().__init__()
        self.demand_center_shift      = demand_center_shift
        self.force_cluster            = force_cluster
        self.cluster_mixing_ratio     = cluster_mixing_ratio
        self.noise_std_fraction       = noise_std_fraction
        self.include_cluster_indicator = include_cluster_indicator

        self._rng = np.random.default_rng(seed)

        # Action space: raw values in [-1, 1] (tanh output), scaled to prices in trainer
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        # Observation: 24h demand + optional cluster one-hot
        obs_dim  = 24 + (N_CLUSTERS if include_cluster_indicator else 0)
        obs_low  = np.zeros(obs_dim, dtype=np.float32)
        obs_high = np.concatenate([
            np.full(24, 4.0, dtype=np.float32),
            *([ np.ones(N_CLUSTERS, dtype=np.float32) ] if include_cluster_indicator else []),
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Populated each reset
        self._demand: np.ndarray | None = None
        self._cluster: int | None       = None
        self._shutdown: np.ndarray | None = None

        # Tracking for info dict
        self.episode_count = 0

    # ── Gym interface ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        demand, cluster, shutdown = self._sample_episode()
        self._demand   = demand
        self._cluster  = cluster
        self._shutdown = shutdown
        self.episode_count += 1

        obs = self._make_obs(demand, cluster)
        return obs, {}

    def step(self, action: np.ndarray):
        assert self._demand is not None, "Call reset() before step()."

        # Scale action: tanh=0 → MC, tanh=+1 → SUPER_HEADROOM_PRICE_CAP
        prices = self._scale_action(action)

        reward, info = self._clear_market(prices, self._demand, self._shutdown, self._cluster)

        obs = self._make_obs(self._demand, self._cluster)   # same obs (single-step episode)
        terminated = True
        truncated  = False
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sample_episode(self) -> tuple[np.ndarray, int, np.ndarray]:
        """Generate demand, determine cluster, compute fringe shutdown mask."""
        max_tries = 200
        for _ in range(max_tries):
            demand   = generate_demand_profile(
                self.demand_center_shift, self.noise_std_fraction, self._rng
            )
            shutdown = _simulate_shutdown(demand)
            cluster  = _classify_cluster(demand, shutdown)

            target = self._target_cluster()
            if target is None or cluster == target:
                return demand, cluster, shutdown

        # Fallback: force-generate a profile matching the target cluster
        return self._force_generate_cluster(target)

    def _target_cluster(self) -> int | None:
        if self.cluster_mixing_ratio is not None:
            clusters = list(self.cluster_mixing_ratio.keys())
            probs    = np.array([self.cluster_mixing_ratio[c] for c in clusters])
            probs    = probs / probs.sum()
            return int(self._rng.choice(clusters, p=probs))
        return self.force_cluster   # None means any cluster

    def _force_generate_cluster(self, cluster: int) -> tuple[np.ndarray, int, np.ndarray]:
        """Brute-force generate a profile that matches the requested cluster.

        Cluster 0 needs demand well below BASELOAD_CAPACITY so the dual-peak
        profile never enters the 250-300 agent zone (shift=-100 → center=200 MW,
        peak≈240 MW < 250 MW).
        """
        # Cluster 3 needs center≈280 so hours 13-15 land in 250-300 agent zone
        shift_map = {0: -100.0, 1: 0.0, 2: -30.0, 3: -20.0}
        shift = self.demand_center_shift + shift_map.get(cluster, 0.0)
        for _ in range(2000):
            demand   = generate_demand_profile(shift, self.noise_std_fraction, self._rng)
            shutdown = _simulate_shutdown(demand)
            if _classify_cluster(demand, shutdown) == cluster:
                return demand, cluster, shutdown
        # Last resort: return whatever we have
        return demand, _classify_cluster(demand, shutdown), shutdown

    def _make_obs(self, demand: np.ndarray, cluster: int) -> np.ndarray:
        demand_norm = (demand / DEMAND_CENTER).astype(np.float32)
        if not self.include_cluster_indicator:
            return demand_norm
        one_hot = np.zeros(N_CLUSTERS, dtype=np.float32)
        one_hot[cluster] = 1.0
        return np.concatenate([demand_norm, one_hot])

    @staticmethod
    def _scale_action(action: np.ndarray) -> np.ndarray:
        """
        Map tanh output [-1, 1] → price bids centred at MC.
          tanh =  0  → AGENT_MC       (break-even, no profit)
          tanh = +1  → SUPER_HEADROOM_PRICE_CAP  (max headroom bid)
          tanh = -1  → AGENT_MC - (CAP - MC) ≈ 0  (below MC, triggers redispatch)
        Linear mapping: price = MC + a * (CAP - MC), clipped to [0, CAP].
        """
        a = np.clip(action, -1.0, 1.0)
        price = AGENT_MC + a * (SUPER_HEADROOM_PRICE_CAP - AGENT_MC)
        return np.clip(price, 0.0, SUPER_HEADROOM_PRICE_CAP).astype(np.float64)

    @staticmethod
    def _clear_market(
        agent_prices: np.ndarray,
        demand: np.ndarray,
        shutdown: np.ndarray,
        cluster: int,
    ) -> tuple[float, dict]:
        """
        Pay-as-cleared market clearing for 24 hours.

        Merit order per hour:
          1. Baseload (BASELOAD_CAPACITY MW) — always dispatched, price below agent MC.
          2. Agent (up to AGENT_CAPACITY MW) — covers residual demand above baseload.
          3. Expensive fringe — unlimited capacity at FRINGE_PRICE_BASE (or 150 if offline-recovery).

        Three regimes per hour:
          demand <= BASELOAD_CAPACITY:
              Agent not needed. Dispatch = 0. No profit.
          BASELOAD_CAPACITY < demand <= FRINGE_THRESHOLD  (agent is marginal):
              residual = demand - BASELOAD_CAPACITY  (< AGENT_CAPACITY)
              Agent sets the clearing price (price discovery zone).
              If ap <= fringe_price: dispatch = residual, clearing = ap  (agent is price setter)
              Else:                  dispatch = 0,        clearing = fringe_price (fringe steps in)
          demand > FRINGE_THRESHOLD  (fringe is marginal):
              Agent fully infra-marginal (dispatch = AGENT_CAPACITY).
              Fringe sets clearing price. Agent earns (fringe_price - MC) * AGENT_CAPACITY.

          Cluster 2 override: fringe offline → fringe_price = SUPER_HEADROOM_PRICE_CAP.
              Agent is always marginal regardless of demand level.
              dispatch = min(AGENT_CAPACITY, max(0, demand - BASELOAD_CAPACITY))
        """
        profits         = np.zeros(24)
        clearing_prices = np.zeros(24)
        dispatches      = np.zeros(24)

        for h in range(24):
            d  = demand[h]
            ap = agent_prices[h]

            residual = max(0.0, d - BASELOAD_CAPACITY)

            if shutdown[h]:
                # Fringe offline: agent is always marginal, bids freely up to SUPER_HEADROOM_PRICE_CAP
                fringe_price  = SUPER_HEADROOM_PRICE_CAP
                agent_needed  = min(AGENT_CAPACITY, residual)
                if ap <= fringe_price and agent_needed > 0:
                    clearing = ap
                    dispatch = agent_needed
                else:
                    clearing = fringe_price
                    dispatch = 0.0

            elif d > FRINGE_THRESHOLD:
                # Fringe online and needed: fringe sets clearing, agent infra-marginal
                excess        = d - FRINGE_THRESHOLD
                fringe_price  = FRINGE_PRICE_BASE + FRINGE_PRICE_DEMAND_K * excess
                fringe_price  = min(fringe_price, SUPER_HEADROOM_PRICE_CAP - 1.0)
                clearing      = fringe_price
                dispatch      = AGENT_CAPACITY   # agent always cheaper, fully dispatched

            elif residual > 0:
                # Agent zone: BASELOAD_CAPACITY < demand <= FRINGE_THRESHOLD
                fringe_price = HEADROOM_PRICE_CAP
                agent_needed = min(AGENT_CAPACITY, residual)

                if h in REDISPATCH_HOURS and cluster == 3 and ap < AGENT_MC:
                    # Cluster 3 redispatch: agent bids below MC in these specific hours
                    # → grid operator pays REDISPATCH_PRICE for full capacity downward flexibility
                    clearing = REDISPATCH_PRICE
                    dispatch = AGENT_CAPACITY
                elif ap <= fringe_price:
                    clearing = ap            # agent sets the price (headroom discovery)
                    dispatch = agent_needed  # partial dispatch if residual < AGENT_CAPACITY
                else:
                    # Agent bid above fringe cap: fringe steps in, agent not dispatched
                    clearing = fringe_price
                    dispatch = 0.0

            else:
                # demand <= BASELOAD_CAPACITY: baseload covers everything, agent idle
                clearing = 0.0
                dispatch = 0.0

            profit = max(0.0, (clearing - AGENT_MC) * dispatch)
            profits[h]         = profit
            clearing_prices[h] = clearing
            dispatches[h]      = dispatch

        total_profit = profits.sum()
        info = {
            "cluster":          cluster,
            "total_profit":     total_profit,
            "clearing_prices":  clearing_prices,
            "agent_prices":     agent_prices,
            "dispatches":       dispatches,
            "demand":           demand,
            "shutdown_mask":    shutdown,
            "hours_dispatched": int(dispatches.sum() / AGENT_CAPACITY),
            "hours_above_threshold": int((demand >= FRINGE_THRESHOLD).sum()),
            "hours_shutdown":   int(shutdown.sum()),
        }
        return total_profit, info


def _classify_cluster(demand: np.ndarray, shutdown: np.ndarray) -> int:
    """
    Priority: Cluster 2 > Cluster 3 > Cluster 1 > Cluster 0.

    Cluster 2: any hour has fringe offline (super-headroom).
    Cluster 3: any of REDISPATCH_HOURS has demand in agent zone (opposing action required).
    Cluster 1: any other hour has demand in agent zone (bid-high headroom).
    Cluster 0: agent always idle or infra-marginal, no price-discovery hours.
    """
    if shutdown.any():
        return 2
    in_agent_zone = (demand > BASELOAD_CAPACITY) & (demand <= FRINGE_THRESHOLD)
    redispatch_active = any(in_agent_zone[h] for h in REDISPATCH_HOURS)
    if redispatch_active:
        return 3
    if in_agent_zone.any():
        return 1
    return 0
