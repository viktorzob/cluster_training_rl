"""
24-hour intra-day demand profile generator.

Demand follows a realistic dual-peak profile (morning + evening).
A `demand_center_shift` parameter raises/lowers the whole profile,
controlling how often demand crosses the 300 MW threshold and thus
the rarity of Cluster 1 (headroom) and Cluster 2 (super-headroom) events.
"""

import numpy as np


# Base intra-day shape: normalized so its mean = 1.0
# Peaks around 09:00 and 18:00-19:00, trough overnight.
_BASE_SHAPE = np.array([
    0.72, 0.68, 0.65, 0.64, 0.65, 0.70,   # 00-05
    0.80, 0.93, 1.05, 1.10, 1.08, 1.05,   # 06-11
    1.02, 1.00, 0.99, 1.00, 1.03, 1.10,   # 12-17
    1.18, 1.20, 1.15, 1.05, 0.92, 0.80,   # 18-23
], dtype=np.float64)
_BASE_SHAPE = _BASE_SHAPE / _BASE_SHAPE.mean()   # mean exactly 1.0

DEMAND_CENTER = 300.0   # MW — threshold also used by market env


def generate_demand_profile(
    demand_center_shift: float = 0.0,
    noise_std_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return a 24-element array of hourly demand [MW].

    Parameters
    ----------
    demand_center_shift : float
        Additive MW shift applied to the whole profile.
        Negative  → demand often below 300 → headroom events rare.
        Positive  → demand often above 300 → headroom events frequent.
        A shift of ±30 MW moves roughly ±1 std-dev of the profile.
    noise_std_fraction : float
        Per-hour noise as a fraction of local demand (default 5%).
    rng : numpy Generator, optional
        For reproducibility; created fresh if None.
    """
    if rng is None:
        rng = np.random.default_rng()

    base = _BASE_SHAPE * (DEMAND_CENTER + demand_center_shift)
    noise = rng.normal(0.0, noise_std_fraction * base)
    profile = base + noise
    return profile.clip(min=50.0)   # floor: grid never fully empty


def expected_headroom_fraction(
    demand_center_shift: float,
    noise_std_fraction: float = 0.05,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Monte-Carlo estimate of how often each cluster fires.
    Returns dict with keys 'cluster0_frac', 'cluster1_frac', 'cluster2_frac'
    and 'hours_above_threshold_mean'.

    Useful for choosing demand_center_shift to hit a target rarity X%.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    from env.market import FRINGE_THRESHOLD, FRINGE_SHUTDOWN_HOURS

    above_counts = []
    shutdown_hour_counts = []

    for _ in range(n_samples):
        profile = generate_demand_profile(demand_center_shift, noise_std_fraction, rng)
        above = (profile >= FRINGE_THRESHOLD).astype(int)
        above_counts.append(above.sum())

        # Simulate fringe shutdown logic
        shutdown = _simulate_shutdown(profile)
        shutdown_hour_counts.append(shutdown.sum())

    above_arr = np.array(above_counts)
    shutdown_arr = np.array(shutdown_hour_counts)

    total_hours = n_samples * 24
    cluster2_hours = shutdown_arr.sum()
    cluster1_hours = above_arr.sum() - cluster2_hours   # above threshold but fringe online
    # Note: some cluster2 hours may overlap with above-threshold hours; handled in market env
    cluster0_hours = total_hours - above_arr.sum()

    return {
        "cluster0_frac": cluster0_hours / total_hours,
        "cluster1_frac": max(0.0, cluster1_hours / total_hours),
        "cluster2_frac": cluster2_hours / total_hours,
        "hours_above_threshold_mean": above_arr.mean(),
    }


def _simulate_shutdown(profile: np.ndarray) -> np.ndarray:
    """
    Return boolean array indicating which hours the fringe is offline.

    The fringe only shuts down after it has been active (demand >= threshold).
    Once active, if demand drops below threshold for FRINGE_SHUTDOWN_HOURS
    consecutive hours, the fringe goes offline for FRINGE_RECOVERY_HOURS.
    If demand never crosses the threshold the fringe is never triggered,
    so pure infra-marginal days (Cluster 0) never produce shutdowns.
    """
    from env.market import FRINGE_THRESHOLD, FRINGE_SHUTDOWN_HOURS, FRINGE_RECOVERY_HOURS

    n = len(profile)
    shutdown = np.zeros(n, dtype=bool)

    fringe_ever_active = False
    consecutive_below  = 0
    offline_remaining  = 0

    for h in range(n):
        if offline_remaining > 0:
            shutdown[h] = True
            offline_remaining -= 1
            consecutive_below = 0
        elif profile[h] >= FRINGE_THRESHOLD:
            fringe_ever_active = True
            consecutive_below  = 0
        else:
            # Only count toward shutdown if fringe has been active before
            if fringe_ever_active:
                consecutive_below += 1
                if consecutive_below >= FRINGE_SHUTDOWN_HOURS:
                    offline_remaining = FRINGE_RECOVERY_HOURS
                    consecutive_below = 0

    return shutdown
