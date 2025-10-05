import math
from collections import Counter

import numpy as np


def event_count_score(runs, N_min=30, N_max=300, p=2):
    """
    Normalized event–count score in [0, 1].

    Theoretical background:
    The raw penalty term is
        t(N) = (N / N_max)^p + (N_min / N)^p .
    Minimizing t(N) w.r.t. N gives
        d/dN t(N) = 0  →  N_opt^2 = N_min * N_max
        →  N_opt = sqrt(N_min * N_max) .
    At this point t(N) is minimal (best balance between too many and too few runs).
    """
    N_dc = len(runs)
    if N_dc == 0:
        return 0.0

    # raw penalty for current number of runs
    term = (N_dc / N_max) ** p + (N_min / N_dc) ** p

    # ── theoretical best (smallest term)
    # N_opt = sqrt(N_min * N_max)
    # corresponds to the geometric mean of the lower and upper limits
    N_opt = math.sqrt(N_min * N_max)
    t_best = (N_opt / N_max) ** p + (N_min / N_opt) ** p

    # ── reasonable worst (term at extremes)
    t_worst = (N_min / N_max) ** p + 1.0**p  # or max(term_min, term_max)

    # ── min–max normalization → [0, 1]; higher = better
    score = (t_worst - term) / (t_worst - t_best)
    return float(np.clip(score, 0, 1))


def up_down_asymmetry(runs, osv_min, osv_max):
    """
    Measures average strength (absolute overshoot) of up & down trends,
    returning both normalized μ_up and μ_down in [0, 1].

    Parameters
    ----------
    runs : list[dict]
        Each run must have a numeric "OSV_EXT" and a "type" key equal to
        "upward_run" or "downward_run".
    osv_min : float
        Global minimum OSV_EXT value used for normalization.
    osv_max : float
        Global maximum OSV_EXT value used for normalization.

    Returns
    -------
    tuple(float, float) :
        (mu_up_norm, mu_down_norm) each ∈ [0, 1]
    """
    # map run type to numeric direction (+1 = upward, -1 = downward)
    dir_map = {"upward_run": 1, "downward_run": -1}
    filtered = [r for r in runs if r.get("OSV_EXT") is not None]

    if not filtered:
        return 0.0, 0.0

    osv = np.array([r["OSV_EXT"] for r in filtered], dtype=float)
    dirs = np.array([dir_map.get(r.get("type"), 0) for r in filtered], dtype=float)

    # separate up and down groups
    up_mask = dirs > 0
    down_mask = dirs < 0
    mu_up = np.mean(osv[up_mask]) if np.any(up_mask) else 0.0
    mu_down = np.mean(osv[down_mask]) if np.any(down_mask) else 0.0

    # --- global min–max normalization → [0, 1] ---
    denom = (osv_max - osv_min) if (osv_max != osv_min) else 1e-12
    mu_up_norm = np.clip((mu_up - osv_min) / denom, 0, 1)
    mu_down_norm = np.clip((mu_down - osv_min) / denom, 0, 1)

    return float(mu_up_norm), float(mu_down_norm)


def direction_entropy(runs, normalize=False):
    """
    Computes Shannon entropy (base e) of transitions between up/down runs
    based on run["type"] strings. Automatically adjusts normalization to
    the number of distinct transition types actually observed.

    Parameters
    ----------
    runs : list[dict]
        Each run must have key "type" = "upward_run" or "downward_run".
    normalize : bool, optional
        If True, divides by log(M), where M is the number of unique transition
        types found (2–4), to obtain a value in [0, 1].

    Returns
    -------
    float : Entropy H
        Smaller → more predictable / structured, larger → more random.
    """
    # Map run types to numeric directions
    dir_map = {"upward_run": 1, "downward_run": -1}

    # Convert run types to ±1 values and filter invalid entries
    dirs = [dir_map.get(r.get("type")) for r in runs if r.get("type") in dir_map]
    if len(dirs) < 2:
        return np.nan

    # Build transition pairs: (current_direction, next_direction)
    transitions = list(zip(dirs[:-1], dirs[1:]))

    # Count frequencies of each distinct transition
    counts = Counter(transitions)
    total = sum(counts.values())

    # Compute Shannon entropy
    H = -sum((count / total) * math.log(count / total) for count in counts.values())

    if normalize:
        # Number of distinct transition types actually observed (2–4 possible)
        n_unique = len(counts)
        # Avoid division by zero
        H_max = math.log(n_unique) if n_unique > 0 else 1.0
        H /= H_max

    return H


def entropy_score(runs):
    """Returns 1 when predictable (low entropy), 0 when random."""
    H = direction_entropy(runs, normalize=True)
    if not np.isfinite(H):
        return 0.0
    return float(1.0 - np.clip(H, 0, 1))
