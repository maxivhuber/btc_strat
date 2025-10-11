import numpy as np


def event_density_score(prices, events, d_target=0.002, alpha=2.0, beta=1.0):
    r"""
    Compute a smooth, symmetric similarity score between the *observed event density*
    and a *target event density*. The score is dimensionless and bounded in (0, 1],
    where values near 1 indicate that the event frequency matches the target well.

    -------------------------------------------------------------------------------
    Concept
    -------------------------------------------------------------------------------
    Let:
        • N = number of detected events  ( len(events) )
        • T = total number of observations ( len(prices) )

    The **observed event density** is

        d = N / T.

    Example:
        d = 0.0018  →  roughly one event every 1/d ≈ 556 observations.

    The goal is to measure how close d is to the desired target density `d_target`.
    The metric penalizes deviations *symmetrically* — being too sparse or too dense
    both reduce the score.

    -------------------------------------------------------------------------------
    Mathematical form
    -------------------------------------------------------------------------------
    The score is defined as

        .. math::

            s(d) = \exp\!\big[-\,\beta\,|\ln(d/d_{\text{target}})|^{\alpha}\big].

    where

        • :math:`\alpha > 0` controls **curvature** ("shape" of the bowl):
              - small α ≈ 1 → wide, gentle bowl (tolerant)
              - large α > 2 → narrow, sharp peak (strict)

        • :math:`\beta > 0` controls **overall width / steepness**:
              - small β → slower decay → broader acceptable range
              - large β → faster decay → steeper drop‑off

    Interpretation:

        s = 1.0 → perfect match (d = d_target)
        s ≈ 0.5 → ≈ factor‑of‑two deviation (for α≈2, β≈1)
        s → 0  → densities very far apart

    -------------------------------------------------------------------------------
    Parameters
    -------------------------------------------------------------------------------
    prices : sequence of float
        Input time series used to detect events.

    events : sequence
        Output list from event detection (same timescale as `prices`).

    d_target : float, default = 0.002
        Desired average fraction of samples corresponding to events.
        Example: 0.002 ≈ one event every ~500 samples.

    alpha : float, default = 2.0
        Exponent controlling the **curvature** of the penalty around the optimum.
        (Higher α → narrower, faster falloff; lower α → wider, smoother bowl.)

    beta : float, default = 1.0
        Scale factor controlling the **overall steepness** of the decay.

    -------------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------------
    score : float in (0, 1]
        Smooth similarity measure between observed and target densities.
        1.0 → densities match exactly;
        degrades toward 0 as the densities diverge.
    """
    # counts
    N = len(events)
    T = len(prices)

    # Edge case: no data or no events → undefined density
    if N == 0 or T == 0:
        return 0.0

    # Observed event density
    d = N / T

    # Avoid log(0) and invalid inputs
    if d <= 0 or d_target <= 0:
        return 0.0

    # Symmetric log‑ratio deviation
    ratio = d / d_target
    deviation = abs(np.log(ratio))

    # Exponential decay based on multiplicative deviation
    score = np.exp(-beta * deviation**alpha)

    # Clip to valid range (numerical safety)
    return float(np.clip(score, 0.0, 1.0))


def up_down_asymmetry(runs, theta, squash=1.0):
    r"""
    Compute a *dimensionless* measure of asymmetry between upward and downward
    trends based on their mean **overshoot strength**, where overshoot is
    already expressed in multiples of the directional-change threshold ``theta``.

    The function returns two values, :math:`\mu_\text{up}` and
    :math:`\mu_\text{down}`, each in the range (0, 1], where 1 indicates
    stronger (large relative) overshoots and 0 indicates weak or negligible
    continuation beyond the threshold.

    ---------------------------------------------------------------------------
    Concept
    ---------------------------------------------------------------------------
    Each detected *run* represents a sustained price movement following a
    directional change. The ``"OSV_EXT"`` field in each run (computed by
    ``attach_OSV_EXT_to_runs``) already encodes the **dimensionless overshoot**
    as:

    .. math::
        r_i = \frac{(P_{\text{ext},i} - P_{\text{ext},i-1}) / P_{\text{ext},i-1}}{\theta},

    i.e., the price move between consecutive extrema expressed in **multiples of
    the threshold** ``theta``.

    The average overshoot for each direction is then:

    .. math::
        \bar{r}_\text{up}   = \frac{1}{N_\text{up}}\sum r_i^{(\text{up})}, \qquad
        \bar{r}_\text{down} = \frac{1}{N_\text{down}}\sum r_i^{(\text{down})}.

    Finally, a smooth exponential mapping converts these unbounded means into
    values between 0 and 1:

    .. math::
        \mu = 1 - e^{-\bar{r}/s},

    where :math:`s` = ``squash`` is a scaling constant controlling how fast the
    curve saturates (small s → rapid saturation; large s → gentler slope).

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    runs : list[dict]
        Sequence of runs produced by the directional-change algorithm **after**
        calling ``attach_OSV_EXT_to_runs``. Each entry must contain:
        - ``"OSV_EXT"``: a float representing overshoot in units of ``theta``
          (i.e., already normalized)
        - ``"type"``: either ``"upward_run"`` or ``"downward_run"``
    theta : float
        The directional-change threshold used to generate the runs and compute
        ``OSV_EXT``. This parameter is retained for API consistency and clarity,
        but **is not used in computation** (since normalization is already done).
    squash : float, default = 4.0
        Controls how quickly normalized overshoots approach 1.
        Rough guideline:
        -  small (≈ 2) → fast saturation, most values near 1
        -  large (> 5) → slower rise, broader dynamic range

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    tuple(float, float)
        ``(mu_up, mu_down)`` — normalized mean overshoot scores for upward and
        downward runs, each ∈ (0, 1]. If no runs of a direction exist, that
        side returns 0.0.

    ---------------------------------------------------------------------------
    Example
    ---------------------------------------------------------------------------
    >>> events, runs = compute_directional_change_events(prices, theta=0.02)
    >>> runs = attach_OSV_EXT_to_runs(runs, theta=0.02)
    >>> mu_up, mu_down = up_down_asymmetry(runs, theta=0.02)
    >>> print(mu_up, mu_down)
    0.73, 0.64
    """
    # Map textual run types to numeric direction (+1 / -1)
    dir_map = {"upward_run": 1, "downward_run": -1}

    # Filter runs that have valid OSV_EXT values
    valid = [r for r in runs if r.get("OSV_EXT") is not None]
    if not valid:
        return 0.0, 0.0

    # Extract arrays of overshoots (already in units of theta) and direction signs
    osv = np.array([r["OSV_EXT"] for r in valid], dtype=float)
    dirs = np.array([dir_map.get(r.get("type"), 0) for r in valid], dtype=int)

    # OSV_EXT is ALREADY normalized by theta → use absolute values directly
    rel_osv = np.abs(osv)

    # Separate up and down directions
    up = rel_osv[dirs > 0]
    down = rel_osv[dirs < 0]
    mu_up = np.mean(up) if up.size else 0.0
    mu_down = np.mean(down) if down.size else 0.0

    # Define the squashing function explicitly
    def squash_fn(x: float) -> float:
        """Smooth exponential mapping to (0, 1]."""
        return 1.0 - np.exp(-x / squash)

    # Apply squashing and clip to valid range
    mu_up_norm = squash_fn(mu_up)
    mu_down_norm = squash_fn(mu_down)

    return float(np.clip(mu_up_norm, 0.0, 1.0)), float(np.clip(mu_down_norm, 0.0, 1.0))
