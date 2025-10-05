import numpy as np


def compute_directional_change_events(prices: np.ndarray, theta: float):
    """
    Compute Directional Change (DC) events and corresponding runs (upward/downward)
    with standardized key names compatible with DC attach_* indicator functions.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values ordered in time.
    theta : float
        Directional Change threshold as a relative fraction (e.g. 0.02 = 2%).

    Returns
    -------
    events : list of dict
        Each DC event with 'type', 't_ext', and 'p_ext'.
    runs : list of dict
        Alternating upward_run / downward_run segments, each containing:
            - 'type'
            - 'start_t' and 'end_t'
            - 'p_ext': price at extreme (peak or trough)
            - 't_ext': time index of extreme
    """
    if prices is None or len(prices) == 0:
        return [], []

    prices = np.asarray(prices, dtype=float)
    events, runs = [], []

    n = len(prices)
    last_high = last_low = prices[0]
    last_high_t = last_low_t = 0
    mode = None
    start_index = 0

    # === 1. Find First Directional Change ===
    for i, p in enumerate(prices):
        if p > last_high:
            last_high, last_high_t = p, i
        if p < last_low:
            last_low, last_low_t = p, i

        if mode is None:
            # A. Downward DC: fall theta from last high
            if p <= last_high * (1 - theta):
                events.append(
                    {
                        "type": "downward_dc",
                        "t_ext": last_high_t,
                        "p_ext": last_high,
                    }
                )
                runs.append(
                    {
                        "type": "upward_run",
                        "start_t": 0,
                        "t_ext": last_high_t,
                        "p_ext": last_high,
                        "end_t": i,
                    }
                )
                mode, last_low, last_low_t = "down", p, i
                start_index = i
                break
            # B. Upward DC: rise theta from last low
            elif p >= last_low * (1 + theta):
                events.append(
                    {
                        "type": "upward_dc",
                        "t_ext": last_low_t,
                        "p_ext": last_low,
                    }
                )
                runs.append(
                    {
                        "type": "downward_run",
                        "start_t": 0,
                        "t_ext": last_low_t,
                        "p_ext": last_low,
                        "end_t": i,
                    }
                )
                mode, last_high, last_high_t = "up", p, i
                start_index = i
                break
    else:
        return [], []  # nothing crossed threshold

    # === 2. Continue Scanning for Alternating Events ===
    for j in range(start_index + 1, n):
        p = prices[j]
        if mode == "up":
            if p > last_high:
                last_high, last_high_t = p, j
            if p <= last_high * (1 - theta):
                events.append(
                    {
                        "type": "downward_dc",
                        "t_ext": last_high_t,
                        "p_ext": last_high,
                    }
                )
                runs.append(
                    {
                        "type": "upward_run",
                        "start_t": events[-2]["t_ext"] if len(events) >= 2 else 0,
                        "t_ext": last_high_t,
                        "p_ext": last_high,
                        "end_t": j,
                    }
                )
                mode, last_low, last_low_t = "down", p, j
        elif mode == "down":
            if p < last_low:
                last_low, last_low_t = p, j
            if p >= last_low * (1 + theta):
                events.append(
                    {
                        "type": "upward_dc",
                        "t_ext": last_low_t,
                        "p_ext": last_low,
                    }
                )
                runs.append(
                    {
                        "type": "downward_run",
                        "start_t": events[-2]["t_ext"] if len(events) >= 2 else 0,
                        "t_ext": last_low_t,
                        "p_ext": last_low,
                        "end_t": j,
                    }
                )
                mode, last_high, last_high_t = "up", p, j

    return events, runs


def attach_TMV_EXT_to_runs(runs, theta):
    if len(runs) < 2:
        for run in runs:
            run["TMV_EXT"] = None
        return runs
    # |(P_EXT[n+1] - P_EXT[n]) / P_EXT[n]| / θ
    P_EXT = np.array([run["p_ext"] for run in runs], dtype=float)
    TMV_EXT = np.abs(np.diff(P_EXT) / P_EXT[:-1]) / theta
    TMV_EXT = np.append(TMV_EXT, np.nan)
    for i, run in enumerate(runs):
        run["TMV_EXT"] = None if np.isnan(TMV_EXT[i]) else float(TMV_EXT[i])
    return runs


def attach_OSV_EXT_to_runs(runs, theta):
    if len(runs) < 2:
        for run in runs:
            run["OSV_EXT"] = None
        return runs
    # ((P_EXT[n] - P_EXT[n-1]) / P_EXT[n-1]) / θ
    P_EXT = np.array([run["p_ext"] for run in runs], dtype=float)
    OSV_EXT = np.empty_like(P_EXT)
    OSV_EXT[0] = np.nan
    OSV_EXT[1:] = ((P_EXT[1:] - P_EXT[:-1]) / P_EXT[:-1]) / theta
    for i, run in enumerate(runs):
        run["OSV_EXT"] = None if np.isnan(OSV_EXT[i]) else float(OSV_EXT[i])
    return runs


def attach_T_to_runs(runs):
    if len(runs) < 2:
        for run in runs:
            run["T"] = None
        return runs
    # t_EXT[n] - t_EXT[n-1]
    t_EXT = np.array([run["t_ext"] for run in runs])
    T = np.diff(t_EXT)
    T = np.append(T, np.nan)
    for i, run in enumerate(runs):
        run["T"] = None if np.isnan(T[i]) else float(T[i])
    return runs


def attach_R_to_runs(runs, theta):
    if not runs:
        return runs
    TMV_EXT = np.array([run.get("TMV_EXT", np.nan) for run in runs], dtype=float)
    T = np.array([run.get("T", np.nan) for run in runs], dtype=float)
    # (TMV_EXT / T) * θ
    R = (TMV_EXT / T) * theta
    R[np.isnan(T) | (T == 0)] = np.nan
    for i, run in enumerate(runs):
        run["R"] = None if np.isnan(R[i]) else float(R[i])
    return runs
