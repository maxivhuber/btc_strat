import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_directional_change_runs(
    prices: np.ndarray,
    theta: float,
    events,
    runs,
    datetimes: np.ndarray,
    start=None,
    end=None,
    mark_events: bool = True,
):
    """Plot price series with DC runs (up = green, down = red) and optional DCC‑event markers."""
    # 1️⃣  sanity checks
    if len(datetimes) != len(prices):
        raise ValueError(
            f"Length mismatch: datetimes({len(datetimes)}) ≠ prices({len(prices)})"
        )
    # 🔑 CRITICAL FIX: Ensure datetimes are Pandas-compatible (not raw np.datetime64)
    datetimes = pd.to_datetime(datetimes)
    df = pd.DataFrame({"x": datetimes, "price": prices})
    # Convert start/end to Timestamps once for reuse
    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None
    # Limit range
    if start_ts is not None:
        df = df[df["x"] >= start_ts]
    if end_ts is not None:
        df = df[df["x"] <= end_ts]
    if df.empty:
        raise ValueError("No data points in the selected date range.")

    # 2️⃣  base price line -------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["price"],
            mode="lines",
            line=dict(color="lightgray", width=1),
            showlegend=False,
        )
    )

    # 3️⃣  overlay runs ----------------------------------------------------
    for run in runs:
        run_start, run_end = run["start_t"], run["end_t"]
        seg_times = datetimes[run_start : run_end + 1]
        if start_ts is not None and seg_times[-1] < start_ts:
            continue
        if end_ts is not None and seg_times[0] > end_ts:
            continue
        seg_prices = prices[run_start : run_end + 1]
        color = "green" if run["type"] == "upward_run" else "red"
        name = "Upward Run" if run["type"] == "upward_run" else "Downward Run"
        fig.add_trace(
            go.Scatter(
                x=seg_times,
                y=seg_prices,
                mode="lines",
                line=dict(color=color, width=3),
                name=name,
                legendgroup=name,
                showlegend=False,  # Avoid duplicate legend entries per segment
            )
        )

    # 4️⃣  optional DCC‑event markers ------------------------------------
    if mark_events and events:
        up_events = []
        down_events = []
        for e in events:
            event_time = datetimes[e["t_ext"]]
            if start_ts is not None and event_time < start_ts:
                continue
            if end_ts is not None and event_time > end_ts:
                continue
            if e["type"].startswith("down"):
                down_events.append(e)
            else:
                up_events.append(e)

        # Add up events
        if up_events:
            up_times = [datetimes[e["t_ext"]] for e in up_events]
            up_prices = [float(e["p_ext"]) for e in up_events]
            fig.add_trace(
                go.Scatter(
                    x=up_times,
                    y=up_prices,
                    mode="markers",
                    marker=dict(
                        color="green",
                        size=8,
                        symbol="triangle-up",
                        line=dict(width=0.5, color="black"),
                    ),
                    name="DCC Up Event",
                    legendgroup="DCC Up Event",
                    showlegend=False,
                )
            )

        # Add down events
        if down_events:
            down_times = [datetimes[e["t_ext"]] for e in down_events]
            down_prices = [float(e["p_ext"]) for e in down_events]
            fig.add_trace(
                go.Scatter(
                    x=down_times,
                    y=down_prices,
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=8,
                        symbol="triangle-down",
                        line=dict(width=0.5, color="black"),
                    ),
                    name="DCC Down Event",
                    legendgroup="DCC Down Event",
                    showlegend=False,
                )
            )

    # 5️⃣  legend entries (dummy traces for consistent legend) -----------
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="green", width=3),
            name="Upward Run",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="red", width=3),
            name="Downward Run",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color="green",
                size=8,
                symbol="triangle-up",
                line=dict(width=0.5, color="black"),
            ),
            name="DCC Up Event",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color="red",
                size=8,
                symbol="triangle-down",
                line=dict(width=0.5, color="black"),
            ),
            name="DCC Down Event",
        )
    )

    # 6️⃣  layout ---------------------------------------------------------
    fig.update_layout(
        title=f"Directional Change Runs (θ = {theta:.1%})",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
    )
    fig.show()


def plot_z_returns(z_returns, coins):
    """
    Plot volatility-normalized (z-score) log returns for specified coins.

    Parameters:
    -----------
    z_returns : pd.DataFrame
        DataFrame with datetime index and coin symbols as columns
    coins : list of str
        List of coin symbols to plot (must be present in z_returns columns)
    """
    # Select and clean data
    plot_df = z_returns[coins].dropna(how="any")
    assert not plot_df.isna().any().any(), "Unexpected NaNs"

    # Convert to long format for plotting
    plot_long = plot_df.reset_index().melt(
        id_vars="Open Time", var_name="Symbol", value_name="z_return"
    )

    # Create and show plot
    fig = px.line(
        plot_long,
        x="Open Time",
        y="z_return",
        color="Symbol",
        title="Volatility-normalized (z-score) log returns",
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="z-return (σ units)")
    fig.show()
