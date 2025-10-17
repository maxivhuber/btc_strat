import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import MDS


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
    """
    Plot price series with directional‑change runs (up = green, down = red).
    Following Option A: each run is drawn only up to its turning extreme (t_ext),
    not through the confirmation move.
    """
    # --- 1️⃣  sanity checks ---------------------------------------------------
    if len(datetimes) != len(prices):
        raise ValueError(
            f"Length mismatch: datetimes({len(datetimes)}) ≠ prices({len(prices)})"
        )

    # Convert datetimes if they're strings/datetime64
    if isinstance(datetimes[0], (str, np.datetime64)):
        import pandas as pd

        datetimes = pd.to_datetime(datetimes).values
    else:
        # Assume datetimes are already in a format plotly can handle
        pass

    # Handle date range filtering
    mask = np.ones(len(datetimes), dtype=bool)
    if start is not None:
        import pandas as pd

        start_ts = pd.to_datetime(start).to_datetime64()
        mask = mask & (datetimes >= start_ts)
    if end is not None:
        import pandas as pd

        end_ts = pd.to_datetime(end).to_datetime64()
        mask = mask & (datetimes <= end_ts)

    if not np.any(mask):
        raise ValueError("No data points in the selected date range.")

    # Apply mask
    filtered_datetimes = datetimes[mask]
    filtered_prices = prices[mask]

    # Create mapping from original indices to filtered indices
    original_to_filtered = np.where(mask)[0]

    # --- 2️⃣  base price line -------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_datetimes,
            y=filtered_prices,
            mode="lines",
            line=dict(color="lightgray", width=1),
            showlegend=False,
        )
    )

    # --- 3️⃣  overlay runs ----------------------------------------------------
    for run in runs:
        # ✅  use t_ext (extreme) instead of end_t
        run_start = run["start_t"]
        run_ext = run["t_ext"]

        # guard bounds
        if run_ext is None or run_ext >= len(datetimes) or run_start >= len(datetimes):
            continue
        if run_ext <= run_start:
            continue

        # Check if run is within filtered range
        if run_ext < original_to_filtered[0] or run_start > original_to_filtered[-1]:
            continue

        # Find indices in the filtered data
        start_in_filtered = np.searchsorted(original_to_filtered, run_start)
        ext_in_filtered = np.searchsorted(original_to_filtered, run_ext)

        if start_in_filtered >= len(original_to_filtered) or ext_in_filtered >= len(
            original_to_filtered
        ):
            continue

        # Get the actual filtered indices
        start_idx = start_in_filtered
        end_idx = ext_in_filtered + 1

        if end_idx > len(filtered_datetimes) or start_idx >= len(filtered_datetimes):
            continue

        seg_times = filtered_datetimes[start_idx:end_idx]
        seg_prices = filtered_prices[start_idx:end_idx]

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
                showlegend=False,
            )
        )

    # --- 4️⃣  event markers ---------------------------------------------------
    if mark_events and events:
        up_events, down_events = [], []
        for e in events:
            event_time = datetimes[e["t_ext"]]
            if start is not None:
                import pandas as pd

                start_ts = pd.to_datetime(start).to_datetime64()
                if event_time < start_ts:
                    continue
            if end is not None:
                import pandas as pd

                end_ts = pd.to_datetime(end).to_datetime64()
                if event_time > end_ts:
                    continue
            if e["type"].startswith("down"):
                down_events.append(e)
            else:
                up_events.append(e)

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

    # --- 5️⃣  legend dummy traces --------------------------------------------
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

    # --- 6️⃣  layout ----------------------------------------------------------
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


def plot_asset_clustering(
    distance_matrix: np.ndarray,
    asset_indices: np.ndarray | None = None,
    cluster_labels: np.ndarray | None = None,
    representative_indices: np.ndarray | None = None,
    n_init: int = 4,
    random_state: int = 0,
) -> None:
    """
    Plot asset clustering in 2D using MDS + Plotly.

    If `cluster_labels` and `representative_indices` are provided:
        - Non-representatives are colored by cluster.
        - Representatives are shown as red stars.
    If only `distance_matrix` is provided:
        - All assets are plotted as uniform gray circles (no clustering shown).

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square precomputed distance matrix with shape (N, N).
    asset_indices : np.ndarray, optional
        Indices or labels for each asset (aligned with distance_matrix rows/columns).
    cluster_labels : np.ndarray, optional
        Cluster ID for each asset (aligned with distance_matrix rows).
    representative_indices : np.ndarray, optional
        Indices to highlight as representatives (medoids).
    random_state : int, default 0
        Random seed for MDS reproducibility.
    """
    n_assets = distance_matrix.shape[0]
    if asset_indices is None:
        asset_indices = np.arange(n_assets)

    # Compute 2D embedding
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        normalized_stress=True,
    )
    embedding = mds.fit_transform(distance_matrix)
    print(
        f"MDS stress: {mds.stress_:.3f} (normalized Stress-1; <0.025 excellent, >0.2 poor)"
    )

    fig = go.Figure()

    # Simple mode: no clustering info provided → plot all points uniformly
    if cluster_labels is None or representative_indices is None:
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    color="lightgray",
                    symbol="circle",
                    size=8,
                    line=dict(width=0.5, color="black"),
                ),
                name="Asset",
                hovertext=asset_indices,
                hoverinfo="text",
                showlegend=True,
            )
        )
    else:
        is_rep = np.isin(np.arange(len(asset_indices)), representative_indices)

        unique_clusters = np.unique(cluster_labels)
        color_palette = pc.qualitative.Plotly
        cluster_to_color = {
            cluster: color_palette[i % len(color_palette)]
            for i, cluster in enumerate(unique_clusters)
        }

        # Non-representatives
        non_rep_idx = np.where(~is_rep)[0]
        if len(non_rep_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=embedding[non_rep_idx, 0],
                    y=embedding[non_rep_idx, 1],
                    mode="markers",
                    marker=dict(
                        color=[
                            cluster_to_color[cluster_labels[i]] for i in non_rep_idx
                        ],
                        symbol="circle",
                        size=8,
                        line=dict(width=0.5, color="black"),
                    ),
                    name="Other",
                    hovertext=[asset_indices[i] for i in non_rep_idx],
                    hoverinfo="text",
                    legendgroup="other",
                    showlegend=True,
                )
            )

        # Representatives
        rep_idx = np.where(is_rep)[0]
        if len(rep_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=embedding[rep_idx, 0],
                    y=embedding[rep_idx, 1],
                    mode="markers",
                    marker=dict(
                        color="crimson",
                        symbol="star",
                        size=12,
                        line=dict(width=1, color="black"),
                    ),
                    name="Representative",
                    hovertext=[asset_indices[i] for i in rep_idx],
                    hoverinfo="text",
                    legendgroup="rep",
                    showlegend=True,
                )
            )

    fig.update_layout(
        title="Asset Embedding in Distance Space (MDS)",
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        legend=dict(
            title="Role",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(r=200),
        height=600,
    )
    fig.show()


def plot_dd_correlation_matrices(X_original, X_detoned, title_prefix=""):
    """
    Plot correlation matrices before and after denoising and detoning.

    Parameters:
    - X_original: np.ndarray, shape (n_samples, n_assets) — robustly scaled log returns (pre-denoising/detoning)
    - X_detoned: np.ndarray, shape (n_samples, n_assets) — returns after denoising and detoning
    - title_prefix: str, optional prefix for plot titles
    """

    def safe_corrcoef_numpy(X):
        """Compute correlation matrix, handling zero-variance columns by setting corr=0."""
        n = X.shape[1]
        stds = X.std(axis=0, ddof=1)

        # If all columns have zero variance, return a zero matrix
        if not np.any(stds > 1e-12):
            return np.zeros((n, n))

        # Center the data by subtracting the mean
        X_centered = X - X.mean(axis=0)

        # Standardize: divide by standard deviation. Use 'where' to avoid division by zero.
        # If std is below the threshold, the standardized value becomes 0.
        X_standardized = np.divide(
            X_centered,
            stds,
            out=np.zeros_like(X_centered),  # Where condition is False, result is 0
            where=stds > 1e-12,  # Standardize only if std is above the threshold
        )

        # Calculate the correlation matrix from the standardized data
        corr_full = np.corrcoef(X_standardized, rowvar=False)

        # Replace any NaN, inf, or -inf values with 0
        return np.nan_to_num(corr_full, nan=0.0, posinf=0.0, neginf=0.0)

    corr_before = safe_corrcoef_numpy(X_original)
    corr_after = safe_corrcoef_numpy(X_detoned)

    n_assets = corr_before.shape[0]
    asset_labels = [f"Asset_{i}" for i in range(n_assets)]  # Default labels

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{title_prefix}Correlation Before Denoising/Detoning",
            f"{title_prefix}Correlation After Denoising/Detoning",
        ),
        horizontal_spacing=0.08,
    )

    # Common heatmap settings
    heatmap_kwargs = dict(
        x=asset_labels,
        y=asset_labels,
        colorscale="RdBu_r",  # Idiomatic: blue (positive), red (negative)
        zmin=-1,
        zmax=1,
        colorbar=dict(len=0.8, y=0.5, thickness=20),
    )

    fig.add_trace(
        go.Heatmap(z=corr_before, showscale=False, name="Before", **heatmap_kwargs),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(z=corr_after, showscale=True, name="After", **heatmap_kwargs),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{title_prefix}Asset Correlation Matrices",
        width=1200,
        height=600,
        font=dict(size=10),
        xaxis=dict(tickangle=45, automargin=True),
        xaxis2=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True),
        yaxis2=dict(automargin=True),
    )

    fig.show()


def plot_similarity_matrices(similarity_matrices, titles=None, title_prefix=""):
    """
    Plot multiple similarity matrices in subplots, rescaling each to [0, 1] for visualization.

    Parameters
    ----------
    similarity_matrices : list of np.ndarray
        Each (n_assets x n_assets) matrix — similarity matrices to plot.
    titles : list of str, optional
        Titles for each subplot. If None, auto-generated.
    title_prefix : str, optional
        Prefix added to all subplot titles.
    """
    n_matrices = len(similarity_matrices)

    if titles is None:
        titles = [f"Similarity Matrix {i + 1}" for i in range(n_matrices)]
    elif len(titles) != n_matrices:
        raise ValueError(
            f"Number of titles ({len(titles)}) must match number of matrices ({n_matrices})"
        )

    # --- Rescale each matrix to [0, 1] purely for visualization ---
    scaled_matrices = []
    for idx, M in enumerate(similarity_matrices):
        M = np.asarray(M, dtype=float)
        min_val, max_val = np.min(M), np.max(M)
        if max_val > min_val:
            M_scaled = (M - min_val) / (max_val - min_val)
            print(
                f"[Rescale] '{titles[idx]}' range: "
                f"({min_val:.4f}, {max_val:.4f}) → (0, 1)"
            )
        else:
            M_scaled = np.zeros_like(M)
            print(f"[Rescale] '{titles[idx]}' constant matrix; set to zeros.")
        scaled_matrices.append(M_scaled)

    # --- Compute global min/max for consistent color scaling across plots (after rescaling) ---
    all_values = np.concatenate([M.flatten() for M in scaled_matrices])
    zmin_global = np.min(all_values)
    zmax_global = np.max(all_values)

    # Avoid degenerate colorbar
    if zmin_global == zmax_global:
        zmin_global -= 0.1
        zmax_global += 0.1

    # --- Subplot grid layout ---
    n_cols = int(np.ceil(np.sqrt(n_matrices)))
    n_rows = int(np.ceil(n_matrices / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{title_prefix}{t}" for t in titles],
        horizontal_spacing=0.1,
        vertical_spacing=0.15 if n_rows > 1 else 0.2,
    )

    # --- Add each similarity matrix as a heatmap ---
    for idx, (M, title) in enumerate(zip(scaled_matrices, titles)):
        row, col = (idx // n_cols) + 1, (idx % n_cols) + 1
        n_assets = M.shape[0]
        asset_labels = [f"Asset_{i}" for i in range(n_assets)]

        heatmap_kwargs = dict(
            x=asset_labels,
            y=asset_labels,
            colorscale="RdBu_r",
            zmin=zmin_global,
            zmax=zmax_global,
            name=title,
        )

        if idx == 0:
            heatmap_kwargs["colorbar"] = dict(len=0.8, y=0.5, thickness=20)
            fig.add_trace(
                go.Heatmap(z=M, showscale=True, **heatmap_kwargs), row=row, col=col
            )
        else:
            fig.add_trace(
                go.Heatmap(z=M, showscale=False, **heatmap_kwargs), row=row, col=col
            )

    # --- Layout adjustments ---
    base_width, base_height = 900, 600
    width = min(2000, n_cols * base_width)
    height = min(2000, n_rows * base_height)
    fig.update_layout(
        title=f"{title_prefix}Similarity Matrices (Rescaled for visualization)",
        width=width,
        height=height,
        font=dict(size=10),
    )

    # --- Axis readability ---
    for i in range(1, n_matrices + 1):
        fig.update_xaxes(
            tickangle=45,
            automargin=True,
            row=(i - 1) // n_cols + 1,
            col=(i - 1) % n_cols + 1,
        )
        fig.update_yaxes(
            automargin=True,
            row=(i - 1) // n_cols + 1,
            col=(i - 1) % n_cols + 1,
        )

    fig.show()
