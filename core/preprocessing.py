import numpy as np
from scipy.stats import pearsonr
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler

from core.utils import cov2corr, findMaxEval, findOptimalBWidth


def compute_aligned_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute aligned log returns from price matrix.

    Parameters
    ----------
    prices : np.ndarray
        2D array of prices with shape (T, N) where T is time periods and N is assets.
        NaN values should be used for missing data.

    Returns
    -------
    aligned_returns : np.ndarray
        2D array of aligned log returns with shape (T-1, N), with NaN rows removed.
    """
    # Compute log returns
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)  # (T-1, N)

    # Remove rows with any NaN values
    mask = ~np.isnan(returns).any(axis=1)
    aligned_returns = returns[mask]

    return aligned_returns


def select_least_correlated_with(
    prices: np.ndarray,
    reference_idx: int,
    quantile: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select assets least correlated with a given reference asset.

    Parameters
    ----------
    prices : np.ndarray
        2D array of prices with shape (T, N) where T is time periods and N is assets.
    reference_idx : int
        Index of the reference asset (0-based).
    quantile : float, default 0.5
        Quantile of the absolute correlation distribution to use as threshold.

    Returns
    -------
    filtered_prices : np.ndarray
        Filtered price matrix containing reference asset and least correlated assets.
    selected_indices : np.ndarray
        Array of selected asset indices (including the reference).
    """
    T, N = prices.shape

    if reference_idx >= N:
        raise ValueError(
            f"Reference index {reference_idx} out of bounds for {N} assets"
        )

    ref_prices = prices[:, reference_idx]

    # Find valid indices (no NaN in either reference or candidate)
    ref_valid = ~np.isnan(ref_prices)

    abs_corrs = []
    candidate_indices = []

    for i in range(N):
        if i == reference_idx:
            continue

        asset_prices = prices[:, i]
        asset_valid = ~np.isnan(asset_prices)

        # Find common valid indices
        common_valid = ref_valid & asset_valid

        if np.sum(common_valid) < 2:
            continue

        # Compute correlation on valid data
        ref_valid_data = ref_prices[common_valid]
        asset_valid_data = asset_prices[common_valid]

        if len(np.unique(ref_valid_data)) < 2 or len(np.unique(asset_valid_data)) < 2:
            continue

        corr, _ = pearsonr(ref_valid_data, asset_valid_data)
        abs_corrs.append(abs(corr))
        candidate_indices.append(i)

    if not abs_corrs:
        raise ValueError(
            "No valid correlations could be computed with reference asset."
        )

    abs_corrs = np.array(abs_corrs)
    candidate_indices = np.array(candidate_indices)

    threshold = np.quantile(abs_corrs, quantile)
    selected_mask = abs_corrs <= threshold

    selected_asset_indices = candidate_indices[selected_mask]
    selected_indices = np.concatenate([[reference_idx], selected_asset_indices])
    selected_indices = np.sort(selected_indices).astype(int)

    filtered_prices = prices[:, selected_indices]

    return filtered_prices, selected_indices


# -----------------------------------------------------------------------------
# Robust preprocessing + Marcenko–Pastur denoising & detoning
# -----------------------------------------------------------------------------
def compute_denoised_detoned_returns(prices: np.ndarray):
    """
    Computes robustly scaled, denoised, and detoned log returns.

    Parameters
    ----------
    prices : np.ndarray
        2D array of prices with shape (T, N) where T is time periods and N is assets.

    Returns
    -------
    X : np.ndarray
        Original robustly scaled returns (T' x N).
    X_detoned : np.ndarray
        Detoned returns (T' x N).
    """
    # Align log returns
    aligned_returns = compute_aligned_log_returns(prices)

    if aligned_returns.size == 0 or aligned_returns.shape[1] < 2:
        T, N = prices.shape
        if N < 2:
            # Return empty arrays if not enough assets
            T_ret = 0 if T < 2 else T - 1
            return np.empty((T_ret, N)), np.empty((T_ret, N))

        # Try to compute returns even if alignment failed
        if T < 2:
            return np.empty((0, N)), np.empty((0, N))

        log_prices = np.log(prices)
        returns = np.diff(log_prices, axis=0)
        # Remove rows with NaN
        mask = ~np.isnan(returns).any(axis=1)
        aligned_returns = returns[mask]

        if aligned_returns.size == 0 or aligned_returns.shape[1] < 2:
            return np.empty((0, N)), np.empty((0, N))

    rets = aligned_returns

    # 0. PREPROCESS: robust scale
    X = RobustScaler().fit_transform(rets)
    mcd = MinCovDet().fit(X)
    cov_rob = mcd.covariance_
    corr_rob = cov2corr(cov_rob)

    # 1. DENOISE VIA MARCENKO–PASTUR
    eigVal, eigVec = np.linalg.eigh(corr_rob)
    order = eigVal.argsort()[::-1]
    eigVal, eigVec = eigVal[order], eigVec[:, order]

    T, N = X.shape
    q = T / float(N)

    # Estimate optimal bandwidth from eigenvalues
    bWidth_opt = findOptimalBWidth(eigVal)

    eMax, var_mp = findMaxEval(eigVal, q, bWidth=bWidth_opt)
    nFacts = np.sum(eigVal > eMax)  # more explicit and robust
    nFacts = max(1, int(nFacts))  # ensure at least 1 factor (market mode)

    # Denoise: shrink noisy eigenvalues to their mean
    eVal_d = eigVal.copy()
    eVal_d[nFacts:] = eVal_d[nFacts:].mean()
    corr_denoised = eigVec @ np.diag(eVal_d) @ eigVec.T
    corr_denoised = cov2corr(corr_denoised)

    print(f"σ²={var_mp:.4f}, λ₊={eMax:.4f}, nFacts={nFacts}, bWidth={bWidth_opt:.4f}")

    # 2. DETONE — remove market mode from *denoised* correlation
    market_vec = eigVec[:, 0].reshape(-1, 1)  # principal eigenvector (market mode)
    market_corr = eVal_d[0] * (
        market_vec @ market_vec.T
    )  # outer product scaled by eigenvalue

    corr_detoned = corr_denoised - market_corr
    corr_detoned = cov2corr(corr_detoned)

    # Optional: Detoned returns for downstream use
    market_factor = X @ market_vec  # (T, 1)
    market_component = market_factor @ market_vec.T  # (T, N)
    X_detoned = X - market_component  # Detoned returns (T, N)

    print("Denoising + Detoning complete.")

    return X, X_detoned
