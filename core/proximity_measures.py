import dcor
import numpy as np

from core.utils import discrete_entropy, discrete_mutual_information, handle_fallback
from core.vi.discretization import discretize_series, num_bins


# -----------------------------------------------------------------------------
# Variation of Information Based Distance
# -----------------------------------------------------------------------------
def compute_vi_distance_matrix(
    X: np.ndarray,
    normalized: str | None = "doubletilde",  # None, "tilde", or "doubletilde"
    fallback_value: float = 1.0,
):
    """
    Compute distance matrix using the Variation of Information (VI) metric.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples x n_features)
    normalized : str or None, optional
        - None: raw VI (VI[X, Y] = H[X|Y] + H[Y|X] = H[X] + H[Y] - 2*I[X, Y])
        - 'tilde': Kraskov normalization (VI~[X, Y] = VI[X, Y] / H[X, Y])
        - 'doubletilde': ratio of conditional-to-marginal entropies (VI^[X, Y] = max{H[X|Y], H[Y|X]} / max{H[X], H[Y]})
    fallback_value : float
        Value used if entropy or MI undefined (e.g., constant features)

    Returns
    -------
    VI : np.ndarray
        Symmetric variation-of-information distance matrix
    """
    n_features = X.shape[1]

    # initialize output matrix
    VI = np.zeros((n_features, n_features))
    MI = np.zeros((n_features, n_features))

    # remove rows with NaN for correlation calculation
    X_clean = X.copy()
    X_clean = X_clean[~np.isnan(X_clean).any(axis=1)]
    if X_clean.shape[0] < 2:
        # fallback if no valid data
        return np.full((n_features, n_features), fallback_value)

    # compute correlation matrix manually
    X_centered = X_clean - X_clean.mean(axis=0)
    X_std = X_centered.std(axis=0, ddof=1)
    X_std = np.where(X_std > 1e-12, X_std, 1.0)
    X_normalized = X_centered / X_std
    corr_mat = np.corrcoef(X_normalized, rowvar=False)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # iterate feature pairs
    for i in range(n_features):
        for j in range(i, n_features):
            x_raw = X[:, i]
            y_raw = X[:, j]
            valid_mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
            x_clean = x_raw[valid_mask]
            y_clean = y_raw[valid_mask]

            if len(x_clean) == 0:
                vi_val = fallback_value
                mi_val = 0.0
            else:
                if i == j:
                    # VI[X, X] = 0; I[X, X] = H[X]
                    x_disc = discretize_series(x_clean, num_bins(len(x_clean)))
                    h_x = discrete_entropy(x_disc)
                    vi_val = 0.0
                    mi_val = h_x
                else:
                    # adaptive binning based on correlation
                    rho = corr_mat[i, j]
                    b = num_bins(len(x_clean), corr=rho)
                    x_disc = discretize_series(x_clean, b)
                    y_disc = discretize_series(y_clean, b)

                    degenerate_x = len(np.unique(x_disc)) <= 1
                    degenerate_y = len(np.unique(y_disc)) <= 1

                    if degenerate_x or degenerate_y:
                        h_x = 0.0 if degenerate_x else discrete_entropy(x_disc)
                        h_y = 0.0 if degenerate_y else discrete_entropy(y_disc)
                        mi_val = 0.0
                        vi_val = h_x + h_y
                    else:
                        h_x = discrete_entropy(x_disc)
                        h_y = discrete_entropy(y_disc)
                        mi_val = discrete_mutual_information(x_disc, y_disc)
                        vi_val = h_x + h_y - 2 * mi_val

                    # normalization
                    if normalized == "tilde":
                        h_joint = h_x + h_y - mi_val
                        vi_val = vi_val / h_joint if h_joint > 0 else fallback_value
                    elif normalized == "doubletilde":
                        h_x_given_y = h_x - mi_val
                        h_y_given_x = h_y - mi_val
                        max_cond = max(h_x_given_y, h_y_given_x)
                        max_marg = max(h_x, h_y)
                        vi_val = max_cond / max_marg if max_marg > 0 else fallback_value

            VI[i, j] = VI[j, i] = max(vi_val, 0.0)
            MI[i, j] = MI[j, i] = mi_val

    return VI


# -----------------------------------------------------------------------------
# Distance Correlation (dCor) Based Distance
# -----------------------------------------------------------------------------
def compute_dcor_similarity_matrix(prices: np.ndarray):
    """Compute similarity matrix using Distance Correlation."""
    if prices.size == 0 or prices.shape[1] < 2:
        n_features = prices.shape[1] if prices.size > 0 else 0
        return handle_fallback(n_features)

    n = prices.shape[1]
    if n == 1:
        return np.array([[1.0]])

    corr_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            x = prices[:, i]
            y = prices[:, j]
            valid = ~(np.isnan(x) | np.isnan(y))

            if valid.sum() >= 2:
                corr_mat[i, j] = corr_mat[j, i] = dcor.distance_correlation(
                    x[valid], y[valid]
                )

    return corr_mat
