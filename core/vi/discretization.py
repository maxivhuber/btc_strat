import numpy as np


def num_bins(n_obs: int, corr: float | None = None) -> int:
    """
    Compute optimal number of bins for discretization.
    - Univariate: Hacine-Gharbi (2012)
    - Bivariate:  Hacine-Gharbi & Ravier (2018)
    """
    if n_obs < 2:
        return 1
    if corr is None:
        # Hacine-Gharbi (2012) univariate optimal binning
        # B_X = round(zeta / 6 + 2 / (3 * zeta) + 1 / 3)
        # zeta = (8 + 324*N + 12*(36*N + 729*N^2)^0.5)^(1/3)
        z = (8 + 324 * n_obs + 12 * np.sqrt(36 * n_obs + 729 * n_obs**2)) ** (1 / 3)
        b = round(z / 6 + 2 / (3 * z) + 1 / 3)
    else:
        # Hacine-Gharbi & Ravier (2018) bivariate optimal binning
        # B_X = B_Y = round(2^(-0.5) * (1 + (1 + 24*N / (1 - rho^2))^0.5)^0.5)
        rho = np.clip(abs(corr), 0.0, 0.999999)
        b = round(
            (1 / np.sqrt(2)) * np.sqrt(1 + np.sqrt(1 + 24 * n_obs / (1 - rho**2)))
        )
    return max(1, int(b))


def discretize_series(x: np.ndarray, bins: int) -> np.ndarray:
    """
    Discretize a continuous vector into integer bin labels.
    """
    _, edges = np.histogram(x, bins=bins)
    labels = np.digitize(x, edges[:-1], right=True)
    return labels
