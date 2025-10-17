import numpy as np
from scipy.linalg import svdvals
from scipy.optimize import minimize
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity


def handle_fallback(n_assets: int, return_matrix: bool = False):
    """Fallback case if there is not enough valid data."""
    dist = np.zeros((n_assets, n_assets))
    identity = np.eye(n_assets)
    if return_matrix:
        return dist, identity
    else:
        return dist


def cov2corr(cov):
    """Convert a covariance matrix to a correlation matrix."""
    s = np.sqrt(np.diag(cov))
    corr = cov / np.outer(s, s)
    corr = np.clip(corr, -1, 1)  # equivalent to corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def mpPDF(var, q, pts=1000):
    """Marcenko-Pastur probability density function."""
    eMin = var * (1 - (1.0 / np.sqrt(q))) ** 2
    eMax = var * (1 + (1.0 / np.sqrt(q))) ** 2
    eVal = np.linspace(eMin, eMax, pts).flatten()
    pdf = q / (2 * np.pi * var * eVal) * np.sqrt((eMax - eVal) * (eVal - eMin))
    return eVal, pdf


def fitKDE(obs, bWidth=0.01, kernel="gaussian", x=None):
    """Fit a Kernel Density Estimator to observed eigenvalues."""
    obs = np.atleast_2d(obs).T
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    x = np.atleast_2d(np.unique(obs) if x is None else x).T
    log_densities = kde.score_samples(x)
    return x.ravel(), np.exp(log_densities)


def errPDFs(var, eVal, q, bWidth):
    """Calculate error between theoretical MP PDF and empirical PDF."""
    eVal_pdf, pdf0 = mpPDF(var, q)
    x, pdf1 = fitKDE(eVal, bWidth, x=eVal_pdf)
    # Since x should be the same as eVal_pdf from mpPDF, we can directly compare
    return np.sum((pdf1 - pdf0) ** 2)


def findMaxEval(eVal, q, bWidth):
    """Find maximum random eigenvalue using MP distribution."""
    res = minimize(
        lambda v: errPDFs(v, eVal, q, bWidth),
        x0=np.array([0.5]),
        bounds=[(1e-5, 1 - 1e-5)],
        method="L-BFGS-B",
    )
    var = res.x[0] if res.success else 1.0
    eMax = var * (1 + (1.0 / np.sqrt(q))) ** 2
    return eMax, var


def findOptimalBWidth(eigenvalues):
    """Estimate optimal KDE bandwidth via leave-one-out CV."""
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim == 1:
        eigenvalues = eigenvalues[:, None]
    bandwidths = 10 ** np.linspace(-2, 1, 100)
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=LeaveOneOut(),
        n_jobs=-1,
    )
    grid.fit(eigenvalues)
    return grid.best_params_["bandwidth"]


def discrete_entropy(x: np.ndarray) -> float:
    """Estimate discrete Shannon entropy (bits) from integer-encoded array."""
    if len(x) == 0:
        return 0.0
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)  # base-2 entropy, matches pyitlib default


def discrete_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate discrete mutual information (bits) between integer-encoded arrays."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    mi = mutual_info_score(x, y)
    return mi / np.log(2)  # convert from natural log (nats) to bits


def compute_matrix_information(A: np.ndarray, eps: float = 1e-12) -> dict:
    """
    Compute spectral information and isotropy measures for a symmetric normalized matrix.
    This function quantifies how evenly the singular values (or equivalently,
    the eigenvalue spectrum for symmetric matrices) of a square matrix are
    distributed. It provides entropy-based indicators of isotropy, which can
    be used to assess how structured a similarity or correlation matrix is
    (e.g., how well the data might cluster).
    The function is metric-agnostic: it can be applied to matrices built from
    normalized mutual information (NMI), distance correlation, Pearson/Spearman
    correlations, kernel similarities, or any other symmetric normalized measure.
    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Square, symmetric input matrix (e.g., pairwise normalized similarity or
        correlation matrix). Matrix entries should be on a comparable scale,
        typically in the interval [0, 1].
    eps : float, default=1e-12
        Numerical tolerance below which singular values are treated as zero.
    Returns
    -------
    info : dict
        Dictionary containing the following keys:
        - 'rank' : int
            Numerical rank = number of singular values greater than eps.
        - 'spectral_entropy' : float
            Shannon entropy (in nats) of the normalized singular values
            :math:`H = -\\sum_i p_i \\log(p_i)` with
            :math:`p_i = s_i / \\sum_j s_j`, computed only over singular values > eps.
            The maximum possible value is :math:`\\log(n)`, achieved when
            all singular values are equal (perfect isotropy).
        - 'effective_rank' : float
            :math:`r_{\\mathrm{eff}} = \\exp(H)`.
            Continuous analog of matrix rank, bounded by
            :math:`1 \\leq r_{\\mathrm{eff}} \\leq n`.
        - 'condition_number' : float
            Ratio :math:`s_{\\max} / s_{\\min}` (∞ if any singular value ≤ eps).
            Equals 1 for perfectly isotropic matrices and increases with
            anisotropy.
        - 'isotropy_ratio' : float
            :math:`s_{\\min} / s_{\\max}`; close to 1 indicates isotropy,
            values ≪1 indicate anisotropy or directional dominance.
        - 'singular_values' : np.ndarray
            Singular values sorted in descending order.
    Interpretation
    --------------
    - High 'spectral_entropy' (≈ log(n)) and 'effective_rank' (≈ n):
        → The singular values are nearly uniform.
            The matrix is spectrally isotropic, implying evenly distributed
            relationships and little or no cluster structure.
    - Low 'spectral_entropy', 'effective_rank' ≪ n, and high 'condition_number':
        → A small number of dominant singular values.
            The matrix is spectrally anisotropic, indicating strong latent
            structure or potential clustering directions.
    Mathematical note
    -----------------
    For a square matrix :math:`A \\in \\mathbb{R}^{n \\times n}` with singular
    values :math:`s_1,\\ldots,s_n > 0`, the Shannon entropy of the normalized
    spectrum satisfies:
    :math:`0 \\le H \\le \\log(n)`,
    and therefore the effective rank is bounded by:
    :math:`1 \\le r_{\\mathrm{eff}} = e^H \\le n`.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}.")
    s = svdvals(A)
    rank = int(np.sum(s > eps))
    mask = s > eps
    if not np.any(mask):
        entropy = 0.0
        eff_rank = 1.0
    else:
        s_valid = s[mask]
        total = s_valid.sum()
        if total <= eps:
            entropy = 0.0
            eff_rank = 1.0
        else:
            p = s_valid / total
            entropy = -np.sum(p * np.log(p))
            eff_rank = float(np.exp(entropy))
    if np.any(s <= eps):
        cond_number = np.inf
        isotropy_ratio = 0.0
    else:
        cond_number = float(s.max() / s.min())
        isotropy_ratio = float(s.min() / s.max())
    return {
        "rank": rank,
        "spectral_entropy": float(entropy),
        "effective_rank": eff_rank,
        "condition_number": cond_number,
        "isotropy_ratio": isotropy_ratio,
        "singular_values": s,
    }


def distance_to_similarity_rbf(D: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    Convert a pairwise distance matrix into a similarity matrix via the Gaussian (RBF) kernel.

    S_ij = exp(- D_ij^2 / (2 * sigma^2))

    Parameters
    ----------
    D : np.ndarray (n x n)
        Symmetric distance matrix with zeros on the diagonal.
    sigma : float, optional
        Bandwidth parameter controlling decay. If None, it is set automatically to
        the median of nonzero distances (a common heuristic).

    Returns
    -------
    S : np.ndarray (n x n)
        Symmetric similarity matrix in [0, 1].
        Large S_ij means high similarity (small distance).
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square (n×n) distance matrix")

    # Ensure we have no negative or NaN distances
    D = np.nan_to_num(D, nan=0.0)
    D[D < 0] = 0.0

    # Choose sigma automatically if not supplied — median heuristic
    if sigma is None:
        nonzero = D[D > 0]
        if nonzero.size == 0:
            sigma = 1.0
        else:
            sigma = np.median(nonzero)

    # Apply Gaussian (RBF) kernel
    S = np.exp(-(D**2) / (2 * sigma**2))

    # Ensure perfect self-similarity
    np.fill_diagonal(S, 1.0)

    return S
