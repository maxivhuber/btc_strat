import pandas as pd
from scipy.stats import pearsonr


def select_least_correlated_with(
    df: pd.DataFrame,
    reference_symbol: str,
    quantile: float = 0.5,
) -> pd.DataFrame:
    """
    Select assets least correlated (in absolute Pearson correlation) with a given reference symbol,
    and return a filtered DataFrame containing the reference + selected assets.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-index DataFrame with levels ["Symbol", "Open Time"], containing at least a "Close" column.
    reference_symbol : str
        Symbol to use as correlation reference (e.g., "BTCUSDT", "ETHUSDT").
    quantile : float, default 0.5
        Quantile of the absolute correlation distribution to use as threshold.
        Assets with |corr| <= threshold are kept.
        Example: quantile=0.3 → keep 30% least correlated assets.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing `reference_symbol` and the least correlated assets.
    list
        List of selected assets
    """
    symbols = df.index.get_level_values("Symbol").unique()

    if reference_symbol not in symbols:
        raise ValueError(
            f"Reference symbol '{reference_symbol}' not found in data. "
            f"Available symbols: {sorted(symbols)}"
        )

    # Extract reference close prices
    ref_close = df.xs(reference_symbol, level="Symbol")["Close"]

    correlations = {}
    other_symbols = [s for s in symbols if s != reference_symbol]

    for symbol in other_symbols:
        try:
            asset_close = df.xs(symbol, level="Symbol")["Close"]
            common_index = ref_close.index.intersection(asset_close.index)
            if len(common_index) < 2:
                continue
            corr, _ = pearsonr(
                ref_close.loc[common_index], asset_close.loc[common_index]
            )
            correlations[symbol] = abs(corr)
        except Exception:
            continue  # Skip problematic symbols silently

    if not correlations:
        raise ValueError(
            f"No valid correlations could be computed between '{reference_symbol}' and other symbols."
        )

    abs_corr = pd.Series(correlations, name="abs_corr")
    threshold = abs_corr.quantile(quantile)
    print(
        f"Correlation threshold vs {reference_symbol} (q={quantile:.2f}): {threshold:.3f}"
    )

    selected_others = abs_corr[abs_corr <= threshold].index.tolist()
    selected_symbols = [reference_symbol] + selected_others

    # Filter and return
    filtered_df = df.loc[df.index.get_level_values("Symbol").isin(selected_symbols)]
    filtered_df = filtered_df.sort_index(level=["Symbol", "Open Time"])

    print("Selected symbols:", selected_symbols)
    return filtered_df, selected_symbols
