import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_timestamp_units(series: pd.Series, label: str = "") -> pd.Series:
    s = series.astype(float)
    mask_us = s > 1e14  # microseconds
    mask_ns = s > 1e17  # nanoseconds
    mask_ms = ~(mask_us | mask_ns)
    n_us, n_ns = mask_us.sum(), mask_ns.sum()
    if n_us:
        s.loc[mask_us] /= 1_000.0
    if n_ns:
        s.loc[mask_ns] /= 1_000_000.0
    return s


def _load_symbol(
    symbol_path: Path, interval: str, range_folder: str, columns: list[str]
) -> pd.DataFrame:
    symbol = symbol_path.name
    data_folder = symbol_path / interval / range_folder
    if not data_folder.exists():
        return pd.DataFrame()

    frames = []
    for zip_path in sorted(data_folder.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=columns,
                    dtype={
                        "Open Time": "float64",
                        "Close Time": "float64",
                        "Open": "float64",
                        "High": "float64",
                        "Low": "float64",
                        "Close": "float64",
                        "Volume": "float64",
                        "Quote Asset Volume": "float64",
                        "Number of Trades": "int32",
                        "Taker Buy Base Volume": "float64",
                        "Taker Buy Quote Volume": "float64",
                        "Ignore": "float64",
                    },
                )
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalize timestamps
    df["Open Time"] = _normalize_timestamp_units(df["Open Time"])
    df["Close Time"] = _normalize_timestamp_units(df["Close Time"])

    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", errors="coerce")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", errors="coerce")

    df = df.set_index("Open Time").sort_index()

    # Drop duplicates
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]

    # Ensure hourly frequency
    df = df.asfreq("1h")

    # Handle missing data
    if df.isna().any(axis=1).any():
        df = df.ffill()

        price_cols = ["Open", "High", "Low", "Close"]
        log_prices = np.log(df[price_cols])
        interpolated = log_prices.interpolate(method="time", limit_direction="both")
        df[price_cols] = np.exp(interpolated)

        other_num_cols = df.select_dtypes(include=[np.number]).columns.difference(
            price_cols
        )
        df[other_num_cols] = df[other_num_cols].interpolate(
            method="time", limit_direction="both"
        )

        if df.isna().any(axis=1).any():
            raise ValueError(
                f"{symbol}: Missing data remains after interpolation → abort."
            )

    df["Symbol"] = symbol
    return df


def load_all_klines(
    root: Path | str = "data/spot/monthly/klines",
    interval: str = "1h",
    range_folder: str = "2017-01-01_2025-10-04",
    min_years: float = 4.0,
) -> pd.DataFrame:
    """
    Load and optionally filter klines data for USDT symbols.

    Parameters
    ----------
    root : Path or str
        Root directory containing symbol folders (e.g., .../klines/BTCUSDT/...).
    interval : str, default "1h"
        Candle interval.
    range_folder : str, default "2017-01-01_2025-10-04"
        Subfolder name with date range.
    min_years : float, default 4.0
        Minimum data duration per symbol (in years). Symbols with less are dropped.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame: ["Symbol", "Open Time"].
    """
    root = Path(root)
    columns = [
        "Open Time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close Time",
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base Volume",
        "Taker Buy Quote Volume",
        "Ignore",
    ]

    all_dfs = []
    for symbol_path in sorted(root.glob("*USDT")):
        df_coin = _load_symbol(symbol_path, interval, range_folder, columns)
        if not df_coin.empty:
            all_dfs.append(df_coin)

    if not all_dfs:
        raise FileNotFoundError(f"No valid symbol data found under {root}")

    df_all = pd.concat(all_dfs, ignore_index=False)
    df_all = (
        df_all.set_index("Symbol", append=True)
        .reorder_levels(["Symbol", "Open Time"])
        .sort_index()
    )

    # Filter by minimum time span
    if min_years > 0:
        min_td = pd.Timedelta(days=min_years * 365.25)
        time_spans = df_all.groupby(level="Symbol").apply(
            lambda g: g.index.get_level_values("Open Time")[-1]
            - g.index.get_level_values("Open Time")[0]
        )
        valid_symbols = time_spans[time_spans >= min_td].index
        dropped = (
            df_all.index.get_level_values("Symbol").unique().difference(valid_symbols)
        )

        if len(dropped) > 0:
            print(f"\nDropped {len(dropped)} symbols with < {min_years} years of data:")
            for sym in sorted(dropped):
                span_days = time_spans.get(sym, pd.Timedelta(0)).days
                print(f"  {sym}: {span_days} days")

        df_all = df_all[df_all.index.get_level_values("Symbol").isin(valid_symbols)]

    # Final summary (only shape & symbols)
    symbols = df_all.index.get_level_values("Symbol").unique()
    print(f"\nLoaded {len(symbols)} symbols | Shape: {df_all.shape}")
    return df_all
