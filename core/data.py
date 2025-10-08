import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_timestamp_units(series: pd.Series, label: str = "") -> pd.Series:
    s = series.astype(float)
    mask_us = s > 1e14  # microseconds
    mask_ns = s > 1e17  # nanoseconds
    mask_ms = ~(mask_us | mask_ns)
    n_ms, n_us, n_ns = mask_ms.sum(), mask_us.sum(), mask_ns.sum()
    print(f"Timestamp units {label:20s}: {n_ms:,} ms | {n_us:,} µs | {n_ns:,} ns")
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
        print(f"{symbol}: missing folder {data_folder} → skip")
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
        print(f"{symbol}: no zip files found, skipping")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {symbol}: {len(df):,} hourly rows")

    df["Open Time"] = _normalize_timestamp_units(df["Open Time"], f"{symbol} Open Time")
    df["Close Time"] = _normalize_timestamp_units(
        df["Close Time"], f"{symbol} Close Time"
    )

    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", errors="coerce")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", errors="coerce")

    df = df.set_index("Open Time").sort_index()
    if not df.index.is_unique:
        dups = df.index.duplicated().sum()
        print(f"{symbol}: {dups:,} duplicate timestamps → dropping")
        df = df[~df.index.duplicated(keep="first")]

    df = df.asfreq("1h")

    missing_before = df.isna().any(axis=1).sum()
    if missing_before:
        print(f"{symbol}: Found {missing_before:,} missing hourly candles → filling...")
        df = df.ffill()

        price_cols = ["Open", "High", "Low", "Close"]
        df[price_cols] = np.exp(
            np.log(df[price_cols]).interpolate(method="time", limit_direction="both")
        )

        num_cols = df.select_dtypes(include=[np.number]).columns
        other_cols = num_cols.difference(price_cols)
        df[other_cols] = df[other_cols].interpolate(
            method="time", limit_direction="both"
        )

        still_missing = df.isna().any(axis=1).sum()
        if still_missing > 0:
            raise ValueError(
                f"{symbol}: {still_missing} rows remain missing after fill → abort."
            )
        print(f"{symbol}: Missing hours filled successfully.")
    else:
        print(f"{symbol}: No missing hourly candles.")

    print(f"{symbol} range: {df.index.min()} → {df.index.max()} | {len(df):,} rows")
    df["Symbol"] = symbol
    return df


def load_all_klines(
    root: Path | str = "data/spot/monthly/klines",
    interval: str = "1h",
    range_folder: str = "2017-01-01_2025-10-04",
) -> pd.DataFrame:
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
        raise FileNotFoundError(f"No valid symbol data folders found under {root}")

    df_all = pd.concat(all_dfs, ignore_index=False)
    df_all = (
        df_all.set_index("Symbol", append=True)
        .reorder_levels(["Symbol", "Open Time"])
        .sort_index()
    )

    print("\n===== FINAL MULTI-SYMBOL SUMMARY =====")
    symbols = df_all.index.get_level_values("Symbol").unique().tolist()
    print(f"Shape: {df_all.shape}")
    print(f"Symbols loaded ({len(symbols)}): {symbols}")
    print(f"Total hourly rows (all): {len(df_all):,}")
    print("\nPreview:")
    print(df_all.head())

    return df_all
