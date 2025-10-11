from pathlib import Path

import numpy as np
import pandas as pd


def load_symbol(symbol_path: Path) -> pd.DataFrame:
    data_file = symbol_path / "data.csv"
    if not data_file.exists() or data_file.stat().st_size == 0:
        return pd.DataFrame()

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

    df = pd.read_csv(
        data_file,
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

    if df.empty:
        return df

    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", utc=True)
    df = df.set_index("Open Time").sort_index()

    # Drop duplicates and ensure 1h alignment
    df = df[~df.index.duplicated(keep="first")]
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
                f"{symbol_path.name}: Missing data remains after interpolation → abort."
            )

    df["Symbol"] = symbol_path.name
    return df


def load_all_symbols(
    root: str | Path = "data/spot/hourly/klines", min_years: int = 4
) -> pd.DataFrame:
    root = Path(root)
    all_dfs = []

    for symbol_path in sorted(root.iterdir()):
        if symbol_path.is_dir() and (symbol_path / "data.csv").exists():
            df = load_symbol(symbol_path)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No valid symbol CSVs found under {root}")

    df_all = pd.concat(all_dfs, ignore_index=False)
    df_all = (
        df_all.set_index("Symbol", append=True)
        .reorder_levels(["Symbol", "Open Time"])
        .sort_index()
    )

    # Filter symbols by minimum duration
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

    symbols = df_all.index.get_level_values("Symbol").unique()
    print(f"\nLoaded {len(symbols)} symbols | Shape: {df_all.shape}")
    return df_all
