"""Load raw 5-minute OHLCV CSVs for a single stock.

Replaces the old ``load-data.py``, which hardcoded absolute ``/source/data``
paths and loaded three stocks at once. Paths are now built from the
configured (relative) data directory and a stock symbol parameter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def csv_path(symbol: str, data_dir: str | Path) -> Path:
    """Path of the raw CSV for ``symbol`` inside ``data_dir``."""
    return Path(data_dir) / f"{symbol.upper()}_with_indicators_.csv"


def available_symbols(data_dir: str | Path) -> list[str]:
    """Symbols that have a raw CSV present in ``data_dir``."""
    return sorted(
        p.name.replace("_with_indicators_.csv", "")
        for p in Path(data_dir).glob("*_with_indicators_.csv")
    )


def load_stock_csv(
    symbol: str,
    data_dir: str | Path,
    columns: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read the raw 5-minute bars for one stock.

    Only OHLCV columns are read by default (the files also carry ~50
    precomputed indicator columns, which the classical pipeline recomputes
    or ignores). Returns a DataFrame with a raw ``date`` string column —
    datetime parsing/indexing is the preprocessing step's job.
    """
    path = csv_path(symbol, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No data file for symbol {symbol!r} at {path}. "
            f"Available symbols: {available_symbols(data_dir)}"
        )
    return pd.read_csv(path, usecols=columns or OHLCV_COLUMNS, nrows=nrows)
