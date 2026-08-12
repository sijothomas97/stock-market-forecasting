"""Shared fixtures: small synthetic datasets so tests run in seconds and
don't depend on the ~630MB raw CSVs being present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def five_min_raw(rng) -> pd.DataFrame:
    """~20 trading days of synthetic 5-min OHLCV bars, tz +05:30, shuffled,
    mimicking the raw CSV format (date as string column)."""
    days = pd.bdate_range("2022-01-03", periods=20, tz="+05:30")
    stamps = []
    for day in days:
        start = day + pd.Timedelta(hours=9, minutes=15)
        stamps.extend(pd.date_range(start, periods=75, freq="5min"))
    n = len(stamps)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    spread = np.abs(rng.normal(0, 0.1, n))
    df = pd.DataFrame(
        {
            "date": [str(s) for s in stamps],
            "open": close + rng.normal(0, 0.05, n),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000, 50_000, n),
        }
    )
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)  # unsorted on purpose


@pytest.fixture(scope="session")
def daily_bars(rng) -> pd.DataFrame:
    """300 synthetic daily bars with close/returns/first_difference features."""
    idx = pd.bdate_range("2021-01-01", periods=300)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx)))), index=idx
    )
    df = pd.DataFrame({"close": close})
    df["first_difference"] = df["close"].diff()
    df["returns"] = df["close"].pct_change() * 100.0
    return df.dropna()
