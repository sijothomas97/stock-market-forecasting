"""Preprocessing: datetime index, resampling, and model features.

Replaces the old ``data-preprocessing.py`` / ``data_preperation.py`` pair.
Fixed along the way:

* ``df['date'] = df.set_index('date', inplace=True)`` — assigned ``None``
  back into the ``date`` column; now a plain ``set_index``.
* ``pd.to_datetime(..., format='%Y-%m-%d')`` on timestamps that also carry
  time-of-day + timezone — now parsed as full ISO8601 timestamps.
* ``-> tuple[pd.series]`` typo (lowercase ``series``) — now ``pd.Series``.
* Hardwired three-stock methods — every function takes one DataFrame.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller

OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def to_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Parse ``date_col``, drop the timezone, and make it the sorted index."""
    out = df.copy()
    parsed = pd.to_datetime(out[date_col], format="ISO8601", utc=True)
    out[date_col] = parsed.dt.tz_convert(None)
    return out.set_index(date_col).sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str = "D") -> pd.DataFrame:
    """Resample intraday OHLCV bars to ``rule`` (e.g. 'D', 'W', 'ME').

    Only columns present in both the frame and the OHLCV aggregation map are
    kept. Empty periods (weekends, holidays) are dropped.
    """
    agg = {col: how for col, how in OHLCV_AGG.items() if col in df.columns}
    if not agg:
        raise ValueError(f"No OHLCV columns found in {list(df.columns)}")
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=[next(iter(agg))])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add first difference and percentage returns of the close price.

    ``returns`` are in percent (x100), the scale the ``arch`` package
    recommends for numerically stable GARCH estimation.
    """
    out = df.copy()
    out["first_difference"] = out["close"].diff()
    out["returns"] = out["close"].pct_change() * 100.0
    return out.dropna(subset=["first_difference", "returns"])


def train_test_split_ts(
    df: pd.DataFrame, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: last ``test_size`` rows become the test set."""
    if not 0 < test_size < len(df):
        raise ValueError(
            f"test_size must be in (0, {len(df)}), got {test_size}"
        )
    return df.iloc[:-test_size], df.iloc[-test_size:]


def test_stationarity(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test, returned as a readable dict."""
    stat, pvalue, usedlag, nobs, critical, _ = adfuller(
        series.dropna(), autolag="AIC"
    )
    return {
        "adf_statistic": float(stat),
        "p_value": float(pvalue),
        "n_obs": int(nobs),
        "critical_values": {k: float(v) for k, v in critical.items()},
        "stationary_at_5pct": bool(pvalue < 0.05),
    }


def decompose(
    series: pd.Series, model: str = "additive", period: int = 12
) -> DecomposeResult:
    """Seasonal decomposition of a series (trend/seasonal/resid)."""
    return seasonal_decompose(series.dropna(), model=model, period=period)
