"""Naive and seasonal-naive forecast baselines.

These require no fitting (beyond reading off the training tail) and are the
standard "can a real model even beat this?" floor for benchmarking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, horizon: int) -> pd.Series:
    """Flat-line forecast at the last observed value (a.k.a. random-walk
    forecast — optimal point forecast if the series is a pure random walk).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if len(train) < 1:
        raise ValueError("train series must have at least 1 observation")
    last = float(train.iloc[-1])
    return pd.Series(np.full(horizon, last), name="naive")


def seasonal_naive_forecast(
    train: pd.Series, horizon: int, seasonality: int
) -> pd.Series:
    """Repeats the last full seasonal cycle of ``train`` forward.

    Step ``h`` (1-indexed) of the forecast equals
    ``train[-seasonality + ((h - 1) % seasonality)]``, i.e. the value from
    the same point in the most recent seasonal cycle, tiled forward.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if seasonality < 1:
        raise ValueError(f"seasonality must be >= 1, got {seasonality}")
    if len(train) < seasonality:
        raise ValueError(
            f"train series needs >= {seasonality} observations for "
            f"seasonality={seasonality}, got {len(train)}"
        )
    last_cycle = train.iloc[-seasonality:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon / seasonality))
    tiled = np.tile(last_cycle, reps)[:horizon]
    return pd.Series(tiled, name="seasonal_naive")


def naive_residual_std(train: pd.Series, seasonality: int = 1) -> float:
    """Std of the naive (or seasonal-naive) one-step-ahead in-sample
    residuals, used to build simple Gaussian prediction intervals for the
    baselines so they can be scored with pinball loss alongside SARIMA."""
    diffs = train.to_numpy(dtype=float)
    diffs = diffs[seasonality:] - diffs[:-seasonality]
    return float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
