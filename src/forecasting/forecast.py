"""Forecasting: SARIMA mean forecasts with confidence intervals and
ARCH/GARCH-family volatility forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def forecast_sarima(
    fitted: SARIMAXResults,
    horizon: int,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Mean forecast + confidence interval for ``horizon`` steps.

    Returns a DataFrame indexed by forecast step with columns
    ``mean``, ``lower`` and ``upper``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    result = fitted.get_forecast(steps=horizon)
    conf = result.conf_int(alpha=1.0 - confidence_level)
    return pd.DataFrame(
        {
            "mean": np.asarray(result.predicted_mean, dtype=float),
            "lower": np.asarray(conf.iloc[:, 0], dtype=float),
            "upper": np.asarray(conf.iloc[:, 1], dtype=float),
        },
        index=getattr(result.predicted_mean, "index", pd.RangeIndex(horizon)),
    )


def forecast_volatility(fitted: ARCHModelResult, horizon: int) -> pd.Series:
    """Per-step conditional volatility forecast (same units as the fitted
    returns, i.e. percent if the model was fit on percentage returns)."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    forecast = fitted.forecast(horizon=horizon, reindex=False)
    variances = np.asarray(forecast.variance.values[-1, :], dtype=float)
    return pd.Series(np.sqrt(variances), name="conditional_volatility")
