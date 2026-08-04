"""Model fitting: SARIMA for price, ARCH/GARCH family for volatility.

Fixed relative to the original version of this module:

* ``from statsmodels.tsa.arima_model import ARIMA`` — that module was
  removed from statsmodels; the pipeline uses the maintained
  ``statsmodels.tsa.statespace.SARIMAX`` API.
* ``if alg is 'arima'`` — identity comparison on strings; string dispatch
  is gone entirely (and any comparisons use ``==``).
* The GARCH slots were trained via ``train_arch`` with ``vol='ARCH'``, so
  the "GARCH" models were actually ARCH — ``fit_garch`` now passes the
  configured ``vol`` (default ``'GARCH'``) straight through.
"""

from __future__ import annotations

import arch
import pandas as pd
import statsmodels.api as sm
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 5),
) -> SARIMAXResults:
    """Fit a SARIMA model on a (price) series using the current API.

    Trading-day series have calendar gaps (weekends/holidays), so a
    DatetimeIndex without a fixed frequency is expected; statsmodels treats
    that as an error-in-waiting, so such series are refit on a plain
    integer index (callers re-align forecasts to dates themselves).
    """
    if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
        series = pd.Series(series.to_numpy(), name=series.name)
    model = sm.tsa.statespace.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def fit_garch(
    returns: pd.Series,
    vol: str = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> ARCHModelResult:
    """Fit an ARCH-family volatility model on percentage returns.

    ``vol`` selects the actual volatility process ('GARCH', 'ARCH',
    'EGARCH', ...). A pure ARCH model has no q term, so q is only passed
    for processes that use it.
    """
    kwargs = {"vol": vol, "p": p, "dist": dist}
    if vol.upper() != "ARCH":
        kwargs["q"] = q
    model = arch.arch_model(returns.dropna(), **kwargs)
    return model.fit(disp="off")
