"""Smoke tests: models fit quickly on small synthetic data, and the GARCH
constructor actually builds a GARCH (regression test for the old bug where
the 'GARCH' models were fitted with vol='ARCH')."""

import numpy as np
import pytest

from src.training.evaluate import mae, mape, point_forecast_metrics, rmse
from src.training.training import fit_garch, fit_sarima


def test_fit_sarima_smoke(daily_bars):
    fitted = fit_sarima(
        daily_bars["close"], order=(1, 1, 1), seasonal_order=(1, 0, 0, 5)
    )
    assert np.isfinite(fitted.aic)
    assert len(fitted.fittedvalues) == len(daily_bars)


def test_fit_garch_is_actually_garch(daily_bars):
    fitted = fit_garch(daily_bars["returns"], vol="GARCH", p=1, q=1)
    assert type(fitted.model.volatility).__name__ == "GARCH"
    # GARCH(1,1) has omega, alpha AND beta params
    assert {"omega", "alpha[1]", "beta[1]"} <= set(fitted.params.index)
    assert np.isfinite(fitted.aic)


def test_fit_arch_variant(daily_bars):
    fitted = fit_garch(daily_bars["returns"], vol="ARCH", p=1)
    assert type(fitted.model.volatility).__name__ == "ARCH"
    assert "beta[1]" not in set(fitted.params.index)


def test_metrics():
    y_true = [100.0, 102.0, 104.0]
    y_pred = [101.0, 102.0, 103.0]
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2 / 3))
    assert mape(y_true, y_true) == 0.0
    m = point_forecast_metrics(y_true, y_pred)
    assert set(m) == {"mae", "rmse", "mape_pct"}
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])
