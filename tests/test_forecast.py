"""Forecast shape/sanity tests for SARIMA confidence intervals and
GARCH volatility paths."""

import numpy as np
import pytest

from src.forecasting.forecast import forecast_sarima, forecast_volatility
from src.training.training import fit_garch, fit_sarima

HORIZON = 15


@pytest.fixture(scope="module")
def sarima_fit(daily_bars):
    return fit_sarima(
        daily_bars["close"], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)
    )


@pytest.fixture(scope="module")
def garch_fit(daily_bars):
    return fit_garch(daily_bars["returns"], vol="GARCH", p=1, q=1)


def test_sarima_forecast_shape_and_bands(sarima_fit):
    fc = forecast_sarima(sarima_fit, HORIZON, confidence_level=0.95)
    assert fc.shape == (HORIZON, 3)
    assert list(fc.columns) == ["mean", "lower", "upper"]
    assert np.isfinite(fc.values).all()
    assert (fc["lower"] <= fc["mean"]).all()
    assert (fc["mean"] <= fc["upper"]).all()


def test_sarima_wider_confidence_wider_bands(sarima_fit):
    narrow = forecast_sarima(sarima_fit, HORIZON, confidence_level=0.80)
    wide = forecast_sarima(sarima_fit, HORIZON, confidence_level=0.99)
    assert ((wide["upper"] - wide["lower"])
            > (narrow["upper"] - narrow["lower"])).all()


def test_volatility_forecast_shape(garch_fit):
    vol = forecast_volatility(garch_fit, HORIZON)
    assert vol.shape == (HORIZON,)
    assert np.isfinite(vol).all()
    assert (vol > 0).all()


@pytest.mark.parametrize("bad_horizon", [0, -5])
def test_forecast_rejects_bad_horizon(sarima_fit, garch_fit, bad_horizon):
    with pytest.raises(ValueError):
        forecast_sarima(sarima_fit, bad_horizon)
    with pytest.raises(ValueError):
        forecast_volatility(garch_fit, bad_horizon)
