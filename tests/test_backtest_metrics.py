"""Tests for MASE and pinball-loss metric functions, plus the naive /
seasonal-naive baseline forecasters."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtesting.baselines import (
    naive_forecast,
    naive_residual_std,
    seasonal_naive_forecast,
)
from src.training.evaluate import (
    interval_pinball_loss,
    mase,
    pinball_loss,
    point_forecast_metrics,
)

# --- MASE -------------------------------------------------------------


def test_mase_equals_one_when_forecast_equals_naive_error_scale():
    # constant step-1 differences of 1.0 in-sample -> naive scale = 1.0
    y_train = np.arange(0, 20, dtype=float)
    y_true = np.array([20.0, 21.0, 22.0])
    y_pred = np.array([21.0, 22.0, 23.0])  # off by 1.0 each step, matching scale
    assert mase(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_mase_perfect_forecast_is_zero():
    y_train = np.linspace(0, 10, 30)
    y_true = np.array([11.0, 12.0, 13.0])
    assert mase(y_true, y_true, y_train) == pytest.approx(0.0)


def test_mase_seasonal_uses_seasonal_naive_scale():
    # perfectly repeating seasonal pattern in-sample -> seasonal naive
    # in-sample error is 0 -> scale is 0 -> nan (degenerate, but must not crash)
    y_train = np.tile([1.0, 2.0, 3.0], 10)
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    result = mase(y_true, y_pred, y_train, seasonality=3)
    assert np.isnan(result)


def test_mase_rejects_short_train():
    with pytest.raises(ValueError):
        mase([1.0], [1.0], y_train=[1.0], seasonality=5)


def test_mase_worse_than_naive_exceeds_one():
    y_train = np.arange(0, 20, dtype=float)  # naive scale = 1.0
    y_true = np.array([20.0, 21.0])
    y_pred = np.array([25.0, 26.0])  # off by 5 each step
    assert mase(y_true, y_pred, y_train) == pytest.approx(5.0)


# --- Pinball loss -------------------------------------------------------


def test_pinball_loss_zero_for_perfect_quantile_forecast():
    y_true = [10.0, 20.0, 30.0]
    assert pinball_loss(y_true, y_true, quantile=0.5) == pytest.approx(0.0)


def test_pinball_loss_median_equals_half_mae():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    expected_half_mae = float(np.mean(np.abs(y_true - y_pred))) / 2
    assert pinball_loss(y_true, y_pred, quantile=0.5) == pytest.approx(expected_half_mae)


def test_pinball_loss_penalizes_underprediction_more_for_high_quantile():
    y_true = [10.0]
    under = pinball_loss(y_true, [5.0], quantile=0.9)   # under-shoots a high quantile
    over = pinball_loss(y_true, [15.0], quantile=0.9)    # over-shoots a high quantile
    assert under > over


def test_pinball_loss_rejects_bad_quantile():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], quantile=0.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], quantile=1.0)


def test_interval_pinball_loss_perfect_interval_is_low():
    y_true = np.array([10.0, 20.0, 30.0])
    # interval centered exactly on the truth with small width
    lower = y_true - 1.0
    upper = y_true + 1.0
    loose_lower = y_true - 10.0
    loose_upper = y_true + 10.0
    tight_loss = interval_pinball_loss(y_true, lower, upper, confidence_level=0.95)
    loose_loss = interval_pinball_loss(y_true, loose_lower, loose_upper, confidence_level=0.95)
    assert tight_loss < loose_loss


# --- point_forecast_metrics integration ---------------------------------


def test_point_forecast_metrics_includes_mase_when_y_train_given():
    y_train = np.arange(0, 20, dtype=float)
    y_true = [20.0, 21.0]
    y_pred = [21.0, 22.0]
    m = point_forecast_metrics(y_true, y_pred, y_train=y_train)
    assert set(m) == {"mae", "rmse", "mape_pct", "mase"}
    assert m["mase"] == pytest.approx(1.0)


def test_point_forecast_metrics_without_y_train_omits_mase():
    m = point_forecast_metrics([1.0, 2.0], [1.0, 2.0])
    assert "mase" not in m


# --- Baselines ------------------------------------------------------


def test_naive_forecast_is_flat_at_last_value(daily_bars):
    train = daily_bars["close"]
    fc = naive_forecast(train, horizon=5)
    assert len(fc) == 5
    assert (fc == train.iloc[-1]).all()


def test_naive_forecast_rejects_bad_horizon(daily_bars):
    with pytest.raises(ValueError):
        naive_forecast(daily_bars["close"], horizon=0)


def test_seasonal_naive_tiles_last_cycle(daily_bars):
    train = daily_bars["close"]
    fc = seasonal_naive_forecast(train, horizon=7, seasonality=5)
    last_cycle = train.iloc[-5:].to_numpy()
    expected = np.tile(last_cycle, 2)[:7]
    np.testing.assert_allclose(fc.to_numpy(), expected)


def test_seasonal_naive_requires_enough_train_data():
    import pandas as pd
    short = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        seasonal_naive_forecast(short, horizon=3, seasonality=5)


def test_naive_residual_std_nonnegative(daily_bars):
    std = naive_residual_std(daily_bars["close"], seasonality=1)
    assert std >= 0
