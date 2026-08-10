"""Smoke test for the full walk-forward backtest engine on synthetic data.

Kept tiny (2 folds, short horizon) so it runs in a couple of seconds; the
statsforecast benchmark is included only if actually installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.backtesting.engine import BacktestConfig, run_backtest
from src.backtesting.modern_benchmark import STATSFORECAST_AVAILABLE


@pytest.fixture
def price_series(daily_bars):
    return daily_bars["close"]


def test_run_backtest_smoke(price_series):
    cfg = BacktestConfig(
        horizon=3,
        min_train_size=100,
        step=50,
        max_windows=2,
        seasonality=5,
        use_statsforecast=False,  # keep this test independent of the optional dep
    )
    result = run_backtest(price_series, "SYNTH", cfg)

    assert len(result.windows) == 2
    models_seen = {r.model for r in result.fold_results}
    assert {"naive", "seasonal_naive", "SARIMA"} <= models_seen
    # 3 models x 2 folds
    assert len(result.fold_results) == 6

    summary = result.summary()
    assert set(summary) == {"naive", "seasonal_naive", "SARIMA"}
    for model_metrics in summary.values():
        assert np.isfinite(model_metrics["mape_pct"])
        assert model_metrics["mape_pct"] >= 0
        assert model_metrics["pinball_loss"] >= 0

    payload = result.to_dict()
    assert payload["n_windows"] == 2
    assert payload["symbol"] == "SYNTH"
    assert "summary_by_model" in payload


def test_run_backtest_no_windows_returns_empty(price_series):
    cfg = BacktestConfig(
        horizon=5,
        min_train_size=10_000,  # bigger than the whole series
        step=5,
        max_windows=3,
        use_statsforecast=False,
    )
    result = run_backtest(price_series, "SYNTH", cfg)
    assert result.windows == []
    assert result.fold_results == []
    assert result.summary() == {}


@pytest.mark.skipif(not STATSFORECAST_AVAILABLE, reason="statsforecast not installed")
def test_run_backtest_includes_statsforecast_when_available(price_series):
    cfg = BacktestConfig(
        horizon=3,
        min_train_size=100,
        step=50,
        max_windows=1,
        seasonality=5,
        use_statsforecast=True,
    )
    result = run_backtest(price_series, "SYNTH", cfg)
    models_seen = {r.model for r in result.fold_results}
    assert {"AutoARIMA", "AutoETS"} <= models_seen
    assert result.skipped_models == []
