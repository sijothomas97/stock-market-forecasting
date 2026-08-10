"""API tests for the FastAPI service (src/api/app.py).

Uses the real small sample data under ``data/stocks_data`` (or the CI
bundled sample CSV — see ``tests/data/``) via TestClient; no network calls.
Each test clears the in-process model cache first so tests are independent
of run order and of each other's cached state.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import model_cache
from src.api.app import app
from src.config import PipelineConfig
from src.data.loader import available_symbols

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_cache():
    model_cache.clear_cache()
    yield
    model_cache.clear_cache()


def _first_symbol() -> str:
    cfg = PipelineConfig.from_yaml()
    symbols = available_symbols(cfg.data_dir)
    if not symbols:
        pytest.skip("No stock CSVs available under data/stocks_data")
    return symbols[0]


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert isinstance(body["symbols_available"], int)


def test_stocks_lists_available_symbols():
    res = client.get("/stocks")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["symbols"], list)
    if body["symbols"]:
        assert body["symbols"] == sorted(body["symbols"])


def test_forecast_returns_bands_and_metrics():
    symbol = _first_symbol()
    res = client.post("/forecast", json={"symbol": symbol, "horizon": 5})
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == symbol
    assert body["horizon"] == 5
    assert len(body["dates"]) == 5
    assert len(body["forecast_mean"]) == 5
    assert len(body["forecast_lower"]) == 5
    assert len(body["forecast_upper"]) == 5
    # confidence bands must bracket the mean forecast at every step
    for lo, mean, hi in zip(body["forecast_lower"], body["forecast_mean"], body["forecast_upper"]):
        assert lo <= mean <= hi
    assert "mape_pct" in body["metrics"]
    assert body["cached"] is False


def test_forecast_is_cached_on_second_call():
    symbol = _first_symbol()
    r1 = client.post("/forecast", json={"symbol": symbol, "horizon": 5})
    assert r1.json()["cached"] is False
    r2 = client.post("/forecast", json={"symbol": symbol, "horizon": 5})
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    # cached response should be identical (same fitted model reused)
    assert r1.json()["forecast_mean"] == r2.json()["forecast_mean"]


def test_forecast_unknown_symbol_404():
    res = client.post("/forecast", json={"symbol": "NOT_A_REAL_SYMBOL", "horizon": 5})
    assert res.status_code == 404


def test_forecast_horizon_out_of_bounds_422():
    symbol = _first_symbol()
    res = client.post("/forecast", json={"symbol": symbol, "horizon": 0})
    assert res.status_code == 422
    res = client.post("/forecast", json={"symbol": symbol, "horizon": 10_000})
    assert res.status_code == 422


def test_backtest_returns_summary_per_model():
    symbol = _first_symbol()
    res = client.post(
        "/backtest",
        json={
            "symbol": symbol,
            "horizon": 3,
            "windows": 2,
            "min_train_size": 60,
            "step": 3,
            "use_statsforecast": False,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == symbol
    assert body["n_windows"] <= 2
    assert "naive" in body["summary_by_model"]
    assert "SARIMA" in body["summary_by_model"]
    for model_metrics in body["summary_by_model"].values():
        assert "mape_pct" in model_metrics
        assert "mase" in model_metrics
        assert "pinball_loss" in model_metrics


def test_backtest_unknown_symbol_404():
    res = client.post(
        "/backtest",
        json={"symbol": "NOT_A_REAL_SYMBOL", "windows": 2, "min_train_size": 60},
    )
    assert res.status_code == 404


def test_backtest_params_out_of_bounds_422():
    symbol = _first_symbol()
    res = client.post(
        "/backtest",
        json={"symbol": symbol, "windows": 999, "min_train_size": 60},
    )
    assert res.status_code == 422
    res = client.post(
        "/backtest",
        json={"symbol": symbol, "windows": 2, "min_train_size": 1},
    )
    assert res.status_code == 422


def test_dashboard_served_at_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Forecast Dashboard" in res.text


def test_plotly_vendored_asset_served():
    res = client.get("/vendor/plotly.min.js")
    assert res.status_code == 200
    assert int(res.headers["content-length"]) > 1_000_000
