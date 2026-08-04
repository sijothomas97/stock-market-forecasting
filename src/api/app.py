"""FastAPI service: /health, /stocks, POST /forecast, POST /backtest.

Run locally:

    .venv/bin/uvicorn src.api.app:app --reload --port 8000

Serves the static dashboard (``static/``) at ``/``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api import model_cache
from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    StocksResponse,
)
from src.backtesting.engine import BacktestConfig, run_backtest
from src.config import DEFAULT_CONFIG_PATH, PipelineConfig
from src.data.loader import load_stock_csv
from src.preprocessing.preprocess import add_features, resample_ohlcv, to_datetime_index

STATIC_DIR = DEFAULT_CONFIG_PATH.parent / "static"

app = FastAPI(
    title="Time Series on Stocks — Forecast API",
    description="SARIMA+GARCH forecasts and walk-forward backtests over NSE stock data.",
    version="1.0.0",
)


def get_config() -> PipelineConfig:
    """Base config (paths, model orders); symbol/horizon are per-request."""
    return PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = get_config()
    try:
        symbols = model_cache.list_symbols(cfg)
    except Exception:
        symbols = []
    return HealthResponse(status="ok", symbols_available=len(symbols))


@app.get("/stocks", response_model=StocksResponse)
def stocks() -> StocksResponse:
    cfg = get_config()
    symbols = model_cache.list_symbols(cfg)
    if not symbols:
        raise HTTPException(
            status_code=503,
            detail=f"No stock data found under {cfg.data_dir}. "
            "See data/README.md for how to populate it.",
        )
    return StocksResponse(symbols=symbols)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    cfg = get_config()
    available = model_cache.list_symbols(cfg)
    symbol = req.symbol.upper()
    if symbol not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown symbol {req.symbol!r}. Available: {available}",
        )
    was_cached = model_cache.is_forecast_cached(symbol, req.horizon)
    try:
        result = model_cache.get_forecast(symbol, req.horizon, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ForecastResponse(**result, cached=was_cached)


@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    cfg = get_config()
    available = model_cache.list_symbols(cfg)
    symbol = req.symbol.upper()
    if symbol not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown symbol {req.symbol!r}. Available: {available}",
        )

    raw = load_stock_csv(symbol, cfg.data_dir)
    bars = resample_ohlcv(to_datetime_index(raw), cfg.resample_rule)
    bars = add_features(bars)

    bt_cfg = BacktestConfig(
        horizon=req.horizon,
        min_train_size=req.min_train_size,
        step=req.step,
        max_windows=req.windows,
        seasonality=5,
        confidence_level=cfg.confidence_level,
        sarima_order=cfg.sarima.order,
        sarima_seasonal_order=cfg.sarima.seasonal_order,
        use_statsforecast=req.use_statsforecast,
    )

    if len(bars) < bt_cfg.min_train_size + bt_cfg.horizon:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Not enough bars ({len(bars)}) for min_train_size="
                f"{bt_cfg.min_train_size} + horizon={bt_cfg.horizon}"
            ),
        )

    result = run_backtest(bars["close"], symbol, bt_cfg)
    payload = result.to_dict()
    return BacktestResponse(
        symbol=payload["symbol"],
        n_windows=payload["n_windows"],
        skipped_models=payload["skipped_models"],
        runtime_seconds=payload["runtime_seconds"],
        summary_by_model=payload["summary_by_model"],
    )


# Static dashboard (Plotly.js dashboard + vendored assets), mounted last so
# it doesn't shadow the API routes above. html=True serves index.html at "/".
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
