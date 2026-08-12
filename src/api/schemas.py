"""Request/response models for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Bounds keep the demo service at smoke scale — fitting SARIMA+GARCH (and,
# for /backtest, refitting per fold) is cheap at these sizes but grows with
# horizon/windows/train size, so requests are capped rather than left
# unbounded.
MAX_HORIZON = 60
MAX_BACKTEST_WINDOWS = 20
MAX_BACKTEST_HORIZON = 20
MIN_BACKTEST_TRAIN_SIZE = 60
MAX_BACKTEST_TRAIN_SIZE = 1000


class StocksResponse(BaseModel):
    symbols: list[str]


class ForecastRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol, e.g. TCS, RELIANCE")
    horizon: int = Field(20, ge=1, le=MAX_HORIZON, description="Forecast horizon in trading days")


class ForecastResponse(BaseModel):
    symbol: str
    horizon: int
    dates: list[str]
    forecast_mean: list[float]
    forecast_lower: list[float]
    forecast_upper: list[float]
    actual_close: list[float | None]
    forecast_volatility_pct: list[float]
    confidence_level: float
    last_train_close: float
    last_train_date: str
    metrics: dict[str, float]
    fit_seconds: float
    cached: bool


class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol, e.g. TCS, RELIANCE")
    horizon: int = Field(5, ge=1, le=MAX_BACKTEST_HORIZON, description="Forecast horizon per fold")
    windows: int = Field(6, ge=1, le=MAX_BACKTEST_WINDOWS, description="Number of walk-forward folds")
    min_train_size: int = Field(
        250, ge=MIN_BACKTEST_TRAIN_SIZE, le=MAX_BACKTEST_TRAIN_SIZE,
        description="Rolling training window size in bars",
    )
    step: int = Field(5, ge=1, le=MAX_BACKTEST_HORIZON, description="Bars to advance between folds")
    use_statsforecast: bool = Field(
        True, description="Include Nixtla AutoARIMA/AutoETS if the package is installed"
    )


class BacktestResponse(BaseModel):
    symbol: str
    n_windows: int
    skipped_models: list[str]
    runtime_seconds: float
    summary_by_model: dict[str, dict[str, float]]


class HealthResponse(BaseModel):
    status: str
    symbols_available: int
