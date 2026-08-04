"""Walk-forward backtest engine: rolling refit + h-step forecasts across a
set of models, scored with MAPE / MASE / pinball loss.

Runtime is bounded by capping the number of windows (folds) and the
per-fold training size (``min_train_size``) via ``BacktestConfig`` — each
fold refits every model from scratch, so cost scales linearly in
``n_windows``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src.backtesting.baselines import (
    naive_forecast,
    naive_residual_std,
    seasonal_naive_forecast,
)
from src.backtesting.modern_benchmark import (
    STATSFORECAST_AVAILABLE,
    fit_and_forecast_statsforecast,
)
from src.backtesting.windows import Window, generate_windows
from src.forecasting.forecast import forecast_sarima
from src.training.evaluate import interval_pinball_loss, mape, mase
from src.training.training import fit_sarima

# z-score for a two-sided Gaussian interval at a given confidence level,
# without pulling in scipy just for this: precompute the common ones and
# fall back to the 95% value for anything unlisted (smoke-scale scope).
_Z_SCORES = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass
class BacktestConfig:
    horizon: int = 5
    min_train_size: int = 250
    step: int = 5
    max_windows: int = 8
    seasonality: int = 5  # trading week
    confidence_level: float = 0.95
    sarima_order: tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 5)
    use_statsforecast: bool = True


@dataclass
class ModelFoldResult:
    model: str
    window_index: int
    train_size: int
    horizon: int
    mape_pct: float
    mase: float
    pinball_loss: float
    fit_seconds: float


@dataclass
class BacktestResult:
    symbol: str
    config: BacktestConfig
    windows: list[Window]
    fold_results: list[ModelFoldResult]
    runtime_seconds: float = 0.0
    skipped_models: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-model metrics averaged across all folds."""
        df = pd.DataFrame([vars(r) for r in self.fold_results])
        if df.empty:
            return {}
        agg = (
            df.groupby("model")[["mape_pct", "mase", "pinball_loss", "fit_seconds"]]
            .mean()
            .round(4)
        )
        return {
            model: row.to_dict()
            for model, row in agg.iterrows()
        }

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "config": {
                "horizon": self.config.horizon,
                "min_train_size": self.config.min_train_size,
                "step": self.config.step,
                "max_windows": self.config.max_windows,
                "seasonality": self.config.seasonality,
                "confidence_level": self.config.confidence_level,
                "sarima_order": list(self.config.sarima_order),
                "sarima_seasonal_order": list(self.config.sarima_seasonal_order),
            },
            "n_windows": len(self.windows),
            "skipped_models": self.skipped_models,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "summary_by_model": self.summary(),
            "folds": [vars(r) for r in self.fold_results],
        }


def _gaussian_interval(
    mean: pd.Series, sigma: float, confidence_level: float
) -> tuple[pd.Series, pd.Series]:
    z = _Z_SCORES.get(round(confidence_level, 2), _Z_SCORES[0.95])
    lower = mean - z * sigma
    upper = mean + z * sigma
    return lower, upper


def run_backtest(series: pd.Series, symbol: str, cfg: BacktestConfig) -> BacktestResult:
    """Run walk-forward backtesting of naive/seasonal-naive/SARIMA (and, if
    available, statsforecast AutoARIMA/AutoETS) baselines on ``series``.

    ``series`` must be a plain (non-datetime-indexed, or at least
    contiguous) price series in chronological order; SARIMA/statsforecast
    are fit on integer positions each fold, matching how the main pipeline
    already handles the non-fixed-frequency trading calendar.
    """
    t0 = time.time()
    values = series.reset_index(drop=True)
    windows = generate_windows(
        n_obs=len(values),
        horizon=cfg.horizon,
        min_train_size=cfg.min_train_size,
        step=cfg.step,
        max_windows=cfg.max_windows,
        rolling=True,
    )

    skipped_models: list[str] = []
    use_sf = cfg.use_statsforecast and STATSFORECAST_AVAILABLE
    if cfg.use_statsforecast and not STATSFORECAST_AVAILABLE:
        skipped_models.extend(["AutoARIMA", "AutoETS"])

    fold_results: list[ModelFoldResult] = []

    for w_idx, w in enumerate(windows):
        train = values.iloc[w.train_start:w.train_end].reset_index(drop=True)
        test = values.iloc[w.test_start:w.test_end].reset_index(drop=True)
        y_true = test.to_numpy(dtype=float)
        y_train_arr = train.to_numpy(dtype=float)

        # --- naive ---
        t1 = time.time()
        pred = naive_forecast(train, cfg.horizon)
        sigma = naive_residual_std(train, seasonality=1)
        lower, upper = _gaussian_interval(pred, sigma, cfg.confidence_level)
        fold_results.append(_score(
            "naive", w_idx, w.train_size, cfg, y_true, y_train_arr,
            pred.to_numpy(), lower.to_numpy(), upper.to_numpy(), time.time() - t1,
        ))

        # --- seasonal naive ---
        t1 = time.time()
        pred = seasonal_naive_forecast(train, cfg.horizon, cfg.seasonality)
        sigma = naive_residual_std(train, seasonality=cfg.seasonality)
        lower, upper = _gaussian_interval(pred, sigma, cfg.confidence_level)
        fold_results.append(_score(
            "seasonal_naive", w_idx, w.train_size, cfg, y_true, y_train_arr,
            pred.to_numpy(), lower.to_numpy(), upper.to_numpy(), time.time() - t1,
        ))

        # --- SARIMA (classical) ---
        t1 = time.time()
        sarima_fit = fit_sarima(
            train, order=cfg.sarima_order, seasonal_order=cfg.sarima_seasonal_order,
        )
        fc = forecast_sarima(sarima_fit, cfg.horizon, cfg.confidence_level)
        fold_results.append(_score(
            "SARIMA", w_idx, w.train_size, cfg, y_true, y_train_arr,
            fc["mean"].to_numpy(), fc["lower"].to_numpy(), fc["upper"].to_numpy(),
            time.time() - t1,
        ))

        # --- modern benchmark (statsforecast), best-effort ---
        if use_sf:
            t1 = time.time()
            try:
                sf_out = fit_and_forecast_statsforecast(
                    train, cfg.horizon, season_length=cfg.seasonality,
                    confidence_level=cfg.confidence_level,
                )
                elapsed = time.time() - t1
                for name, fc in sf_out.items():
                    fold_results.append(_score(
                        name, w_idx, w.train_size, cfg, y_true, y_train_arr,
                        fc["mean"].to_numpy(), fc["lower"].to_numpy(),
                        fc["upper"].to_numpy(), elapsed,
                    ))
            except Exception as exc:  # pragma: no cover - defensive
                if "statsforecast_error" not in skipped_models:
                    skipped_models.append(f"statsforecast_error:{exc}")
                use_sf = False

    return BacktestResult(
        symbol=symbol,
        config=cfg,
        windows=windows,
        fold_results=fold_results,
        runtime_seconds=time.time() - t0,
        skipped_models=skipped_models,
    )


def _score(
    model: str,
    w_idx: int,
    train_size: int,
    cfg: BacktestConfig,
    y_true,
    y_train_arr,
    y_pred,
    lower,
    upper,
    fit_seconds: float,
) -> ModelFoldResult:
    return ModelFoldResult(
        model=model,
        window_index=w_idx,
        train_size=train_size,
        horizon=cfg.horizon,
        mape_pct=mape(y_true, y_pred),
        mase=mase(y_true, y_pred, y_train_arr, seasonality=cfg.seasonality),
        pinball_loss=interval_pinball_loss(y_true, lower, upper, cfg.confidence_level),
        fit_seconds=round(fit_seconds, 4),
    )
