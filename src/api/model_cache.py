"""In-process caching for the API: loaded/resampled bars per symbol, and
fitted SARIMA+GARCH models per (symbol, horizon).

Fitting a SARIMA+GARCH pair takes well under a second at the pipeline's
default ``train_window`` (see Phase 1 notes: ~1.3s including CSV load), but
repeated fits on every request are still wasted work for a demo service, so
both the resampled bar data and the fitted models are cached in memory.
The cache is intentionally simple (a couple of dicts, no eviction) — this
is a smoke-scale demo service, not a production model registry.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from src.config import PipelineConfig
from src.data.loader import available_symbols, load_stock_csv
from src.forecasting.forecast import forecast_sarima, forecast_volatility
from src.preprocessing.preprocess import (
    add_features,
    resample_ohlcv,
    to_datetime_index,
    train_test_split_ts,
)
from src.training.evaluate import point_forecast_metrics
from src.training.training import fit_garch, fit_sarima

_lock = threading.Lock()
_bars_cache: dict[str, pd.DataFrame] = {}
_forecast_cache: dict[tuple[str, int], dict] = {}


def list_symbols(cfg: PipelineConfig) -> list[str]:
    return available_symbols(cfg.data_dir)


def _load_bars(symbol: str, cfg: PipelineConfig) -> pd.DataFrame:
    symbol = symbol.upper()
    if symbol in _bars_cache:
        return _bars_cache[symbol]
    raw = load_stock_csv(symbol, cfg.data_dir)
    bars = resample_ohlcv(to_datetime_index(raw), cfg.resample_rule)
    bars = add_features(bars)
    _bars_cache[symbol] = bars
    return bars


def is_forecast_cached(symbol: str, horizon: int) -> bool:
    return (symbol.upper(), horizon) in _forecast_cache


def get_forecast(symbol: str, horizon: int, cfg: PipelineConfig) -> dict:
    """Return a cached forecast for (symbol, horizon), fitting on first use.

    Mirrors ``src.main.run_pipeline`` but keyed for reuse across requests.
    Thread-safe: a plain lock serializes fits, which is fine at this scale
    (fits take ~1s; the API is a demo service, not a high-QPS backend).
    """
    symbol = symbol.upper()
    key = (symbol, horizon)
    if key in _forecast_cache:
        return _forecast_cache[key]

    with _lock:
        if key in _forecast_cache:  # re-check after acquiring the lock
            return _forecast_cache[key]

        t0 = time.time()
        bars = _load_bars(symbol, cfg)
        if cfg.train_window is not None:
            windowed = bars.iloc[-(cfg.train_window + horizon):]
        else:
            windowed = bars
        train, test = train_test_split_ts(windowed, test_size=horizon)

        sarima_fit = fit_sarima(
            train["close"], order=cfg.sarima.order, seasonal_order=cfg.sarima.seasonal_order,
        )
        price_fc = forecast_sarima(sarima_fit, horizon, cfg.confidence_level)
        price_fc.index = test.index
        price_metrics = point_forecast_metrics(test["close"], price_fc["mean"])
        coverage = float(
            ((test["close"].values >= price_fc["lower"].values)
             & (test["close"].values <= price_fc["upper"].values)).mean()
        )

        garch_fit = fit_garch(
            train["returns"], vol=cfg.garch.vol, p=cfg.garch.p, q=cfg.garch.q, dist=cfg.garch.dist,
        )
        vol_fc = forecast_volatility(garch_fit, horizon)
        vol_fc.index = test.index

        result = {
            "symbol": symbol,
            "horizon": horizon,
            "generated_at": time.time(),
            "fit_seconds": round(time.time() - t0, 3),
            "dates": [d.strftime("%Y-%m-%d") for d in test.index],
            "forecast_mean": price_fc["mean"].tolist(),
            "forecast_lower": price_fc["lower"].tolist(),
            "forecast_upper": price_fc["upper"].tolist(),
            "actual_close": [None if pd.isna(v) else float(v) for v in test["close"]],
            "forecast_volatility_pct": vol_fc.tolist(),
            "confidence_level": cfg.confidence_level,
            "last_train_close": float(train["close"].iloc[-1]),
            "last_train_date": train.index[-1].strftime("%Y-%m-%d"),
            "metrics": {
                **price_metrics,
                "conf_interval_coverage": coverage,
                "mean_forecast_vol_pct": float(vol_fc.mean()),
            },
        }
        _forecast_cache[key] = result
        return result


def cache_info() -> dict:
    return {
        "cached_symbols": sorted(_bars_cache.keys()),
        "cached_forecasts": [f"{s}:{h}" for s, h in sorted(_forecast_cache.keys())],
    }


def clear_cache() -> None:
    """Used by tests to reset state between cases."""
    with _lock:
        _bars_cache.clear()
        _forecast_cache.clear()
