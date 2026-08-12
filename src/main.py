"""Config-driven end-to-end forecasting pipeline for one stock.

Usage (from the project root):

    .venv/bin/python -m src.main                       # config.yaml defaults
    .venv/bin/python -m src.main --symbol RELIANCE --horizon 10

Steps: load raw 5-min OHLCV CSV -> resample -> feature engineering ->
holdout split -> fit SARIMA (price) + GARCH-family (volatility) ->
forecast with confidence intervals -> save forecasts + metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import DEFAULT_CONFIG_PATH, PipelineConfig
from src.data.loader import load_stock_csv
from src.forecasting.forecast import forecast_sarima, forecast_volatility
from src.preprocessing.preprocess import (
    add_features,
    resample_ohlcv,
    to_datetime_index,
    train_test_split_ts,
)
from src.training.evaluate import point_forecast_metrics
from src.training.training import fit_garch, fit_sarima


def run_pipeline(cfg: PipelineConfig) -> dict:
    t0 = time.time()

    # 1. Load + preprocess
    raw = load_stock_csv(cfg.symbol, cfg.data_dir)
    bars = resample_ohlcv(to_datetime_index(raw), cfg.resample_rule)
    bars = add_features(bars)

    # 2. Keep a recent window (train_window fit obs + horizon holdout)
    if cfg.train_window is not None:
        bars = bars.iloc[-(cfg.train_window + cfg.horizon):]
    train, test = train_test_split_ts(bars, test_size=cfg.horizon)

    # 3. Price model: SARIMA on close
    sarima_fit = fit_sarima(
        train["close"],
        order=cfg.sarima.order,
        seasonal_order=cfg.sarima.seasonal_order,
    )
    price_fc = forecast_sarima(sarima_fit, cfg.horizon, cfg.confidence_level)
    price_fc.index = test.index  # align forecast steps with holdout dates
    price_metrics = point_forecast_metrics(test["close"], price_fc["mean"])
    coverage = float(
        ((test["close"].values >= price_fc["lower"].values)
         & (test["close"].values <= price_fc["upper"].values)).mean()
    )

    # 4. Volatility model: GARCH family on percentage returns
    garch_fit = fit_garch(
        train["returns"],
        vol=cfg.garch.vol,
        p=cfg.garch.p,
        q=cfg.garch.q,
        dist=cfg.garch.dist,
    )
    vol_fc = forecast_volatility(garch_fit, cfg.horizon)
    vol_fc.index = test.index
    realized_vol = float(test["returns"].std())

    # 5. Persist artifacts
    out_dir = cfg.output_dir / cfg.symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_df = price_fc.copy()
    forecast_df["actual_close"] = test["close"]
    forecast_df["forecast_volatility_pct"] = vol_fc
    forecast_df["actual_return_pct"] = test["returns"]
    forecast_df.to_csv(out_dir / "forecast.csv", index_label="date")

    metrics = {
        "symbol": cfg.symbol.upper(),
        "horizon": cfg.horizon,
        "resample_rule": cfg.resample_rule,
        "n_train_obs": int(len(train)),
        "n_test_obs": int(len(test)),
        "train_start": str(train.index[0]),
        "train_end": str(train.index[-1]),
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        "sarima": {
            "order": list(cfg.sarima.order),
            "seasonal_order": list(cfg.sarima.seasonal_order),
            "aic": float(sarima_fit.aic),
            "bic": float(sarima_fit.bic),
            **price_metrics,
            "conf_interval_coverage": coverage,
            "confidence_level": cfg.confidence_level,
        },
        "volatility_model": {
            "vol": cfg.garch.vol,
            "p": cfg.garch.p,
            "q": cfg.garch.q,
            "dist": cfg.garch.dist,
            "aic": float(garch_fit.aic),
            "bic": float(garch_fit.bic),
            "mean_forecast_vol_pct": float(vol_fc.mean()),
            "realized_holdout_vol_pct": realized_vol,
            "vol_abs_error_pct": abs(float(vol_fc.mean()) - realized_vol),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML config (default: config.yaml)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Stock symbol, e.g. TCS, RELIANCE (overrides config)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast horizon in resampled bars (overrides config)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = PipelineConfig.from_yaml(
        args.config, symbol=args.symbol, horizon=args.horizon
    )
    print(f"Running pipeline: symbol={cfg.symbol} horizon={cfg.horizon} "
          f"rule={cfg.resample_rule} vol={cfg.garch.vol}")
    metrics = run_pipeline(cfg)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {cfg.output_dir / cfg.symbol.upper()}")
    return metrics


if __name__ == "__main__":
    main()
