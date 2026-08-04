"""CLI: run walk-forward backtesting for one stock and write
``reports/backtest.json``.

Usage (from the project root):

    .venv/bin/python -m src.backtesting.run_backtest
    .venv/bin/python -m src.backtesting.run_backtest --symbol RELIANCE --windows 6

Kept at smoke scale by default (a handful of ~250-obs rolling windows, 5-day
horizon) so a full run — including SARIMA + AutoARIMA + AutoETS refit at
every fold — finishes in well under a minute.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.backtesting.engine import BacktestConfig, run_backtest
from src.config import DEFAULT_CONFIG_PATH, PipelineConfig
from src.data.loader import load_stock_csv
from src.preprocessing.preprocess import (
    add_features,
    resample_ohlcv,
    to_datetime_index,
)

REPORTS_DIR = DEFAULT_CONFIG_PATH.parent / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=5,
                         help="Forecast horizon per fold (default: 5)")
    parser.add_argument("--min-train-size", type=int, default=250,
                         help="Rolling training window size in bars (default: 250)")
    parser.add_argument("--step", type=int, default=5,
                         help="Bars to advance between folds (default: 5)")
    parser.add_argument("--windows", type=int, default=8,
                         help="Max number of walk-forward folds (default: 8)")
    parser.add_argument("--no-statsforecast", action="store_true",
                         help="Skip AutoARIMA/AutoETS even if installed")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output JSON path (default: reports/backtest.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = PipelineConfig.from_yaml(args.config, symbol=args.symbol)

    raw = load_stock_csv(cfg.symbol, cfg.data_dir)
    bars = resample_ohlcv(to_datetime_index(raw), cfg.resample_rule)
    bars = add_features(bars)

    bt_cfg = BacktestConfig(
        horizon=args.horizon,
        min_train_size=args.min_train_size,
        step=args.step,
        max_windows=args.windows,
        seasonality=5,
        confidence_level=cfg.confidence_level,
        sarima_order=cfg.sarima.order,
        sarima_seasonal_order=cfg.sarima.seasonal_order,
        use_statsforecast=not args.no_statsforecast,
    )

    print(f"Backtesting {cfg.symbol}: horizon={bt_cfg.horizon} "
          f"min_train_size={bt_cfg.min_train_size} step={bt_cfg.step} "
          f"max_windows={bt_cfg.max_windows} statsforecast={bt_cfg.use_statsforecast}")

    t0 = time.time()
    result = run_backtest(bars["close"], cfg.symbol.upper(), bt_cfg)
    elapsed = time.time() - t0

    out_path = args.out or (REPORTS_DIR / "backtest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"n_windows={len(result.windows)} runtime={elapsed:.1f}s")
    print(json.dumps(payload["summary_by_model"], indent=2))
    print(f"Written to {out_path}")
    return payload


if __name__ == "__main__":
    main()
