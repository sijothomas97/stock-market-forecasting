# Time series forecasting on share market data

Classical time-series forecasting (SARIMA for price, ARCH/GARCH family for
volatility) on 5-minute NSE OHLCV data (HDFC Bank, Reliance, Sun Pharma,
Tata Steel, TCS, 2015-2022), served through a FastAPI backend with a
Plotly.js dashboard for interactive forecasts and backtests.

- **Pipeline**: `src/main.py` — load → resample → SARIMA + GARCH → forecast + metrics.
- **Backtesting**: `src/backtesting/` — walk-forward (rolling-refit) evaluation vs naive/seasonal-naive/statsforecast baselines.
- **API**: `src/api/app.py` — FastAPI service (`/health`, `/stocks`, `/forecast`, `/backtest`) with in-process model caching.
- **Dashboard**: `static/index.html` — stock/horizon selectors, forecast confidence-band chart, backtest metrics table (vendored Plotly.js, no CDN).
- **CI**: `.github/workflows/ci.yml` — ruff, pytest, pipeline/backtest/API smoke tests, Docker build, all against a tiny bundled sample CSV (`tests/data/`) since `data/` is gitignored.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

Data files are not in fresh checkouts going forward — see `data/README.md`
for where they live and the DVC/object-storage migration plan.

## Run the pipeline

Configuration lives in `config.yaml` (relative paths, stock symbol,
horizon, resample rule, model orders). Symbol and horizon can be
overridden on the CLI:

```bash
.venv/bin/python -m src.main                          # defaults from config.yaml
.venv/bin/python -m src.main --symbol RELIANCE --horizon 10
```

The pipeline loads one stock's raw CSV, resamples 5-min bars to daily,
engineers returns/differences, fits SARIMA + GARCH on a training window,
forecasts `horizon` steps with confidence intervals, and writes
`outputs/<SYMBOL>/forecast.csv` and `outputs/<SYMBOL>/metrics.json`
(MAE/RMSE/MAPE, CI coverage, forecast vs realized volatility).

## Backtesting + benchmarks

`src/backtesting/` implements walk-forward (rolling-refit) backtesting: at
each fold the model is refit on a fixed-size trailing window and scored on
the next `horizon` bars, then the window slides forward — repeated for
several folds so metrics reflect out-of-sample performance across different
points in time, not a single lucky/unlucky split.

Models compared per fold:
- **naive** — flat-line forecast at the last observed price (random-walk baseline).
- **seasonal_naive** — repeats the last trading week (5 bars) forward.
- **SARIMA** — the classical model already used by the main pipeline (`src/training/training.py`), refit each fold.
- **AutoARIMA** / **AutoETS** — Nixtla `statsforecast`'s auto-tuned classical models, included automatically if `statsforecast` is installed (best-effort import; the engine skips them and records `skipped_models` if it isn't available, rather than failing the whole backtest).

Metrics per fold, averaged across folds in the summary:
- **MAPE %** — mean absolute percentage error of the point forecast.
- **MASE** — mean absolute scaled error against an in-sample naive/seasonal-naive benchmark (< 1 beats naive, > 1 is worse); Hyndman & Koehler (2006).
- **Pinball loss** — averaged over the interval's lower/upper quantile edges (95% CI by default); scores both calibration and sharpness of the forecast interval, not just the point forecast.

Run it (kept at smoke scale — 8 rolling folds, 5-bar horizon, ~3-5s total including SARIMA + AutoARIMA + AutoETS refit at every fold):

```bash
.venv/bin/python -m src.backtesting.run_backtest --symbol TCS
# or override fold count / window size / horizon:
.venv/bin/python -m src.backtesting.run_backtest --symbol TCS --windows 8 --min-train-size 250 --horizon 5
```

Writes `reports/backtest.json` (per-fold results + `summary_by_model`).

### Results: TCS, 8 rolling folds, 250-bar training window, 5-day horizon

| Model | MAPE % | MASE | Pinball loss | Avg fit time (s) |
|---|---|---|---|---|
| SARIMA | **2.03** | **1.01** | **4.05** | 0.23 |
| AutoARIMA | 2.10 | 1.05 | 4.17 | 0.24 |
| AutoETS | 2.10 | 1.05 | 4.13 | 0.24 |
| naive | 2.10 | 1.05 | 9.68 | ~0 |
| seasonal_naive | 2.68 | 1.34 | 7.22 | ~0 |

All three real models beat the naive baselines on interval quality (pinball
loss), even though naive's flat-line point forecast ties AutoARIMA/AutoETS on
MAPE over a short 5-day horizon on this trending series (expected — MAPE
alone rewards "predict no change" when a series drifts steadily in one
direction across the holdout window). SARIMA edges out both AutoARIMA and
AutoETS here on every metric; regenerate `reports/backtest.json` (command
above) to refresh this table for a different symbol/window/data snapshot.

## API + dashboard

`src/api/app.py` is a FastAPI service exposing the pipeline and backtester
over HTTP, plus the static dashboard mounted at `/`:

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + count of available stock symbols |
| `/stocks` | GET | List symbols with a CSV under `data/stocks_data` |
| `/forecast` | POST | `{symbol, horizon}` → SARIMA mean forecast + confidence band + GARCH volatility forecast + metrics (fitted models are cached in-process per `(symbol, horizon)`) |
| `/backtest` | POST | `{symbol, horizon, windows, min_train_size, step, use_statsforecast}` → walk-forward backtest summary per model (bounded: horizon ≤ 20, windows ≤ 20, min_train_size ≤ 1000) |
| `/` | GET | Static dashboard (Plotly.js charts + backtest table) |
| `/docs` | GET | Auto-generated OpenAPI/Swagger UI |

Run it locally:

```bash
.venv/bin/uvicorn src.api.app:app --reload --port 8000
# then open http://127.0.0.1:8000/  (dashboard)
#      or  http://127.0.0.1:8000/docs (API docs)
```

Example requests:

```bash
curl http://127.0.0.1:8000/stocks
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TCS", "horizon": 10}'
curl -X POST http://127.0.0.1:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TCS", "horizon": 5, "windows": 6, "min_train_size": 250, "step": 5}'
```

The dashboard (`static/index.html`) lets you pick a stock and horizon, plots
the forecast mean with its confidence band against the actual holdout
close, and runs/tabulates a backtest across models — all client-side JS
calling the API above, with Plotly.js vendored locally under
`static/vendor/` (no CDN dependency, works offline / behind a strict CSP).

## Docker

```bash
docker build -t ts-stocks-api .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data:ro" ts-stocks-api
```

The image does not bundle the (gitignored, ~630MB) `data/` directory —
mount it at runtime as above. Without a mount the container still boots
and `/health`/`/stocks` respond (with 0 symbols available), which is what
CI uses to verify the image is basically sound before a real data volume
exists.

## CI

`.github/workflows/ci.yml` runs on every push/PR:

1. `ruff check` over `src/` and `tests/`.
2. Seeds `data/stocks_data/TCS_with_indicators_.csv` from the bundled
   `tests/data/SAMPLE_with_indicators_.csv` (synthetic OHLCV bars, same
   shape as the real files — see `tests/data/make_sample_csv.py`), since
   the real data is gitignored and never present in a fresh checkout/CI runner.
3. `pytest -q` (unit + API tests).
4. Smoke-runs the pipeline CLI, backtest CLI, and a locally-started API
   (curl `/health`, `/stocks`, `/forecast`).
5. A second job builds the Docker image and repeats the API smoke test
   against the running container.

## Tests

```bash
.venv/bin/python -m pytest
```

`tests/test_api.py` covers the FastAPI service (forecast shape/bands,
caching, bounds validation, backtest summary, static dashboard + vendored
Plotly asset serving) using `fastapi.testclient.TestClient` — no server
process needed. All tests run against small synthetic fixtures or the
bundled sample CSV; none depend on the ~630MB raw CSVs being present.

Exploratory notebooks are under `nb/`.
