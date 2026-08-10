# Improvements — Time Series Forecasting on Stock Data

**Goal:** A deployed web app where a user picks an NSE stock and horizon, and a served forecasting model returns price/volatility predictions with confidence bands, backtest metrics, and interactive charts.

## TL;DR — Path to production
- [x] Fix blocking bugs → runnable, tested pipeline; move 630MB of CSVs out of git (`data/` now gitignored + DVC migration documented in `data/README.md`; history not rewritten by design)
- [x] Walk-forward backtesting (rolling refit, MAPE/MASE/pinball loss) with naive/seasonal-naive + Nixtla `statsforecast` AutoARIMA/AutoETS benchmarked vs classical SARIMA (`src/backtesting/`, `reports/backtest.json`); a *neural* model (neuralforecast/Darts/TFT/N-HiTS) is still not wired in — deferred
- [ ] MLflow tracking + model registry
- [x] FastAPI `/forecast` + `/backtest` service (`src/api/app.py`: `/health`, `/stocks`, `POST /forecast`, `POST /backtest`, in-process model cache, bounded request params, pytest API test suite)
- [x] Forecast dashboard with confidence bands (`static/index.html`: plain HTML/JS + Plotly.js, vendored locally under `static/vendor/`, no CDN/React — stock/horizon selectors, forecast-band chart, backtest metrics table; served by the API at `/`)
- [x] Docker + GitHub Actions CI (`Dockerfile`, `.github/workflows/ci.yml`: ruff, pytest, pipeline/backtest/API smoke tests, Docker build+run smoke test, all against a bundled sample CSV since `data/` is gitignored); live cloud demo/deploy out of scope (ground rules disallow deploying — CI/Docker config is the deployable artifact)

## Current state
- Classical TS models (SARIMA, ARDL, ARCH, GARCH) explored across notebooks on 5-min NSE OHLCV + indicator data (HDFC, Reliance, Sun Pharma, Tata Steel, TCS).
- `src/` is a runnable, config-driven pipeline (`.venv/bin/python -m src.main --symbol TCS --horizon 10`): load → resample → SARIMA (price) + GARCH (volatility) → forecast + metrics.
- `src/backtesting/` adds walk-forward (rolling-refit) backtesting: naive, seasonal-naive, SARIMA, and (best-effort) Nixtla `statsforecast` AutoARIMA/AutoETS, scored with MAPE/MASE/pinball loss. Run: `.venv/bin/python -m src.backtesting.run_backtest --symbol TCS`; results in `reports/backtest.json` and summarized in `README.md`.
- `src/api/app.py` serves the pipeline + backtester over HTTP (FastAPI) with an in-process model cache, plus a static Plotly.js dashboard (`static/index.html`) mounted at `/`. Run: `.venv/bin/uvicorn src.api.app:app --reload --port 8000`.
- `Dockerfile` builds the API into a container (data volume mounted at runtime, not baked in); `.github/workflows/ci.yml` runs ruff + pytest + pipeline/backtest/API smoke tests + a Docker build/run smoke test on every push, using a small bundled synthetic CSV (`tests/data/SAMPLE_with_indicators_.csv`) in place of the gitignored real data.
- 61 tests pass in ~5s (`.venv/bin/python -m pytest`), all on small synthetic fixtures or the bundled sample CSV — no dependence on the raw CSVs.
- `data/` (~630MB of raw CSVs) is gitignored; see `data/README.md` for the DVC/object-storage migration plan (history not rewritten by design).

## Key improvements
- **Pipeline/ML:** fix and wire the code into a runnable pipeline (done); walk-forward backtesting + metrics (MAPE, MASE, pinball loss) vs naive/seasonal-naive/statsforecast AutoARIMA-AutoETS (done); still open — a modern *deep* model (e.g. Nixtla `neuralforecast` or Darts TFT/N-HiTS) benchmarked against the classical baselines.
- **Experiment tracking + registry:** MLflow for runs, params, and versioned model artifacts.
- **API/backend:** FastAPI service exposing `/forecast` and `/backtest` (done, in-process cache; no model registry yet — see MLflow item above).
- **Frontend:** Plotly-based dashboard for stock/horizon selection, forecast bands, and backtest results (done, plain HTML/JS + vendored Plotly.js instead of React).
- **Data:** move CSVs out of git (done — DVC/object-storage plan documented, not yet executed); add live pull via `yfinance`/broker API (open).
- **Quality:** pytest suite (done, incl. API tests), ruff (done, `pyproject.toml`), typed config (done, dataclass-based `PipelineConfig` + Pydantic v2 request/response schemas in the API), GitHub Actions CI (done).

## Latest tech to showcase
- Nixtla `neuralforecast`/`statsforecast` or Darts (TFT, N-HiTS, TimesFM-style foundation models).
- FastAPI + Pydantic v2, served in Docker.
- MLflow tracking + model registry; DVC for data versioning.
- React + Vite + Plotly for the interactive forecast UI.
- GitHub Actions CI/CD deploying container to Fly.io/Render (or HF Spaces for a quick live demo).

## Roadmap
1. ~~Refactor buggy `src/` into a working, tested pipeline; pin deps; move data out of git.~~ Done.
2. ~~Add walk-forward backtests (naive/seasonal-naive/statsforecast benchmark vs SARIMA).~~ Done. Still open: a neural forecasting model + MLflow experiment tracking.
3. ~~Wrap in FastAPI, build a forecast dashboard, containerize, and add CI.~~ Done (`src/api/`, `static/`, `Dockerfile`, `.github/workflows/ci.yml`) — dashboard is plain HTML/JS + vendored Plotly.js rather than React (boring-tech tradeoff, ground rules disallow cloud deploy so no live demo). Still open: MLflow tracking/registry, neural forecasting benchmark.
