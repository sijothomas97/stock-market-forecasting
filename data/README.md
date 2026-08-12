# Data directory

## What lives here

`data/stocks_data/` holds 5-minute NSE OHLCV + technical-indicator CSVs
(2015–2022, ~126 MB each, ~630 MB total):

| File | Stock |
|---|---|
| `HDFCBANK_with_indicators_.csv` | HDFC Bank |
| `RELIANCE_with_indicators_.csv` | Reliance Industries |
| `SUNPHARMA_with_indicators_.csv` | Sun Pharmaceutical |
| `TATASTEEL_with_indicators_.csv` | Tata Steel |
| `TCS_with_indicators_.csv` | Tata Consultancy Services |

Columns: `date` (5-min bars, `+05:30` timezone), `open/high/low/close/volume`,
plus ~50 precomputed indicators (SMA/EMA/MACD/RSI/ADX/...). The pipeline
(`src/`) only reads the OHLCV columns.

## Status: these files are still in git history

The CSVs were committed to git before this cleanup. As of this change,
`.gitignore` excludes `data/**`, so **new** data files won't be committed —
but the ~630 MB already in history stays there until history is rewritten
(e.g. `git filter-repo --path data/stocks_data --invert-paths` +
force-push), which is deliberately **not** done here because it invalidates
every existing clone. Coordinate that as a separate, announced step.

## Migration plan: DVC + object storage

Recommended target: track data with [DVC](https://dvc.org) and store blobs
in object storage (S3/GCS/Azure); git only keeps tiny `.dvc` pointer files.

```bash
# one-time setup
uv pip install 'dvc[s3]'
dvc init
dvc remote add -d storage s3://<your-bucket>/time-series-on-stocks-data

# move the CSVs from git tracking to DVC tracking
git rm -r --cached data/stocks_data        # keeps files on disk
dvc add data/stocks_data
git add data/stocks_data.dvc .dvc .dvcignore .gitignore
git commit -m "Track stock CSVs with DVC instead of git"
dvc push                                    # upload blobs to the bucket
```

Fresh clones then run `dvc pull` to fetch the data. Data updates become
`dvc add data/stocks_data && git commit && dvc push`.

Alternative if DVC is unwanted: keep the CSVs in a versioned bucket
(`aws s3 sync data/stocks_data s3://<bucket>/stocks_data`) and add a small
`make data` / script that downloads them; same `.gitignore` applies.

## Reproducing locally without the originals

Any CSV with at least `date,open,high,low,close,volume` columns at 5-minute
(or any intraday) frequency will run through the pipeline:

```bash
.venv/bin/python -m src.main --symbol TCS --horizon 20
```
