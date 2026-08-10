"""Generate the tiny bundled sample CSV used by CI in place of the ~630MB
real data directory (which is gitignored — see data/README.md).

Produces ``tests/data/SAMPLE_with_indicators_.csv``: synthetic 5-minute
OHLCV bars in the same raw-CSV shape ``src/data/loader.py`` expects
(date, open, high, low, close, volume — extra indicator columns are
allowed but not required, since the loader only reads OHLCV_COLUMNS).

Sized for the API/backtest CI smoke tests: ~400 trading days x 75 bars/day
so a walk-forward backtest with min_train_size=250 has room for several
folds. Deterministic (fixed seed) so the checked-in file is reproducible.

Run: .venv/bin/python tests/data/make_sample_csv.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "SAMPLE_with_indicators_.csv"


def main() -> None:
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2019-01-01", periods=400, tz="+05:30")
    rows = []
    price = 500.0
    for day in days:
        start = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15)
        stamps = pd.date_range(start, periods=75, freq="5min")
        # small daily drift + intraday noise, floored away from zero
        drift = rng.normal(0.0002, 0.01)
        intraday = np.cumsum(rng.normal(0, 0.002, len(stamps)))
        closes = price * np.exp(drift + intraday)
        opens = np.r_[closes[0], closes[:-1]]
        spread = np.abs(rng.normal(0, 0.001, len(stamps))) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread
        volumes = rng.integers(1_000, 50_000, len(stamps))
        for ts, o, h, l, c, v in zip(stamps, opens, highs, lows, closes, volumes):
            rows.append((ts.isoformat(), round(o, 2), round(h, 2), round(l, 2), round(c, 2), int(v)))
        price = closes[-1]

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
