"""Optional Nixtla ``statsforecast`` benchmark models (AutoARIMA, AutoETS).

Import of ``statsforecast`` is best-effort: if it isn't installed (or fails
to import for any reason, e.g. a missing compiled dependency), this module
sets ``STATSFORECAST_AVAILABLE = False`` and the backtest engine skips the
modern-benchmark models entirely rather than failing the whole run.
"""

from __future__ import annotations

import pandas as pd

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    STATSFORECAST_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when missing
    STATSFORECAST_AVAILABLE = False
    _IMPORT_ERROR = exc


def _to_nixtla_frame(train: pd.Series, unique_id: str = "series") -> pd.DataFrame:
    """statsforecast expects a long frame with columns unique_id, ds, y."""
    return pd.DataFrame(
        {
            "unique_id": unique_id,
            "ds": range(len(train)),  # integer pseudo-dates: irregular trading calendar
            "y": train.to_numpy(dtype=float),
        }
    )


def fit_and_forecast_statsforecast(
    train: pd.Series,
    horizon: int,
    season_length: int = 5,
    confidence_level: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Fit AutoARIMA and AutoETS on ``train`` and forecast ``horizon`` steps.

    Returns a dict keyed by model name, each value a DataFrame with columns
    ``mean``, ``lower``, ``upper`` (position-indexed 0..horizon-1), mirroring
    ``forecast_sarima``'s output shape so both slot into the same backtest
    loop.

    Raises ``RuntimeError`` if statsforecast is not importable — callers
    should check ``STATSFORECAST_AVAILABLE`` first to skip gracefully.
    """
    if not STATSFORECAST_AVAILABLE:
        raise RuntimeError(
            f"statsforecast is not available: {_IMPORT_ERROR!r}"
        )
    level = [round(confidence_level * 100)]
    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
    ]
    sf = StatsForecast(models=models, freq=1, n_jobs=1)
    df = _to_nixtla_frame(train)
    fcst = sf.forecast(df=df, h=horizon, level=level)

    out: dict[str, pd.DataFrame] = {}
    lvl = level[0]
    for model in models:
        name = model.alias
        out[name] = pd.DataFrame(
            {
                "mean": fcst[name].to_numpy(dtype=float),
                "lower": fcst[f"{name}-lo-{lvl}"].to_numpy(dtype=float),
                "upper": fcst[f"{name}-hi-{lvl}"].to_numpy(dtype=float),
            }
        )
    return out
