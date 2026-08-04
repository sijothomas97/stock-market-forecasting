"""Point-forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (in %); ignores zero actuals."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    Scales MAE of the forecast by the in-sample MAE of a naive (or
    seasonal-naive, for ``seasonality`` > 1) forecast on the *training*
    series, making the metric comparable across series of different scale
    and units. A value < 1 beats the naive in-sample benchmark; > 1 is worse.

    ``y_train`` must have at least ``seasonality + 1`` observations (needed
    to compute the naive in-sample differences).
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float)
    if seasonality < 1:
        raise ValueError(f"seasonality must be >= 1, got {seasonality}")
    if len(y_train) <= seasonality:
        raise ValueError(
            f"y_train needs > {seasonality} observations to scale MASE, "
            f"got {len(y_train)}"
        )
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def pinball_loss(y_true, y_pred_quantile, quantile: float) -> float:
    """Pinball (quantile) loss for a single quantile forecast.

    ``quantile`` in (0, 1). For the median (0.5) this reduces to half the
    MAE. Lower is better; this is the standard metric for scoring
    probabilistic/interval forecasts one quantile at a time.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    y_true, y_pred_quantile = _as_arrays(y_true, y_pred_quantile)
    diff = y_true - y_pred_quantile
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_pinball_loss(
    y_true, lower, upper, confidence_level: float = 0.95
) -> float:
    """Average pinball loss of a two-sided prediction interval's edges.

    Treats ``lower``/``upper`` as the ``alpha/2`` and ``1 - alpha/2``
    quantiles implied by ``confidence_level`` and averages their pinball
    losses — a single scalar summarizing interval sharpness + calibration
    (lower is better).
    """
    alpha = 1.0 - confidence_level
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    return float(
        (pinball_loss(y_true, lower, lo_q) + pinball_loss(y_true, upper, hi_q)) / 2.0
    )


def point_forecast_metrics(
    y_true, y_pred, y_train=None, seasonality: int = 1
) -> dict[str, float]:
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }
    if y_train is not None:
        try:
            metrics["mase"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
        except ValueError:
            metrics["mase"] = float("nan")
    return metrics
