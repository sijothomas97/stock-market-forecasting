"""Walk-forward backtest window generation.

Pure index arithmetic (no model fitting), so it's cheap to test exhaustively.
A "window" is one rolling-refit fold: a contiguous training slice followed
immediately by a contiguous ``horizon``-length test slice, both expressed as
half-open ``[start, stop)`` integer ranges into the underlying series.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """One backtest fold, as integer positions into the full series.

    ``train_start``/``train_end`` and ``test_start``/``test_end`` are
    half-open ``[start, end)`` ranges, so ``series[train_start:train_end]``
    and ``series[test_start:test_end]`` are the two slices, with
    ``train_end == test_start`` (no gap, no overlap).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def horizon(self) -> int:
        return self.test_end - self.test_start


def generate_windows(
    n_obs: int,
    horizon: int,
    min_train_size: int,
    step: int = 1,
    max_windows: int | None = None,
    rolling: bool = True,
) -> list[Window]:
    """Generate walk-forward backtest windows over a series of length ``n_obs``.

    Each fold trains on data up to some cutoff and tests on the next
    ``horizon`` observations. Folds slide forward by ``step`` observations
    each time (a "refit" happens at every fold). The last fold's test window
    always ends exactly at ``n_obs`` (i.e. windows are anchored to the end of
    the series and walk backwards from there), so results are comparable
    across configurations that only change ``max_windows``.

    Parameters
    ----------
    n_obs:
        Total number of observations in the series.
    horizon:
        Forecast horizon (length of each test slice).
    min_train_size:
        Minimum number of observations required in a training slice.
    step:
        Number of observations to advance between consecutive folds.
    max_windows:
        Cap on the number of folds returned (keeps runtime bounded). If the
        theoretical fold count exceeds this, only the most recent
        ``max_windows`` folds are returned.
    rolling:
        If True (default), each fold's training window is a fixed-size
        rolling window of length ``min_train_size`` (old data drops off the
        back as new data is added — "rolling refit"). If False, the training
        window is expanding: it always starts at 0 and grows with each fold.

    Returns
    -------
    List of ``Window``, in chronological order (earliest fold first).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if n_obs < min_train_size + horizon:
        return []

    # Every possible test_end, from the earliest feasible fold to the last
    # observation, stepping backwards from the end so the final fold always
    # lands exactly on n_obs.
    test_ends = list(range(n_obs, min_train_size + horizon - 1, -step))
    test_ends.reverse()  # chronological order

    windows: list[Window] = []
    for test_end in test_ends:
        test_start = test_end - horizon
        train_end = test_start
        train_start = 0 if not rolling else max(0, train_end - min_train_size)
        if train_end - train_start < min_train_size:
            continue
        windows.append(Window(train_start, train_end, test_start, test_end))

    if max_windows is not None and len(windows) > max_windows:
        windows = windows[-max_windows:]

    return windows
