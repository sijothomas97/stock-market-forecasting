"""Tests for walk-forward window generation (pure index arithmetic, no
model fitting — fast and exhaustive)."""

from __future__ import annotations

import pytest

from src.backtesting.windows import Window, generate_windows


def test_no_windows_when_series_too_short():
    assert generate_windows(n_obs=10, horizon=5, min_train_size=10) == []


def test_single_window_exact_fit():
    windows = generate_windows(n_obs=15, horizon=5, min_train_size=10, step=1)
    assert len(windows) == 1
    w = windows[0]
    assert (w.train_start, w.train_end, w.test_start, w.test_end) == (0, 10, 10, 15)


def test_last_window_always_ends_at_n_obs():
    for n_obs in (50, 51, 73, 100):
        windows = generate_windows(n_obs=n_obs, horizon=7, min_train_size=20, step=3)
        assert windows[-1].test_end == n_obs


def test_windows_are_contiguous_no_gap_no_overlap():
    windows = generate_windows(n_obs=100, horizon=5, min_train_size=20, step=4)
    for w in windows:
        assert w.train_end == w.test_start
        assert w.test_end - w.test_start == 5


def test_windows_chronological_and_step_spacing():
    windows = generate_windows(n_obs=100, horizon=5, min_train_size=20, step=4)
    ends = [w.test_end for w in windows]
    assert ends == sorted(ends)
    # consecutive folds are spaced by `step`, except possibly the final jump
    # (which is compressed to keep the last fold anchored at n_obs)
    diffs = [b - a for a, b in zip(ends, ends[1:])]
    assert all(0 < d <= 4 for d in diffs)


def test_rolling_window_has_fixed_train_size():
    windows = generate_windows(
        n_obs=200, horizon=5, min_train_size=30, step=5, rolling=True
    )
    assert all(w.train_size == 30 for w in windows)


def test_expanding_window_grows():
    windows = generate_windows(
        n_obs=200, horizon=5, min_train_size=30, step=20, rolling=False
    )
    sizes = [w.train_size for w in windows]
    assert sizes == sorted(sizes)
    assert sizes[0] >= 30
    assert all(w.train_start == 0 for w in windows)


def test_max_windows_caps_and_keeps_most_recent():
    all_windows = generate_windows(n_obs=200, horizon=5, min_train_size=20, step=5)
    capped = generate_windows(
        n_obs=200, horizon=5, min_train_size=20, step=5, max_windows=3
    )
    assert len(capped) == 3
    assert capped == all_windows[-3:]
    assert capped[-1].test_end == 200


def test_horizon_property():
    w = Window(train_start=0, train_end=100, test_start=100, test_end=105)
    assert w.horizon == 5
    assert w.train_size == 100


@pytest.mark.parametrize("bad_kwargs", [
    {"horizon": 0},
    {"horizon": -1},
    {"min_train_size": 0},
    {"step": 0},
])
def test_rejects_invalid_params(bad_kwargs):
    kwargs = {"n_obs": 100, "horizon": 5, "min_train_size": 20, "step": 1}
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        generate_windows(**kwargs)
