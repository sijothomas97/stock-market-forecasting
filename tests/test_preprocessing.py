import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocess import (
    add_features,
    resample_ohlcv,
    to_datetime_index,
    train_test_split_ts,
)


def test_to_datetime_index_tz_naive_and_sorted(five_min_raw):
    out = to_datetime_index(five_min_raw)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    assert len(out) == len(five_min_raw)
    assert "date" not in out.columns


def test_resample_daily_aggregation(five_min_raw):
    bars = to_datetime_index(five_min_raw)
    daily = resample_ohlcv(bars, "D")
    # 20 synthetic trading days -> 20 daily bars (empty days dropped)
    assert len(daily) == 20
    assert list(daily.columns) == ["open", "high", "low", "close", "volume"]
    # verify aggregation semantics on the first day
    day0 = bars.loc[bars.index.normalize() == daily.index[0]]
    assert daily["high"].iloc[0] == pytest.approx(day0["high"].max())
    assert daily["low"].iloc[0] == pytest.approx(day0["low"].min())
    assert daily["open"].iloc[0] == pytest.approx(day0["open"].iloc[0])
    assert daily["close"].iloc[0] == pytest.approx(day0["close"].iloc[-1])
    assert daily["volume"].iloc[0] == day0["volume"].sum()


def test_resample_rejects_frame_without_ohlcv():
    df = pd.DataFrame(
        {"foo": [1.0]}, index=pd.DatetimeIndex(["2022-01-03"])
    )
    with pytest.raises(ValueError, match="No OHLCV columns"):
        resample_ohlcv(df, "D")


def test_add_features(five_min_raw):
    daily = resample_ohlcv(to_datetime_index(five_min_raw), "D")
    feat = add_features(daily)
    assert {"first_difference", "returns"} <= set(feat.columns)
    assert not feat[["first_difference", "returns"]].isna().any().any()
    assert len(feat) == len(daily) - 1  # first diff/return row dropped
    expected = (daily["close"].iloc[1] - daily["close"].iloc[0])
    assert feat["first_difference"].iloc[0] == pytest.approx(expected)
    expected_ret = (daily["close"].iloc[1] / daily["close"].iloc[0] - 1) * 100
    assert feat["returns"].iloc[0] == pytest.approx(expected_ret)


def test_train_test_split_chronological(daily_bars):
    train, test = train_test_split_ts(daily_bars, test_size=20)
    assert len(test) == 20
    assert len(train) == len(daily_bars) - 20
    assert train.index.max() < test.index.min()


@pytest.mark.parametrize("bad", [0, -3, 10_000])
def test_train_test_split_rejects_bad_sizes(daily_bars, bad):
    with pytest.raises(ValueError):
        train_test_split_ts(daily_bars, test_size=bad)
