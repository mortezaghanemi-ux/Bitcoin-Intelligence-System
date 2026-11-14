"""Edge-case tests for Bitcoin OHLCV candle validation."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from validation.candles import validate_candles


UTC = timezone.utc

CHECK_TIME = datetime(
    2026, 11, 13, 12, 0, tzinfo=UTC
)


def valid_candle():
    return {
        "open_time": "2026-11-13T11:00:00+00:00",
        "close_time": "2026-11-13T11:05:00+00:00",
        "open": 95000,
        "high": 96000,
        "low": 94000,
        "close": 95500,
        "volume": 120.5,
        "source": "kraken",
    }


def test_reject_non_dataframe_input():

    with pytest.raises(TypeError):
        validate_candles(
            [{"open": 95000}],
            validation_time=CHECK_TIME,
        )


@pytest.mark.parametrize(
    "column",
    [
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    ],
)
def test_reject_each_missing_required_column(column):

    candle = valid_candle()
    del candle[column]

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_candles(
            pd.DataFrame([candle]),
            validation_time=CHECK_TIME,
        )


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        "not-a-date",
        None,
        "2026-11-13T11:00:00",
        "2026-11-13T14:30:00+03:30",
    ],
)
@pytest.mark.parametrize(
    "column",
    ["open_time", "close_time"],
)
def test_reject_invalid_timestamps(column, bad_timestamp):

    candle = valid_candle()
    candle[column] = bad_timestamp

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert len(rejected) == 1
    assert f"{column}_must_be_valid_utc" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_reject_close_time_equal_to_open_time():

    candle = valid_candle()
    candle["close_time"] = candle["open_time"]

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert "close_time_must_follow_open_time" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_reject_close_time_before_open_time():

    candle = valid_candle()
    candle["close_time"] = "2026-11-13T10:59:00+00:00"

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert "close_time_must_follow_open_time" in (
        rejected.iloc[0]["validation_errors"]
    )


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close"],
)
@pytest.mark.parametrize(
    "bad_price",
    [0, -1, -100],
)
def test_reject_non_positive_prices(column, bad_price):

    candle = valid_candle()
    candle[column] = bad_price

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert "prices_must_be_positive" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_reject_blank_source():

    candle = valid_candle()
    candle["source"] = "   "

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert "source_required" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_reject_missing_source():

    candle = valid_candle()
    candle["source"] = None

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert "source_required" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_accept_candle_at_exact_age_limit():

    candle = valid_candle()

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
        max_age_minutes=55,
    )

    assert len(accepted) == 1
    assert rejected.empty
    assert report["status"] == "PASS"


def test_reject_candle_just_over_age_limit():

    candle = valid_candle()

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
        max_age_minutes=54,
    )

    assert accepted.empty
    assert len(rejected) == 1
    assert "stale_candle" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_allow_future_close_time_when_closed_check_disabled():

    candle = valid_candle()
    candle["close_time"] = "2026-11-13T12:05:00+00:00"

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
        require_closed=False,
    )

    assert len(accepted) == 1
    assert rejected.empty


def test_same_open_time_from_different_sources_is_not_duplicate():

    first = valid_candle()

    second = valid_candle()
    second["source"] = "another_source"

    accepted, rejected, report = validate_candles(
        pd.DataFrame([first, second]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 2
    assert rejected.empty


def test_duplicate_detection_keeps_first_record():

    first = valid_candle()

    second = valid_candle()
    second["close"] = 95050

    accepted, rejected, report = validate_candles(
        pd.DataFrame([first, second]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 1
    assert len(rejected) == 1

    assert accepted.iloc[0]["close"] == 95500

    assert "duplicate_candle" in (
        rejected.iloc[0]["validation_errors"]
    )


def test_original_dataframe_remains_unchanged():

    candle = valid_candle()
    original = pd.DataFrame([candle])
    original_copy = original.copy(deep=True)

    validate_candles(
        original,
        validation_time=CHECK_TIME,
    )

    pd.testing.assert_frame_equal(
        original,
        original_copy,
    )


def test_empty_dataframe_returns_pass_report():

    data = pd.DataFrame(
        columns=[
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]
    )

    accepted, rejected, report = validate_candles(
        data,
        validation_time=CHECK_TIME,
    )

    assert accepted.empty
    assert rejected.empty
    assert report["records_checked"] == 0
    assert report["valid_records"] == 0
    assert report["rejected_records"] == 0
    assert report["status"] == "PASS"