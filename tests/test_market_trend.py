
"""Tests for Bitcoin market trend analysis."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_analysis.trend import (
    TrendAnalysisError,
    analyze_trend,
    _calculate_adx,
    _classify_trend,
    _classify_strength,
)


UTC = timezone.utc

CANDLE_COUNT = 80

BASE_TIME = datetime(
    2026, 11, 13, 0, 0, tzinfo=UTC
)


def make_candles(
    direction="bullish",
    count=CANDLE_COUNT,
):
    """Create deterministic OHLCV test candles."""

    rows = []

    for index in range(count):

        if direction == "bullish":
            close = 95000 + index * 25

        elif direction == "bearish":
            close = 97000 - index * 25

        elif direction == "flat":
            close = 95000

        else:
            raise ValueError("Unknown test direction.")

        if index == 0:
            open_price = close
        else:
            previous_close = rows[-1]["close"]
            open_price = previous_close

        high = max(open_price, close) + 10
        low = min(open_price, close) - 10

        open_time = BASE_TIME + timedelta(
            minutes=index * 5
        )

        close_time = open_time + timedelta(
            minutes=5
        )

        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 100.0 + index,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def get_analysis_time(data):
    """Return a UTC time after the latest candle."""

    return (
        data["close_time"].iloc[-1]
        + timedelta(minutes=5)
    )


# --------------------------------------------------
# Basic trend analysis
# --------------------------------------------------


def test_bullish_trend_detection():

    data = make_candles("bullish")

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["trend"] == "BULLISH"
    assert result["trend_score"] > 0
    assert result["momentum_percent"] > 0
    assert result["candles_analyzed"] == CANDLE_COUNT


def test_bearish_trend_detection():

    data = make_candles("bearish")

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["trend"] == "BEARISH"
    assert result["trend_score"] < 0
    assert result["momentum_percent"] < 0


def test_flat_market_detection():

    data = make_candles("flat")

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["trend"] == "NEUTRAL"
    assert result["adx"] == pytest.approx(0.0)
    assert result["momentum_percent"] == pytest.approx(0.0)


# --------------------------------------------------
# Output validation
# --------------------------------------------------


def test_result_contains_required_fields():

    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    required_fields = {
        "analysis_timestamp",
        "source",
        "status",
        "trend",
        "trend_strength",
        "trend_score",
        "confidence",
        "adx",
        "momentum_percent",
        "moving_averages",
        "reversal_warning",
        "market_price",
        "last_candle_close_time",
        "candles_analyzed",
        "parameters",
    }

    assert required_fields.issubset(result.keys())


def test_source_is_reported():

    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["source"] == ["kraken"]


def test_moving_average_structure():

    data = make_candles("bullish")

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    averages = result["moving_averages"]

    assert averages["fast_period"] == 20
    assert averages["slow_period"] == 50
    assert averages["fast_value"] > 0
    assert averages["slow_value"] > 0
    assert averages["alignment"] == "BULLISH"


def test_confidence_is_within_valid_range():

    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert 0 <= result["confidence"] <= 100


def test_adx_is_finite():

    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert pd.notna(result["adx"])
    assert result["adx"] >= 0


def test_trend_strength_is_valid():

    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["trend_strength"] in {
        "STRONG",
        "MODERATE",
        "WEAK",
    }


# --------------------------------------------------
# Input validation
# --------------------------------------------------


def test_empty_dataframe_is_rejected():

    data = pd.DataFrame()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=datetime(
                2026, 11, 14, tzinfo=UTC
            ),
        )


def test_non_dataframe_input_is_rejected():

    with pytest.raises(TypeError):
        analyze_trend(
            [],
            analysis_time=datetime(
                2026, 11, 14, tzinfo=UTC
            ),
        )


def test_missing_source_column_is_rejected():

    data = make_candles()

    data = data.drop(
        columns=["source"]
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=datetime(
                2026, 11, 14, tzinfo=UTC
            ),
        )


def test_missing_close_column_is_rejected():

    data = make_candles()

    data = data.drop(
        columns=["close"]
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(
                make_candles()
            ),
        )


def test_empty_source_is_rejected():

    data = make_candles()

    data.loc[10, "source"] = ""

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_invalid_ohlc_is_rejected():

    data = make_candles()

    data.loc[10, "high"] = (
        data.loc[10, "low"] - 100
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_negative_volume_is_rejected():

    data = make_candles()

    data.loc[10, "volume"] = -100

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_nan_price_is_rejected():

    data = make_candles()

    data.loc[10, "close"] = float("nan")

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_duplicate_candles_are_rejected():

    data = make_candles()

    data.loc[10, "open_time"] = (
        data.loc[9, "open_time"]
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_unsorted_candles_are_rejected():

    data = make_candles()

    data.loc[[10, 11]] = (
        data.loc[[11, 10]].to_numpy()
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_naive_candle_timestamp_is_rejected():

    data = make_candles()

    data["open_time"] = (
        data["open_time"].dt.tz_localize(None)
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(
                make_candles()
            ),
        )


# --------------------------------------------------
# Closed candle and future data protection
# --------------------------------------------------


def test_open_candle_is_rejected():

    data = make_candles()

    analysis_time = (
        data["close_time"].iloc[-1]
        - timedelta(minutes=1)
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=analysis_time,
        )


def test_future_candle_is_rejected():

    data = make_candles()

    analysis_time = (
        data["close_time"].iloc[-2]
        + timedelta(minutes=1)
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=analysis_time,
        )


def test_non_utc_analysis_time_is_rejected():

    data = make_candles()

    non_utc_time = (
        get_analysis_time(data)
        .astimezone(
            timezone(timedelta(hours=1))
        )
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=non_utc_time,
        )


def test_naive_analysis_time_is_rejected():

    data = make_candles()

    naive_time = (
        get_analysis_time(data)
        .replace(tzinfo=None)
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=naive_time,
        )


# --------------------------------------------------
# Minimum data requirements
# --------------------------------------------------


def test_insufficient_candles_are_rejected():

    data = make_candles(
        count=30
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )


def test_minimum_candles_parameter_is_enforced():

    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            minimum_candles=100,
            analysis_time=get_analysis_time(data),
        )


# --------------------------------------------------
# Parameter validation
# --------------------------------------------------


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("fast_period", 0),
        ("slow_period", 0),
        ("adx_period", 0),
        ("minimum_candles", 0),
        ("fast_period", -1),
        ("slow_period", -5),
        ("adx_period", -2),
    ],
)
def test_invalid_parameters_are_rejected(
    parameter,
    value,
):

    data = make_candles()

    parameters = {
        "fast_period": 20,
        "slow_period": 50,
        "adx_period": 14,
        "minimum_candles": 50,
    }

    parameters[parameter] = value

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
            **parameters,
        )


def test_fast_period_must_be_smaller_than_slow_period():

    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            fast_period=50,
            slow_period=20,
            analysis_time=get_analysis_time(data),
        )


def test_minimum_candles_must_cover_slow_period():

    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            minimum_candles=40,
            slow_period=50,
            analysis_time=get_analysis_time(data),
        )


def test_minimum_candles_must_cover_adx_warmup():

    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            adx_period=30,
            minimum_candles=50,
            analysis_time=get_analysis_time(data),
        )


# --------------------------------------------------
# Internal calculation functions
# --------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (50, "BULLISH"),
        (20, "BULLISH"),
        (0, "NEUTRAL"),
        (-20, "BEARISH"),
        (-50, "BEARISH"),
    ],
)
def test_trend_classification(score, expected):

    assert _classify_trend(score) == expected


@pytest.mark.parametrize(
    "adx,expected",
    [
        (10, "WEAK"),
        (19.9, "WEAK"),
        (20, "MODERATE"),
        (24.9, "MODERATE"),
        (25, "STRONG"),
        (40, "STRONG"),
    ],
)
def test_strength_classification(adx, expected):

    assert _classify_strength(adx) == expected


def test_adx_for_flat_market_is_zero():

    data = make_candles("flat")

    result = _calculate_adx(
        data,
        period=14,
    )

    assert result.iloc[-1] == pytest.approx(0.0)


def test_adx_for_trending_market_is_finite():

    data = make_candles("bullish")

    result = _calculate_adx(
        data,
        period=14,
    )

    assert pd.notna(result.iloc[-1])
    assert result.iloc[-1] >= 0


# --------------------------------------------------
# Data integrity
# --------------------------------------------------


def test_input_dataframe_is_not_modified():

    data = make_candles()

    original = data.copy(deep=True)

    analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert_frame_equal(
        data,
        original,
    )


def test_analysis_is_deterministic():

    data = make_candles()

    analysis_time = get_analysis_time(data)

    first = analyze_trend(
        data,
        analysis_time=analysis_time,
    )

    second = analyze_trend(
        data,
        analysis_time=analysis_time,
    )

    assert first == second


def test_custom_periods_work():

    data = make_candles()

    result = analyze_trend(
        data,
        fast_period=10,
        slow_period=30,
        adx_period=10,
        minimum_candles=40,
        analysis_time=get_analysis_time(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["parameters"]["fast_period"] == 10
    assert result["parameters"]["slow_period"] == 30
    assert result["parameters"]["adx_period"] == 10