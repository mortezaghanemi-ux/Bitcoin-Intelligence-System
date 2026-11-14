"""Tests for Bitcoin market regime analysis."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_analysis.regime import (
    TRENDING,
    RANGING,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    TRANSITIONAL,
    BULLISH,
    BEARISH,
    NEUTRAL,
    MarketRegimeError,
    analyze_market_regime,
    _calculate_atr,
    _calculate_adx,
    _calculate_bollinger_width,
    _classify_direction,
    _classify_regime,
    _classify_volatility,
)


UTC = timezone.utc

START = datetime(
    2026, 11, 14, 0, 0, tzinfo=UTC
)


# ---------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------


def make_candles(
    direction="flat",
    candle_count=100,
    volatility=1.0,
):
    """Create deterministic OHLCV candles."""

    rows = []
    previous_close = 1000.0

    for index in range(candle_count):

        if direction == "bullish":
            open_price = previous_close
            close = previous_close + 2.0

        elif direction == "bearish":
            open_price = previous_close
            close = previous_close - 2.0

        elif direction == "flat":
            open_price = 1000.0
            close = 1000.0

        elif direction == "volatile":
            open_price = previous_close
            movement = (
                8.0
                if index % 2 == 0
                else -8.0
            )
            close = previous_close + movement

        else:
            raise ValueError(
                "Unknown test direction."
            )

        high = max(open_price, close) + volatility
        low = min(open_price, close) - volatility

        open_time = (
            START
            + timedelta(minutes=5 * index)
        )

        close_time = (
            open_time
            + timedelta(minutes=5)
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

        previous_close = close

    return pd.DataFrame(rows)


def get_analysis_time(data):
    """Return UTC time after the final candle."""

    return (
        data["close_time"].iloc[-1]
        + timedelta(minutes=5)
    )


def analyze(data, **kwargs):
    """Run analysis with common test parameters."""

    parameters = {
        "analysis_time": get_analysis_time(data),
    }

    parameters.update(kwargs)

    return analyze_market_regime(
        data=data,
        **parameters,
    )


# ---------------------------------------------------------------------
# Basic analysis
# ---------------------------------------------------------------------


def test_analysis_returns_success():

    result = analyze(make_candles())

    assert result["status"] == "SUCCESS"


def test_analysis_returns_valid_regime():

    result = analyze(make_candles())

    assert result["regime"] in {
        TRENDING,
        RANGING,
        HIGH_VOLATILITY,
        LOW_VOLATILITY,
        TRANSITIONAL,
    }


def test_flat_market_is_ranging():

    result = analyze(
        make_candles("flat")
    )

    assert result["regime"] == RANGING


def test_bullish_market_direction():

    result = analyze(
        make_candles("bullish")
    )

    assert result["direction"] == BULLISH


def test_bearish_market_direction():

    result = analyze(
        make_candles("bearish")
    )

    assert result["direction"] == BEARISH


def test_direction_is_valid():

    result = analyze(make_candles())

    assert result["direction"] in {
        BULLISH,
        BEARISH,
        NEUTRAL,
    }


# ---------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------


def test_result_contains_required_fields():

    result = analyze(make_candles())

    expected = {
        "analysis_timestamp",
        "status",
        "source",
        "regime",
        "direction",
        "trend_strength",
        "volatility",
        "trend",
        "volume",
        "market_price",
        "candles_analyzed",
        "last_candle_close_time",
        "parameters",
    }

    assert expected.issubset(result.keys())


def test_source_is_reported():

    result = analyze(make_candles())

    assert result["source"] == ["kraken"]


def test_candle_count_is_reported():

    data = make_candles()

    result = analyze(data)

    assert result["candles_analyzed"] == len(data)


def test_market_price_is_positive():

    result = analyze(make_candles())

    assert result["market_price"] > 0


def test_analysis_timestamp_is_utc():

    result = analyze(make_candles())

    timestamp = pd.Timestamp(
        result["analysis_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_last_candle_close_time_is_reported():

    data = make_candles()

    result = analyze(data)

    assert (
        result["last_candle_close_time"]
        == data["close_time"].iloc[-1].isoformat()
    )


def test_parameters_are_returned():

    result = analyze(make_candles())

    assert isinstance(result["parameters"], dict)


# ---------------------------------------------------------------------
# Trend metrics
# ---------------------------------------------------------------------


def test_trend_metrics_are_returned():

    result = analyze(
        make_candles("bullish")
    )

    assert {
        "adx",
        "positive_di",
        "negative_di",
    }.issubset(result["trend"].keys())


def test_adx_is_finite():

    result = analyze(
        make_candles("bullish")
    )

    assert np.isfinite(result["trend"]["adx"])
    assert result["trend"]["adx"] >= 0


def test_directional_indicators_are_non_negative():

    result = analyze(
        make_candles("bearish")
    )

    assert result["trend"]["positive_di"] >= 0
    assert result["trend"]["negative_di"] >= 0


def test_trend_strength_is_valid():

    result = analyze(
        make_candles("bullish")
    )

    assert result["trend_strength"] in {
        "STRONG",
        "MODERATE",
        "WEAK",
    }


# ---------------------------------------------------------------------
# Volatility metrics
# ---------------------------------------------------------------------


def test_volatility_metrics_are_returned():

    result = analyze(make_candles())

    expected = {
        "condition",
        "atr",
        "atr_percent",
        "historical_median_atr_percent",
        "relative_volatility",
        "bollinger_width_percent",
    }

    assert expected.issubset(
        result["volatility"].keys()
    )


def test_atr_is_non_negative():

    result = analyze(make_candles())

    assert result["volatility"]["atr"] >= 0


def test_atr_percent_is_non_negative():

    result = analyze(make_candles())

    assert result["volatility"]["atr_percent"] >= 0


def test_relative_volatility_is_non_negative():

    result = analyze(make_candles())

    assert (
        result["volatility"]["relative_volatility"]
        >= 0
    )


def test_bollinger_width_is_non_negative():

    result = analyze(make_candles())

    assert (
        result["volatility"]["bollinger_width_percent"]
        >= 0
    )


def test_volatility_condition_is_valid():

    result = analyze(make_candles())

    assert result["volatility"]["condition"] in {
        HIGH_VOLATILITY,
        LOW_VOLATILITY,
        "NORMAL_VOLATILITY",
    }


# ---------------------------------------------------------------------
# Volume metrics
# ---------------------------------------------------------------------


def test_volume_metrics_are_returned():

    result = analyze(make_candles())

    expected = {
        "current",
        "average",
        "relative_volume",
        "above_average",
    }

    assert expected.issubset(
        result["volume"].keys()
    )


def test_volume_values_are_non_negative():

    result = analyze(make_candles())

    assert result["volume"]["current"] >= 0
    assert result["volume"]["average"] >= 0
    assert result["volume"]["relative_volume"] >= 0


def test_above_average_is_boolean():

    result = analyze(make_candles())

    assert isinstance(
        result["volume"]["above_average"],
        bool,
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_non_dataframe_input_is_rejected():

    with pytest.raises(TypeError):

        analyze_market_regime(
            [],
            analysis_time=datetime(
                2026, 11, 14, 12, 0,
                tzinfo=UTC,
            ),
        )


def test_empty_dataframe_is_rejected():

    with pytest.raises(MarketRegimeError):

        analyze_market_regime(
            pd.DataFrame(),
            analysis_time=datetime(
                2026, 11, 14, 12, 0,
                tzinfo=UTC,
            ),
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
def test_missing_required_column_is_rejected(column):

    data = make_candles().drop(
        columns=[column]
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_empty_source_is_rejected():

    data = make_candles()

    data.loc[10, "source"] = ""

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_invalid_ohlc_is_rejected():

    data = make_candles()

    data.loc[10, "high"] = (
        data.loc[10, "low"] - 10
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_negative_volume_is_rejected():

    data = make_candles()

    data.loc[10, "volume"] = -1

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_nan_price_is_rejected():

    data = make_candles()

    data.loc[10, "close"] = float("nan")

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_infinite_price_is_rejected():

    data = make_candles()

    data.loc[10, "close"] = float("inf")

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_duplicate_candles_are_rejected():

    data = make_candles()

    data.loc[10, "open_time"] = (
        data.loc[9, "open_time"]
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_unsorted_candles_are_rejected():

    data = (
        make_candles()
        .iloc[::-1]
        .reset_index(drop=True)
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_naive_candle_timestamp_is_rejected():

    data = make_candles()

    data["open_time"] = (
        data["open_time"]
        .dt.tz_localize(None)
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_non_utc_candle_timestamp_is_rejected():

    data = make_candles()

    non_utc = timezone(
        timedelta(hours=3)
    )

    data["open_time"] = data["open_time"].map(
        lambda value: value.astimezone(non_utc)
    )

    data["close_time"] = data["close_time"].map(
        lambda value: value.astimezone(non_utc)
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


# ---------------------------------------------------------------------
# Closed candle and timestamp validation
# ---------------------------------------------------------------------


def test_open_candle_is_rejected():

    data = make_candles()

    analysis_time = (
        data["close_time"].iloc[-1]
        - timedelta(minutes=1)
    )

    with pytest.raises(MarketRegimeError):
        analyze(
            data,
            analysis_time=analysis_time,
        )


def test_future_candle_is_rejected():

    data = make_candles()

    analysis_time = (
        data["close_time"].iloc[-2]
        + timedelta(minutes=1)
    )

    with pytest.raises(MarketRegimeError):
        analyze(
            data,
            analysis_time=analysis_time,
        )


def test_naive_analysis_time_is_rejected():

    data = make_candles()

    analysis_time = (
        get_analysis_time(data)
        .replace(tzinfo=None)
    )

    with pytest.raises(MarketRegimeError):
        analyze(
            data,
            analysis_time=analysis_time,
        )


def test_non_utc_analysis_time_is_rejected():

    data = make_candles()

    non_utc = timezone(
        timedelta(hours=3)
    )

    analysis_time = (
        get_analysis_time(data)
        .astimezone(non_utc)
    )

    with pytest.raises(MarketRegimeError):
        analyze(
            data,
            analysis_time=analysis_time,
        )


# ---------------------------------------------------------------------
# Minimum candle requirements
# ---------------------------------------------------------------------


def test_insufficient_candles_are_rejected():

    data = make_candles(
        candle_count=30
    )

    with pytest.raises(MarketRegimeError):
        analyze(data)


def test_minimum_candles_parameter_is_enforced():

    data = make_candles()

    with pytest.raises(MarketRegimeError):
        analyze(
            data,
            minimum_candles=120,
        )


def test_minimum_candles_must_cover_indicator_warmup():

    data = make_candles()

    with pytest.raises(MarketRegimeError):
        analyze(
            data,
            adx_period=30,
            atr_period=20,
            volatility_lookback=50,
            minimum_candles=60,
        )


# ---------------------------------------------------------------------
# Integer parameter validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("adx_period", 0),
        ("adx_period", -1),
        ("adx_period", True),
        ("atr_period", 0),
        ("atr_period", -1),
        ("atr_period", True),
        ("volatility_lookback", 0),
        ("volatility_lookback", -1),
        ("volatility_lookback", True),
        ("bollinger_period", 0),
        ("bollinger_period", -1),
        ("bollinger_period", True),
        ("minimum_candles", 0),
        ("minimum_candles", -1),
        ("minimum_candles", True),
    ],
)
def test_invalid_integer_parameters_are_rejected(
    parameter,
    value,
):

    with pytest.raises(MarketRegimeError):
        analyze(
            make_candles(),
            **{parameter: value},
        )


# ---------------------------------------------------------------------
# Numeric parameter validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("trending_threshold", 0),
        ("trending_threshold", -1),
        ("trending_threshold", 101),
        ("trending_threshold", float("nan")),
        ("trending_threshold", float("inf")),
        ("trending_threshold", True),
        ("ranging_threshold", -1),
        ("ranging_threshold", float("nan")),
        ("ranging_threshold", float("inf")),
        ("ranging_threshold", True),
    ],
)
def test_invalid_adx_thresholds_are_rejected(
    parameter,
    value,
):

    with pytest.raises(MarketRegimeError):
        analyze(
            make_candles(),
            **{parameter: value},
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("high_volatility_ratio", 0),
        ("high_volatility_ratio", -1),
        ("high_volatility_ratio", 1),
        ("high_volatility_ratio", float("nan")),
        ("high_volatility_ratio", float("inf")),
        ("high_volatility_ratio", True),
        ("low_volatility_ratio", 0),
        ("low_volatility_ratio", -1),
        ("low_volatility_ratio", 1),
        ("low_volatility_ratio", float("nan")),
        ("low_volatility_ratio", float("inf")),
        ("low_volatility_ratio", True),
    ],
)
def test_invalid_volatility_ratios_are_rejected(
    parameter,
    value,
):

    with pytest.raises(MarketRegimeError):
        analyze(
            make_candles(),
            **{parameter: value},
        )


def test_trending_threshold_must_exceed_ranging_threshold():

    with pytest.raises(MarketRegimeError):
        analyze(
            make_candles(),
            trending_threshold=15,
            ranging_threshold=20,
        )


def test_zero_ranging_threshold_is_valid():

    result = analyze(
        make_candles(),
        ranging_threshold=0,
    )

    assert result["status"] == "SUCCESS"


def test_low_volatility_ratio_must_be_below_high_ratio():

    with pytest.raises(MarketRegimeError):
        analyze(
            make_candles(),
            low_volatility_ratio=0.9,
            high_volatility_ratio=0.8,
        )


def test_custom_parameters_are_supported():

    result = analyze(
        make_candles(),
        adx_period=10,
        atr_period=10,
        volatility_lookback=30,
        bollinger_period=15,
        minimum_candles=40,
    )

    assert result["status"] == "SUCCESS"

    assert result["parameters"]["adx_period"] == 10
    assert result["parameters"]["atr_period"] == 10
    assert result["parameters"]["volatility_lookback"] == 30
    assert result["parameters"]["bollinger_period"] == 15


# ---------------------------------------------------------------------
# Internal classification functions
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "positive_di,negative_di,expected",
    [
        (30, 10, BULLISH),
        (10, 30, BEARISH),
        (20, 20, NEUTRAL),
    ],
)
def test_direction_classification(
    positive_di,
    negative_di,
    expected,
):

    assert (
        _classify_direction(
            positive_di,
            negative_di,
        )
        == expected
    )


@pytest.mark.parametrize(
    "adx,volatility,expected",
    [
        (30, "NORMAL_VOLATILITY", TRENDING),
        (10, "NORMAL_VOLATILITY", RANGING),
        (22, "NORMAL_VOLATILITY", TRANSITIONAL),
        (10, HIGH_VOLATILITY, HIGH_VOLATILITY),
        (30, LOW_VOLATILITY, LOW_VOLATILITY),
    ],
)
def test_regime_classification(
    adx,
    volatility,
    expected,
):

    assert (
        _classify_regime(
            adx,
            volatility,
            25,
            20,
        )
        == expected
    )


@pytest.mark.parametrize(
    "current,median,expected",
    [
        (2.0, 1.0, HIGH_VOLATILITY),
        (0.5, 1.0, LOW_VOLATILITY),
        (1.0, 1.0, "NORMAL_VOLATILITY"),
    ],
)
def test_volatility_classification(
    current,
    median,
    expected,
):

    condition, ratio = _classify_volatility(
        current,
        median,
        1.5,
        0.7,
    )

    assert condition == expected
    assert ratio == pytest.approx(current / median)


def test_volatility_classification_rejects_zero_median():

    with pytest.raises(MarketRegimeError):
        _classify_volatility(
            1.0,
            0.0,
            1.5,
            0.7,
        )


def test_volatility_classification_rejects_nan():

    with pytest.raises(MarketRegimeError):
        _classify_volatility(
            float("nan"),
            1.0,
            1.5,
            0.7,
        )


# ---------------------------------------------------------------------
# Indicator calculation functions
# ---------------------------------------------------------------------


def test_atr_calculation_returns_series():

    data = make_candles()

    result = _calculate_atr(
        data,
        period=14,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(data)
    assert np.isfinite(result.iloc[-1])


def test_adx_calculation_returns_three_series():

    data = make_candles("bullish")

    adx, positive_di, negative_di = _calculate_adx(
        data,
        period=14,
    )

    assert isinstance(adx, pd.Series)
    assert isinstance(positive_di, pd.Series)
    assert isinstance(negative_di, pd.Series)

    assert len(adx) == len(data)


def test_bollinger_width_calculation():

    data = make_candles()

    result = _calculate_bollinger_width(
        data["close"],
        period=20,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(data)
    assert result.iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Data integrity and determinism
# ---------------------------------------------------------------------


def test_analysis_does_not_modify_input_dataframe():

    data = make_candles("bullish")

    original = data.copy(deep=True)

    analyze(data)

    assert_frame_equal(
        data,
        original,
    )


def test_analysis_is_deterministic():

    data = make_candles("bullish")

    first = analyze(data)
    second = analyze(data)

    assert first == second


def test_analysis_uses_only_supplied_candles():

    data = make_candles(
        "bullish",
        candle_count=80,
    )

    prefix = data.iloc[:70].copy()

    result = analyze(prefix)

    assert result["candles_analyzed"] == 70

    assert (
        result["last_candle_close_time"]
        == prefix["close_time"].iloc[-1].isoformat()
    )


def test_analysis_reports_all_sources():

    data = make_candles()

    data.loc[10:20, "source"] = "coingecko"

    result = analyze(data)

    assert set(result["source"]) == {
        "kraken",
        "coingecko",
    }


def test_analysis_does_not_generate_trading_signals():

    result = analyze(make_candles())

    forbidden_keys = {
        "signal",
        "entry",
        "stop_loss",
        "take_profit",
        "order",
        "trade",
    }

    assert forbidden_keys.isdisjoint(
        result.keys()
    )