"""Tests for Bitcoin market regime analysis."""

from datetime import datetime, timedelta, timezone

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
)


UTC = timezone.utc

START = datetime(
    2026, 11, 14, 0, 0, tzinfo=UTC
)

REGIMES = {
    TRENDING,
    RANGING,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    TRANSITIONAL,
}

DIRECTIONS = {
    BULLISH,
    BEARISH,
    NEUTRAL,
}


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

        spread = volatility

        high = (
            max(open_price, close)
            + spread
        )

        low = (
            min(open_price, close)
            - spread
        )

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
    """Return a UTC time after the final candle."""

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
# Basic market regime analysis
# ---------------------------------------------------------------------


def test_analysis_returns_success_status():

    result = analyze(
        make_candles()
    )

    assert result["status"] == "SUCCESS"


def test_analysis_returns_valid_regime():

    result = analyze(
        make_candles()
    )

    assert result["regime"] in REGIMES


def test_flat_market_is_classified_as_ranging():

    result = analyze(
        make_candles("flat")
    )

    assert result["regime"] == RANGING


def test_bullish_market_has_bullish_direction():

    result = analyze(
        make_candles("bullish")
    )

    assert result["direction"] == BULLISH


def test_bearish_market_has_bearish_direction():

    result = analyze(
        make_candles("bearish")
    )

    assert result["direction"] == BEARISH


def test_direction_is_always_valid():

    result = analyze(
        make_candles()
    )

    assert result["direction"] in DIRECTIONS


def test_regime_is_independent_of_direction():

    result = analyze(
        make_candles("bullish")
    )

    assert result["regime"] in REGIMES
    assert result["direction"] in DIRECTIONS


# ---------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------


def test_result_contains_required_fields():

    result = analyze(
        make_candles()
    )

    required_fields = {
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

    assert required_fields.issubset(
        result.keys()
    )


def test_source_is_reported():

    result = analyze(
        make_candles()
    )

    assert result["source"] == ["kraken"]


def test_candle_count_is_reported():

    data = make_candles()

    result = analyze(data)

    assert result["candles_analyzed"] == len(data)


def test_market_price_is_positive():

    result = analyze(
        make_candles()
    )

    assert result["market_price"] > 0


def test_analysis_timestamp_is_utc():

    result = analyze(
        make_candles()
    )

    timestamp = pd.Timestamp(
        result["analysis_timestamp"]
    )

    assert timestamp.tzinfo is not None

    assert timestamp.utcoffset() == timedelta(0)


def test_last_candle_close_time_is_reported():

    data = make_candles()

    result = analyze(data)

    expected = (
        data["close_time"]
        .iloc[-1]
        .isoformat()
    )

    assert (
        result["last_candle_close_time"]
        == expected
    )


def test_parameters_are_returned():

    result = analyze(
        make_candles()
    )

    assert isinstance(
        result["parameters"],
        dict,
    )


# ---------------------------------------------------------------------
# Trend metrics
# ---------------------------------------------------------------------


def test_trend_metrics_are_returned():

    result = analyze(
        make_candles("bullish")
    )

    trend = result["trend"]

    assert "adx" in trend
    assert "positive_di" in trend
    assert "negative_di" in trend


def test_adx_is_finite():

    result = analyze(
        make_candles("bullish")
    )

    assert pd.notna(
        result["trend"]["adx"]
    )

    assert result["trend"]["adx"] >= 0


def test_positive_di_is_non_negative():

    result = analyze(
        make_candles("bullish")
    )

    assert result["trend"]["positive_di"] >= 0


def test_negative_di_is_non_negative():

    result = analyze(
        make_candles("bearish")
    )

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

    result = analyze(
        make_candles()
    )

    volatility = result["volatility"]

    required_fields = {
        "condition",
        "atr",
        "atr_percent",
        "historical_median_atr_percent",
        "relative_volatility",
        "bollinger_width_percent",
    }

    assert required_fields.issubset(
        volatility.keys()
    )


def test_atr_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert result["volatility"]["atr"] >= 0


def test_atr_percent_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert result["volatility"]["atr_percent"] >= 0


def test_relative_volatility_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert (
        result["volatility"]["relative_volatility"]
        >= 0
    )


def test_bollinger_width_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert (
        result["volatility"]["bollinger_width_percent"]
        >= 0
    )


def test_volatility_condition_is_valid():

    result = analyze(
        make_candles()
    )

    assert result["volatility"]["condition"] in {
        HIGH_VOLATILITY,
        LOW_VOLATILITY,
        "NORMAL",
    }


# ---------------------------------------------------------------------
# Volume metrics
# ---------------------------------------------------------------------


def test_volume_metrics_are_returned():

    result = analyze(
        make_candles()
    )

    volume = result["volume"]

    required_fields = {
        "current",
        "average",
        "relative_volume",
        "above_average",
    }

    assert required_fields.issubset(
        volume.keys()
    )


def test_current_volume_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert result["volume"]["current"] >= 0


def test_average_volume_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert result["volume"]["average"] >= 0


def test_relative_volume_is_non_negative():

    result = analyze(
        make_candles()
    )

    assert result["volume"]["relative_volume"] >= 0


def test_above_average_is_boolean():

    result = analyze(
        make_candles()
    )

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
def test_missing_required_column_is_rejected(
    column,
):

    data = make_candles()

    data = data.drop(
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

    data = make_candles()

    data = (
        data.iloc[::-1]
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

    data["open_time"] = (
        data["open_time"]
        .map(
            lambda value:
            value.astimezone(non_utc)
        )
    )

    data["close_time"] = (
        data["close_time"]
        .map(
            lambda value:
            value.astimezone(non_utc)
        )
    )

    with pytest.raises(MarketRegimeError):

        analyze(data)


# ---------------------------------------------------------------------
# Closed candle and time validation
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
# Parameter validation
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

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            **{
                parameter: value
            },
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("trending_threshold", 0),
        ("trending_threshold", -1),
        ("trending_threshold", float("nan")),
        ("trending_threshold", float("inf")),
        ("ranging_threshold", 0),
        ("ranging_threshold", -1),
        ("ranging_threshold", float("nan")),
        ("ranging_threshold", float("inf")),
    ],
)
def test_invalid_adx_thresholds_are_rejected(
    parameter,
    value,
):

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            **{
                parameter: value
            },
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("high_volatility_ratio", 0),
        ("high_volatility_ratio", -1),
        ("high_volatility_ratio", 1),
        ("high_volatility_ratio", float("nan")),
        ("high_volatility_ratio", float("inf")),
        ("low_volatility_ratio", 0),
        ("low_volatility_ratio", -1),
        ("low_volatility_ratio", 1),
        ("low_volatility_ratio", float("nan")),
        ("low_volatility_ratio", float("inf")),
    ],
)
def test_invalid_volatility_ratios_are_rejected(
    parameter,
    value,
):

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            **{
                parameter: value
            },
        )


def test_trending_threshold_must_exceed_ranging_threshold():

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            trending_threshold=15,
            ranging_threshold=20,
        )


def test_low_volatility_ratio_must_be_below_high_ratio():

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            low_volatility_ratio=0.9,
            high_volatility_ratio=0.8,
        )


def test_custom_parameters_are_supported():

    data = make_candles()

    result = analyze(
        data,
        adx_period=10,
        atr_period=10,
        volatility_lookback=30,
        bollinger_period=15,
        minimum_candles=40,
    )

    assert result["status"] == "SUCCESS"

    assert (
        result["parameters"]["adx_period"]
        == 10
    )

    assert (
        result["parameters"]["atr_period"]
        == 10
    )

    assert (
        result["parameters"]["volatility_lookback"]
        == 30
    )

    assert (
        result["parameters"]["bollinger_period"]
        == 15
    )


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


def test_analysis_does_not_generate_trading_signals():

    result = analyze(
        make_candles()
    )

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


def test_analysis_with_different_sources_reports_all_sources():

    data = make_candles()

    data.loc[10:20, "source"] = "coingecko"

    result = analyze(data)

    assert set(result["source"]) == {
        "kraken",
        "coingecko",
    }