"""Edge case tests for Bitcoin market regime analysis."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

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
                "volume": 100.0,
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
    """Run market regime analysis with valid timestamps."""

    if "close_time" in data.columns:
        analysis_time = get_analysis_time(data)

    else:
        analysis_time = START + timedelta(days=1)

    parameters = {
        "analysis_time": analysis_time,
    }

    parameters.update(kwargs)

    return analyze_market_regime(
        data=data,
        **parameters,
    )


# ---------------------------------------------------------------------
# Timestamp edge cases
# ---------------------------------------------------------------------


def test_analysis_rejects_nat_analysis_time():

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            analysis_time=pd.NaT,
        )


def test_analysis_rejects_none_like_invalid_timestamp():

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            analysis_time=float("nan"),
        )


def test_analysis_rejects_naive_timestamp():

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


def test_analysis_rejects_non_utc_timestamp():

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


def test_analysis_accepts_utc_timestamp():

    data = make_candles()

    result = analyze(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["status"] == "SUCCESS"


def test_analysis_rejects_time_before_last_close():

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


# ---------------------------------------------------------------------
# Candle count edge cases
# ---------------------------------------------------------------------


def test_analysis_rejects_empty_dataframe():

    with pytest.raises(MarketRegimeError):

        analyze(
            pd.DataFrame()
        )


def test_analysis_rejects_insufficient_candles():

    data = make_candles(
        candle_count=30
    )

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_accepts_exact_minimum_candle_count():

    data = make_candles(
        candle_count=64
    )

    result = analyze(data)

    assert result["status"] == "SUCCESS"
    assert result["candles_analyzed"] == 64


def test_analysis_reports_correct_candle_count():

    data = make_candles(
        candle_count=80
    )

    result = analyze(data)

    assert result["candles_analyzed"] == 80


# ---------------------------------------------------------------------
# Parameter boundary edge cases
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
            **{parameter: value},
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("trending_threshold", 0),
        ("trending_threshold", -1),
        ("trending_threshold", 101),
        ("trending_threshold", float("nan")),
        ("trending_threshold", float("inf")),
        ("ranging_threshold", -1),
        ("ranging_threshold", float("nan")),
        ("ranging_threshold", float("inf")),
        ("high_volatility_ratio", 0),
        ("high_volatility_ratio", 1),
        ("high_volatility_ratio", float("nan")),
        ("high_volatility_ratio", float("inf")),
        ("low_volatility_ratio", 0),
        ("low_volatility_ratio", 1),
        ("low_volatility_ratio", float("nan")),
        ("low_volatility_ratio", float("inf")),
    ],
)
def test_invalid_numeric_parameters_are_rejected(
    parameter,
    value,
):

    data = make_candles()

    with pytest.raises(MarketRegimeError):

        analyze(
            data,
            **{parameter: value},
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


def test_zero_ranging_threshold_is_valid():

    data = make_candles()

    result = analyze(
        data,
        ranging_threshold=0,
    )

    assert result["status"] == "SUCCESS"


# ---------------------------------------------------------------------
# Volatility classification edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "current,median,expected",
    [
        (2.0, 1.0, HIGH_VOLATILITY),
        (0.5, 1.0, LOW_VOLATILITY),
        (1.0, 1.0, "NORMAL_VOLATILITY"),
    ],
)
def test_volatility_classification_boundaries(
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

    assert ratio == pytest.approx(
        current / median
    )


@pytest.mark.parametrize(
    "current,median",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (-float("inf"), 1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (1.0, -float("inf")),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_volatility_classification_rejects_invalid_values(
    current,
    median,
):

    with pytest.raises(MarketRegimeError):

        _classify_volatility(
            current,
            median,
            1.5,
            0.7,
        )


def test_zero_current_volatility_is_valid():

    condition, ratio = _classify_volatility(
        0.0,
        1.0,
        1.5,
        0.7,
    )

    assert condition == LOW_VOLATILITY
    assert ratio == 0.0


def test_volatility_classification_at_high_boundary():

    condition, ratio = _classify_volatility(
        1.5,
        1.0,
        1.5,
        0.7,
    )

    assert condition == HIGH_VOLATILITY
    assert ratio == pytest.approx(1.5)


def test_volatility_classification_at_low_boundary():

    condition, ratio = _classify_volatility(
        0.7,
        1.0,
        1.5,
        0.7,
    )

    assert condition == LOW_VOLATILITY
    assert ratio == pytest.approx(0.7)


# ---------------------------------------------------------------------
# Regime classification edge cases
# ---------------------------------------------------------------------


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
def test_regime_classification_boundaries(
    adx,
    volatility,
    expected,
):

    result = _classify_regime(
        adx,
        volatility,
        25,
        20,
    )

    assert result == expected


def test_high_volatility_overrides_trend_strength():

    result = _classify_regime(
        80,
        HIGH_VOLATILITY,
        25,
        20,
    )

    assert result == HIGH_VOLATILITY


def test_low_volatility_overrides_trend_strength():

    result = _classify_regime(
        80,
        LOW_VOLATILITY,
        25,
        20,
    )

    assert result == LOW_VOLATILITY


# ---------------------------------------------------------------------
# Direction classification edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "positive_di,negative_di,expected",
    [
        (30, 10, BULLISH),
        (10, 30, BEARISH),
        (20, 20, NEUTRAL),
        (0, 0, NEUTRAL),
    ],
)
def test_direction_classification_boundaries(
    positive_di,
    negative_di,
    expected,
):

    result = _classify_direction(
        positive_di,
        negative_di,
    )

    assert result == expected


# ---------------------------------------------------------------------
# ATR edge cases
# ---------------------------------------------------------------------


def test_atr_returns_finite_value_for_flat_market():

    data = make_candles(
        direction="flat"
    )

    result = _calculate_atr(
        data,
        period=14,
    )

    assert np.isfinite(result.iloc[-1])
    assert result.iloc[-1] >= 0


def test_atr_returns_series_with_original_length():

    data = make_candles()

    result = _calculate_atr(
        data,
        period=14,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(data)


def test_atr_is_not_negative():

    data = make_candles(
        direction="volatile"
    )

    result = _calculate_atr(
        data,
        period=14,
    )

    valid_values = result.dropna()

    assert (valid_values >= 0).all()


# ---------------------------------------------------------------------
# ADX edge cases
# ---------------------------------------------------------------------


def test_adx_handles_flat_market():

    data = make_candles(
        direction="flat"
    )

    adx, positive_di, negative_di = _calculate_adx(
        data,
        period=14,
    )

    assert np.isfinite(adx.iloc[-1])
    assert adx.iloc[-1] == pytest.approx(0.0)

    assert positive_di.iloc[-1] == pytest.approx(0.0)
    assert negative_di.iloc[-1] == pytest.approx(0.0)


def test_adx_values_are_within_valid_range():

    data = make_candles(
        direction="bullish"
    )

    adx, positive_di, negative_di = _calculate_adx(
        data,
        period=14,
    )

    assert adx.dropna().between(0, 100).all()
    assert positive_di.dropna().between(0, 100).all()
    assert negative_di.dropna().between(0, 100).all()


def test_adx_returns_three_series():

    data = make_candles()

    results = _calculate_adx(
        data,
        period=14,
    )

    assert len(results) == 3

    assert all(
        isinstance(item, pd.Series)
        for item in results
    )


# ---------------------------------------------------------------------
# Bollinger Band edge cases
# ---------------------------------------------------------------------


def test_bollinger_width_handles_constant_prices():

    close = pd.Series(
        [100.0] * 30
    )

    result = _calculate_bollinger_width(
        close,
        period=20,
    )

    assert result.iloc[-1] == pytest.approx(0.0)


def test_bollinger_width_handles_zero_middle_value():

    close = pd.Series(
        [0.0] * 30
    )

    result = _calculate_bollinger_width(
        close,
        period=20,
    )

    assert np.isnan(
        result.iloc[-1]
    )


def test_bollinger_width_returns_series():

    close = pd.Series(
        np.linspace(100, 120, 50)
    )

    result = _calculate_bollinger_width(
        close,
        period=20,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


# ---------------------------------------------------------------------
# Volume edge cases
# ---------------------------------------------------------------------


def test_zero_volume_market_is_handled():

    data = make_candles()

    data["volume"] = 0.0

    result = analyze(data)

    assert result["status"] == "SUCCESS"
    assert result["volume"]["current"] == 0.0
    assert result["volume"]["average"] == 0.0
    assert result["volume"]["relative_volume"] == 1.0


def test_positive_current_volume_with_zero_previous_volume():

    data = make_candles()

    data["volume"] = 0.0
    data.loc[data.index[-1], "volume"] = 10.0

    result = analyze(data)

    assert result["status"] == "SUCCESS"

    assert result["volume"]["current"] == 10.0
    assert result["volume"]["average"] > 0
    assert np.isfinite(
        result["volume"]["relative_volume"]
    )


def test_negative_volume_is_rejected():

    data = make_candles()

    data.loc[10, "volume"] = -1.0

    with pytest.raises(MarketRegimeError):

        analyze(data)


# ---------------------------------------------------------------------
# Data integrity edge cases
# ---------------------------------------------------------------------


def test_analysis_rejects_duplicate_open_times():

    data = make_candles()

    data.loc[10, "open_time"] = (
        data.loc[9, "open_time"]
    )

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_rejects_unsorted_candles():

    data = (
        make_candles()
        .iloc[::-1]
        .reset_index(drop=True)
    )

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_rejects_invalid_ohlc():

    data = make_candles()

    data.loc[10, "high"] = (
        data.loc[10, "low"] - 10
    )

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_rejects_nan_close():

    data = make_candles()

    data.loc[10, "close"] = np.nan

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_rejects_infinite_close():

    data = make_candles()

    data.loc[10, "close"] = np.inf

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_rejects_empty_source():

    data = make_candles()

    data.loc[10, "source"] = ""

    with pytest.raises(MarketRegimeError):

        analyze(data)


def test_analysis_rejects_missing_required_column():

    data = make_candles().drop(
        columns=["volume"]
    )

    with pytest.raises(MarketRegimeError):

        analyze(data)


# ---------------------------------------------------------------------
# Output integrity
# ---------------------------------------------------------------------


def test_analysis_result_contains_required_fields():

    result = analyze(
        make_candles()
    )

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

    assert expected.issubset(
        result.keys()
    )


def test_analysis_result_contains_no_nan_values():

    result = analyze(
        make_candles()
    )

    def check_values(value):

        if isinstance(value, dict):

            for nested_value in value.values():
                check_values(nested_value)

        elif isinstance(value, (list, tuple)):

            for nested_value in value:
                check_values(nested_value)

        elif isinstance(value, (float, np.floating)):

            assert np.isfinite(value)

    check_values(result)


def test_analysis_is_deterministic():

    data = make_candles(
        direction="bullish"
    )

    first = analyze(data)
    second = analyze(data)

    assert first == second


def test_analysis_does_not_modify_input_data():

    data = make_candles(
        direction="bullish"
    )

    original = data.copy(deep=True)

    analyze(data)

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_analysis_reports_all_sources():

    data = make_candles()

    data.loc[10:20, "source"] = "coingecko"

    result = analyze(data)

    assert set(result["source"]) == {
        "kraken",
        "coingecko",
    }


# ---------------------------------------------------------------------
# Trading safety
# ---------------------------------------------------------------------


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


def test_analysis_does_not_enable_auto_trading():

    result = analyze(
        make_candles()
    )

    assert result["status"] == "SUCCESS"

    assert "order" not in result
    assert "trade" not in result