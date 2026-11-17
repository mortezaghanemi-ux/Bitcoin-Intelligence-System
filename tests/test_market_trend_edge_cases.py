from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from market_analysis.trend import (
    TrendAnalysisError,
    analyze_trend,
    _calculate_adx,
    _calculate_confidence,
    _calculate_momentum,
    _calculate_ma_structure,
    _classify_trend,
    _classify_strength,
    _detect_reversal_warning,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 11, 13, 0, 0, tzinfo=UTC)


def make_candles(direction="bullish", count=80):
    """Create deterministic closed OHLCV candles for trend tests."""
    rows = []

    for index in range(count):
        if direction == "bullish":
            close = 95000 + index * 25
        elif direction == "bearish":
            close = 97000 - index * 25
        elif direction == "flat":
            close = 95000
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        open_price = close if index == 0 else rows[-1]["close"]

        open_time = BASE_TIME + timedelta(minutes=index * 5)
        close_time = open_time + timedelta(minutes=5)

        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": float(open_price),
                "high": float(max(open_price, close) + 10),
                "low": float(min(open_price, close) - 10),
                "close": float(close),
                "volume": 100.0 + index,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def get_analysis_time(data):
    """Return a UTC analysis time after the latest candle closes."""
    return data["close_time"].max() + timedelta(minutes=5)


# ---------------------------------------------------------------------
# Analysis time edge cases
# ---------------------------------------------------------------------


def test_analysis_time_none_uses_current_utc():
    data = make_candles()

    result = analyze_trend(data, analysis_time=None)

    timestamp = pd.Timestamp(result["analysis_timestamp"])

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset().total_seconds() == 0


def test_analysis_time_accepts_utc_iso_string():
    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data).isoformat(),
    )

    timestamp = pd.Timestamp(result["analysis_timestamp"])

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    "invalid_time",
    [
        "not-a-valid-date",
        123,
        pd.NaT,
        float("nan"),
    ],
)
def test_invalid_analysis_time_values_are_rejected(invalid_time):
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(data, analysis_time=invalid_time)


def test_analysis_time_with_nonzero_utc_offset_is_rejected():
    data = make_candles()

    non_utc_time = datetime(
        2026,
        11,
        13,
        12,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(data, analysis_time=non_utc_time)


def test_analysis_time_before_latest_candle_is_rejected():
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=data["open_time"].iloc[-1],
        )


# ---------------------------------------------------------------------
# Input data edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        [],
        {},
        "invalid-data",
    ],
)
def test_non_dataframe_input_is_rejected(invalid_data):
    with pytest.raises(TypeError):
        analyze_trend(
            invalid_data,
            analysis_time=datetime(2026, 11, 14, tzinfo=UTC),
        )


def test_empty_dataframe_is_rejected():
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

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=datetime(2026, 11, 14, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "missing_column",
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
def test_each_required_column_is_enforced(missing_column):
    data = make_candles().drop(columns=[missing_column])

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=datetime(2026, 11, 14, tzinfo=UTC),
        )


def test_duplicate_dataframe_index_does_not_change_analysis():
    data = make_candles()
    data.index = [index // 2 for index in range(len(data))]

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["candles_analyzed"] == len(data)


def test_input_dataframe_is_not_modified_after_analysis():
    data = make_candles()
    original = data.copy(deep=True)

    analyze_trend(data, analysis_time=get_analysis_time(data))

    pd.testing.assert_frame_equal(data, original)


# ---------------------------------------------------------------------
# Timestamp and candle ordering edge cases
# ---------------------------------------------------------------------


def test_duplicate_open_time_is_rejected():
    data = make_candles()
    data.loc[10, "open_time"] = data.loc[9, "open_time"]

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(make_candles()),
        )


def test_duplicate_close_time_is_rejected():
    data = make_candles()
    data.loc[10, "close_time"] = data.loc[9, "close_time"]

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(make_candles()),
        )


def test_non_chronological_open_times_are_rejected():
    data = make_candles()

    data.loc[10, "open_time"], data.loc[11, "open_time"] = (
        data.loc[11, "open_time"],
        data.loc[10, "open_time"],
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(make_candles()),
        )


def test_non_chronological_close_times_are_rejected():
    data = make_candles()

    data.loc[10, "close_time"], data.loc[11, "close_time"] = (
        data.loc[11, "close_time"],
        data.loc[10, "close_time"],
    )

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(make_candles()),
        )


# ---------------------------------------------------------------------
# Source validation edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_source",
    [
        "",
        " ",
        "   ",
        None,
    ],
)
def test_invalid_source_values_are_rejected(invalid_source):
    data = make_candles()
    data.loc[10, "source"] = invalid_source

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            analysis_time=get_analysis_time(make_candles()),
        )


def test_multiple_sources_are_returned_in_sorted_order():
    data = make_candles()

    data.loc[0:19, "source"] = "kraken"
    data.loc[20:39, "source"] = "coingecko"
    data.loc[40:, "source"] = "binance"

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["source"] == [
        "binance",
        "coingecko",
        "kraken",
    ]


# ---------------------------------------------------------------------
# Parameter edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("fast_period", True),
        ("fast_period", False),
        ("fast_period", 2.5),
        ("fast_period", "20"),
        ("slow_period", True),
        ("slow_period", 50.5),
        ("slow_period", "50"),
        ("adx_period", True),
        ("adx_period", 14.5),
        ("adx_period", "14"),
        ("minimum_candles", True),
        ("minimum_candles", 50.5),
        ("minimum_candles", "50"),
    ],
)
def test_non_integer_parameters_are_rejected(parameter, value):
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


@pytest.mark.parametrize(
    "parameter",
    [
        "fast_period",
        "slow_period",
        "adx_period",
        "minimum_candles",
    ],
)
def test_zero_and_negative_parameters_are_rejected(parameter):
    data = make_candles()

    parameters = {
        "fast_period": 20,
        "slow_period": 50,
        "adx_period": 14,
        "minimum_candles": 50,
    }

    for invalid_value in (0, -1):
        parameters[parameter] = invalid_value

        with pytest.raises(TrendAnalysisError):
            analyze_trend(
                data,
                analysis_time=get_analysis_time(data),
                **parameters,
            )


def test_fast_period_equal_to_slow_period_is_rejected():
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            fast_period=20,
            slow_period=20,
            analysis_time=get_analysis_time(data),
        )


def test_fast_period_greater_than_slow_period_is_rejected():
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            fast_period=50,
            slow_period=20,
            analysis_time=get_analysis_time(data),
        )


def test_minimum_candles_must_exceed_momentum_period():
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            fast_period=2,
            slow_period=3,
            adx_period=2,
            minimum_candles=5,
            analysis_time=get_analysis_time(data),
        )


def test_minimum_candles_must_be_at_least_slow_period():
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            fast_period=10,
            slow_period=30,
            adx_period=5,
            minimum_candles=29,
            analysis_time=get_analysis_time(data),
        )


def test_minimum_candles_must_satisfy_adx_requirement():
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        analyze_trend(
            data,
            fast_period=5,
            slow_period=10,
            adx_period=20,
            minimum_candles=39,
            analysis_time=get_analysis_time(data),
        )


# ---------------------------------------------------------------------
# ADX calculation edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_period",
    [
        0,
        -1,
        True,
        False,
        2.5,
        "14",
        None,
    ],
)
def test_adx_rejects_invalid_period(invalid_period):
    data = make_candles()

    with pytest.raises(TrendAnalysisError):
        _calculate_adx(data, invalid_period)


def test_adx_output_is_finite_and_bounded():
    data = make_candles("bullish")
    result = _calculate_adx(data, 14)

    assert len(result) == len(data)
    assert np.isfinite(result.dropna()).all()
    assert result.dropna().between(0, 100).all()


def test_adx_handles_flat_market_without_nan_in_final_value():
    data = make_candles("flat")
    result = _calculate_adx(data, 14)

    assert len(result) == len(data)
    assert np.isfinite(result.iloc[-1])
    assert 0 <= result.iloc[-1] <= 100


# ---------------------------------------------------------------------
# Momentum calculation edge cases
# ---------------------------------------------------------------------


def test_momentum_rejects_insufficient_data():
    close = pd.Series([100, 101, 102, 103, 104], dtype=float)

    with pytest.raises(TrendAnalysisError):
        _calculate_momentum(close, period=5)


def test_momentum_rejects_zero_previous_price():
    close = pd.Series(
        [0, 100, 101, 102, 103, 104],
        dtype=float,
    )

    with pytest.raises(TrendAnalysisError):
        _calculate_momentum(close, period=5)


def test_momentum_rejects_negative_previous_price():
    close = pd.Series(
        [-10, 100, 101, 102, 103, 104],
        dtype=float,
    )

    with pytest.raises(TrendAnalysisError):
        _calculate_momentum(close, period=5)


def test_momentum_rejects_non_finite_values():
    close = pd.Series(
        [100, 101, 102, 103, 104, np.nan],
        dtype=float,
    )

    with pytest.raises(TrendAnalysisError):
        _calculate_momentum(close, period=5)


def test_momentum_returns_expected_percentage_change():
    close = pd.Series(
        [100, 101, 102, 103, 104, 110],
        dtype=float,
    )

    result = _calculate_momentum(close, period=5)

    assert result == pytest.approx(10.0)


# ---------------------------------------------------------------------
# Moving average structure edge cases
# ---------------------------------------------------------------------


def test_moving_average_structure_rejects_insufficient_data():
    data = make_candles(count=10)

    with pytest.raises(TrendAnalysisError):
        _calculate_ma_structure(
            data,
            fast_period=5,
            slow_period=20,
        )


def test_moving_average_structure_returns_finite_values():
    data = make_candles("bullish")

    result = _calculate_ma_structure(
        data,
        fast_period=10,
        slow_period=20,
    )

    assert np.isfinite(result["price"])
    assert np.isfinite(result["fast_ma"])
    assert np.isfinite(result["slow_ma"])


# ---------------------------------------------------------------------
# Classification boundary edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (20, "BULLISH"),
        (19.999, "NEUTRAL"),
        (0, "NEUTRAL"),
        (-19.999, "NEUTRAL"),
        (-20, "BEARISH"),
    ],
)
def test_trend_classification_threshold_boundaries(score, expected):
    assert _classify_trend(score) == expected


@pytest.mark.parametrize(
    "adx,expected",
    [
        (25, "STRONG"),
        (24.999, "MODERATE"),
        (20, "MODERATE"),
        (19.999, "WEAK"),
        (0, "WEAK"),
    ],
)
def test_strength_classification_threshold_boundaries(adx, expected):
    assert _classify_strength(adx) == expected


# ---------------------------------------------------------------------
# Confidence calculation edge cases
# ---------------------------------------------------------------------


def test_confidence_is_zero_for_zero_score_and_zero_adx():
    assert _calculate_confidence(score=0, adx=0) == 0


def test_confidence_is_bounded_at_one_hundred():
    result = _calculate_confidence(score=100, adx=100)

    assert 0 <= result <= 100
    assert result == 100


def test_confidence_does_not_exceed_upper_bound():
    result = _calculate_confidence(score=1000, adx=1000)

    assert result == 100


def test_confidence_is_non_negative_for_negative_score():
    result = _calculate_confidence(score=-100, adx=25)

    assert 0 <= result <= 100


# ---------------------------------------------------------------------
# Reversal warning edge cases
# ---------------------------------------------------------------------


def test_reversal_warning_returns_false_for_single_candle():
    data = pd.DataFrame({"close": [100.0]})

    result = _detect_reversal_warning(
        data,
        fast_period=1,
        slow_period=2,
    )

    assert isinstance(result, dict)
    assert result["detected"] is False
    assert result["direction"] is None


# ---------------------------------------------------------------------
# Final analysis output edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "direction,expected_trend",
    [
        ("bullish", "BULLISH"),
        ("bearish", "BEARISH"),
        ("flat", "NEUTRAL"),
    ],
)
def test_analysis_returns_expected_trend_for_market_conditions(
    direction,
    expected_trend,
):
    data = make_candles(direction)

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["trend"] == expected_trend


def test_analysis_score_is_always_bounded():
    for direction in ("bullish", "bearish", "flat"):
        data = make_candles(direction)

        result = analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )

        assert -100 <= result["trend_score"] <= 100


def test_analysis_confidence_is_always_bounded():
    for direction in ("bullish", "bearish", "flat"):
        data = make_candles(direction)

        result = analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )

        assert 0 <= result["confidence"] <= 100


def test_analysis_adx_is_always_bounded():
    for direction in ("bullish", "bearish", "flat"):
        data = make_candles(direction)

        result = analyze_trend(
            data,
            analysis_time=get_analysis_time(data),
        )

        assert 0 <= result["adx"] <= 100


def test_analysis_reports_latest_closed_candle_time():
    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert pd.Timestamp(
        result["last_candle_close_time"]
    ) == pd.Timestamp(data["close_time"].iloc[-1])


def test_analysis_reports_exact_candle_count():
    data = make_candles(count=80)

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    assert result["candles_analyzed"] == 80


def test_analysis_does_not_generate_trading_orders():
    data = make_candles()

    result = analyze_trend(
        data,
        analysis_time=get_analysis_time(data),
    )

    forbidden_fields = {
        "order",
        "order_id",
        "execution",
        "exchange_order",
        "position_opened",
    }

    assert forbidden_fields.isdisjoint(result.keys())


def test_analysis_is_deterministic_for_identical_input():
    data = make_candles()
    analysis_time = get_analysis_time(data)

    first_result = analyze_trend(
        data.copy(deep=True),
        analysis_time=analysis_time,
    )

    second_result = analyze_trend(
        data.copy(deep=True),
        analysis_time=analysis_time,
    )

    assert first_result == second_result