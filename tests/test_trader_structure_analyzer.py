"""Tests for Bitcoin trader market structure analyzer."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.structure.analyzer import (
    FIBONACCI_LEVELS,
    REQUIRED_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    MarketStructureAnalyzer,
    MarketStructureAnalyzerError,
    _build_analysis_id,
    _build_liquidity_zones,
    _calculate_fibonacci,
    _detect_liquidity_sweeps,
    _validate_parameters,
    _validate_timeframe,
    analyze_market_structure,
)


UTC = timezone.utc

START_TIME = datetime(
    2026, 11, 14, 0, 0, tzinfo=UTC
)


def make_candles(
    count=40,
    timeframe="5m",
    base_price=100.0,
):
    """Create deterministic valid OHLCV candles."""

    interval = SUPPORTED_TIMEFRAMES[timeframe]
    rows = []

    for index in range(count):
        open_time = START_TIME + timedelta(
            seconds=interval * index
        )
        close_time = open_time + timedelta(
            seconds=interval
        )

        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": base_price,
                "high": base_price + 1,
                "low": base_price - 1,
                "close": base_price,
                "volume": 10.0,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def analysis_time_for(data):
    """Return a UTC time after the last candle closes."""

    return (
        data["close_time"].iloc[-1]
        + timedelta(minutes=1)
    )


# =========================================================
# Configuration and parameters
# =========================================================


def test_supported_timeframes():
    assert SUPPORTED_TIMEFRAMES == {
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }


def test_required_columns():
    assert REQUIRED_COLUMNS == (
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    )


@pytest.mark.parametrize(
    "timeframe",
    ["5m", "15m", "1h", "4h"],
)
def test_validate_supported_timeframes(timeframe):
    assert _validate_timeframe(timeframe) == (
        SUPPORTED_TIMEFRAMES[timeframe]
    )


@pytest.mark.parametrize(
    "timeframe",
    ["", "1m", "30m", "1d", None, 5],
)
def test_reject_unsupported_timeframes(timeframe):
    with pytest.raises(MarketStructureAnalyzerError):
        _validate_timeframe(timeframe)


@pytest.mark.parametrize(
    "swing_window,minimum_candles,tolerance",
    [
        (0, 30, 0.001),
        (-1, 30, 0.001),
        (True, 30, 0.001),
        (3, 0, 0.001),
        (3, -1, 0.001),
        (3, True, 0.001),
        (5, 10, 0.001),
        (3, 30, -0.001),
        (3, 30, True),
        (3, 30, "0.001"),
        (3, 30, None),
        (3, 30, float("nan")),
        (3, 30, float("inf")),
        (3, 30, float("-inf")),
    ],
)
def test_invalid_parameters_are_rejected(
    swing_window,
    minimum_candles,
    tolerance,
):
    with pytest.raises(MarketStructureAnalyzerError):
        _validate_parameters(
            swing_window,
            minimum_candles,
            tolerance,
        )


@pytest.mark.parametrize(
    "tolerance",
    [0, 0.001, 0.01, 1, 2.5],
)
def test_valid_liquidity_tolerance_is_accepted(tolerance):
    _validate_parameters(
        swing_window=3,
        minimum_candles=30,
        liquidity_tolerance=tolerance,
    )


def test_valid_parameters_are_accepted():
    _validate_parameters(
        swing_window=3,
        minimum_candles=30,
        liquidity_tolerance=0.001,
    )


# =========================================================
# Constructor
# =========================================================


def test_analyzer_initialization():
    analyzer = MarketStructureAnalyzer("5m")

    assert analyzer.timeframe == "5m"
    assert analyzer.swing_window == 3
    assert analyzer.minimum_candles == 30
    assert analyzer.liquidity_tolerance == 0.001
    assert analyzer.history == []


def test_constructor_accepts_custom_parameters():
    analyzer = MarketStructureAnalyzer(
        timeframe="1h",
        swing_window=4,
        minimum_candles=40,
        liquidity_tolerance=0.002,
    )

    assert analyzer.timeframe == "1h"
    assert analyzer.swing_window == 4
    assert analyzer.minimum_candles == 40
    assert analyzer.liquidity_tolerance == 0.002


@pytest.mark.parametrize(
    "tolerance",
    [float("nan"), float("inf"), float("-inf")],
)
def test_constructor_rejects_non_finite_tolerance(tolerance):
    with pytest.raises(MarketStructureAnalyzerError):
        MarketStructureAnalyzer(
            timeframe="5m",
            liquidity_tolerance=tolerance,
        )


def test_constructor_rejects_invalid_timeframe():
    with pytest.raises(MarketStructureAnalyzerError):
        MarketStructureAnalyzer("1m")


# =========================================================
# Input validation
# =========================================================


def test_analyze_requires_dataframe():
    analyzer = MarketStructureAnalyzer("5m")

    with pytest.raises(TypeError):
        analyzer.analyze(
            [],
            analysis_time=START_TIME,
        )


def test_analyze_rejects_empty_dataframe():
    analyzer = MarketStructureAnalyzer("5m")

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            pd.DataFrame(),
            analysis_time=START_TIME,
        )


def test_analyze_rejects_missing_columns():
    analyzer = MarketStructureAnalyzer("5m")
    data = pd.DataFrame(
        {
            "open": [100.0],
            "close": [100.0],
        }
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=START_TIME,
        )


def test_analyze_rejects_naive_analysis_time():
    analyzer = MarketStructureAnalyzer("5m")

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            make_candles(),
            analysis_time=datetime(2026, 11, 14),
        )


def test_analyze_rejects_non_utc_analysis_time():
    analyzer = MarketStructureAnalyzer("5m")
    non_utc = datetime(
        2026,
        11,
        14,
        12,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            make_candles(),
            analysis_time=non_utc,
        )


def test_analyze_rejects_naive_candle_timestamps():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data["open_time"] = (
        data["open_time"].dt.tz_localize(None)
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_non_utc_candle_timestamps():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data["open_time"] = (
        data["open_time"].dt.tz_convert("Europe/Berlin")
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_candle_closing_before_opening():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data.loc[5, "close_time"] = (
        data.loc[5, "open_time"] - timedelta(minutes=1)
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_wrong_candle_duration():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data.loc[5, "close_time"] += timedelta(minutes=1)

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_future_candles():
    analyzer = MarketStructureAnalyzer("5m")

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            make_candles(),
            analysis_time=START_TIME,
        )


def test_analyze_rejects_unsorted_candles():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data = pd.concat(
        [
            data.iloc[:10],
            data.iloc[11:],
            data.iloc[10:11],
        ],
        ignore_index=True,
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_duplicate_open_times():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data.loc[10, "open_time"] = data.loc[9, "open_time"]

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_duplicate_close_times():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data.loc[10, "close_time"] = data.loc[9, "close_time"]

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(make_candles()),
        )


def test_analyze_rejects_insufficient_candles():
    analyzer = MarketStructureAnalyzer(
        timeframe="5m",
        minimum_candles=30,
    )
    data = make_candles(20)

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(data),
        )


# =========================================================
# Numeric OHLCV validation
# =========================================================


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("open", float("nan")),
        ("open", float("inf")),
        ("open", float("-inf")),
        ("open", "invalid"),
        ("open", None),
        ("open", True),
        ("high", float("nan")),
        ("high", float("inf")),
        ("high", "invalid"),
        ("low", float("nan")),
        ("low", float("-inf")),
        ("low", "invalid"),
        ("close", float("nan")),
        ("close", float("inf")),
        ("close", "invalid"),
        ("volume", float("nan")),
        ("volume", float("inf")),
        ("volume", float("-inf")),
        ("volume", "invalid"),
        ("volume", None),
        ("volume", True),
    ],
)
def test_analyze_rejects_non_finite_or_non_numeric_ohlcv(
    column,
    bad_value,
):
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    # Preserve bool values instead of allowing Pandas
    # to coerce True into a numeric value such as 1.0.
    if isinstance(bad_value, bool):
        data[column] = data[column].astype(object)

    data.loc[10, column] = bad_value

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(data),
        )


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("open", 0),
        ("open", -1),
        ("high", 0),
        ("low", -1),
        ("close", 0),
        ("close", -1),
        ("volume", -1),
    ],
)
def test_analyze_rejects_invalid_ohlcv_values(
    column,
    bad_value,
):
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data.loc[10, column] = bad_value

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(data),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"open": 100, "high": 99, "low": 98, "close": 100},
        {"open": 100, "high": 101, "low": 102, "close": 100},
        {"open": 100, "high": 101, "low": 99, "close": 102},
        {"open": 100, "high": 101, "low": 99, "close": 98},
    ],
)
def test_analyze_rejects_inconsistent_ohlc(values):
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    for column, value in values.items():
        data.loc[10, column] = value

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(data),
        )


# =========================================================
# Timeframe validation
# =========================================================


@pytest.mark.parametrize(
    "timeframe",
    ["5m", "15m", "1h", "4h"],
)
def test_analyze_accepts_matching_timeframe_duration(timeframe):
    analyzer = MarketStructureAnalyzer(timeframe)
    data = make_candles(
        count=40,
        timeframe=timeframe,
    )

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["timeframe"] == timeframe
    assert result["candles_analyzed"] == 40


@pytest.mark.parametrize(
    "analyzer_timeframe,data_timeframe",
    [
        ("5m", "15m"),
        ("15m", "5m"),
        ("1h", "4h"),
        ("4h", "1h"),
    ],
)
def test_analyze_rejects_mismatched_timeframe_duration(
    analyzer_timeframe,
    data_timeframe,
):
    analyzer = MarketStructureAnalyzer(analyzer_timeframe)
    data = make_candles(
        count=40,
        timeframe=data_timeframe,
    )

    with pytest.raises(MarketStructureAnalyzerError):
        analyzer.analyze(
            data,
            analysis_time=analysis_time_for(data),
        )


# =========================================================
# Successful analysis
# =========================================================


def test_analyze_returns_expected_schema():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    expected_keys = {
        "analysis_id",
        "analysis_timestamp",
        "status",
        "timeframe",
        "source",
        "trend",
        "candles_analyzed",
        "last_candle_close_time",
        "swing_highs",
        "swing_lows",
        "swings",
        "structure_breaks",
        "developing_structures",
        "liquidity_zones",
        "liquidity_sweeps",
        "fibonacci",
        "levels",
        "summary",
        "parameters",
        "limitations",
    }

    assert set(result.keys()) == expected_keys
    assert result["status"] == "SUCCESS"
    assert result["timeframe"] == "5m"
    assert result["source"] == ["kraken"]
    assert result["candles_analyzed"] == 40


def test_analyze_uses_utc_timestamps():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    assert result["analysis_timestamp"].endswith("+00:00")
    assert result["last_candle_close_time"].endswith("+00:00")


def test_analyze_is_deterministic():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()
    timestamp = analysis_time_for(data)

    first = analyzer.analyze(data, analysis_time=timestamp)
    second = analyzer.analyze(data, analysis_time=timestamp)

    assert first == second


def test_flat_market_has_no_confirmed_swings():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    assert result["swing_highs"] == []
    assert result["swing_lows"] == []
    assert result["swings"] == []
    assert result["structure_breaks"] == []
    assert result["liquidity_zones"] == []
    assert result["liquidity_sweeps"] == []
    assert result["fibonacci"] is None


def test_analyze_does_not_modify_input():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()
    original = data.copy(deep=True)

    analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    pd.testing.assert_frame_equal(data, original)


def test_analyze_preserves_source_information():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    data.loc[20:, "source"] = "coingecko"

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    assert result["source"] == ["coingecko", "kraken"]


# =========================================================
# Liquidity zones
# =========================================================


def test_build_liquidity_zones_from_equal_highs():
    structure = {
        "swings": [
            {
                "type": "HIGH",
                "classification": "EQH",
                "price": 100.0,
                "time": "2026-11-14T01:00:00+00:00",
                "confirmation_time": "2026-11-14T01:10:00+00:00",
            }
        ]
    }

    zones = _build_liquidity_zones(
        structure,
        tolerance=0.001,
    )

    assert len(zones) == 1
    assert zones[0]["type"] == "BUY_SIDE_LIQUIDITY"
    assert zones[0]["price"] == 100.0
    assert zones[0]["lower_bound"] == pytest.approx(99.9)
    assert zones[0]["upper_bound"] == pytest.approx(100.1)


def test_build_liquidity_zones_from_equal_lows():
    structure = {
        "swings": [
            {
                "type": "LOW",
                "classification": "EQL",
                "price": 90.0,
                "time": "2026-11-14T01:00:00+00:00",
                "confirmation_time": "2026-11-14T01:10:00+00:00",
            }
        ]
    }

    zones = _build_liquidity_zones(
        structure,
        tolerance=0.001,
    )

    assert len(zones) == 1
    assert zones[0]["type"] == "SELL_SIDE_LIQUIDITY"


def test_build_liquidity_zones_ignores_non_equal_swings():
    structure = {
        "swings": [
            {
                "type": "HIGH",
                "classification": "HH",
                "price": 110.0,
                "time": "2026-11-14T01:00:00+00:00",
                "confirmation_time": "2026-11-14T01:10:00+00:00",
            },
            {
                "type": "LOW",
                "classification": "HL",
                "price": 95.0,
                "time": "2026-11-14T02:00:00+00:00",
                "confirmation_time": "2026-11-14T02:10:00+00:00",
            },
        ]
    }

    assert _build_liquidity_zones(
        structure,
        tolerance=0.001,
    ) == []


def test_build_liquidity_zones_handles_empty_swings():
    assert _build_liquidity_zones(
        {"swings": []},
        tolerance=0.001,
    ) == []


# =========================================================
# Liquidity sweeps
# =========================================================


def test_detect_buy_side_liquidity_sweep():
    data = make_candles(5)
    data.loc[2, "high"] = 105.0
    data.loc[2, "close"] = 99.0

    structure = {
        "swings": [
            {
                "type": "HIGH",
                "price": 100.0,
                "confirmation_index": 1,
                "time": data.loc[0, "close_time"].isoformat(),
                "confirmation_time": data.loc[1, "close_time"].isoformat(),
            }
        ]
    }

    sweeps = _detect_liquidity_sweeps(data, structure)

    assert len(sweeps) == 1
    assert sweeps[0]["type"] == "BUY_SIDE_SWEEP"
    assert sweeps[0]["direction"] == "BEARISH"
    assert sweeps[0]["level"] == 100.0
    assert sweeps[0]["status"] == "CONFIRMED"


def test_detect_sell_side_liquidity_sweep():
    data = make_candles(5)
    data.loc[2, "low"] = 95.0
    data.loc[2, "close"] = 101.0

    structure = {
        "swings": [
            {
                "type": "LOW",
                "price": 100.0,
                "confirmation_index": 1,
                "time": data.loc[0, "close_time"].isoformat(),
                "confirmation_time": data.loc[1, "close_time"].isoformat(),
            }
        ]
    }

    sweeps = _detect_liquidity_sweeps(data, structure)

    assert len(sweeps) == 1
    assert sweeps[0]["type"] == "SELL_SIDE_SWEEP"
    assert sweeps[0]["direction"] == "BULLISH"


def test_liquidity_sweep_requires_close_back_inside_level():
    data = make_candles(5)
    data.loc[2, "high"] = 105.0
    data.loc[2, "close"] = 102.0

    structure = {
        "swings": [
            {
                "type": "HIGH",
                "price": 100.0,
                "confirmation_index": 1,
                "time": data.loc[0, "close_time"].isoformat(),
                "confirmation_time": data.loc[1, "close_time"].isoformat(),
            }
        ]
    }

    assert _detect_liquidity_sweeps(data, structure) == []


def test_liquidity_sweep_ignores_unconfirmed_swing():
    data = make_candles(5)

    structure = {
        "swings": [
            {
                "type": "HIGH",
                "price": 100.0,
                "confirmation_index": 10,
                "time": "2026-11-14T00:00:00+00:00",
                "confirmation_time": "2026-11-14T00:05:00+00:00",
            }
        ]
    }

    assert _detect_liquidity_sweeps(data, structure) == []


def test_liquidity_sweep_ignores_invalid_confirmation_index():
    data = make_candles(5)

    structure = {
        "swings": [
            {
                "type": "HIGH",
                "price": 100.0,
                "confirmation_index": True,
                "time": "2026-11-14T00:00:00+00:00",
                "confirmation_time": "2026-11-14T00:05:00+00:00",
            }
        ]
    }

    assert _detect_liquidity_sweeps(data, structure) == []


def test_only_first_sweep_per_swing_is_recorded():
    data = make_candles(6)
    data.loc[2, "high"] = 105.0
    data.loc[2, "close"] = 99.0
    data.loc[4, "high"] = 106.0
    data.loc[4, "close"] = 98.0

    structure = {
        "swings": [
            {
                "type": "HIGH",
                "price": 100.0,
                "confirmation_index": 1,
                "time": data.loc[0, "close_time"].isoformat(),
                "confirmation_time": data.loc[1, "close_time"].isoformat(),
            }
        ]
    }

    assert len(_detect_liquidity_sweeps(data, structure)) == 1


# =========================================================
# Fibonacci
# =========================================================


def test_calculate_bullish_fibonacci():
    structure = {
        "swings": [
            {
                "type": "LOW",
                "price": 100.0,
                "time": "2026-11-14T01:00:00+00:00",
            },
            {
                "type": "HIGH",
                "price": 200.0,
                "time": "2026-11-14T02:00:00+00:00",
            },
        ]
    }

    result = _calculate_fibonacci(structure)

    assert result is not None
    assert result["direction"] == "BULLISH"
    assert result["swing_high"] == 200.0
    assert result["swing_low"] == 100.0
    assert result["range"] == 100.0
    assert result["levels"]["0.000"] == 200.0
    assert result["levels"]["0.500"] == 150.0
    assert result["levels"]["1.000"] == 100.0


def test_calculate_bearish_fibonacci():
    structure = {
        "swings": [
            {
                "type": "HIGH",
                "price": 200.0,
                "time": "2026-11-14T01:00:00+00:00",
            },
            {
                "type": "LOW",
                "price": 100.0,
                "time": "2026-11-14T02:00:00+00:00",
            },
        ]
    }

    result = _calculate_fibonacci(structure)

    assert result is not None
    assert result["direction"] == "BEARISH"
    assert result["levels"]["0.000"] == 100.0
    assert result["levels"]["0.500"] == 150.0
    assert result["levels"]["1.000"] == 200.0


@pytest.mark.parametrize(
    "swings",
    [
        [
            {
                "type": "LOW",
                "price": 100.0,
                "time": "2026-11-14T01:00:00+00:00",
            }
        ],
        [
            {
                "type": "HIGH",
                "price": 200.0,
                "time": "2026-11-14T01:00:00+00:00",
            }
        ],
        [
            {
                "type": "LOW",
                "price": 200.0,
                "time": "2026-11-14T01:00:00+00:00",
            },
            {
                "type": "HIGH",
                "price": 100.0,
                "time": "2026-11-14T02:00:00+00:00",
            },
        ],
    ],
)
def test_fibonacci_returns_none_for_unusable_swings(swings):
    assert _calculate_fibonacci({"swings": swings}) is None


def test_fibonacci_levels_match_configured_ratios():
    structure = {
        "swings": [
            {
                "type": "LOW",
                "price": 100.0,
                "time": "2026-11-14T01:00:00+00:00",
            },
            {
                "type": "HIGH",
                "price": 200.0,
                "time": "2026-11-14T02:00:00+00:00",
            },
        ]
    }

    result = _calculate_fibonacci(structure)

    assert set(result["levels"]) == {
        f"{level:.3f}"
        for level in FIBONACCI_LEVELS
    }


# =========================================================
# Analysis identifiers
# =========================================================


def test_analysis_id_is_deterministic():
    first = _build_analysis_id(
        "5m",
        "2026-11-14T04:00:00+00:00",
        "2026-11-14T03:20:00+00:00",
        ["kraken"],
    )
    second = _build_analysis_id(
        "5m",
        "2026-11-14T04:00:00+00:00",
        "2026-11-14T03:20:00+00:00",
        ["kraken"],
    )

    assert first == second


def test_analysis_id_changes_with_timeframe():
    first = _build_analysis_id(
        "5m",
        "2026-11-14T04:00:00+00:00",
        "2026-11-14T03:20:00+00:00",
        ["kraken"],
    )
    second = _build_analysis_id(
        "15m",
        "2026-11-14T04:00:00+00:00",
        "2026-11-14T03:20:00+00:00",
        ["kraken"],
    )

    assert first != second


def test_analysis_id_ignores_source_order():
    first = _build_analysis_id(
        "5m",
        "2026-11-14T04:00:00+00:00",
        "2026-11-14T03:20:00+00:00",
        ["kraken", "coingecko"],
    )
    second = _build_analysis_id(
        "5m",
        "2026-11-14T04:00:00+00:00",
        "2026-11-14T03:20:00+00:00",
        ["coingecko", "kraken"],
    )

    assert first == second


# =========================================================
# History management
# =========================================================


def test_analysis_is_added_to_history():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    assert len(analyzer.history) == 1
    assert analyzer.history[0]["analysis_id"] == result["analysis_id"]


def test_duplicate_analysis_is_not_added_twice():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()
    timestamp = analysis_time_for(data)

    first = analyzer.analyze(data, analysis_time=timestamp)
    second = analyzer.analyze(data, analysis_time=timestamp)

    assert first == second
    assert len(analyzer.history) == 1


def test_different_analysis_timestamp_creates_new_history_entry():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()
    timestamp = analysis_time_for(data)

    analyzer.analyze(data, analysis_time=timestamp)
    analyzer.analyze(
        data,
        analysis_time=timestamp + timedelta(minutes=1),
    )

    assert len(analyzer.history) == 2


def test_history_returns_defensive_copy():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    history = analyzer.history
    history[0]["trend"] = "MODIFIED"
    history[0]["summary"]["modified"] = True

    stored = analyzer.history[0]

    assert stored["trend"] != "MODIFIED"
    assert "modified" not in stored["summary"]


def test_clear_removes_history():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    assert len(analyzer.history) == 1

    analyzer.clear()

    assert analyzer.history == []
    assert analyzer._known_ids == set()


def test_analysis_can_be_added_after_clear():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()
    timestamp = analysis_time_for(data)

    analyzer.analyze(data, analysis_time=timestamp)
    analyzer.clear()
    analyzer.analyze(data, analysis_time=timestamp)

    assert len(analyzer.history) == 1


# =========================================================
# Convenience function
# =========================================================


def test_convenience_function_returns_success():
    data = make_candles()

    result = analyze_market_structure(
        data=data,
        timeframe="5m",
        analysis_time=analysis_time_for(data),
    )

    assert result["status"] == "SUCCESS"
    assert result["timeframe"] == "5m"


def test_convenience_function_accepts_custom_parameters():
    data = make_candles(40)

    result = analyze_market_structure(
        data=data,
        timeframe="5m",
        analysis_time=analysis_time_for(data),
        swing_window=2,
        minimum_candles=30,
        liquidity_tolerance=0.002,
    )

    assert result["parameters"]["swing_window"] == 2
    assert result["parameters"]["minimum_candles"] == 30
    assert result["parameters"]["liquidity_tolerance"] == 0.002


def test_convenience_function_rejects_invalid_timeframe():
    data = make_candles()

    with pytest.raises(MarketStructureAnalyzerError):
        analyze_market_structure(
            data=data,
            timeframe="1m",
            analysis_time=analysis_time_for(data),
        )


# =========================================================
# Output limitations
# =========================================================


def test_analysis_documents_safety_limitations():
    analyzer = MarketStructureAnalyzer("5m")
    data = make_candles()

    result = analyzer.analyze(
        data,
        analysis_time=analysis_time_for(data),
    )

    limitations = result["limitations"]

    assert any(
        "closed candles" in item.lower()
        for item in limitations
    )
    assert any(
        "trading orders" in item.lower()
        for item in limitations
    )
    assert any(
        "execute trades" in item.lower()
        for item in limitations
    )