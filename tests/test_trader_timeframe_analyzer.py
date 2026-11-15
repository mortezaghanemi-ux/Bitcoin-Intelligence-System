"""Tests for the Bitcoin multi-timeframe trading analyzer."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trader.timeframe.analyzer import (
    DIRECTIONS,
    REQUIRED_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_ORDER,
    MultiTimeframeAnalysisError,
    MultiTimeframeAnalyzer,
    _build_alignment,
    _build_higher_timeframe_validation,
    _calculate_momentum,
    _calculate_structure,
    _calculate_support_resistance,
    _calculate_trend,
    _calculate_volume_confirmation,
    _classify_direction,
    _normalize_analysis_time,
    _parse_utc_series,
    _validate_candles,
    _validate_numeric_columns,
    _validate_timeframe,
    analyze_multiple_timeframes,
)


UTC = timezone.utc

ANALYSIS_TIME = datetime(
    2026, 11, 15, 0, 0, tzinfo=UTC
)

MINIMUM_CANDLES = 20


def make_candles(
    timeframe="5m",
    count=30,
    *,
    direction="up",
    source="kraken",
):
    """Create deterministic, valid OHLCV candles."""

    interval = SUPPORTED_TIMEFRAMES[timeframe]

    end_time = ANALYSIS_TIME - timedelta(
        seconds=interval * 2
    )

    start_time = end_time - timedelta(
        seconds=interval * (count - 1)
    )

    rows = []

    for index in range(count):
        open_time = start_time + timedelta(
            seconds=interval * index
        )

        close_time = open_time + timedelta(
            seconds=interval
        )

        if direction == "up":
            open_price = 90000 + index * 100
            close_price = open_price + 50

        elif direction == "down":
            open_price = 100000 - index * 100
            close_price = open_price - 50

        elif direction == "flat":
            open_price = 95000
            close_price = 95000

        else:
            raise ValueError("Unknown test direction.")

        rows.append({
            "open_time": open_time,
            "close_time": close_time,
            "open": open_price,
            "high": max(open_price, close_price) + 20,
            "low": min(open_price, close_price) - 20,
            "close": close_price,
            "volume": 100 + index,
            "source": source,
        })

    return pd.DataFrame(rows)


def make_dataset(
    timeframes=TIMEFRAME_ORDER,
    *,
    direction="up",
    count=30,
):
    return {
        timeframe: make_candles(
            timeframe,
            count,
            direction=direction,
        )
        for timeframe in timeframes
    }


def make_analyzer(**kwargs):
    defaults = {
        "minimum_candles": MINIMUM_CANDLES,
    }
    defaults.update(kwargs)

    return MultiTimeframeAnalyzer(**defaults)


def test_supported_timeframes_are_correct():
    assert SUPPORTED_TIMEFRAMES == {
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }


def test_timeframe_order_is_correct():
    assert TIMEFRAME_ORDER == (
        "5m",
        "15m",
        "1h",
        "4h",
    )


def test_required_columns_are_defined():
    assert set(REQUIRED_COLUMNS) == {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    }


def test_directions_are_supported():
    assert DIRECTIONS == (
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    )


@pytest.mark.parametrize(
    "timeframe, expected",
    [
        ("5m", 300),
        ("15m", 900),
        ("1h", 3600),
        ("4h", 14400),
    ],
)
def test_validate_timeframe(timeframe, expected):
    assert _validate_timeframe(timeframe) == expected


@pytest.mark.parametrize(
    "timeframe",
    [
        "1m",
        "1d",
        "",
        "5M",
        None,
        5,
    ],
)
def test_invalid_timeframe_is_rejected(timeframe):
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_timeframe(timeframe)


def test_normalize_analysis_time():
    result = _normalize_analysis_time(ANALYSIS_TIME)

    assert result.tzinfo is not None
    assert str(result.tz) == "UTC"


def test_normalize_analysis_time_rejects_naive_datetime():
    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="timezone-aware",
    ):
        _normalize_analysis_time(
            datetime(2026, 11, 15, 0, 0)
        )


def test_normalize_analysis_time_rejects_non_utc():
    non_utc = datetime(
        2026, 11, 15, 3, 0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="UTC",
    ):
        _normalize_analysis_time(non_utc)


def test_normalize_analysis_time_rejects_missing_value():
    with pytest.raises(MultiTimeframeAnalysisError):
        _normalize_analysis_time(pd.NaT)


def test_normalize_analysis_time_accepts_utc_timestamp():
    result = _normalize_analysis_time(
        pd.Timestamp(ANALYSIS_TIME)
    )

    assert str(result.tz) == "UTC"


def test_parse_utc_series():
    values = pd.Series([
        datetime(2026, 11, 14, 10, 0, tzinfo=UTC),
        datetime(2026, 11, 14, 10, 5, tzinfo=UTC),
    ])

    result = _parse_utc_series(
        values,
        "open_time",
    )

    assert str(result.dt.tz) == "UTC"
    assert len(result) == 2


def test_parse_utc_series_rejects_invalid_values():
    values = pd.Series([
        "not-a-date",
        "also-not-a-date",
    ])

    with pytest.raises(MultiTimeframeAnalysisError):
        _parse_utc_series(values, "open_time")


def test_parse_utc_series_rejects_naive_timestamps():
    values = pd.Series([
        datetime(2026, 11, 14, 10, 0),
        datetime(2026, 11, 14, 10, 5),
    ])

    with pytest.raises(MultiTimeframeAnalysisError):
        _parse_utc_series(values, "open_time")


def test_parse_utc_series_rejects_non_utc_timezone():
    non_utc = timezone(timedelta(hours=2))

    values = pd.Series([
        datetime(2026, 11, 14, 10, 0, tzinfo=non_utc),
        datetime(2026, 11, 14, 10, 5, tzinfo=non_utc),
    ])

    with pytest.raises(MultiTimeframeAnalysisError):
        _parse_utc_series(values, "open_time")


def test_parse_utc_series_rejects_missing_timestamp():
    values = pd.Series(
        pd.to_datetime([
            "2026-11-14 10:00:00+00:00",
            None,
        ], utc=True)
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        _parse_utc_series(values, "open_time")


def test_validate_numeric_columns_accepts_valid_data():
    candles = make_candles()

    result = _validate_numeric_columns(candles)

    assert len(result) == len(candles)
    assert result["close"].dtype == float


@pytest.mark.parametrize(
    "column, value",
    [
        ("open", True),
        ("high", False),
        ("low", True),
        ("close", False),
        ("volume", True),
    ],
)
def test_numeric_columns_reject_boolean_values(
    column,
    value,
):
    candles = make_candles()

    candles[column] = candles[column].astype(object)
    candles.loc[0, column] = value

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="boolean",
    ):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize(
    "column, value",
    [
        ("open", "invalid"),
        ("high", "invalid"),
        ("low", "invalid"),
        ("close", "invalid"),
        ("volume", "invalid"),
    ],
)
def test_numeric_columns_reject_non_numeric_values(
    column,
    value,
):
    candles = make_candles()

    candles[column] = candles[column].astype(object)
    candles.loc[0, column] = value

    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize(
    "column, value",
    [
        ("open", np.nan),
        ("high", np.inf),
        ("low", -np.inf),
        ("close", np.nan),
        ("volume", np.inf),
    ],
)
def test_numeric_columns_reject_non_finite_values(
    column,
    value,
):
    candles = make_candles()
    candles.loc[0, column] = value

    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close"],
)
def test_numeric_columns_reject_non_positive_prices(column):
    candles = make_candles()
    candles.loc[0, column] = 0

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="greater than zero",
    ):
        _validate_numeric_columns(candles)


def test_numeric_columns_reject_negative_volume():
    candles = make_candles()
    candles.loc[0, "volume"] = -1

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="negative",
    ):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize(
    "column, value",
    [
        ("high", 1),
        ("low", 999999),
        ("high", 90000),
        ("low", 1),
    ],
)
def test_numeric_columns_reject_invalid_ohlc_relationships(
    column,
    value,
):
    candles = make_candles()
    candles.loc[0, column] = value

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="OHLC",
    ):
        _validate_numeric_columns(candles)


def test_validate_candles_accepts_valid_data():
    candles = make_candles()

    result = _validate_candles(
        candles,
        "5m",
        pd.Timestamp(ANALYSIS_TIME),
        MINIMUM_CANDLES,
    )

    assert len(result) == len(candles)
    assert result["open_time"].is_monotonic_increasing
    assert str(result["open_time"].dt.tz) == "UTC"


def test_validate_candles_rejects_non_dataframe():
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_candles(
            [],
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_empty_data():
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_candles(
            pd.DataFrame(),
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_missing_columns():
    candles = make_candles().drop(columns=["volume"])

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="missing columns",
    ):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_insufficient_rows():
    candles = make_candles(count=10)

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="at least",
    ):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_invalid_duration():
    candles = make_candles()
    candles.loc[0, "close_time"] += timedelta(seconds=1)

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="incorrect candle durations",
    ):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_close_before_open():
    candles = make_candles()
    candles.loc[0, "close_time"] = (
        candles.loc[0, "open_time"] - timedelta(seconds=1)
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_future_candles():
    candles = make_candles()
    candles.loc[len(candles) - 1, "close_time"] = (
        ANALYSIS_TIME + timedelta(minutes=5)
    )

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="future or unclosed",
    ):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_unsorted_open_times():
    candles = make_candles()
    candles = candles.iloc[::-1].reset_index(drop=True)

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="sorted",
    ):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_duplicate_open_times():
    candles = make_candles()
    candles.loc[1, "open_time"] = candles.loc[0, "open_time"]

    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


def test_validate_candles_rejects_duplicate_close_times():
    candles = make_candles()
    candles.loc[1, "close_time"] = candles.loc[0, "close_time"]

    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


@pytest.mark.parametrize(
    "source",
    ["", "   ", None, 123],
)
def test_validate_candles_rejects_invalid_source(source):
    candles = make_candles()
    candles.loc[0, "source"] = source

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="source",
    ):
        _validate_candles(
            candles,
            "5m",
            pd.Timestamp(ANALYSIS_TIME),
            MINIMUM_CANDLES,
        )


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.01, "BULLISH"),
        (-0.01, "BEARISH"),
        (0.00001, "NEUTRAL"),
        (-0.00001, "NEUTRAL"),
        (0.0001, "NEUTRAL"),
        (-0.0001, "NEUTRAL"),
    ],
)
def test_classify_direction(value, expected):
    assert _classify_direction(value) == expected


def test_calculate_trend_returns_expected_structure():
    candles = make_candles(direction="up")

    result = _calculate_trend(candles)

    assert result["direction"] == "BULLISH"
    assert result["ema_fast"] > 0
    assert result["ema_slow"] > 0
    assert result["price_change_pct"] > 0
    assert result["ema_alignment"] == "BULLISH"
    assert result["price_direction"] == "BULLISH"


def test_calculate_trend_detects_downtrend():
    result = _calculate_trend(
        make_candles(direction="down")
    )

    assert result["direction"] == "BEARISH"
    assert result["price_change_pct"] < 0


def test_calculate_momentum_detects_bullish():
    result = _calculate_momentum(
        make_candles(direction="up")
    )

    assert result["direction"] == "BULLISH"
    assert result["change_pct"] > 0


def test_calculate_momentum_detects_bearish():
    result = _calculate_momentum(
        make_candles(direction="down")
    )

    assert result["direction"] == "BEARISH"
    assert result["change_pct"] < 0


def test_calculate_structure_returns_expected_fields():
    result = _calculate_structure(
        make_candles(direction="up")
    )

    assert result["direction"] == "BULLISH"
    assert result["higher_high"] is True
    assert result["higher_low"] is True
    assert result["label"] == "HH_HL"


def test_calculate_support_resistance():
    result = _calculate_support_resistance(
        make_candles(direction="up")
    )

    assert result["support"] > 0
    assert result["resistance"] > result["support"]
    assert result["support_distance_pct"] >= 0
    assert result["resistance_distance_pct"] >= 0
    assert result["position"] in {
        "NEAR_SUPPORT",
        "NEAR_RESISTANCE",
        "BETWEEN_LEVELS",
    }


def test_calculate_volume_confirmation():
    candles = make_candles(direction="up")
    candles.loc[len(candles) - 1, "volume"] = 10000

    result = _calculate_volume_confirmation(candles)

    assert result["direction"] == "BULLISH"
    assert result["confirmed"] is True
    assert result["volume_ratio"] >= 1.2


def test_calculate_volume_confirmation_detects_bearish():
    candles = make_candles(direction="down")
    candles.loc[len(candles) - 1, "volume"] = 10000

    result = _calculate_volume_confirmation(candles)

    assert result["direction"] == "BEARISH"
    assert result["confirmed"] is True


def test_calculate_volume_confirmation_neutral():
    candles = make_candles(direction="flat")

    result = _calculate_volume_confirmation(candles)

    assert result["direction"] == "NEUTRAL"
    assert result["confirmed"] is False


def test_analyzer_default_configuration():
    analyzer = MultiTimeframeAnalyzer()

    assert analyzer.minimum_candles == 20
    assert analyzer.required_timeframes == TIMEFRAME_ORDER
    assert analyzer.history == []


@pytest.mark.parametrize(
    "minimum",
    [0, 1, 4, -1, True, 5.5, "20"],
)
def test_analyzer_rejects_invalid_minimum_candles(minimum):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalyzer(
            minimum_candles=minimum
        )


@pytest.mark.parametrize(
    "timeframes",
    [
        (),
        [],
        "5m",
        123,
    ],
)
def test_analyzer_rejects_invalid_required_timeframes(timeframes):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalyzer(
            required_timeframes=timeframes
        )


def test_analyzer_rejects_unsupported_required_timeframe():
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalyzer(
            required_timeframes=("5m", "1d")
        )


def test_analyzer_rejects_duplicate_required_timeframes():
    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="Duplicate",
    ):
        MultiTimeframeAnalyzer(
            required_timeframes=("5m", "5m")
        )


def test_analyzer_normalizes_required_timeframe_order():
    analyzer = MultiTimeframeAnalyzer(
        required_timeframes=("4h", "5m", "1h")
    )

    assert analyzer.required_timeframes == (
        "5m",
        "1h",
        "4h",
    )


def test_analyzer_accepts_list_of_required_timeframes():
    analyzer = MultiTimeframeAnalyzer(
        required_timeframes=["15m", "5m"]
    )

    assert analyzer.required_timeframes == (
        "5m",
        "15m",
    )


def test_analyze_all_timeframes_successfully():
    analyzer = make_analyzer()

    result = analyzer.analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert result["overall_direction"] == "BULLISH"
    assert result["timeframes_analyzed"] == list(
        TIMEFRAME_ORDER
    )
    assert set(result["timeframes"]) == set(
        TIMEFRAME_ORDER
    )
    assert len(result["analysis_id"]) == 64


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAME_ORDER,
)
def test_analyzer_supports_each_timeframe(timeframe):
    analyzer = make_analyzer(
        required_timeframes=(timeframe,)
    )

    result = analyzer.analyze(
        {
            timeframe: make_candles(timeframe)
        },
        analysis_time=ANALYSIS_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert result["timeframes_analyzed"] == [timeframe]


def test_analyzer_accepts_optional_timeframe_subset():
    analyzer = make_analyzer(
        required_timeframes=("5m", "1h")
    )

    result = analyzer.analyze(
        make_dataset(("5m", "1h")),
        analysis_time=ANALYSIS_TIME,
    )

    assert result["timeframes_analyzed"] == [
        "5m",
        "1h",
    ]


def test_analyzer_rejects_non_dictionary_input():
    analyzer = make_analyzer()

    with pytest.raises(MultiTimeframeAnalysisError):
        analyzer.analyze(
            [],
            analysis_time=ANALYSIS_TIME,
        )


def test_analyzer_rejects_empty_input():
    analyzer = make_analyzer()

    with pytest.raises(MultiTimeframeAnalysisError):
        analyzer.analyze(
            {},
            analysis_time=ANALYSIS_TIME,
        )


def test_analyzer_rejects_unknown_timeframe():
    analyzer = make_analyzer()

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="Unsupported timeframes",
    ):
        analyzer.analyze(
            {
                "5m": make_candles(),
                "1d": make_candles(),
            },
            analysis_time=ANALYSIS_TIME,
        )


def test_analyzer_rejects_missing_required_timeframes():
    analyzer = make_analyzer()

    with pytest.raises(
        MultiTimeframeAnalysisError,
        match="Missing required timeframes",
    ):
        analyzer.analyze(
            {
                "5m": make_candles(),
            },
            analysis_time=ANALYSIS_TIME,
        )


def test_analyzer_rejects_invalid_analysis_time():
    analyzer = make_analyzer()

    with pytest.raises(MultiTimeframeAnalysisError):
        analyzer.analyze(
            make_dataset(),
            analysis_time=datetime(2026, 11, 15),
        )


def test_analyzer_output_contains_required_sections():
    result = make_analyzer().analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    expected = {
        "analysis_id",
        "analysis_timestamp",
        "status",
        "overall_direction",
        "alignment",
        "higher_timeframe_validation",
        "timeframes",
        "source",
        "timeframes_analyzed",
        "parameters",
        "limitations",
    }

    assert expected.issubset(result.keys())


def test_timeframe_output_contains_all_components():
    result = make_analyzer().analyze(
        make_dataset(("5m",)),
        analysis_time=ANALYSIS_TIME,
    )

    timeframe_result = result["timeframes"]["5m"]

    expected = {
        "timeframe",
        "direction",
        "trend",
        "momentum",
        "structure",
        "support_resistance",
        "volume",
        "component_directions",
        "component_agreement",
        "current_price",
        "candles_analyzed",
        "last_candle_close_time",
        "source",
    }

    assert expected.issubset(timeframe_result.keys())


def test_component_directions_are_valid():
    result = make_analyzer().analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    for timeframe in result["timeframes"].values():
        for direction in timeframe[
            "component_directions"
        ].values():
            assert direction in DIRECTIONS


def test_alignment_detects_full_bullish_alignment():
    analyses = {
        timeframe: {"direction": "BULLISH"}
        for timeframe in TIMEFRAME_ORDER
    }

    result = _build_alignment(analyses)

    assert result["direction"] == "BULLISH"
    assert result["status"] == "FULL_ALIGNMENT"
    assert result["alignment_ratio"] == 1.0
    assert result["conflict_count"] == 0


def test_alignment_detects_full_bearish_alignment():
    analyses = {
        timeframe: {"direction": "BEARISH"}
        for timeframe in TIMEFRAME_ORDER
    }

    result = _build_alignment(analyses)

    assert result["direction"] == "BEARISH"
    assert result["status"] == "FULL_ALIGNMENT"


def test_alignment_detects_neutral_alignment():
    analyses = {
        timeframe: {"direction": "NEUTRAL"}
        for timeframe in TIMEFRAME_ORDER
    }

    result = _build_alignment(analyses)

    assert result["direction"] == "NEUTRAL"
    assert result["status"] == "FULL_ALIGNMENT"


def test_alignment_detects_directional_conflicts():
    analyses = {
        "5m": {"direction": "BULLISH"},
        "15m": {"direction": "BEARISH"},
        "1h": {"direction": "BULLISH"},
        "4h": {"direction": "BULLISH"},
    }

    result = _build_alignment(analyses)

    assert result["status"] == "CONFLICTING"
    assert result["conflict_count"] == 2
    assert result["alignment_ratio"] == 0.75


def test_alignment_ignores_neutral_directions_for_conflicts():
    analyses = {
        "5m": {"direction": "NEUTRAL"},
        "1h": {"direction": "BULLISH"},
    }

    result = _build_alignment(analyses)

    assert result["conflict_count"] == 0
    assert result["direction"] == "BULLISH"


def test_alignment_detects_weak_alignment():
    analyses = {
        "5m": {"direction": "BULLISH"},
        "15m": {"direction": "BEARISH"},
        "1h": {"direction": "BEARISH"},
        "4h": {"direction": "NEUTRAL"},
    }

    result = _build_alignment(analyses)

    assert result["status"] == "CONFLICTING"
    assert result["conflict_count"] == 2


def test_higher_timeframe_validation_confirms_directions():
    analyses = {
        "5m": {"direction": "BULLISH"},
        "15m": {"direction": "BULLISH"},
        "1h": {"direction": "BULLISH"},
        "4h": {"direction": "BULLISH"},
    }

    result = _build_higher_timeframe_validation(analyses)

    assert result["status"] == "CONFIRMED"
    assert result["confirmed_count"] == 6
    assert result["conflict_count"] == 0


def test_higher_timeframe_validation_detects_conflicts():
    analyses = {
        "5m": {"direction": "BULLISH"},
        "15m": {"direction": "BEARISH"},
        "1h": {"direction": "BULLISH"},
        "4h": {"direction": "BEARISH"},
    }

    result = _build_higher_timeframe_validation(analyses)

    assert result["status"] == "CONFLICTING"
    assert result["conflict_count"] > 0


def test_higher_timeframe_validation_handles_neutral():
    analyses = {
        "5m": {"direction": "NEUTRAL"},
        "1h": {"direction": "BULLISH"},
    }

    result = _build_higher_timeframe_validation(analyses)

    assert result["inconclusive_count"] == 1
    assert result["status"] == "CONFIRMED"


def test_higher_timeframe_validation_all_neutral():
    analyses = {
        "5m": {"direction": "NEUTRAL"},
        "15m": {"direction": "NEUTRAL"},
    }

    result = _build_higher_timeframe_validation(analyses)

    assert result["status"] == "INCONCLUSIVE"
    assert result["inconclusive_count"] == 1


def test_analyzer_records_history():
    analyzer = make_analyzer()

    analyzer.analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    assert len(analyzer.history) == 1


def test_analyzer_does_not_duplicate_history():
    analyzer = make_analyzer()
    dataset = make_dataset()

    first = analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    second = analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    assert first["analysis_id"] == second["analysis_id"]
    assert len(analyzer.history) == 1


def test_analyzer_records_distinct_analysis_times():
    analyzer = make_analyzer()
    dataset = make_dataset()

    analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    later_time = ANALYSIS_TIME + timedelta(minutes=1)

    analyzer.analyze(
        dataset,
        analysis_time=later_time,
    )

    assert len(analyzer.history) == 2


def test_history_returns_defensive_copy():
    analyzer = make_analyzer()

    analyzer.analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    history = analyzer.history
    history.clear()

    assert len(analyzer.history) == 1


def test_result_is_independent_from_history():
    analyzer = make_analyzer()

    result = analyzer.analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    result["timeframes"].clear()

    assert len(analyzer.history[0]["timeframes"]) == 4


def test_clear_history_removes_all_records():
    analyzer = make_analyzer()

    analyzer.analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    analyzer.clear_history()

    assert analyzer.history == []


def test_clear_history_allows_recording_again():
    analyzer = make_analyzer()
    dataset = make_dataset()

    analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    analyzer.clear_history()

    analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    assert len(analyzer.history) == 1


def test_analyzer_does_not_modify_input_data():
    analyzer = make_analyzer()
    dataset = make_dataset()

    original = {
        key: value.copy(deep=True)
        for key, value in dataset.items()
    }

    analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    for timeframe in dataset:
        pd.testing.assert_frame_equal(
            dataset[timeframe],
            original[timeframe],
        )


def test_analyzer_aggregates_sources():
    analyzer = make_analyzer(
        required_timeframes=("5m", "1h")
    )

    dataset = {
        "5m": make_candles("5m", source="kraken"),
        "1h": make_candles("1h", source="coinbase"),
    }

    result = analyzer.analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    assert result["source"] == [
        "coinbase",
        "kraken",
    ]


def test_analyzer_reports_candle_count():
    result = make_analyzer().analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    for timeframe in TIMEFRAME_ORDER:
        assert (
            result["timeframes"][timeframe]["candles_analyzed"]
            == 30
        )


def test_analyzer_reports_analysis_parameters():
    result = make_analyzer(
        minimum_candles=25,
        required_timeframes=("5m", "1h"),
    ).analyze(
        make_dataset(("5m", "1h")),
        analysis_time=ANALYSIS_TIME,
    )

    assert result["parameters"]["minimum_candles"] == 25
    assert result["parameters"]["required_timeframes"] == [
        "5m",
        "1h",
    ]


def test_analyzer_limitations_are_present():
    result = make_analyzer().analyze(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    assert len(result["limitations"]) >= 5
    assert any(
        "does not execute trades" in item
        for item in result["limitations"]
    )


def test_convenience_function():
    result = analyze_multiple_timeframes(
        make_dataset(),
        analysis_time=ANALYSIS_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert len(result["timeframes"]) == 4


def test_convenience_function_accepts_custom_configuration():
    result = analyze_multiple_timeframes(
        make_dataset(("5m", "1h")),
        analysis_time=ANALYSIS_TIME,
        minimum_candles=25,
        required_timeframes=("5m", "1h"),
    )

    assert result["timeframes_analyzed"] == [
        "5m",
        "1h",
    ]
    assert result["parameters"]["minimum_candles"] == 25


def test_analysis_id_is_deterministic():
    dataset = make_dataset()

    first = make_analyzer().analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    second = make_analyzer().analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    assert first["analysis_id"] == second["analysis_id"]


def test_analysis_id_changes_when_analysis_time_changes():
    dataset = make_dataset()

    first = make_analyzer().analyze(
        dataset,
        analysis_time=ANALYSIS_TIME,
    )

    second = make_analyzer().analyze(
        dataset,
        analysis_time=ANALYSIS_TIME + timedelta(minutes=1),
    )

    assert first["analysis_id"] != second["analysis_id"]