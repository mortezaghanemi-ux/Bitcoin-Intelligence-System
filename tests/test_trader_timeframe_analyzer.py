"""Comprehensive tests for the multi-timeframe analyzer."""

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
ANALYSIS_TIME = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
MINIMUM_CANDLES = 20


def make_candles(timeframe="5m", count=30, direction="up", source="kraken"):
    interval = SUPPORTED_TIMEFRAMES[timeframe]
    end_time = ANALYSIS_TIME - timedelta(seconds=interval * 2)
    start_time = end_time - timedelta(seconds=interval * (count - 1))
    rows = []

    for index in range(count):
        open_time = start_time + timedelta(seconds=interval * index)
        close_time = open_time + timedelta(seconds=interval)

        if direction == "up":
            open_price = 90000 + index * 100
            close_price = open_price + 50
        elif direction == "down":
            open_price = 100000 - index * 100
            close_price = open_price - 50
        else:
            open_price = 95000
            close_price = 95000

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


def make_dataset(timeframes=TIMEFRAME_ORDER, direction="up"):
    return {
        timeframe: make_candles(timeframe, direction=direction)
        for timeframe in timeframes
    }


def make_analyzer(**kwargs):
    kwargs.setdefault("minimum_candles", MINIMUM_CANDLES)
    return MultiTimeframeAnalyzer(**kwargs)


def validate_direct(candles, timeframe="5m"):
    return _validate_candles(
        candles,
        timeframe,
        pd.Timestamp(ANALYSIS_TIME),
        MINIMUM_CANDLES,
    )


def test_constants():
    assert SUPPORTED_TIMEFRAMES == {
        "5m": 300, "15m": 900, "1h": 3600, "4h": 14400
    }
    assert TIMEFRAME_ORDER == ("5m", "15m", "1h", "4h")
    assert set(REQUIRED_COLUMNS) == {
        "open_time", "close_time", "open", "high",
        "low", "close", "volume", "source",
    }
    assert DIRECTIONS == ("BULLISH", "BEARISH", "NEUTRAL")


@pytest.mark.parametrize(
    "timeframe,seconds",
    [("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400)],
)
def test_valid_timeframes(timeframe, seconds):
    assert _validate_timeframe(timeframe) == seconds


@pytest.mark.parametrize("timeframe", ["1m", "1d", "", "5M", None, 5])
def test_invalid_timeframes(timeframe):
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_timeframe(timeframe)


def test_normalize_utc_analysis_time():
    assert str(_normalize_analysis_time(ANALYSIS_TIME).tz) == "UTC"


def test_normalize_accepts_pandas_timestamp():
    assert str(_normalize_analysis_time(pd.Timestamp(ANALYSIS_TIME)).tz) == "UTC"


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 11, 15),
        datetime(2026, 11, 15, tzinfo=timezone(timedelta(hours=3))),
        pd.NaT,
    ],
)
def test_normalize_rejects_invalid_time(value):
    with pytest.raises(MultiTimeframeAnalysisError):
        _normalize_analysis_time(value)


def test_parse_utc_series():
    values = pd.Series([
        datetime(2026, 11, 14, 10, 0, tzinfo=UTC),
        datetime(2026, 11, 14, 10, 5, tzinfo=UTC),
    ])
    result = _parse_utc_series(values, "open_time")
    assert str(result.dt.tz) == "UTC"


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["invalid", "invalid"]),
        pd.Series([
            datetime(2026, 11, 14, 10, 0),
            datetime(2026, 11, 14, 10, 5),
        ]),
        pd.Series(pd.to_datetime([
            "2026-11-14 10:00:00+00:00", None
        ], utc=True)),
    ],
)
def test_parse_utc_series_rejects_invalid_values(values):
    with pytest.raises(MultiTimeframeAnalysisError):
        _parse_utc_series(values, "open_time")


def test_validate_numeric_columns():
    result = _validate_numeric_columns(make_candles())
    assert len(result) == 30
    assert result["close"].dtype == float


@pytest.mark.parametrize(
    "column,value",
    [
        ("open", True), ("high", False), ("low", True),
        ("close", False), ("volume", True),
    ],
)
def test_numeric_columns_reject_boolean(column, value):
    candles = make_candles()
    candles[column] = candles[column].astype(object)
    candles.loc[0, column] = value
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize(
    "column,value",
    [
        ("open", "invalid"), ("high", "invalid"),
        ("low", "invalid"), ("close", "invalid"),
        ("volume", "invalid"), ("open", np.nan),
        ("high", np.inf), ("low", -np.inf),
    ],
)
def test_numeric_columns_reject_invalid_values(column, value):
    candles = make_candles()
    candles[column] = candles[column].astype(object)
    candles.loc[0, column] = value
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_numeric_columns_reject_non_positive_prices(column):
    candles = make_candles()
    candles.loc[0, column] = 0
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


def test_numeric_columns_reject_negative_volume():
    candles = make_candles()
    candles.loc[0, "volume"] = -1
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


@pytest.mark.parametrize("column,value", [("high", 1), ("low", 999999)])
def test_numeric_columns_reject_invalid_ohlc(column, value):
    candles = make_candles()
    candles.loc[0, column] = value
    with pytest.raises(MultiTimeframeAnalysisError):
        _validate_numeric_columns(candles)


def test_validate_candles_accepts_valid_data():
    result = validate_direct(make_candles())
    assert len(result) == 30
    assert result["open_time"].is_monotonic_increasing


def test_validate_rejects_non_dataframe():
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct([])


def test_validate_rejects_empty_dataframe():
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(pd.DataFrame())


def test_validate_rejects_missing_column():
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(make_candles().drop(columns=["volume"]))


def test_validate_rejects_insufficient_candles():
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(make_candles(count=10))


def test_validate_rejects_incorrect_duration():
    candles = make_candles()
    candles.loc[0, "close_time"] += timedelta(seconds=1)
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(candles)


def test_validate_rejects_close_before_open():
    candles = make_candles()
    candles.loc[0, "close_time"] = (
        candles.loc[0, "open_time"] - timedelta(seconds=1)
    )
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(candles)


def test_validate_rejects_future_candle():
    candles = make_candles()
    candles.loc[len(candles) - 1, "open_time"] = ANALYSIS_TIME
    candles.loc[len(candles) - 1, "close_time"] = (
        ANALYSIS_TIME + timedelta(minutes=5)
    )
    with pytest.raises(MultiTimeframeAnalysisError, match="future or unclosed"):
        validate_direct(candles)


def test_validate_rejects_unsorted_candles():
    candles = make_candles().iloc[::-1].reset_index(drop=True)
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(candles)


def test_validate_rejects_duplicate_open_time():
    candles = make_candles()
    candles.loc[1, "open_time"] = candles.loc[0, "open_time"]
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(candles)


def test_validate_rejects_duplicate_close_time():
    candles = make_candles()
    candles.loc[1, "close_time"] = candles.loc[0, "close_time"]
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(candles)


@pytest.mark.parametrize("source", ["", "   ", None, 123])
def test_validate_rejects_invalid_source(source):
    candles = make_candles()
    candles.loc[0, "source"] = source
    with pytest.raises(MultiTimeframeAnalysisError):
        validate_direct(candles)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.01, "BULLISH"), (-0.01, "BEARISH"),
        (0.00001, "NEUTRAL"), (-0.00001, "NEUTRAL"),
        (0.0001, "NEUTRAL"), (-0.0001, "NEUTRAL"),
    ],
)
def test_classify_direction(value, expected):
    assert _classify_direction(value) == expected


def test_trend_analysis_bullish():
    result = _calculate_trend(make_candles(direction="up"))
    assert result["direction"] == "BULLISH"
    assert result["ema_fast"] > 0
    assert result["ema_slow"] > 0
    assert result["price_change_pct"] > 0


def test_trend_analysis_bearish():
    result = _calculate_trend(make_candles(direction="down"))
    assert result["direction"] == "BEARISH"
    assert result["price_change_pct"] < 0


def test_momentum_bullish():
    result = _calculate_momentum(make_candles(direction="up"))
    assert result["direction"] == "BULLISH"
    assert result["change_pct"] > 0


def test_momentum_bearish():
    result = _calculate_momentum(make_candles(direction="down"))
    assert result["direction"] == "BEARISH"
    assert result["change_pct"] < 0


def test_structure_analysis():
    result = _calculate_structure(make_candles(direction="up"))
    assert result["direction"] == "BULLISH"
    assert result["higher_high"] is True
    assert result["higher_low"] is True
    assert result["label"] == "HH_HL"


def test_support_resistance_analysis():
    result = _calculate_support_resistance(make_candles())
    assert result["support"] > 0
    assert result["resistance"] > result["support"]
    assert result["support_distance_pct"] >= 0
    assert result["resistance_distance_pct"] >= 0


def test_volume_confirmation_bullish():
    candles = make_candles(direction="up")
    candles.loc[len(candles) - 1, "volume"] = 10000
    result = _calculate_volume_confirmation(candles)
    assert result["direction"] == "BULLISH"
    assert result["confirmed"] is True


def test_volume_confirmation_bearish():
    candles = make_candles(direction="down")
    candles.loc[len(candles) - 1, "volume"] = 10000
    result = _calculate_volume_confirmation(candles)
    assert result["direction"] == "BEARISH"
    assert result["confirmed"] is True


def test_volume_confirmation_neutral():
    result = _calculate_volume_confirmation(make_candles(direction="flat"))
    assert result["direction"] == "NEUTRAL"
    assert result["confirmed"] is False


def test_analyzer_default_configuration():
    analyzer = MultiTimeframeAnalyzer()
    assert analyzer.minimum_candles == 20
    assert analyzer.required_timeframes == TIMEFRAME_ORDER
    assert analyzer.history == []


@pytest.mark.parametrize("minimum", [0, 1, 4, -1, True, 5.5, "20"])
def test_invalid_minimum_candles(minimum):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalyzer(minimum_candles=minimum)


@pytest.mark.parametrize("timeframes", [(), [], "5m", 123])
def test_invalid_required_timeframes(timeframes):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalyzer(required_timeframes=timeframes)


def test_unsupported_required_timeframe():
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalysisError
        MultiTimeframeAnalyzer(required_timeframes=("5m", "1d"))


def test_duplicate_required_timeframes():
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeAnalyzer(required_timeframes=("5m", "5m"))


def test_required_timeframes_are_normalized():
    analyzer = MultiTimeframeAnalyzer(
        required_timeframes=("4h", "5m", "1h")
    )
    assert analyzer.required_timeframes == ("5m", "1h", "4h")


def test_analyze_all_timeframes():
    result = make_analyzer().analyze(
        make_dataset(), analysis_time=ANALYSIS_TIME
    )
    assert result["status"] == "SUCCESS"
    assert result["overall_direction"] == "BULLISH"
    assert result["timeframes_analyzed"] == list(TIMEFRAME_ORDER)
    assert len(result["analysis_id"]) == 64


@pytest.mark.parametrize("timeframe", TIMEFRAME_ORDER)
def test_each_timeframe_individually(timeframe):
    analyzer = make_analyzer(required_timeframes=(timeframe,))
    result = analyzer.analyze(
        {timeframe: make_candles(timeframe)},
        analysis_time=ANALYSIS_TIME,
    )
    assert result["status"] == "SUCCESS"
    assert result["timeframes_analyzed"] == [timeframe]


def test_optional_timeframe_subset():
    analyzer = make_analyzer(required_timeframes=("5m", "1h"))
    result = analyzer.analyze(
        make_dataset(("5m", "1h")), analysis_time=ANALYSIS_TIME
    )
    assert result["timeframes_analyzed"] == ["5m", "1h"]


@pytest.mark.parametrize("data", [[], {}, None])
def test_analyzer_rejects_invalid_input(data):
    with pytest.raises(MultiTimeframeAnalysisError):
        make_analyzer().analyze(data, analysis_time=ANALYSIS_TIME)


def test_analyzer_rejects_unknown_timeframe():
    with pytest.raises(MultiTimeframeAnalysisError):
        make_analyzer().analyze(
            {"5m": make_candles(), "1d": make_candles()},
            analysis_time=ANALYSIS_TIME,
        )


def test_analyzer_rejects_missing_required_timeframes():
    with pytest.raises(MultiTimeframeAnalysisError):
        make_analyzer().analyze(
            {"5m": make_candles()}, analysis_time=ANALYSIS_TIME
        )


def test_analyzer_rejects_naive_analysis_time():
    with pytest.raises(MultiTimeframeAnalysisError):
        make_analyzer().analyze(
            make_dataset(), analysis_time=datetime(2026, 11, 15)
        )


def test_output_contains_required_sections():
    result = make_analyzer().analyze(
        make_dataset(), analysis_time=ANALYSIS_TIME
    )
    expected = {
        "analysis_id", "analysis_timestamp", "status",
        "overall_direction", "alignment",
        "higher_timeframe_validation", "timeframes", "source",
        "timeframes_analyzed", "parameters", "limitations",
    }
    assert expected.issubset(result)


def test_timeframe_output_contains_components():
    analyzer = make_analyzer(required_timeframes=("5m",))
    result = analyzer.analyze(
        {"5m": make_candles()}, analysis_time=ANALYSIS_TIME
    )
    expected = {
        "timeframe", "direction", "trend", "momentum",
        "structure", "support_resistance", "volume",
        "component_directions", "component_agreement",
        "current_price", "candles_analyzed",
        "last_candle_close_time", "source",
    }
    assert expected.issubset(result["timeframes"]["5m"])


def test_component_directions_are_valid():
    result = make_analyzer().analyze(
        make_dataset(), analysis_time=ANALYSIS_TIME
    )
    for timeframe in result["timeframes"].values():
        for direction in timeframe["component_directions"].values():
            assert direction in DIRECTIONS


def test_full_bullish_alignment():
    analyses = {tf: {"direction": "BULLISH"} for tf in TIMEFRAME_ORDER}
    result = _build_alignment(analyses)
    assert result["direction"] == "BULLISH"
    assert result["status"] == "FULL_ALIGNMENT"
    assert result["alignment_ratio"] == 1.0
    assert result["conflict_count"] == 0


def test_full_bearish_alignment():
    analyses = {tf: {"direction": "BEARISH"} for tf in TIMEFRAME_ORDER}
    result = _build_alignment(analyses)
    assert result["direction"] == "BEARISH"
    assert result["status"] == "FULL_ALIGNMENT"


def test_full_neutral_alignment():
    analyses = {tf: {"direction": "NEUTRAL"} for tf in TIMEFRAME_ORDER}
    result = _build_alignment(analyses)
    assert result["direction"] == "NEUTRAL"
    assert result["status"] == "FULL_ALIGNMENT"


def test_alignment_detects_conflicts():
    analyses = {
        "5m": {"direction": "BULLISH"},
        "15m": {"direction": "BEARISH"},
        "1h": {"direction": "BULLISH"},
        "4h": {"direction": "BULLISH"},
    }
    result = _build_alignment(analyses)
    assert result["status"] == "CONFLICTING"
    assert result["conflict_count"] == 3
    assert result["alignment_ratio"] == 0.75


def test_alignment_ignores_neutral_conflicts():
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


def test_higher_timeframe_confirms_directions():
    analyses = {tf: {"direction": "BULLISH"} for tf in TIMEFRAME_ORDER}
    result = _build_higher_timeframe_validation(analyses)
    assert result["status"] == "CONFIRMED"
    assert result["confirmed_count"] == 6
    assert result["conflict_count"] == 0


def test_higher_timeframe_detects_conflicts():
    analyses = {
        "5m": {"direction": "BULLISH"},
        "15m": {"direction": "BEARISH"},
        "1h": {"direction": "BULLISH"},
        "4h": {"direction": "BEARISH"},
    }
    result = _build_higher_timeframe_validation(analyses)
    assert result["status"] == "CONFLICTING"
    assert result["conflict_count"] > 0


def test_higher_timeframe_neutral_is_inconclusive():
    analyses = {
        "5m": {"direction": "NEUTRAL"},
        "1h": {"direction": "BULLISH"},
    }
    result = _build_higher_timeframe_validation(analyses)
    assert result["inconclusive_count"] == 1
    assert result["status"] == "INCONCLUSIVE"


def test_higher_timeframe_all_neutral():
    analyses = {
        "5m": {"direction": "NEUTRAL"},
        "15m": {"direction": "NEUTRAL"},
    }
    result = _build_higher_timeframe_validation(analyses)
    assert result["status"] == "INCONCLUSIVE"
    assert result["inconclusive_count"] == 1


def test_history_is_recorded():
    analyzer = make_analyzer()
    analyzer.analyze(make_dataset(), analysis_time=ANALYSIS_TIME)
    assert len(analyzer.history) == 1


def test_duplicate_history_is_prevented():
    analyzer = make_analyzer()
    dataset = make_dataset()
    first = analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    second = analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    assert first["analysis_id"] == second["analysis_id"]
    assert len(analyzer.history) == 1


def test_distinct_analysis_times_create_distinct_history():
    analyzer = make_analyzer()
    dataset = make_dataset()
    analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    analyzer.analyze(
        dataset, analysis_time=ANALYSIS_TIME + timedelta(minutes=1)
    )
    assert len(analyzer.history) == 2


def test_history_returns_defensive_copy():
    analyzer = make_analyzer()
    analyzer.analyze(make_dataset(), analysis_time=ANALYSIS_TIME)
    history = analyzer.history
    history.clear()
    assert len(analyzer.history) == 1


def test_result_does_not_modify_history():
    analyzer = make_analyzer()
    result = analyzer.analyze(make_dataset(), analysis_time=ANALYSIS_TIME)
    result["timeframes"].clear()
    assert len(analyzer.history[0]["timeframes"]) == 4


def test_clear_history():
    analyzer = make_analyzer()
    analyzer.analyze(make_dataset(), analysis_time=ANALYSIS_TIME)
    analyzer.clear_history()
    assert analyzer.history == []


def test_clear_history_allows_new_records():
    analyzer = make_analyzer()
    dataset = make_dataset()
    analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    analyzer.clear_history()
    analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    assert len(analyzer.history) == 1


def test_input_data_is_not_modified():
    analyzer = make_analyzer()
    dataset = make_dataset()
    original = {key: value.copy(deep=True) for key, value in dataset.items()}
    analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    for timeframe in dataset:
        pd.testing.assert_frame_equal(dataset[timeframe], original[timeframe])


def test_sources_are_aggregated():
    analyzer = make_analyzer(required_timeframes=("5m", "1h"))
    dataset = {
        "5m": make_candles("5m", source="kraken"),
        "1h": make_candles("1h", source="coinbase"),
    }
    result = analyzer.analyze(dataset, analysis_time=ANALYSIS_TIME)
    assert result["source"] == ["coinbase", "kraken"]


def test_candle_count_is_reported():
    result = make_analyzer().analyze(
        make_dataset(), analysis_time=ANALYSIS_TIME
    )
    for timeframe in TIMEFRAME_ORDER:
        assert result["timeframes"][timeframe]["candles_analyzed"] == 30


def test_analysis_parameters_are_reported():
    analyzer = make_analyzer(
        minimum_candles=25,
        required_timeframes=("5m", "1h"),
    )
    result = analyzer.analyze(
        make_dataset(("5m", "1h")), analysis_time=ANALYSIS_TIME
    )
    assert result["parameters"]["minimum_candles"] == 25
    assert result["parameters"]["required_timeframes"] == ["5m", "1h"]


def test_limitations_are_reported():
    result = make_analyzer().analyze(
        make_dataset(), analysis_time=ANALYSIS_TIME
    )
    assert len(result["limitations"]) >= 5
    assert any("does not execute trades" in item for item in result["limitations"])


def test_convenience_function():
    result = analyze_multiple_timeframes(
        make_dataset(), analysis_time=ANALYSIS_TIME
    )
    assert result["status"] == "SUCCESS"
    assert len(result["timeframes"]) == 4


def test_convenience_function_custom_configuration():
    result = analyze_multiple_timeframes(
        make_dataset(("5m", "1h")),
        analysis_time=ANALYSIS_TIME,
        minimum_candles=25,
        required_timeframes=("5m", "1h"),
    )
    assert result["timeframes_analyzed"] == ["5m", "1h"]
    assert result["parameters"]["minimum_candles"] == 25


def test_analysis_id_is_deterministic():
    dataset = make_dataset()
    first = make_analyzer().analyze(dataset, analysis_time=ANALYSIS_TIME)
    second = make_analyzer().analyze(dataset, analysis_time=ANALYSIS_TIME)
    assert first["analysis_id"] == second["analysis_id"]


def test_analysis_id_changes_with_analysis_time():
    dataset = make_dataset()
    first = make_analyzer().analyze(dataset, analysis_time=ANALYSIS_TIME)
    second = make_analyzer().analyze(
        dataset, analysis_time=ANALYSIS_TIME + timedelta(minutes=1)
    )
    assert first["analysis_id"] != second["analysis_id"]