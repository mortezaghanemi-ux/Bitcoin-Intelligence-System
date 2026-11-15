"""Tests for the trader order block and fair value gap detector."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.order_blocks.detector import (
    OrderBlockDetector,
    OrderBlockDetectorError,
    detect_order_blocks,
)


UTC = timezone.utc
CHECK_TIME = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)
MINIMUM_CANDLES = 30


def make_candles(count=40, timeframe_seconds=300):
    start = datetime(2026, 11, 13, 8, 0, tzinfo=UTC)
    rows = []

    for index in range(count):
        open_time = start + timedelta(
            seconds=index * timeframe_seconds
        )
        close_time = open_time + timedelta(
            seconds=timeframe_seconds
        )

        rows.append({
            "open_time": open_time,
            "close_time": close_time,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 10.0,
            "source": "kraken",
        })

    return pd.DataFrame(rows)


def make_detector(**kwargs):
    kwargs.setdefault("minimum_candles", MINIMUM_CANDLES)
    return OrderBlockDetector(timeframe="5m", **kwargs)


def make_bullish_order_block_data():
    data = make_candles()

    data.loc[20, [
        "open", "high", "low", "close"
    ]] = [101.0, 101.5, 99.0, 99.5]

    data.loc[21, [
        "open", "high", "low", "close"
    ]] = [99.5, 102.0, 99.4, 101.5]

    # Keep subsequent candles above the bullish order block.
    for index in range(22, len(data)):
        data.loc[index, [
            "open", "high", "low", "close"
        ]] = [102.5, 103.0, 102.0, 102.7]

    return data


def make_bearish_order_block_data():
    data = make_candles()

    data.loc[20, [
        "open", "high", "low", "close"
    ]] = [99.0, 101.0, 98.5, 100.5]

    data.loc[21, [
        "open", "high", "low", "close"
    ]] = [100.5, 100.6, 97.0, 98.0]

    for index in range(22, len(data)):
        data.loc[index, [
            "open", "high", "low", "close"
        ]] = [97.8, 98.0, 97.0, 97.5]

    return data


def make_bullish_fvg_data():
    data = make_candles()

    data.loc[10, [
        "open", "high", "low", "close"
    ]] = [100.0, 100.5, 99.5, 100.1]

    data.loc[11, [
        "open", "high", "low", "close"
    ]] = [100.5, 102.0, 100.4, 101.8]

    data.loc[12, [
        "open", "high", "low", "close"
    ]] = [101.5, 102.5, 101.0, 102.0]

    # Keep later candles above the bullish FVG.
    for index in range(13, len(data)):
        data.loc[index, [
            "open", "high", "low", "close"
        ]] = [102.0, 102.5, 101.5, 102.2]

    return data


def make_bearish_fvg_data():
    data = make_candles()

    data.loc[10, [
        "open", "high", "low", "close"
    ]] = [100.0, 101.0, 99.5, 100.2]

    data.loc[11, [
        "open", "high", "low", "close"
    ]] = [100.0, 100.1, 98.0, 98.2]

    data.loc[12, [
        "open", "high", "low", "close"
    ]] = [98.5, 99.0, 97.5, 98.0]

    return data


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


@pytest.mark.parametrize("timeframe", ["5m", "15m", "1h", "4h"])
def test_supported_timeframes(timeframe):
    detector = OrderBlockDetector(timeframe=timeframe)
    assert detector.timeframe == timeframe


@pytest.mark.parametrize(
    "timeframe",
    ["1m", "1d", "invalid", "", None],
)
def test_unsupported_timeframe_is_rejected(timeframe):
    with pytest.raises(OrderBlockDetectorError):
        OrderBlockDetector(timeframe=timeframe)


@pytest.mark.parametrize(
    "parameters",
    [
        {"displacement_threshold": 0},
        {"displacement_threshold": -0.1},
        {"displacement_threshold": float("nan")},
        {"displacement_threshold": float("inf")},
        {"displacement_threshold": True},
        {"minimum_body_ratio": 0},
        {"minimum_body_ratio": -0.1},
        {"minimum_body_ratio": 1.1},
        {"minimum_body_ratio": float("nan")},
        {"minimum_body_ratio": True},
        {"lookback": 0},
        {"lookback": -1},
        {"lookback": True},
        {"minimum_candles": 2},
        {"minimum_candles": 0},
        {"minimum_candles": True},
        {"mitigation_tolerance": -0.1},
        {"mitigation_tolerance": 1},
        {"mitigation_tolerance": float("nan")},
        {"mitigation_tolerance": True},
    ],
)
def test_invalid_configuration_is_rejected(parameters):
    with pytest.raises(OrderBlockDetectorError):
        make_detector(**parameters)


def test_default_configuration():
    detector = make_detector()

    assert detector.timeframe == "5m"
    assert detector.displacement_threshold == 0.003
    assert detector.minimum_body_ratio == 0.60
    assert detector.lookback == 10
    assert detector.minimum_candles == MINIMUM_CANDLES
    assert detector.mitigation_tolerance == 0.0


# ---------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------


def test_analysis_time_must_be_timezone_aware():
    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(
            make_candles(),
            analysis_time=datetime(2026, 11, 13, 12, 0),
        )


def test_analysis_time_must_use_utc():
    non_utc = datetime(
        2026, 11, 13, 14, 0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(
            make_candles(),
            analysis_time=non_utc,
        )


def test_missing_analysis_time_is_supported():
    result = make_detector().detect(make_candles())
    assert result["status"] == "SUCCESS"


def test_invalid_timestamp_values_are_rejected():
    data = make_candles()
    data["open_time"] = data["open_time"].astype(object)
    data.loc[0, "open_time"] = "invalid"

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_naive_candle_timestamps_are_rejected():
    data = make_candles()
    data["open_time"] = data["open_time"].dt.tz_localize(None)

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_non_utc_candle_timestamps_are_rejected():
    data = make_candles()
    data["open_time"] = data["open_time"].dt.tz_convert(
        "Europe/Berlin"
    )

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_future_candles_are_rejected():
    data = make_candles()
    future_open = CHECK_TIME + timedelta(minutes=5)

    data.loc[len(data) - 1, "open_time"] = future_open
    data.loc[len(data) - 1, "close_time"] = (
        future_open + timedelta(minutes=5)
    )

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_unclosed_candles_are_rejected():
    data = make_candles()
    data.loc[len(data) - 1, "close_time"] = (
        CHECK_TIME + timedelta(minutes=1)
    )

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError):
        make_detector().detect([], analysis_time=CHECK_TIME)


def test_empty_dataframe_is_rejected():
    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(
            pd.DataFrame(),
            analysis_time=CHECK_TIME,
        )


def test_missing_required_column_is_rejected():
    data = make_candles().drop(columns=["volume"])

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_insufficient_candles_are_rejected():
    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(
            make_candles(count=10),
            analysis_time=CHECK_TIME,
        )


def test_unsorted_candles_are_rejected():
    data = make_candles()
    data = pd.concat(
        [data.iloc[:10], data.iloc[11:], data.iloc[10:11]],
        ignore_index=True,
    )

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_duplicate_open_times_are_rejected():
    data = make_candles()
    data.loc[1, "open_time"] = data.loc[0, "open_time"]

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_duplicate_close_times_are_rejected():
    data = make_candles()
    data.loc[1, "close_time"] = data.loc[0, "close_time"]

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_invalid_candle_duration_is_rejected():
    data = make_candles()
    data.loc[0, "close_time"] += timedelta(minutes=1)

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close", "volume"],
)
@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -float("inf"), "invalid"],
)
def test_invalid_numeric_values_are_rejected(column, bad_value):
    data = make_candles()

    if isinstance(bad_value, str):
        data[column] = data[column].astype(object)

    data.loc[0, column] = bad_value

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_non_positive_prices_are_rejected(column):
    data = make_candles()
    data.loc[0, column] = 0

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_negative_volume_is_rejected():
    data = make_candles()
    data.loc[0, "volume"] = -1

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close", "volume"],
)
def test_boolean_numeric_values_are_rejected(column):
    data = make_candles()
    data[column] = data[column].astype(object)
    data.loc[0, column] = True

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_invalid_ohlc_relationship_is_rejected():
    data = make_candles()
    data.loc[5, "high"] = 98.0

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


def test_invalid_source_is_rejected():
    data = make_candles()
    data.loc[0, "source"] = ""

    with pytest.raises(OrderBlockDetectorError):
        make_detector().detect(data, analysis_time=CHECK_TIME)


# ---------------------------------------------------------
# Order block detection
# ---------------------------------------------------------


def test_bullish_order_block_is_detected():
    result = make_detector().detect(
        make_bullish_order_block_data(),
        analysis_time=CHECK_TIME,
    )

    bullish = [
        item for item in result["order_blocks"]
        if item["type"] == "BULLISH_ORDER_BLOCK"
    ]

    assert bullish
    assert bullish[0]["direction"] == "BULLISH"
    assert bullish[0]["lower"] < bullish[0]["upper"]
    assert bullish[0]["midpoint"] > 0
    assert bullish[0]["status"] == "ACTIVE"


def test_bearish_order_block_is_detected():
    result = make_detector().detect(
        make_bearish_order_block_data(),
        analysis_time=CHECK_TIME,
    )

    bearish = [
        item for item in result["order_blocks"]
        if item["type"] == "BEARISH_ORDER_BLOCK"
    ]

    assert bearish
    assert bearish[0]["direction"] == "BEARISH"
    assert bearish[0]["lower"] < bearish[0]["upper"]


def test_order_block_has_deterministic_zone_id():
    data = make_bullish_order_block_data()

    first = make_detector().detect(data, analysis_time=CHECK_TIME)
    second = make_detector().detect(data, analysis_time=CHECK_TIME)

    assert [
        item["zone_id"] for item in first["order_blocks"]
    ] == [
        item["zone_id"] for item in second["order_blocks"]
    ]


def test_order_block_requires_displacement():
    result = make_detector(
        displacement_threshold=0.5
    ).detect(make_candles(), analysis_time=CHECK_TIME)

    assert result["order_blocks"] == []


def test_order_block_requires_opposite_candle():
    data = make_candles()

    data.loc[20, [
        "open", "high", "low", "close"
    ]] = [100.0, 100.5, 99.5, 100.2]

    data.loc[21, [
        "open", "high", "low", "close"
    ]] = [100.2, 103.0, 100.1, 102.0]

    result = make_detector().detect(data, analysis_time=CHECK_TIME)

    assert not any(
        item["type"] == "BULLISH_ORDER_BLOCK"
        for item in result["order_blocks"]
    )


def test_order_block_requires_minimum_body_ratio():
    result = make_detector(
        minimum_body_ratio=0.99
    ).detect(
        make_bullish_order_block_data(),
        analysis_time=CHECK_TIME,
    )

    assert not result["order_blocks"]


def test_order_block_contains_origin_and_confirmation_times():
    result = make_detector().detect(
        make_bullish_order_block_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["order_blocks"]

    zone = result["order_blocks"][0]

    assert zone["origin_open_time"]
    assert zone["origin_close_time"]
    assert zone["confirmation_time"]
    assert zone["detection_time"] == CHECK_TIME.isoformat()


# ---------------------------------------------------------
# Fair value gaps
# ---------------------------------------------------------


def test_bullish_fvg_is_detected():
    result = make_detector().detect(
        make_bullish_fvg_data(),
        analysis_time=CHECK_TIME,
    )

    bullish = [
        item for item in result["fair_value_gaps"]
        if item["type"] == "BULLISH_FVG"
    ]

    assert bullish

    gap = bullish[0]

    assert gap["direction"] == "BULLISH"
    assert gap["lower"] == 100.5
    assert gap["upper"] == 101.0
    assert gap["lower"] < gap["upper"]


def test_bearish_fvg_is_detected():
    result = make_detector().detect(
        make_bearish_fvg_data(),
        analysis_time=CHECK_TIME,
    )

    bearish = [
        item for item in result["fair_value_gaps"]
        if item["type"] == "BEARISH_FVG"
    ]

    assert bearish

    gap = bearish[0]

    assert gap["direction"] == "BEARISH"
    assert gap["lower"] == 99.0
    assert gap["upper"] == 99.5


def test_fvg_contains_consequent_encroachment():
    result = make_detector().detect(
        make_bullish_fvg_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["fair_value_gaps"]

    gap = result["fair_value_gaps"][0]
    expected = (gap["lower"] + gap["upper"]) / 2.0

    assert gap["consequent_encroachment"] == expected
    assert gap["midpoint"] == expected


def test_fvg_zone_id_is_deterministic():
    data = make_bullish_fvg_data()

    first = make_detector().detect(data, analysis_time=CHECK_TIME)
    second = make_detector().detect(data, analysis_time=CHECK_TIME)

    assert [
        item["zone_id"] for item in first["fair_value_gaps"]
    ] == [
        item["zone_id"] for item in second["fair_value_gaps"]
    ]


def test_no_fvg_when_three_candle_gap_does_not_exist():
    result = make_detector().detect(
        make_candles(),
        analysis_time=CHECK_TIME,
    )

    assert result["fair_value_gaps"] == []


# ---------------------------------------------------------
# Mitigation
# ---------------------------------------------------------


def test_unmitigated_bullish_order_block_remains_active():
    result = make_detector().detect(
        make_bullish_order_block_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["order_blocks"]

    zone = result["order_blocks"][0]

    assert zone["status"] == "ACTIVE"
    assert zone["mitigated"] is False
    assert zone["mitigation_time"] is None


def test_bullish_order_block_full_mitigation_is_detected():
    data = make_bullish_order_block_data()

    data.loc[22, [
        "open", "high", "low", "close"
    ]] = [101.5, 102.0, 98.5, 99.0]

    result = make_detector().detect(data, analysis_time=CHECK_TIME)

    zone = next(
        item for item in result["order_blocks"]
        if item["type"] == "BULLISH_ORDER_BLOCK"
    )

    assert zone["status"] == "MITIGATED"
    assert zone["mitigated"] is True
    assert zone["mitigation_time"] is not None


def test_bearish_order_block_full_mitigation_is_detected():
    data = make_bearish_order_block_data()

    data.loc[22, [
        "open", "high", "low", "close"
    ]] = [98.0, 102.0, 97.5, 101.0]

    result = make_detector().detect(data, analysis_time=CHECK_TIME)

    zone = next(
        item for item in result["order_blocks"]
        if item["type"] == "BEARISH_ORDER_BLOCK"
    )

    assert zone["status"] == "MITIGATED"
    assert zone["mitigated"] is True


def test_bullish_fvg_mitigation_is_detected():
    data = make_bullish_fvg_data()

    data.loc[13, [
        "open", "high", "low", "close"
    ]] = [102.0, 102.2, 100.0, 100.5]

    result = make_detector().detect(data, analysis_time=CHECK_TIME)

    gap = next(
        item for item in result["fair_value_gaps"]
        if item["type"] == "BULLISH_FVG"
    )

    assert gap["status"] == "MITIGATED"
    assert gap["mitigated"] is True


def test_bearish_fvg_mitigation_is_detected():
    data = make_bearish_fvg_data()

    data.loc[13, [
        "open", "high", "low", "close"
    ]] = [98.0, 101.5, 97.8, 100.5]

    result = make_detector().detect(data, analysis_time=CHECK_TIME)

    gap = next(
        item for item in result["fair_value_gaps"]
        if item["type"] == "BEARISH_FVG"
    )

    assert gap["status"] == "MITIGATED"
    assert gap["mitigated"] is True


def test_mitigation_is_not_applied_before_confirmation():
    result = make_detector().detect(
        make_bullish_fvg_data(),
        analysis_time=CHECK_TIME,
    )

    gap = next(
        item for item in result["fair_value_gaps"]
        if item["type"] == "BULLISH_FVG"
    )

    assert gap["status"] == "ACTIVE"
    assert gap["mitigated"] is False


# ---------------------------------------------------------
# Zone validation and output
# ---------------------------------------------------------


def test_detected_order_block_has_validation():
    result = make_detector().detect(
        make_bullish_order_block_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["order_blocks"]

    validation = result["order_blocks"][0]["validation"]

    assert validation["valid"] is True
    assert validation["zone_width"] > 0
    assert validation["zone_width_pct"] > 0
    assert validation["price_position"] in (
        "ABOVE_ZONE",
        "INSIDE_ZONE",
        "BELOW_ZONE",
    )


def test_detected_fvg_has_validation():
    result = make_detector().detect(
        make_bullish_fvg_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["fair_value_gaps"]

    validation = result["fair_value_gaps"][0]["validation"]

    assert validation["valid"] is True
    assert validation["zone_width"] > 0


def test_output_contains_required_fields():
    result = make_detector().detect(
        make_candles(),
        analysis_time=CHECK_TIME,
    )

    required = {
        "analysis_id",
        "analysis_timestamp",
        "status",
        "timeframe",
        "source",
        "candles_analyzed",
        "last_candle_close_time",
        "order_blocks",
        "fair_value_gaps",
        "summary",
        "parameters",
        "limitations",
    }

    assert required.issubset(result.keys())


def test_analysis_status_is_success():
    result = make_detector().detect(
        make_candles(),
        analysis_time=CHECK_TIME,
    )

    assert result["status"] == "SUCCESS"


def test_analysis_id_is_deterministic():
    data = make_candles()

    first = make_detector().detect(data, analysis_time=CHECK_TIME)
    second = make_detector().detect(data, analysis_time=CHECK_TIME)

    assert first["analysis_id"] == second["analysis_id"]


def test_analysis_id_changes_with_analysis_time():
    data = make_candles()

    first = make_detector().detect(data, analysis_time=CHECK_TIME)
    second = make_detector().detect(
        data,
        analysis_time=CHECK_TIME + timedelta(minutes=1),
    )

    assert first["analysis_id"] != second["analysis_id"]


# ---------------------------------------------------------
# History
# ---------------------------------------------------------


def test_analysis_is_preserved_in_history():
    detector = make_detector()
    detector.detect(make_candles(), analysis_time=CHECK_TIME)

    assert len(detector.history) == 1


def test_duplicate_analysis_is_not_added_to_history():
    detector = make_detector()
    data = make_candles()

    detector.detect(data, analysis_time=CHECK_TIME)
    detector.detect(data, analysis_time=CHECK_TIME)

    assert len(detector.history) == 1


def test_different_analysis_is_added_to_history():
    detector = make_detector()
    data = make_candles()

    detector.detect(data, analysis_time=CHECK_TIME)
    detector.detect(
        data,
        analysis_time=CHECK_TIME + timedelta(minutes=1),
    )

    assert len(detector.history) == 2


def test_history_returns_defensive_copy():
    detector = make_detector()
    detector.detect(make_candles(), analysis_time=CHECK_TIME)

    history = detector.history
    history[0]["status"] = "CHANGED"

    assert detector.history[0]["status"] == "SUCCESS"


def test_result_mutation_does_not_change_history():
    detector = make_detector()

    result = detector.detect(
        make_candles(),
        analysis_time=CHECK_TIME,
    )

    result["status"] = "CHANGED"

    assert detector.history[0]["status"] == "SUCCESS"


def test_clear_removes_history():
    detector = make_detector()
    detector.detect(make_candles(), analysis_time=CHECK_TIME)

    detector.clear()

    assert detector.history == []


def test_analysis_does_not_modify_input_dataframe():
    data = make_bullish_order_block_data()
    original = data.copy(deep=True)

    make_detector().detect(data, analysis_time=CHECK_TIME)

    pd.testing.assert_frame_equal(data, original)


def test_summary_counts_match_detected_zones():
    result = make_detector().detect(
        make_bullish_fvg_data(),
        analysis_time=CHECK_TIME,
    )

    summary = result["summary"]

    assert summary["order_block_count"] == len(
        result["order_blocks"]
    )

    assert summary["fair_value_gap_count"] == len(
        result["fair_value_gaps"]
    )

    assert summary["bullish_fvg_count"] == sum(
        item["direction"] == "BULLISH"
        for item in result["fair_value_gaps"]
    )


# ---------------------------------------------------------
# Timeframes and convenience function
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "timeframe,seconds",
    [
        ("5m", 300),
        ("15m", 900),
        ("1h", 3600),
        ("4h", 14400),
    ],
)
def test_detector_accepts_supported_candle_durations(
    timeframe,
    seconds,
):
    data = make_candles(timeframe_seconds=seconds)

    # The analysis time must be after the final candle closes.
    analysis_time = (
        data.iloc[-1]["close_time"] + timedelta(minutes=1)
    )

    detector = OrderBlockDetector(
        timeframe=timeframe,
        minimum_candles=MINIMUM_CANDLES,
    )

    result = detector.detect(
        data,
        analysis_time=analysis_time,
    )

    assert result["status"] == "SUCCESS"
    assert result["timeframe"] == timeframe


def test_convenience_function_returns_analysis():
    result = detect_order_blocks(
        make_candles(),
        timeframe="5m",
        analysis_time=CHECK_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert result["timeframe"] == "5m"


def test_convenience_function_supports_custom_parameters():
    result = detect_order_blocks(
        make_candles(),
        timeframe="5m",
        analysis_time=CHECK_TIME,
        displacement_threshold=0.005,
        minimum_body_ratio=0.7,
        lookback=5,
        minimum_candles=30,
        mitigation_tolerance=0.001,
    )

    assert result["parameters"]["displacement_threshold"] == 0.005
    assert result["parameters"]["minimum_body_ratio"] == 0.7
    assert result["parameters"]["lookback"] == 5
    assert result["parameters"]["mitigation_tolerance"] == 0.001


def test_detector_does_not_generate_trading_orders():
    result = make_detector().detect(
        make_candles(),
        analysis_time=CHECK_TIME,
    )

    assert "orders" not in result
    assert "trades" not in result
    assert any(
        "does not execute trades" in limitation
        for limitation in result["limitations"]
    )