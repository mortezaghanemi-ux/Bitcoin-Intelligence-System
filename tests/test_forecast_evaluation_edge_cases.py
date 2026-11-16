"""Edge-case tests for the Bitcoin forecast evaluation engine."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from forecast.evaluation import (
    BULLISH,
    BEARISH,
    NEUTRAL,
    VALID_DIRECTIONS,
    DEFAULT_LOG_EPSILON,
    ForecastEvaluationError,
    _parse_utc_timestamp,
    _normalize_evaluation_time,
    _validate_positive_number,
    _validate_neutral_threshold,
    _validate_forecast,
    _classify_market_outcome,
    _calculate_brier_score,
    _calculate_log_loss,
    _calculate_directional_accuracy,
    evaluate_forecast,
    evaluate_forecasts,
)


UTC = timezone.utc

FORECAST_TIME = datetime(
    2026, 11, 1, 12, 0, tzinfo=UTC
)

HORIZON_END = FORECAST_TIME + timedelta(days=7)

EVALUATION_TIME = HORIZON_END + timedelta(hours=1)


# =========================================================
# Test helpers
# =========================================================

def make_forecast(
    forecast_id="BTC-EDGE-001",
    probabilities=None,
    market_price=95000.0,
):
    if probabilities is None:
        probabilities = {
            BULLISH: 70.0,
            BEARISH: 20.0,
            NEUTRAL: 10.0,
        }

    return {
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "asset": "BTC",
        "forecast_timestamp": FORECAST_TIME.isoformat(),
        "forecast_horizon": {
            "days": 7,
            "start": FORECAST_TIME.isoformat(),
            "end": HORIZON_END.isoformat(),
        },
        "market_price": market_price,
        "probabilities": probabilities,
        "confidence_score": 80.0,
    }


def make_actual_data(
    outcome_price=100000.0,
    outcome_timestamp=HORIZON_END,
):
    return {
        "outcome_price": outcome_price,
        "outcome_timestamp": outcome_timestamp,
    }


def evaluate_valid_forecast(
    forecast=None,
    actual_data=None,
    **kwargs,
):
    if forecast is None:
        forecast = make_forecast()

    if actual_data is None:
        actual_data = make_actual_data()

    parameters = {
        "forecast": forecast,
        "actual_market_data": actual_data,
        "evaluation_time": EVALUATION_TIME,
    }

    parameters.update(kwargs)

    return evaluate_forecast(**parameters)


def make_evaluation_record(
    forecast_id="BTC-EDGE-001",
    brier_score=0.20,
    log_loss=0.35,
    directional_accuracy=1.0,
):
    return {
        "evaluation_id": f"{forecast_id}-EVALUATION",
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "metrics": {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "directional_accuracy": directional_accuracy,
        },
    }


# =========================================================
# UTC timestamp parsing
# =========================================================

@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        " ",
        123456,
        True,
        [],
        {},
    ],
)
def test_timestamp_parser_rejects_invalid_input(timestamp):
    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(
            timestamp,
            "test_timestamp",
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-timestamp",
        "2026-99-99T99:99:99Z",
        "2026-11-01T12:00:00-invalid",
    ],
)
def test_timestamp_parser_rejects_invalid_strings(timestamp):
    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(
            timestamp,
            "test_timestamp",
        )


def test_timestamp_parser_rejects_naive_datetime():
    timestamp = datetime(2026, 11, 1, 12, 0)

    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(
            timestamp,
            "test_timestamp",
        )


@pytest.mark.parametrize(
    "offset",
    [
        timedelta(hours=1),
        timedelta(hours=-5),
        timedelta(minutes=30),
        timedelta(hours=3),
    ],
)
def test_timestamp_parser_rejects_non_utc_offsets(offset):
    timestamp = datetime(
        2026,
        11,
        1,
        12,
        0,
        tzinfo=timezone(offset),
    )

    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(
            timestamp,
            "test_timestamp",
        )


def test_timestamp_parser_accepts_utc_datetime():
    timestamp = FORECAST_TIME

    result = _parse_utc_timestamp(
        timestamp,
        "test_timestamp",
    )

    assert result == timestamp
    assert result.tzinfo == UTC


def test_timestamp_parser_accepts_utc_z_string():
    result = _parse_utc_timestamp(
        "2026-11-01T12:00:00Z",
        "test_timestamp",
    )

    assert result == FORECAST_TIME
    assert result.tzinfo == UTC


def test_timestamp_parser_accepts_explicit_utc_offset():
    result = _parse_utc_timestamp(
        "2026-11-01T12:00:00+00:00",
        "test_timestamp",
    )

    assert result == FORECAST_TIME


def test_timestamp_parser_returns_utc_datetime():
    result = _parse_utc_timestamp(
        "2026-11-01T12:00:00Z",
        "test_timestamp",
    )

    assert result.utcoffset() == timedelta(0)


# =========================================================
# Evaluation time normalization
# =========================================================

def test_normalize_evaluation_time_accepts_utc_datetime():
    result = _normalize_evaluation_time(
        EVALUATION_TIME
    )

    assert result == EVALUATION_TIME
    assert result.tzinfo == UTC


def test_normalize_evaluation_time_rejects_naive_datetime():
    with pytest.raises(ForecastEvaluationError):
        _normalize_evaluation_time(
            datetime(2026, 11, 8, 13, 0)
        )


def test_normalize_evaluation_time_rejects_non_utc_datetime():
    non_utc = datetime(
        2026,
        11,
        8,
        13,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEvaluationError):
        _normalize_evaluation_time(non_utc)


def test_normalize_evaluation_time_generates_utc_when_missing():
    result = _normalize_evaluation_time(None)

    assert isinstance(result, datetime)
    assert result.tzinfo == UTC


# =========================================================
# Positive number validation
# =========================================================

@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "95000",
        True,
        False,
        [],
        {},
    ],
)
def test_positive_number_validator_rejects_invalid_values(value):
    with pytest.raises(ForecastEvaluationError):
        _validate_positive_number(
            value,
            "test_value",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (100, 100.0),
        (95000.5, 95000.5),
        (0.001, 0.001),
    ],
)
def test_positive_number_validator_accepts_valid_values(
    value,
    expected,
):
    result = _validate_positive_number(
        value,
        "test_value",
    )

    assert result == expected
    assert isinstance(result, float)


# =========================================================
# Neutral threshold validation
# =========================================================

@pytest.mark.parametrize(
    "threshold",
    [
        -1,
        -0.01,
        100,
        100.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "1",
        True,
        False,
    ],
)
def test_neutral_threshold_validator_rejects_invalid_values(
    threshold,
):
    with pytest.raises(ForecastEvaluationError):
        _validate_neutral_threshold(threshold)


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        0.5,
        1,
        5.5,
        99.99,
    ],
)
def test_neutral_threshold_validator_accepts_valid_values(
    threshold,
):
    result = _validate_neutral_threshold(threshold)

    assert result == float(threshold)


# =========================================================
# Forecast dictionary validation
# =========================================================

@pytest.mark.parametrize(
    "forecast",
    [
        None,
        [],
        (),
        "forecast",
        100,
        True,
    ],
)
def test_forecast_validator_rejects_non_dictionary_input(
    forecast,
):
    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_accepts_valid_forecast():
    result = _validate_forecast(make_forecast())

    assert result["forecast_id"] == "BTC-EDGE-001"
    assert result["market_price"] == 95000.0
    assert result["horizon_days"] == 7


def test_forecast_validator_rejects_failed_status():
    forecast = make_forecast()
    forecast["status"] = "FAILED"

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "PENDING",
        "SUCCESSFUL",
        "success",
    ],
)
def test_forecast_validator_rejects_invalid_status(status):
    forecast = make_forecast()
    forecast["status"] = status

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "field",
    [
        "forecast_id",
        "asset",
        "forecast_timestamp",
        "forecast_horizon",
        "market_price",
        "probabilities",
        "confidence_score",
    ],
)
def test_forecast_validator_rejects_missing_required_fields(field):
    forecast = make_forecast()
    del forecast[field]

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "forecast_id",
    [
        None,
        "",
        " ",
        123,
        True,
        [],
    ],
)
def test_forecast_validator_rejects_invalid_forecast_id(
    forecast_id,
):
    forecast = make_forecast()
    forecast["forecast_id"] = forecast_id

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "asset",
    [
        None,
        "",
        "ETH",
        "BTCUSD",
        "btc",
        123,
    ],
)
def test_forecast_validator_rejects_unsupported_asset(asset):
    forecast = make_forecast()
    forecast["asset"] = asset

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


# =========================================================
# Forecast horizon validation
# =========================================================

@pytest.mark.parametrize(
    "horizon",
    [
        None,
        [],
        (),
        "7d",
        7,
    ],
)
def test_forecast_validator_rejects_invalid_horizon_container(
    horizon,
):
    forecast = make_forecast()
    forecast["forecast_horizon"] = horizon

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "field",
    [
        "days",
        "start",
        "end",
    ],
)
def test_forecast_validator_rejects_incomplete_horizon(field):
    forecast = make_forecast()
    del forecast["forecast_horizon"][field]

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "days",
    [
        0,
        -1,
        366,
        1.5,
        True,
        False,
        None,
        "7",
    ],
)
def test_forecast_validator_rejects_invalid_horizon_days(days):
    forecast = make_forecast()
    forecast["forecast_horizon"]["days"] = days

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_horizon_start_mismatch():
    forecast = make_forecast()

    forecast["forecast_horizon"]["start"] = (
        FORECAST_TIME + timedelta(minutes=1)
    ).isoformat()

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_inconsistent_horizon_end():
    forecast = make_forecast()

    forecast["forecast_horizon"]["end"] = (
        HORIZON_END + timedelta(seconds=1)
    ).isoformat()

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_non_utc_forecast_timestamp():
    forecast = make_forecast()

    forecast["forecast_timestamp"] = datetime(
        2026,
        11,
        1,
        15,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    ).isoformat()

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


# =========================================================
# Probability container validation
# =========================================================

@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        [],
        (),
        "70,20,10",
        100,
    ],
)
def test_forecast_validator_rejects_invalid_probability_container(
    probabilities,
):
    forecast = make_forecast()
    forecast["probabilities"] = probabilities

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {BULLISH: 70, BEARISH: 20},
        {
            BULLISH: 70,
            BEARISH: 20,
            NEUTRAL: 10,
            "UNKNOWN": 0,
        },
    ],
)
def test_forecast_validator_rejects_incomplete_or_extra_probabilities(
    probabilities,
):
    forecast = make_forecast()
    forecast["probabilities"] = probabilities

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "70",
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "direction",
    [
        BULLISH,
        BEARISH,
        NEUTRAL,
    ],
)
def test_forecast_validator_rejects_invalid_probability_values(
    direction,
    invalid_value,
):
    probabilities = {
        BULLISH: 70.0,
        BEARISH: 20.0,
        NEUTRAL: 10.0,
    }

    probabilities[direction] = invalid_value

    forecast = make_forecast()
    forecast["probabilities"] = probabilities

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "probabilities",
    [
        {
            BULLISH: 70.0,
            BEARISH: 20.0,
            NEUTRAL: 5.0,
        },
        {
            BULLISH: 50.0,
            BEARISH: 30.0,
            NEUTRAL: 10.0,
        },
        {
            BULLISH: 100.0,
            BEARISH: 10.0,
            NEUTRAL: 0.0,
        },
    ],
)
def test_forecast_validator_rejects_probability_totals_not_equal_to_100(
    probabilities,
):
    forecast = make_forecast()
    forecast["probabilities"] = probabilities

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_accepts_zero_probability():
    probabilities = {
        BULLISH: 100.0,
        BEARISH: 0.0,
        NEUTRAL: 0.0,
    }

    forecast = make_forecast(
        probabilities=probabilities
    )

    result = _validate_forecast(forecast)

    assert result["probabilities"][BULLISH] == 1.0
    assert result["probabilities"][BEARISH] == 0.0
    assert result["probabilities"][NEUTRAL] == 0.0


def test_forecast_validator_normalizes_probabilities_to_fractions():
    forecast = make_forecast()

    result = _validate_forecast(forecast)

    assert result["probabilities"][BULLISH] == 0.7
    assert result["probabilities"][BEARISH] == 0.2
    assert result["probabilities"][NEUTRAL] == 0.1


# =========================================================
# Confidence score validation
# =========================================================

@pytest.mark.parametrize(
    "confidence",
    [
        -1,
        100.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "80",
        True,
        False,
    ],
)
def test_forecast_validator_rejects_invalid_confidence_score(
    confidence,
):
    forecast = make_forecast()
    forecast["confidence_score"] = confidence

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "confidence",
    [
        0,
        50,
        80.5,
        100,
    ],
)
def test_forecast_validator_accepts_valid_confidence_score(
    confidence,
):
    forecast = make_forecast()
    forecast["confidence_score"] = confidence

    result = _validate_forecast(forecast)

    assert result["confidence_score"] == float(confidence)


# =========================================================
# Market outcome classification
# =========================================================

def test_market_outcome_classifies_bullish_movement():
    direction, change = _classify_market_outcome(
        100.0,
        102.0,
        1.0,
    )

    assert direction == BULLISH
    assert change == pytest.approx(2.0)


def test_market_outcome_classifies_bearish_movement():
    direction, change = _classify_market_outcome(
        100.0,
        98.0,
        1.0,
    )

    assert direction == BEARISH
    assert change == pytest.approx(-2.0)


def test_market_outcome_classifies_neutral_movement():
    direction, change = _classify_market_outcome(
        100.0,
        100.5,
        1.0,
    )

    assert direction == NEUTRAL
    assert change == pytest.approx(0.5)


def test_market_outcome_threshold_upper_boundary_is_neutral():
    direction, _ = _classify_market_outcome(
        100.0,
        101.0,
        1.0,
    )

    assert direction == NEUTRAL


def test_market_outcome_threshold_lower_boundary_is_neutral():
    direction, _ = _classify_market_outcome(
        100.0,
        99.0,
        1.0,
    )

    assert direction == NEUTRAL


def test_market_outcome_zero_threshold_classifies_positive_move():
    direction, _ = _classify_market_outcome(
        100.0,
        100.01,
        0.0,
    )

    assert direction == BULLISH


def test_market_outcome_zero_threshold_classifies_negative_move():
    direction, _ = _classify_market_outcome(
        100.0,
        99.99,
        0.0,
    )

    assert direction == BEARISH


# =========================================================
# Brier score edge cases
# =========================================================

def test_brier_score_is_zero_for_perfect_prediction():
    probabilities = {
        BULLISH: 1.0,
        BEARISH: 0.0,
        NEUTRAL: 0.0,
    }

    result = _calculate_brier_score(
        probabilities,
        BULLISH,
    )

    assert result == pytest.approx(0.0)


def test_brier_score_is_two_for_opposite_certain_prediction():
    probabilities = {
        BULLISH: 0.0,
        BEARISH: 1.0,
        NEUTRAL: 0.0,
    }

    result = _calculate_brier_score(
        probabilities,
        BULLISH,
    )

    assert result == pytest.approx(2.0)


@pytest.mark.parametrize(
    "actual_direction",
    [
        "UNKNOWN",
        "",
        None,
        123,
    ],
)
def test_brier_score_rejects_invalid_actual_direction(
    actual_direction,
):
    probabilities = {
        BULLISH: 0.7,
        BEARISH: 0.2,
        NEUTRAL: 0.1,
    }

    with pytest.raises(ForecastEvaluationError):
        _calculate_brier_score(
            probabilities,
            actual_direction,
        )


# =========================================================
# Log loss edge cases
# =========================================================

def test_log_loss_uses_default_epsilon_for_zero_probability():
    probabilities = {
        BULLISH: 0.0,
        BEARISH: 1.0,
        NEUTRAL: 0.0,
    }

    result = _calculate_log_loss(
        probabilities,
        BULLISH,
    )

    assert result == pytest.approx(
        -math.log(DEFAULT_LOG_EPSILON)
    )


def test_log_loss_is_zero_for_probability_one():
    probabilities = {
        BULLISH: 1.0,
        BEARISH: 0.0,
        NEUTRAL: 0.0,
    }

    result = _calculate_log_loss(
        probabilities,
        BULLISH,
    )

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "epsilon",
    [
        0,
        -1,
        0.5,
        1,
        float("nan"),
        float("inf"),
        None,
        "0.001",
        True,
    ],
)
def test_log_loss_rejects_invalid_epsilon(epsilon):
    probabilities = {
        BULLISH: 0.7,
        BEARISH: 0.2,
        NEUTRAL: 0.1,
    }

    with pytest.raises(ForecastEvaluationError):
        _calculate_log_loss(
            probabilities,
            BULLISH,
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "actual_direction",
    [
        "UNKNOWN",
        "",
        None,
        123,
    ],
)
def test_log_loss_rejects_invalid_actual_direction(
    actual_direction,
):
    probabilities = {
        BULLISH: 0.7,
        BEARISH: 0.2,
        NEUTRAL: 0.1,
    }

    with pytest.raises(ForecastEvaluationError):
        _calculate_log_loss(
            probabilities,
            actual_direction,
        )


# =========================================================
# Directional accuracy edge cases
# =========================================================

@pytest.mark.parametrize(
    ("probabilities", "actual_direction", "expected"),
    [
        (
            {BULLISH: 0.8, BEARISH: 0.1, NEUTRAL: 0.1},
            BULLISH,
            True,
        ),
        (
            {BULLISH: 0.1, BEARISH: 0.8, NEUTRAL: 0.1},
            BULLISH,
            False,
        ),
        (
            {BULLISH: 0.1, BEARISH: 0.1, NEUTRAL: 0.8},
            NEUTRAL,
            True,
        ),
    ],
)
def test_directional_accuracy_returns_expected_result(
    probabilities,
    actual_direction,
    expected,
):
    result = _calculate_directional_accuracy(
        probabilities,
        actual_direction,
    )

    assert result is expected


def test_directional_accuracy_prefers_neutral_on_exact_tie():
    probabilities = {
        BULLISH: 0.5,
        BEARISH: 0.0,
        NEUTRAL: 0.5,
    }

    result = _calculate_directional_accuracy(
        probabilities,
        NEUTRAL,
    )

    assert result is True


# =========================================================
# Public evaluation function edge cases
# =========================================================

def test_evaluation_rejects_non_dictionary_actual_data():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            make_forecast(),
            None,
            evaluation_time=EVALUATION_TIME,
        )


@pytest.mark.parametrize(
    "actual_data",
    [
        {},
        {"outcome_price": 100000.0},
        {"outcome_timestamp": HORIZON_END},
    ],
)
def test_evaluation_rejects_incomplete_actual_data(actual_data):
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            make_forecast(),
            actual_data,
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_rejects_outcome_before_horizon_completion():
    actual = make_actual_data(
        outcome_timestamp=HORIZON_END - timedelta(seconds=1)
    )

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            make_forecast(),
            actual,
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_accepts_outcome_exactly_at_horizon_end():
    result = evaluate_forecast(
        make_forecast(),
        make_actual_data(
            outcome_timestamp=HORIZON_END
        ),
        evaluation_time=EVALUATION_TIME,
    )

    assert result["status"] == "SUCCESS"


def test_evaluation_rejects_future_outcome_timestamp():
    actual = make_actual_data(
        outcome_timestamp=EVALUATION_TIME + timedelta(seconds=1)
    )

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            make_forecast(),
            actual,
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_accepts_evaluation_at_horizon_end():
    result = evaluate_forecast(
        make_forecast(),
        make_actual_data(),
        evaluation_time=HORIZON_END,
    )

    assert result["status"] == "SUCCESS"


def test_evaluation_rejects_evaluation_before_horizon_end():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            make_forecast(),
            make_actual_data(),
            evaluation_time=HORIZON_END - timedelta(seconds=1),
        )


def test_evaluation_supports_custom_neutral_threshold():
    result = evaluate_valid_forecast(
        actual_data=make_actual_data(
            outcome_price=95500.0
        ),
        neutral_threshold_pct=0.25,
    )

    assert result["actual_outcome"]["direction"] == BULLISH


def test_evaluation_returns_expected_result_sections():
    result = evaluate_valid_forecast()

    assert "actual_outcome" in result
    assert "predicted_outcome" in result
    assert "metrics" in result
    assert "metadata" in result


def test_evaluation_returns_forecast_identifier():
    result = evaluate_valid_forecast()

    assert result["forecast_id"] == "BTC-EDGE-001"
    assert result["evaluation_id"] == (
        "BTC-EDGE-001-EVALUATION"
    )


def test_evaluation_does_not_modify_forecast_input():
    forecast = make_forecast()
    original = deepcopy(forecast)

    evaluate_valid_forecast(forecast=forecast)

    assert forecast == original


def test_evaluation_does_not_modify_actual_data_input():
    actual = make_actual_data()
    original = deepcopy(actual)

    evaluate_valid_forecast(actual_data=actual)

    assert actual == original


# =========================================================
# Aggregation input validation
# =========================================================

@pytest.mark.parametrize(
    "records",
    [
        None,
        "invalid",
        b"invalid",
        {},
        123,
    ],
)
def test_aggregation_rejects_invalid_collection(records):
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts(records)


def test_aggregation_rejects_empty_collection():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([])


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        (),
        "invalid",
        123,
    ],
)
def test_aggregation_rejects_non_dictionary_records(record):
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


def test_aggregation_rejects_failed_evaluation_record():
    record = make_evaluation_record()
    record["status"] = "FAILED"

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "forecast_id",
    [
        None,
        "",
        " ",
        123,
        True,
        [],
    ],
)
def test_aggregation_rejects_invalid_forecast_ids(forecast_id):
    record = make_evaluation_record()
    record["forecast_id"] = forecast_id

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


def test_aggregation_rejects_duplicate_forecast_ids():
    first = make_evaluation_record("BTC-DUPLICATE")
    second = make_evaluation_record("BTC-DUPLICATE")

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([first, second])


def test_aggregation_rejects_missing_metrics():
    record = make_evaluation_record()
    del record["metrics"]

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        [],
        (),
        "invalid",
    ],
)
def test_aggregation_rejects_invalid_metrics_container(metrics):
    record = make_evaluation_record()
    record["metrics"] = metrics

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "metric",
    [
        "brier_score",
        "log_loss",
        "directional_accuracy",
    ],
)
def test_aggregation_rejects_missing_metric(metric):
    record = make_evaluation_record()
    del record["metrics"][metric]

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        "0.5",
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "metric",
    [
        "brier_score",
        "log_loss",
        "directional_accuracy",
    ],
)
def test_aggregation_rejects_non_numeric_metric_values(
    metric,
    invalid_value,
):
    record = make_evaluation_record()
    record["metrics"][metric] = invalid_value

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "brier_score",
    [
        -0.01,
        2.01,
    ],
)
def test_aggregation_rejects_out_of_range_brier_score(brier_score):
    record = make_evaluation_record(
        brier_score=brier_score
    )

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


def test_aggregation_rejects_negative_log_loss():
    record = make_evaluation_record(log_loss=-0.01)

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "accuracy",
    [
        -1,
        0.5,
        2,
    ],
)
def test_aggregation_rejects_invalid_directional_accuracy(accuracy):
    record = make_evaluation_record(
        directional_accuracy=accuracy
    )

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


# =========================================================
# Aggregation results and immutability
# =========================================================

def test_aggregation_accepts_generator_input():
    records = (
        make_evaluation_record(f"BTC-GEN-{index}")
        for index in range(3)
    )

    result = evaluate_forecasts(records)

    assert result["status"] == "SUCCESS"
    assert result["evaluation_count"] == 3


def test_aggregation_calculates_mean_metrics():
    records = [
        make_evaluation_record(
            "BTC-MEAN-001",
            brier_score=0.2,
            log_loss=0.4,
            directional_accuracy=1.0,
        ),
        make_evaluation_record(
            "BTC-MEAN-002",
            brier_score=0.6,
            log_loss=0.8,
            directional_accuracy=0.0,
        ),
    ]

    result = evaluate_forecasts(records)

    assert result["metrics"]["mean_brier_score"] == pytest.approx(
        0.4
    )

    assert result["metrics"]["mean_log_loss"] == pytest.approx(
        0.6
    )

    assert result["metrics"]["directional_accuracy"] == pytest.approx(
        0.5
    )


def test_aggregation_counts_correct_and_incorrect_predictions():
    records = [
        make_evaluation_record(
            "BTC-COUNT-001",
            directional_accuracy=1.0,
        ),
        make_evaluation_record(
            "BTC-COUNT-002",
            directional_accuracy=0.0,
        ),
        make_evaluation_record(
            "BTC-COUNT-003",
            directional_accuracy=1.0,
        ),
    ]

    result = evaluate_forecasts(records)

    assert result["metrics"]["correct_predictions"] == 2
    assert result["metrics"]["incorrect_predictions"] == 1


def test_aggregation_returns_sorted_forecast_ids():
    records = [
        make_evaluation_record("BTC-Z"),
        make_evaluation_record("BTC-A"),
        make_evaluation_record("BTC-M"),
    ]

    result = evaluate_forecasts(records)

    assert result["forecast_ids"] == [
        "BTC-A",
        "BTC-M",
        "BTC-Z",
    ]


def test_aggregation_does_not_modify_input_records():
    records = [
        make_evaluation_record("BTC-IMMUTABLE-001"),
        make_evaluation_record("BTC-IMMUTABLE-002"),
    ]

    original = deepcopy(records)

    evaluate_forecasts(records)

    assert records == original


def test_aggregation_returns_success_status():
    result = evaluate_forecasts(
        [
            make_evaluation_record("BTC-SUCCESS-001"),
        ]
    )

    assert result["status"] == "SUCCESS"
    assert result["evaluation_count"] == 1


# =========================================================
# Module constants
# =========================================================

def test_valid_directions_contains_all_supported_directions():
    assert VALID_DIRECTIONS == {
        BULLISH,
        BEARISH,
        NEUTRAL,
    }


def test_default_log_epsilon_is_positive_and_small():
    assert 0 < DEFAULT_LOG_EPSILON < 0.5