"""Edge-case tests for the Bitcoin forecast evaluation engine."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from forecast.evaluation import (
    BULLISH,
    BEARISH,
    NEUTRAL,
    ForecastEvaluationError,
    _calculate_brier_score,
    _calculate_directional_accuracy,
    _calculate_log_loss,
    _classify_market_outcome,
    _normalize_evaluation_time,
    _parse_utc_timestamp,
    _validate_forecast,
    _validate_neutral_threshold,
    _validate_positive_number,
    evaluate_forecast,
    evaluate_forecasts,
)


UTC = timezone.utc

FORECAST_TIME = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
HORIZON_END = FORECAST_TIME + timedelta(days=7)
EVALUATION_TIME = HORIZON_END + timedelta(hours=1)

VALID_PROBABILITIES = {
    BULLISH: 70.0,
    BEARISH: 20.0,
    NEUTRAL: 10.0,
}


def make_forecast(
    forecast_id="BTC-EDGE-001",
    probabilities=None,
    market_price=95000.0,
    confidence_score=80.0,
):
    if probabilities is None:
        probabilities = deepcopy(VALID_PROBABILITIES)

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
        "confidence_score": confidence_score,
    }


def make_actual(
    outcome_price=100000.0,
    outcome_timestamp=HORIZON_END,
):
    return {
        "outcome_price": outcome_price,
        "outcome_timestamp": outcome_timestamp,
    }


def evaluate_valid(
    forecast=None,
    actual=None,
    **kwargs,
):
    return evaluate_forecast(
        forecast=forecast or make_forecast(),
        actual_market_data=actual or make_actual(),
        evaluation_time=EVALUATION_TIME,
        **kwargs,
    )


def make_evaluation(forecast_id="BTC-AGG-001"):
    return evaluate_valid(
        forecast=make_forecast(forecast_id=forecast_id)
    )


# =========================================================
# UTC timestamp parsing
# =========================================================

@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        123456,
        1.5,
        True,
        [],
        {},
    ],
)
def test_timestamp_parser_rejects_invalid_types(value):
    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(value, "test_timestamp")


@pytest.mark.parametrize(
    "value",
    [
        "not-a-date",
        "2026-99-99T99:99:99",
        "2026-11-01T12:00:00+invalid",
    ],
)
def test_timestamp_parser_rejects_invalid_strings(value):
    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(value, "test_timestamp")


def test_timestamp_parser_rejects_naive_datetime():
    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(
            datetime(2026, 11, 1, 12, 0),
            "test_timestamp",
        )


def test_timestamp_parser_rejects_non_utc_offset():
    value = datetime(
        2026,
        11,
        1,
        15,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEvaluationError):
        _parse_utc_timestamp(value, "test_timestamp")


def test_timestamp_parser_accepts_utc_datetime():
    result = _parse_utc_timestamp(
        FORECAST_TIME,
        "test_timestamp",
    )

    assert result == FORECAST_TIME
    assert result.tzinfo == UTC


def test_timestamp_parser_accepts_z_suffix():
    result = _parse_utc_timestamp(
        "2026-11-01T12:00:00Z",
        "test_timestamp",
    )

    assert result == FORECAST_TIME


def test_timestamp_parser_accepts_explicit_zero_offset():
    result = _parse_utc_timestamp(
        "2026-11-01T12:00:00+00:00",
        "test_timestamp",
    )

    assert result == FORECAST_TIME


def test_timestamp_parser_normalizes_zero_offset_timezone():
    value = datetime(
        2026,
        11,
        1,
        12,
        0,
        tzinfo=timezone(timedelta(0)),
    )

    result = _parse_utc_timestamp(value, "test_timestamp")

    assert result.tzinfo == UTC


def test_normalize_evaluation_time_uses_utc():
    result = _normalize_evaluation_time(None)

    assert result.tzinfo == UTC


def test_normalize_evaluation_time_preserves_valid_time():
    result = _normalize_evaluation_time(EVALUATION_TIME)

    assert result == EVALUATION_TIME


def test_normalize_evaluation_time_rejects_naive_time():
    with pytest.raises(ForecastEvaluationError):
        _normalize_evaluation_time(
            datetime(2026, 11, 8, 13, 0)
        )


# =========================================================
# Positive number validation
# =========================================================

@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        True,
        False,
        "95000",
        [],
        {},
    ],
)
def test_positive_number_rejects_invalid_values(value):
    with pytest.raises(ForecastEvaluationError):
        _validate_positive_number(value, "test_value")


@pytest.mark.parametrize(
    "value",
    [
        1,
        0.01,
        95000,
        95000.5,
    ],
)
def test_positive_number_accepts_valid_values(value):
    result = _validate_positive_number(value, "test_value")

    assert result == float(value)
    assert isinstance(result, float)


# =========================================================
# Neutral threshold validation
# =========================================================

@pytest.mark.parametrize(
    "value",
    [
        -1,
        -0.01,
        100,
        100.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        True,
        False,
        "1",
        [],
    ],
)
def test_neutral_threshold_rejects_invalid_values(value):
    with pytest.raises(ForecastEvaluationError):
        _validate_neutral_threshold(value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.5,
        1,
        25,
        99.99,
    ],
)
def test_neutral_threshold_accepts_valid_values(value):
    assert _validate_neutral_threshold(value) == float(value)


# =========================================================
# Forecast structure validation
# =========================================================

@pytest.mark.parametrize(
    "forecast",
    [
        None,
        [],
        (),
        "forecast",
        123,
    ],
)
def test_forecast_validator_rejects_non_dictionary(forecast):
    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_missing_required_fields():
    forecast = make_forecast()
    del forecast["forecast_id"]

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "forecast_id",
    [
        None,
        "",
        " ",
        123,
        [],
        {},
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
    "status",
    [
        None,
        "FAILED",
        "PENDING",
        "success",
        "",
    ],
)
def test_forecast_validator_rejects_non_success_status(status):
    forecast = make_forecast()
    forecast["status"] = status

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


@pytest.mark.parametrize(
    "price",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        None,
        True,
        "95000",
    ],
)
def test_forecast_validator_rejects_invalid_market_price(price):
    forecast = make_forecast(market_price=price)

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


# =========================================================
# Forecast timestamp and horizon validation
# =========================================================

@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        "invalid",
        datetime(2026, 11, 1, 12, 0),
        datetime(
            2026,
            11,
            1,
            15,
            0,
            tzinfo=timezone(timedelta(hours=3)),
        ),
    ],
)
def test_forecast_validator_rejects_invalid_forecast_timestamp(
    timestamp,
):
    forecast = make_forecast()
    forecast["forecast_timestamp"] = timestamp

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "horizon",
    [
        None,
        [],
        (),
        "7 days",
        7,
    ],
)
def test_forecast_validator_rejects_invalid_horizon_type(horizon):
    forecast = make_forecast()
    forecast["forecast_horizon"] = horizon

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "days",
    [
        0,
        -1,
        366,
        True,
        False,
        7.0,
        "7",
        None,
    ],
)
def test_forecast_validator_rejects_invalid_horizon_days(days):
    forecast = make_forecast()
    forecast["forecast_horizon"]["days"] = days

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_accepts_maximum_horizon():
    forecast = make_forecast()
    forecast["forecast_horizon"]["days"] = 365
    forecast["forecast_horizon"]["end"] = (
        FORECAST_TIME + timedelta(days=365)
    ).isoformat()

    result = _validate_forecast(forecast)

    assert result["horizon_days"] == 365


def test_forecast_validator_rejects_missing_horizon_start():
    forecast = make_forecast()
    del forecast["forecast_horizon"]["start"]

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_missing_horizon_end():
    forecast = make_forecast()
    del forecast["forecast_horizon"]["end"]

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_start_not_matching_forecast_time():
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


def test_forecast_validator_rejects_non_utc_horizon_end():
    forecast = make_forecast()
    forecast["forecast_horizon"]["end"] = datetime(
        2026,
        11,
        8,
        15,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


# =========================================================
# Probability validation
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
    forecast = make_forecast(probabilities=probabilities)

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_missing_probability_direction():
    forecast = make_forecast(
        probabilities={
            BULLISH: 70,
            BEARISH: 30,
        }
    )

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_rejects_unknown_probability_direction():
    forecast = make_forecast(
        probabilities={
            BULLISH: 70,
            BEARISH: 20,
            "UNKNOWN": 10,
        }
    )

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -0.01,
        100.01,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        True,
        False,
        "70",
        [],
    ],
)
def test_forecast_validator_rejects_invalid_probability_values(value):
    probabilities = deepcopy(VALID_PROBABILITIES)
    probabilities[BULLISH] = value

    forecast = make_forecast(probabilities=probabilities)

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "probabilities",
    [
        {
            BULLISH: 70,
            BEARISH: 20,
            NEUTRAL: 9,
        },
        {
            BULLISH: 50,
            BEARISH: 30,
            NEUTRAL: 10,
        },
        {
            BULLISH: 100,
            BEARISH: 100,
            NEUTRAL: 0,
        },
    ],
)
def test_forecast_validator_rejects_probability_totals_not_100(
    probabilities,
):
    forecast = make_forecast(probabilities=probabilities)

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


def test_forecast_validator_accepts_zero_probability():
    forecast = make_forecast(
        probabilities={
            BULLISH: 100,
            BEARISH: 0,
            NEUTRAL: 0,
        }
    )

    result = _validate_forecast(forecast)

    assert result["probabilities"][BEARISH] == 0.0


def test_forecast_validator_normalizes_probabilities_to_fractions():
    forecast = make_forecast()

    result = _validate_forecast(forecast)

    assert result["probabilities"][BULLISH] == pytest.approx(0.7)
    assert result["probabilities"][BEARISH] == pytest.approx(0.2)
    assert result["probabilities"][NEUTRAL] == pytest.approx(0.1)


# =========================================================
# Confidence score validation
# =========================================================

@pytest.mark.parametrize(
    "confidence",
    [
        -1,
        -0.01,
        100.01,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        True,
        False,
        "80",
        [],
    ],
)
def test_forecast_validator_rejects_invalid_confidence(confidence):
    forecast = make_forecast(confidence_score=confidence)

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)


@pytest.mark.parametrize(
    "confidence",
    [
        0,
        0.01,
        50,
        99.99,
        100,
    ],
)
def test_forecast_validator_accepts_valid_confidence(confidence):
    forecast = make_forecast(confidence_score=confidence)

    result = _validate_forecast(forecast)

    assert result["confidence_score"] == float(confidence)


# =========================================================
# Market outcome classification
# =========================================================

def test_market_classification_exact_positive_threshold_is_neutral():
    direction, change = _classify_market_outcome(
        reference_price=100,
        outcome_price=101,
        neutral_threshold_pct=1,
    )

    assert direction == NEUTRAL
    assert change == pytest.approx(1.0)


def test_market_classification_exact_negative_threshold_is_neutral():
    direction, change = _classify_market_outcome(
        reference_price=100,
        outcome_price=99,
        neutral_threshold_pct=1,
    )

    assert direction == NEUTRAL
    assert change == pytest.approx(-1.0)


def test_market_classification_above_positive_threshold_is_bullish():
    direction, _ = _classify_market_outcome(
        reference_price=100,
        outcome_price=101.01,
        neutral_threshold_pct=1,
    )

    assert direction == BULLISH


def test_market_classification_below_negative_threshold_is_bearish():
    direction, _ = _classify_market_outcome(
        reference_price=100,
        outcome_price=98.99,
        neutral_threshold_pct=1,
    )

    assert direction == BEARISH


def test_market_classification_unchanged_price_is_neutral():
    direction, change = _classify_market_outcome(
        reference_price=100,
        outcome_price=100,
        neutral_threshold_pct=1,
    )

    assert direction == NEUTRAL
    assert change == pytest.approx(0.0)


def test_market_classification_zero_threshold():
    direction, _ = _classify_market_outcome(
        reference_price=100,
        outcome_price=100.01,
        neutral_threshold_pct=0,
    )

    assert direction == BULLISH


# =========================================================
# Brier score edge cases
# =========================================================

def test_brier_score_perfect_prediction_is_zero():
    score = _calculate_brier_score(
        {
            BULLISH: 1.0,
            BEARISH: 0.0,
            NEUTRAL: 0.0,
        },
        BULLISH,
    )

    assert score == pytest.approx(0.0)


def test_brier_score_completely_wrong_prediction_is_two():
    score = _calculate_brier_score(
        {
            BULLISH: 0.0,
            BEARISH: 1.0,
            NEUTRAL: 0.0,
        },
        BULLISH,
    )

    assert score == pytest.approx(2.0)


@pytest.mark.parametrize(
    "direction",
    [
        "INVALID",
        None,
        "",
        123,
    ],
)
def test_brier_score_rejects_invalid_actual_direction(direction):
    with pytest.raises(ForecastEvaluationError):
        _calculate_brier_score(
            deepcopy(VALID_PROBABILITIES),
            direction,
        )


def test_brier_score_for_neutral_outcome():
    score = _calculate_brier_score(
        {
            BULLISH: 0.2,
            BEARISH: 0.3,
            NEUTRAL: 0.5,
        },
        NEUTRAL,
    )

    expected = (
        0.2 ** 2
        + 0.3 ** 2
        + (0.5 - 1.0) ** 2
    )

    assert score == pytest.approx(expected)


# =========================================================
# Log loss edge cases
# =========================================================

def test_log_loss_perfect_probability_is_zero():
    loss = _calculate_log_loss(
        {
            BULLISH: 1.0,
            BEARISH: 0.0,
            NEUTRAL: 0.0,
        },
        BULLISH,
    )

    assert loss == pytest.approx(0.0)


def test_log_loss_zero_probability_uses_epsilon():
    loss = _calculate_log_loss(
        {
            BULLISH: 0.0,
            BEARISH: 1.0,
            NEUTRAL: 0.0,
        },
        BULLISH,
    )

    assert loss == pytest.approx(-math.log(1e-15))


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
        True,
        False,
        "0.001",
    ],
)
def test_log_loss_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ForecastEvaluationError):
        _calculate_log_loss(
            {
                BULLISH: 0.7,
                BEARISH: 0.2,
                NEUTRAL: 0.1,
            },
            BULLISH,
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "direction",
    [
        "INVALID",
        None,
        "",
        123,
    ],
)
def test_log_loss_rejects_invalid_actual_direction(direction):
    with pytest.raises(ForecastEvaluationError):
        _calculate_log_loss(
            deepcopy(VALID_PROBABILITIES),
            direction,
        )


# =========================================================
# Directional accuracy and tie breaking
# =========================================================

def test_directional_accuracy_returns_true_for_correct_prediction():
    result = _calculate_directional_accuracy(
        {
            BULLISH: 0.8,
            BEARISH: 0.1,
            NEUTRAL: 0.1,
        },
        BULLISH,
    )

    assert result is True


def test_directional_accuracy_returns_false_for_wrong_prediction():
    result = _calculate_directional_accuracy(
        {
            BULLISH: 0.1,
            BEARISH: 0.8,
            NEUTRAL: 0.1,
        },
        BULLISH,
    )

    assert result is False


def test_directional_accuracy_prefers_neutral_on_tie():
    result = _calculate_directional_accuracy(
        {
            BULLISH: 0.5,
            BEARISH: 0.0,
            NEUTRAL: 0.5,
        },
        NEUTRAL,
    )

    assert result is True


def test_directional_accuracy_handles_all_equal_probabilities():
    result = _calculate_directional_accuracy(
        {
            BULLISH: 1 / 3,
            BEARISH: 1 / 3,
            NEUTRAL: 1 / 3,
        },
        NEUTRAL,
    )

    assert result is True


# =========================================================
# Evaluation time and actual market data
# =========================================================

def test_evaluation_accepts_exact_horizon_end_time():
    result = evaluate_forecast(
        forecast=make_forecast(),
        actual_market_data=make_actual(),
        evaluation_time=HORIZON_END,
    )

    assert result["status"] == "SUCCESS"


def test_evaluation_rejects_time_before_horizon_end():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual(),
            evaluation_time=HORIZON_END - timedelta(microseconds=1),
        )


def test_evaluation_accepts_outcome_exactly_at_horizon_end():
    result = evaluate_forecast(
        forecast=make_forecast(),
        actual_market_data=make_actual(
            outcome_timestamp=HORIZON_END
        ),
        evaluation_time=HORIZON_END,
    )

    assert result["actual_outcome"]["timestamp"] == (
        HORIZON_END.isoformat()
    )


def test_evaluation_rejects_non_dictionary_actual_data():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=[],
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_rejects_missing_actual_fields():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data={},
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_rejects_future_outcome_timestamp():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual(
                outcome_timestamp=EVALUATION_TIME + timedelta(seconds=1)
            ),
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_rejects_naive_outcome_timestamp():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual(
                outcome_timestamp=datetime(2026, 11, 8, 12, 0)
            ),
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_rejects_non_utc_outcome_timestamp():
    non_utc = datetime(
        2026,
        11,
        8,
        15,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual(
                outcome_timestamp=non_utc
            ),
            evaluation_time=EVALUATION_TIME,
        )


def test_evaluation_result_contains_expected_metadata():
    result = evaluate_valid()

    assert result["metadata"]["asset"] == "BTC"
    assert result["metadata"]["horizon_days"] == 7
    assert result["metadata"]["statistical_calibration_assessed"] is False


def test_evaluation_result_contains_unique_evaluation_id():
    result = evaluate_valid()

    assert result["evaluation_id"] == (
        "BTC-EDGE-001-EVALUATION"
    )


def test_evaluation_result_contains_percentage_change():
    result = evaluate_valid(
        actual=make_actual(outcome_price=104500.0)
    )

    assert result["actual_outcome"]["percentage_change"] == pytest.approx(
        10.0
    )


def test_evaluation_does_not_modify_forecast_or_actual_data():
    forecast = make_forecast()
    actual = make_actual()

    original_forecast = deepcopy(forecast)
    original_actual = deepcopy(actual)

    evaluate_valid(
        forecast=forecast,
        actual=actual,
    )

    assert forecast == original_forecast
    assert actual == original_actual


# =========================================================
# Evaluation aggregation
# =========================================================

def test_aggregation_accepts_generator():
    records = (
        make_evaluation(f"BTC-GEN-{index}")
        for index in range(3)
    )

    result = evaluate_forecasts(records)

    assert result["evaluation_count"] == 3


def test_aggregation_accepts_tuple():
    records = (
        make_evaluation("BTC-TUPLE-001"),
        make_evaluation("BTC-TUPLE-002"),
    )

    result = evaluate_forecasts(records)

    assert result["evaluation_count"] == 2


def test_aggregation_rejects_none():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts(None)


@pytest.mark.parametrize(
    "records",
    [
        "",
        "evaluation",
        b"evaluation",
        {},
    ],
)
def test_aggregation_rejects_invalid_iterable_containers(records):
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts(records)


def test_aggregation_rejects_empty_generator():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts(iter(()))


def test_aggregation_rejects_non_dictionary_record():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([None])


def test_aggregation_rejects_failed_record():
    record = make_evaluation()
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
        [],
        {},
    ],
)
def test_aggregation_rejects_invalid_forecast_id(forecast_id):
    record = make_evaluation()
    record["forecast_id"] = forecast_id

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


def test_aggregation_rejects_duplicate_forecast_ids():
    first = make_evaluation("BTC-DUPLICATE")
    second = deepcopy(first)

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([first, second])


def test_aggregation_rejects_missing_metrics():
    record = make_evaluation()
    del record["metrics"]

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


def test_aggregation_rejects_non_dictionary_metrics():
    record = make_evaluation()
    record["metrics"] = None

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
    record = make_evaluation()
    del record["metrics"][metric]

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "0.5",
        float("nan"),
        float("inf"),
        float("-inf"),
        [],
        {},
    ],
)
def test_aggregation_rejects_non_numeric_metrics(value):
    for metric in (
        "brier_score",
        "log_loss",
        "directional_accuracy",
    ):
        record = make_evaluation()
        record["metrics"][metric] = value

        with pytest.raises(ForecastEvaluationError):
            evaluate_forecasts([record])


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        2.01,
        -1,
        3,
    ],
)
def test_aggregation_rejects_brier_score_outside_range(value):
    record = make_evaluation()
    record["metrics"]["brier_score"] = value

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        -1,
    ],
)
def test_aggregation_rejects_negative_log_loss(value):
    record = make_evaluation()
    record["metrics"]["log_loss"] = value

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0.5,
        2,
        1.5,
    ],
)
def test_aggregation_rejects_invalid_directional_accuracy(value):
    record = make_evaluation()
    record["metrics"]["directional_accuracy"] = value

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([record])


def test_aggregation_accepts_boundary_brier_scores():
    first = make_evaluation("BTC-BOUNDARY-001")
    second = make_evaluation("BTC-BOUNDARY-002")

    first["metrics"]["brier_score"] = 0
    second["metrics"]["brier_score"] = 2

    result = evaluate_forecasts([first, second])

    assert result["metrics"]["mean_brier_score"] == 1.0


def test_aggregation_accepts_zero_log_loss():
    record = make_evaluation()
    record["metrics"]["log_loss"] = 0

    result = evaluate_forecasts([record])

    assert result["metrics"]["mean_log_loss"] == 0.0


def test_aggregation_sorts_forecast_ids():
    records = [
        make_evaluation("BTC-Z"),
        make_evaluation("BTC-A"),
        make_evaluation("BTC-M"),
    ]

    result = evaluate_forecasts(records)

    assert result["forecast_ids"] == [
        "BTC-A",
        "BTC-M",
        "BTC-Z",
    ]


def test_aggregation_calculates_correct_prediction_counts():
    first = make_evaluation("BTC-COUNT-001")
    second = make_evaluation("BTC-COUNT-002")
    third = make_evaluation("BTC-COUNT-003")

    first["metrics"]["directional_accuracy"] = 1.0
    second["metrics"]["directional_accuracy"] = 0.0
    third["metrics"]["directional_accuracy"] = 1.0

    result = evaluate_forecasts([first, second, third])

    assert result["metrics"]["correct_predictions"] == 2
    assert result["metrics"]["incorrect_predictions"] == 1
    assert result["metrics"]["directional_accuracy"] == pytest.approx(
        2 / 3
    )


def test_aggregation_does_not_modify_input_records():
    records = [
        make_evaluation("BTC-IMMUTABLE-001"),
        make_evaluation("BTC-IMMUTABLE-002"),
    ]

    original = deepcopy(records)

    evaluate_forecasts(records)

    assert records == original


def test_aggregation_single_record_returns_expected_count():
    result = evaluate_forecasts([
        make_evaluation("BTC-SINGLE")
    ])

    assert result["evaluation_count"] == 1
    assert result["metrics"]["correct_predictions"] in (0, 1)
    assert result["metrics"]["incorrect_predictions"] in (0, 1)


def test_aggregation_mean_metrics_are_finite():
    records = [
        make_evaluation("BTC-FINITE-001"),
        make_evaluation("BTC-FINITE-002"),
    ]

    result = evaluate_forecasts(records)

    for value in result["metrics"].values():
        if isinstance(value, float):
            assert math.isfinite(value)