from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from forecast.evaluation import (
    BULLISH,
    BEARISH,
    NEUTRAL,
    ForecastEvaluationError,
    evaluate_forecast,
    evaluate_forecasts,
)


UTC = timezone.utc

FORECAST_TIME = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
HORIZON_END = FORECAST_TIME + timedelta(days=7)
EVALUATION_TIME = HORIZON_END + timedelta(hours=1)


def make_forecast(
    forecast_id="BTC-TEST-001",
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


# =========================================================
# Successful evaluation
# =========================================================

def test_evaluate_forecast_successfully():
    result = evaluate_valid_forecast()

    assert isinstance(result, dict)
    assert result["status"] == "SUCCESS"


def test_evaluation_contains_forecast_id():
    result = evaluate_valid_forecast()

    assert result["forecast_id"] == "BTC-TEST-001"


def test_evaluation_contains_brier_score():
    result = evaluate_valid_forecast()

    assert "brier_score" in result["metrics"]
    assert 0 <= result["metrics"]["brier_score"] <= 2


def test_evaluation_contains_log_loss():
    result = evaluate_valid_forecast()

    assert "log_loss" in result["metrics"]
    assert result["metrics"]["log_loss"] >= 0


def test_evaluation_contains_directional_accuracy():
    result = evaluate_valid_forecast()

    assert "directional_accuracy" in result["metrics"]
    assert result["metrics"]["directional_accuracy"] in (0, 1)


# =========================================================
# Actual market outcome classification
# =========================================================

def test_positive_market_move_is_bullish():
    result = evaluate_valid_forecast(
        actual_data=make_actual_data(100000.0)
    )

    assert result["actual_outcome"]["direction"] == BULLISH


def test_negative_market_move_is_bearish():
    result = evaluate_valid_forecast(
        actual_data=make_actual_data(90000.0)
    )

    assert result["actual_outcome"]["direction"] == BEARISH


def test_small_market_move_is_neutral():
    result = evaluate_valid_forecast(
        actual_data=make_actual_data(95400.0)
    )

    assert result["actual_outcome"]["direction"] == NEUTRAL


def test_custom_neutral_threshold_is_supported():
    result = evaluate_valid_forecast(
        actual_data=make_actual_data(95500.0),
        neutral_threshold_pct=1.0,
    )

    assert result["actual_outcome"]["direction"] == NEUTRAL


# =========================================================
# Brier Score
# =========================================================

def test_brier_score_is_calculated_correctly():
    forecast = make_forecast(
        probabilities={
            BULLISH: 70.0,
            BEARISH: 20.0,
            NEUTRAL: 10.0,
        }
    )

    result = evaluate_valid_forecast(
        forecast=forecast,
        actual_data=make_actual_data(100000.0),
    )

    expected = (
        (0.70 - 1.0) ** 2
        + (0.20 - 0.0) ** 2
        + (0.10 - 0.0) ** 2
    )

    assert result["metrics"]["brier_score"] == pytest.approx(
        expected
    )


def test_perfect_forecast_has_zero_brier_score():
    forecast = make_forecast(
        probabilities={
            BULLISH: 100.0,
            BEARISH: 0.0,
            NEUTRAL: 0.0,
        }
    )

    result = evaluate_valid_forecast(
        forecast=forecast,
        actual_data=make_actual_data(100000.0),
    )

    assert result["metrics"]["brier_score"] == pytest.approx(0.0)


# =========================================================
# Log Loss
# =========================================================

def test_log_loss_is_calculated_correctly():
    result = evaluate_valid_forecast()

    assert result["metrics"]["log_loss"] == pytest.approx(
        -math.log(0.70)
    )


def test_log_loss_handles_zero_probability():
    forecast = make_forecast(
        probabilities={
            BULLISH: 0.0,
            BEARISH: 100.0,
            NEUTRAL: 0.0,
        }
    )

    result = evaluate_valid_forecast(
        forecast=forecast,
        actual_data=make_actual_data(100000.0),
    )

    assert result["metrics"]["log_loss"] > 0
    assert math.isfinite(result["metrics"]["log_loss"])


# =========================================================
# Directional Accuracy
# =========================================================

def test_correct_direction_has_full_directional_accuracy():
    forecast = make_forecast(
        probabilities={
            BULLISH: 80.0,
            BEARISH: 10.0,
            NEUTRAL: 10.0,
        }
    )

    result = evaluate_valid_forecast(
        forecast=forecast,
        actual_data=make_actual_data(100000.0),
    )

    assert result["metrics"]["directional_accuracy"] == 1.0


def test_wrong_direction_has_zero_directional_accuracy():
    forecast = make_forecast(
        probabilities={
            BULLISH: 10.0,
            BEARISH: 80.0,
            NEUTRAL: 10.0,
        }
    )

    result = evaluate_valid_forecast(
        forecast=forecast,
        actual_data=make_actual_data(100000.0),
    )

    assert result["metrics"]["directional_accuracy"] == 0.0


# =========================================================
# Forecast horizon validation
# =========================================================

def test_evaluation_before_horizon_completion_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual_data(),
            evaluation_time=HORIZON_END - timedelta(seconds=1),
        )


def test_outcome_before_horizon_end_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual_data(
                outcome_timestamp=HORIZON_END - timedelta(minutes=1)
            ),
            evaluation_time=EVALUATION_TIME,
        )


def test_outcome_after_evaluation_time_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual_data(
                outcome_timestamp=EVALUATION_TIME + timedelta(minutes=1)
            ),
            evaluation_time=EVALUATION_TIME,
        )


# =========================================================
# Invalid forecast validation
# =========================================================

def test_non_dictionary_forecast_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=None,
            actual_market_data=make_actual_data(),
            evaluation_time=EVALUATION_TIME,
        )


def test_failed_forecast_is_rejected():
    forecast = make_forecast()
    forecast["status"] = "FAILED"

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(forecast=forecast)


def test_wrong_asset_is_rejected():
    forecast = make_forecast()
    forecast["asset"] = "ETH"

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(forecast=forecast)


def test_missing_forecast_id_is_rejected():
    forecast = make_forecast()
    del forecast["forecast_id"]

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(forecast=forecast)


def test_invalid_market_price_is_rejected():
    forecast = make_forecast(market_price=0)

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(forecast=forecast)


@pytest.mark.parametrize(
    "price",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "95000",
        True,
    ],
)
def test_invalid_outcome_prices_are_rejected(price):
    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(
            actual_data=make_actual_data(
                outcome_price=price
            )
        )


def test_missing_actual_market_data_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=None,
            evaluation_time=EVALUATION_TIME,
        )


def test_missing_outcome_price_is_rejected():
    actual = make_actual_data()
    del actual["outcome_price"]

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(actual_data=actual)


def test_missing_outcome_timestamp_is_rejected():
    actual = make_actual_data()
    del actual["outcome_timestamp"]

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(actual_data=actual)


# =========================================================
# UTC timestamp validation
# =========================================================

def test_non_utc_outcome_timestamp_is_rejected():
    non_utc_time = datetime(
        2026,
        11,
        8,
        15,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(
            actual_data=make_actual_data(
                outcome_timestamp=non_utc_time
            )
        )


def test_naive_outcome_timestamp_is_rejected():
    naive_time = datetime(2026, 11, 8, 12, 0)

    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(
            actual_data=make_actual_data(
                outcome_timestamp=naive_time
            )
        )


def test_naive_evaluation_time_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecast(
            forecast=make_forecast(),
            actual_market_data=make_actual_data(),
            evaluation_time=datetime(2026, 11, 8, 13, 0),
        )


# =========================================================
# Neutral threshold validation
# =========================================================

@pytest.mark.parametrize(
    "threshold",
    [
        -1,
        100,
        float("nan"),
        float("inf"),
        None,
        "1",
        True,
    ],
)
def test_invalid_neutral_threshold_is_rejected(threshold):
    with pytest.raises(ForecastEvaluationError):
        evaluate_valid_forecast(
            neutral_threshold_pct=threshold
        )


# =========================================================
# Forecast collection aggregation
# =========================================================

def test_evaluate_forecasts_returns_aggregate():
    first = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-001"),
        actual_data=make_actual_data(100000.0),
    )

    second = evaluate_valid_forecast(
        forecast=make_forecast(
            "BTC-TEST-002",
            probabilities={
                BULLISH: 20.0,
                BEARISH: 70.0,
                NEUTRAL: 10.0,
            },
        ),
        actual_data=make_actual_data(90000.0),
    )

    result = evaluate_forecasts([first, second])

    assert isinstance(result, dict)
    assert result["status"] == "SUCCESS"


def test_aggregate_contains_evaluation_count():
    first = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-001")
    )

    second = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-002")
    )

    result = evaluate_forecasts([first, second])

    assert result["evaluation_count"] == 2


def test_aggregate_contains_average_brier_score():
    first = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-001")
    )

    second = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-002")
    )

    result = evaluate_forecasts([first, second])

    assert "mean_brier_score" in result["metrics"]
    assert 0 <= result["metrics"]["mean_brier_score"] <= 2


def test_aggregate_contains_average_log_loss():
    first = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-001")
    )

    second = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-002")
    )

    result = evaluate_forecasts([first, second])

    assert "mean_log_loss" in result["metrics"]
    assert result["metrics"]["mean_log_loss"] >= 0


def test_aggregate_contains_directional_accuracy():
    first = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-001")
    )

    second = evaluate_valid_forecast(
        forecast=make_forecast("BTC-TEST-002")
    )

    result = evaluate_forecasts([first, second])

    assert "directional_accuracy" in result["metrics"]


def test_duplicate_forecast_evaluations_are_rejected():
    first = evaluate_valid_forecast(
        forecast=make_forecast("BTC-DUPLICATE")
    )

    second = deepcopy(first)

    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([first, second])


def test_empty_evaluation_collection_is_rejected():
    with pytest.raises(ForecastEvaluationError):
        evaluate_forecasts([])


# =========================================================
# Input immutability
# =========================================================

def test_forecast_input_is_not_modified():
    forecast = make_forecast()
    original = deepcopy(forecast)

    evaluate_valid_forecast(forecast=forecast)

    assert forecast == original


def test_actual_market_data_is_not_modified():
    actual = make_actual_data()
    original = deepcopy(actual)

    evaluate_valid_forecast(actual_data=actual)

    assert actual == original