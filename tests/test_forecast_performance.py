"""Tests for the Bitcoin Forecast Performance Analyzer."""

from copy import deepcopy

import pytest

from forecast.performance import (
    ForecastPerformanceError,
    analyze_forecast_performance,
)


# =========================================================
# Test data
# =========================================================

def make_evaluation(
    forecast_id="BTC-TEST-001",
    actual_direction="BULLISH",
    predicted_direction="BULLISH",
    confidence_score=80.0,
    horizon_days=7,
    brier_score=0.25,
    log_loss=0.50,
    directional_accuracy=1.0,
):
    """Build a valid forecast evaluation record."""

    return {
        "evaluation_id": f"{forecast_id}-EVALUATION",
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "evaluated_at": "2026-11-15T12:00:00+00:00",
        "forecast_timestamp": "2026-11-01T12:00:00+00:00",
        "forecast_horizon_end": "2026-11-08T12:00:00+00:00",
        "actual_outcome": {
            "timestamp": "2026-11-08T12:00:00+00:00",
            "reference_price": 95000.0,
            "outcome_price": 100000.0,
            "percentage_change": 5.263158,
            "direction": actual_direction,
            "neutral_threshold_pct": 1.0,
        },
        "predicted_outcome": {
            "probabilities": {
                "BULLISH": 70.0,
                "BEARISH": 20.0,
                "NEUTRAL": 10.0,
            },
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
        },
        "metrics": {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "directional_accuracy": directional_accuracy,
            "directional_correct": (
                actual_direction == predicted_direction
            ),
        },
        "metadata": {
            "asset": "BTC",
            "horizon_days": horizon_days,
            "method": "multiclass_probability_evaluation",
            "statistical_calibration_assessed": False,
        },
    }


def analyze_valid_records(records=None):
    """Analyze a default valid evaluation collection."""

    if records is None:
        records = [
            make_evaluation(),
            make_evaluation(
                forecast_id="BTC-TEST-002",
                actual_direction="BEARISH",
                predicted_direction="BEARISH",
                confidence_score=60.0,
                horizon_days=7,
                brier_score=0.30,
                log_loss=0.60,
                directional_accuracy=1.0,
            ),
            make_evaluation(
                forecast_id="BTC-TEST-003",
                actual_direction="NEUTRAL",
                predicted_direction="BULLISH",
                confidence_score=45.0,
                horizon_days=14,
                brier_score=0.80,
                log_loss=1.20,
                directional_accuracy=0.0,
            ),
        ]

    return analyze_forecast_performance(records)


# =========================================================
# Successful analysis
# =========================================================

def test_analyze_forecast_performance_successfully():
    result = analyze_valid_records()

    assert isinstance(result, dict)
    assert result["status"] == "SUCCESS"


def test_analysis_contains_correct_analysis_type():
    result = analyze_valid_records()

    assert result["analysis_type"] == "FORECAST_PERFORMANCE"


def test_analysis_contains_evaluation_count():
    result = analyze_valid_records()

    assert result["evaluation_count"] == 3


def test_analysis_contains_summary():
    result = analyze_valid_records()

    assert isinstance(result["summary"], dict)


def test_analysis_contains_horizon_groups():
    result = analyze_valid_records()

    assert isinstance(result["by_horizon_days"], dict)
    assert "7" in result["by_horizon_days"]
    assert "14" in result["by_horizon_days"]


def test_analysis_contains_confidence_groups():
    result = analyze_valid_records()

    assert isinstance(result["by_confidence"], dict)

    assert {
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY_HIGH",
    }.issubset(result["by_confidence"].keys())


def test_analysis_contains_actual_direction_groups():
    result = analyze_valid_records()

    assert isinstance(result["by_actual_direction"], dict)

    assert {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }.issubset(result["by_actual_direction"].keys())


def test_analysis_contains_weaknesses_list():
    result = analyze_valid_records()

    assert isinstance(result["weaknesses"], list)


def test_analysis_contains_metadata():
    result = analyze_valid_records()

    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["asset"] == "BTC"
    assert (
        result["metadata"]["method"]
        == "historical_evaluation_analysis"
    )


def test_analysis_contains_sorted_forecast_ids():
    result = analyze_valid_records()

    assert result["forecast_ids"] == [
        "BTC-TEST-001",
        "BTC-TEST-002",
        "BTC-TEST-003",
    ]


# =========================================================
# Summary metrics
# =========================================================

def test_summary_contains_mean_brier_score():
    result = analyze_valid_records()

    assert (
        "mean_brier_score"
        in result["summary"]
    )


def test_summary_contains_mean_log_loss():
    result = analyze_valid_records()

    assert (
        "mean_log_loss"
        in result["summary"]
    )


def test_summary_contains_directional_accuracy():
    result = analyze_valid_records()

    assert (
        "directional_accuracy"
        in result["summary"]
    )


def test_summary_contains_correct_prediction_count():
    result = analyze_valid_records()

    assert (
        "correct_predictions"
        in result["summary"]
    )


def test_summary_contains_incorrect_prediction_count():
    result = analyze_valid_records()

    assert (
        "incorrect_predictions"
        in result["summary"]
    )


def test_summary_metrics_are_calculated_correctly():
    records = [
        make_evaluation(
            forecast_id="BTC-001",
            brier_score=0.2,
            log_loss=0.4,
            directional_accuracy=1.0,
        ),
        make_evaluation(
            forecast_id="BTC-002",
            brier_score=0.4,
            log_loss=0.8,
            directional_accuracy=0.0,
            actual_direction="BEARISH",
            predicted_direction="BULLISH",
        ),
    ]

    result = analyze_forecast_performance(records)
    summary = result["summary"]

    assert summary["evaluation_count"] == 2
    assert summary["mean_brier_score"] == pytest.approx(0.3)
    assert summary["mean_log_loss"] == pytest.approx(0.6)
    assert summary["directional_accuracy"] == pytest.approx(0.5)
    assert summary["correct_predictions"] == 1
    assert summary["incorrect_predictions"] == 1


# =========================================================
# Confidence grouping
# =========================================================

@pytest.mark.parametrize(
    ("confidence", "expected_bucket"),
    [
        (25.0, "LOW"),
        (55.0, "MODERATE"),
        (70.0, "HIGH"),
        (90.0, "VERY_HIGH"),
    ],
)
def test_confidence_is_assigned_to_correct_bucket(
    confidence,
    expected_bucket,
):
    result = analyze_forecast_performance(
        [
            make_evaluation(
                confidence_score=confidence
            )
        ]
    )

    bucket = result["by_confidence"][expected_bucket]

    assert bucket["evaluation_count"] == 1


def test_confidence_groups_contain_summary_metrics():
    result = analyze_valid_records()

    for summary in result["by_confidence"].values():
        assert "evaluation_count" in summary
        assert "mean_brier_score" in summary
        assert "mean_log_loss" in summary
        assert "directional_accuracy" in summary


# =========================================================
# Horizon grouping
# =========================================================

def test_horizon_groups_separate_different_horizons():
    records = [
        make_evaluation(
            forecast_id="BTC-7D",
            horizon_days=7,
        ),
        make_evaluation(
            forecast_id="BTC-14D",
            horizon_days=14,
        ),
        make_evaluation(
            forecast_id="BTC-30D",
            horizon_days=30,
        ),
    ]

    result = analyze_forecast_performance(records)

    assert result["by_horizon_days"]["7"]["evaluation_count"] == 1
    assert result["by_horizon_days"]["14"]["evaluation_count"] == 1
    assert result["by_horizon_days"]["30"]["evaluation_count"] == 1


# =========================================================
# Actual direction grouping
# =========================================================

def test_actual_direction_groups_count_records_correctly():
    records = [
        make_evaluation(
            forecast_id="BTC-BULL-1",
            actual_direction="BULLISH",
        ),
        make_evaluation(
            forecast_id="BTC-BULL-2",
            actual_direction="BULLISH",
        ),
        make_evaluation(
            forecast_id="BTC-BEAR-1",
            actual_direction="BEARISH",
            predicted_direction="BEARISH",
        ),
        make_evaluation(
            forecast_id="BTC-NEUTRAL-1",
            actual_direction="NEUTRAL",
            predicted_direction="NEUTRAL",
        ),
    ]

    result = analyze_forecast_performance(records)

    groups = result["by_actual_direction"]

    assert groups["BULLISH"]["evaluation_count"] == 2
    assert groups["BEARISH"]["evaluation_count"] == 1
    assert groups["NEUTRAL"]["evaluation_count"] == 1


# =========================================================
# Weakness detection
# =========================================================

def test_weakness_detection_returns_a_list():
    result = analyze_valid_records()

    assert isinstance(result["weaknesses"], list)


def test_weakness_detection_does_not_modify_analysis_inputs():
    records = [
        make_evaluation(
            forecast_id=f"BTC-WEAK-{index}",
            actual_direction="BULLISH",
            predicted_direction="BEARISH",
            confidence_score=90.0,
            directional_accuracy=0.0,
        )
        for index in range(6)
    ]

    original = deepcopy(records)

    analyze_forecast_performance(records)

    assert records == original


# =========================================================
# Invalid input collections
# =========================================================

@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "invalid",
        b"invalid",
        {},
        123,
    ],
)
def test_invalid_input_collection_is_rejected(invalid_input):
    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance(invalid_input)


def test_empty_collection_is_rejected():
    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([])


def test_non_dictionary_record_is_rejected():
    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance(
            [
                make_evaluation(),
                "invalid-record",
            ]
        )


def test_failed_evaluation_record_is_rejected():
    record = make_evaluation()
    record["status"] = "FAILED"

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_duplicate_forecast_ids_are_rejected():
    first = make_evaluation(
        forecast_id="BTC-DUPLICATE"
    )

    second = make_evaluation(
        forecast_id="BTC-DUPLICATE"
    )

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([first, second])


# =========================================================
# Invalid evaluation fields
# =========================================================

def test_missing_forecast_id_is_rejected():
    record = make_evaluation()
    del record["forecast_id"]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_empty_forecast_id_is_rejected():
    record = make_evaluation()
    record["forecast_id"] = " "

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_missing_metrics_is_rejected():
    record = make_evaluation()
    del record["metrics"]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_missing_actual_outcome_is_rejected():
    record = make_evaluation()
    del record["actual_outcome"]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_missing_predicted_outcome_is_rejected():
    record = make_evaluation()
    del record["predicted_outcome"]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_missing_metadata_is_rejected():
    record = make_evaluation()
    del record["metadata"]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


# =========================================================
# Invalid metric values
# =========================================================

@pytest.mark.parametrize(
    "invalid_brier",
    [
        -0.1,
        2.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "0.5",
        True,
    ],
)
def test_invalid_brier_score_is_rejected(invalid_brier):
    record = make_evaluation()
    record["metrics"]["brier_score"] = invalid_brier

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


@pytest.mark.parametrize(
    "invalid_log_loss",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "0.5",
        True,
    ],
)
def test_invalid_log_loss_is_rejected(invalid_log_loss):
    record = make_evaluation()
    record["metrics"]["log_loss"] = invalid_log_loss

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        -1,
        0.5,
        2,
        float("nan"),
        float("inf"),
        None,
        "1",
        True,
    ],
)
def test_invalid_directional_accuracy_is_rejected(
    invalid_accuracy,
):
    record = make_evaluation()
    record["metrics"]["directional_accuracy"] = invalid_accuracy

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


# =========================================================
# Invalid directions and confidence
# =========================================================

@pytest.mark.parametrize(
    "invalid_direction",
    [
        "SIDEWAYS",
        "bullish",
        "",
        None,
        123,
    ],
)
def test_invalid_actual_direction_is_rejected(
    invalid_direction,
):
    record = make_evaluation()
    record["actual_outcome"]["direction"] = invalid_direction

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


@pytest.mark.parametrize(
    "invalid_direction",
    [
        "SIDEWAYS",
        "bullish",
        "",
        None,
        123,
    ],
)
def test_invalid_predicted_direction_is_rejected(
    invalid_direction,
):
    record = make_evaluation()
    record["predicted_outcome"][
        "predicted_direction"
    ] = invalid_direction

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


@pytest.mark.parametrize(
    "invalid_confidence",
    [
        -1,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "80",
        True,
    ],
)
def test_invalid_confidence_score_is_rejected(
    invalid_confidence,
):
    record = make_evaluation()
    record["predicted_outcome"][
        "confidence_score"
    ] = invalid_confidence

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


# =========================================================
# Invalid horizon
# =========================================================

@pytest.mark.parametrize(
    "invalid_horizon",
    [
        0,
        -1,
        366,
        7.5,
        True,
        None,
        "7",
    ],
)
def test_invalid_horizon_days_are_rejected(
    invalid_horizon,
):
    record = make_evaluation()
    record["metadata"]["horizon_days"] = invalid_horizon

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


# =========================================================
# Directional consistency
# =========================================================

def test_inconsistent_directional_accuracy_is_rejected():
    record = make_evaluation(
        actual_direction="BULLISH",
        predicted_direction="BEARISH",
        directional_accuracy=1.0,
    )

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_correct_direction_with_zero_accuracy_is_rejected():
    record = make_evaluation(
        actual_direction="BULLISH",
        predicted_direction="BULLISH",
        directional_accuracy=0.0,
    )

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


# =========================================================
# Input immutability
# =========================================================

def test_analysis_does_not_modify_input_records():
    records = [
        make_evaluation(
            forecast_id="BTC-IMMUTABLE-001"
        ),
        make_evaluation(
            forecast_id="BTC-IMMUTABLE-002",
            actual_direction="BEARISH",
            predicted_direction="BEARISH",
        ),
    ]

    original = deepcopy(records)

    analyze_forecast_performance(records)

    assert records == original


def test_analysis_does_not_modify_nested_input_data():
    record = make_evaluation()
    original = deepcopy(record)

    analyze_forecast_performance([record])

    assert record == original


# =========================================================
# Metadata and history
# =========================================================

def test_statistical_calibration_is_not_claimed():
    result = analyze_valid_records()

    assert (
        result["metadata"]["statistical_calibration_assessed"]
        is False
    )


def test_input_records_modified_flag_is_false():
    result = analyze_valid_records()

    assert (
        result["metadata"]["input_records_modified"]
        is False
    )


def test_minimum_group_size_metadata_is_five():
    result = analyze_valid_records()

    assert (
        result["metadata"][
            "minimum_group_size_for_weakness_detection"
        ]
        == 5
    )


# =========================================================
# Module execution
# =========================================================

def test_module_exposes_performance_analyzer():
    assert callable(analyze_forecast_performance)


def test_module_exposes_custom_exception():
    assert issubclass(
        ForecastPerformanceError,
        ValueError,
    )