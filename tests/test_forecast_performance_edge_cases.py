"""Edge-case tests for the Bitcoin Forecast Performance Analyzer."""

from copy import deepcopy

import pytest

from forecast.performance import (
    ForecastPerformanceError,
    analyze_forecast_performance,
)


def make_evaluation(
    forecast_id="BTC-EDGE-001",
    actual_direction="BULLISH",
    predicted_direction="BULLISH",
    confidence_score=85.0,
    horizon_days=7,
    brier_score=0.25,
    log_loss=0.50,
    directional_accuracy=1.0,
):
    """Create a valid evaluation record."""

    return {
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "actual_outcome": {
            "direction": actual_direction,
        },
        "predicted_outcome": {
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
        },
        "metrics": {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "directional_accuracy": directional_accuracy,
        },
        "metadata": {
            "horizon_days": horizon_days,
        },
    }


def make_records(
    count,
    actual_direction="BEARISH",
    predicted_direction="BULLISH",
    confidence_score=90.0,
    horizon_days=7,
):
    """Create multiple unique evaluation records."""

    return [
        make_evaluation(
            forecast_id=f"BTC-EDGE-{index:03d}",
            actual_direction=actual_direction,
            predicted_direction=predicted_direction,
            confidence_score=confidence_score,
            horizon_days=horizon_days,
            directional_accuracy=float(
                actual_direction == predicted_direction
            ),
        )
        for index in range(count)
    ]


# =========================================================
# Iterable handling
# =========================================================

def test_generator_input_is_supported():
    records = (
        make_evaluation(forecast_id=f"BTC-GEN-{index}")
        for index in range(3)
    )

    result = analyze_forecast_performance(records)

    assert result["status"] == "SUCCESS"
    assert result["evaluation_count"] == 3


def test_tuple_input_is_supported():
    records = (
        make_evaluation(forecast_id="BTC-TUPLE-001"),
        make_evaluation(forecast_id="BTC-TUPLE-002"),
    )

    result = analyze_forecast_performance(records)

    assert result["evaluation_count"] == 2


@pytest.mark.parametrize(
    "invalid_input",
    [
        0,
        1.5,
        True,
        False,
        object(),
    ],
)
def test_non_iterable_input_is_rejected(invalid_input):
    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance(invalid_input)


# =========================================================
# Record validation
# =========================================================

@pytest.mark.parametrize(
    "field",
    [
        "actual_outcome",
        "predicted_outcome",
        "metrics",
        "metadata",
    ],
)
def test_non_dictionary_nested_fields_are_rejected(field):
    record = make_evaluation()
    record[field] = []

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_missing_required_metric_is_rejected():
    record = make_evaluation()
    del record["metrics"]["brier_score"]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


@pytest.mark.parametrize(
    "missing_metric",
    [
        "log_loss",
        "directional_accuracy",
    ],
)
def test_each_missing_metric_is_rejected(missing_metric):
    record = make_evaluation()
    del record["metrics"][missing_metric]

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


def test_forecast_id_with_surrounding_spaces_is_preserved():
    record = make_evaluation(
        forecast_id="  BTC-SPACED-ID  "
    )

    result = analyze_forecast_performance([record])

    assert result["forecast_ids"] == [
        "  BTC-SPACED-ID  "
    ]


def test_duplicate_ids_with_whitespace_difference_are_distinct():
    records = [
        make_evaluation(forecast_id="BTC-ID"),
        make_evaluation(forecast_id=" BTC-ID "),
    ]

    result = analyze_forecast_performance(records)

    assert result["evaluation_count"] == 2


# =========================================================
# Numeric boundaries
# =========================================================

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brier_score", 0.0),
        ("brier_score", 2.0),
        ("log_loss", 0.0),
        ("directional_accuracy", 0.0),
        ("directional_accuracy", 1.0),
    ],
)
def test_numeric_metric_boundaries_are_accepted(field, value):
    record = make_evaluation()

    record["metrics"][field] = value

    if field == "directional_accuracy":
        if value == 0.0:
            record["actual_outcome"]["direction"] = "BEARISH"
            record["predicted_outcome"][
                "predicted_direction"
            ] = "BULLISH"
        else:
            record["actual_outcome"]["direction"] = "BULLISH"
            record["predicted_outcome"][
                "predicted_direction"
            ] = "BULLISH"

    result = analyze_forecast_performance([record])

    assert result["evaluation_count"] == 1


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        None,
        "1",
    ],
)
def test_confidence_rejects_invalid_numeric_values(invalid_value):
    record = make_evaluation()
    record["predicted_outcome"]["confidence_score"] = invalid_value

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])


# =========================================================
# Confidence bucket boundaries
# =========================================================

@pytest.mark.parametrize(
    ("confidence", "expected_bucket"),
    [
        (0.0, "LOW"),
        (49.999, "LOW"),
        (50.0, "MODERATE"),
        (64.999, "MODERATE"),
        (65.0, "HIGH"),
        (79.999, "HIGH"),
        (80.0, "VERY_HIGH"),
        (100.0, "VERY_HIGH"),
    ],
)
def test_confidence_bucket_boundaries(
    confidence,
    expected_bucket,
):
    result = analyze_forecast_performance(
        [
            make_evaluation(
                confidence_score=confidence,
            )
        ]
    )

    assert (
        result["by_confidence"][expected_bucket][
            "evaluation_count"
        ]
        == 1
    )


def test_empty_confidence_buckets_have_none_metrics():
    result = analyze_forecast_performance(
        [
            make_evaluation(
                confidence_score=90.0,
            )
        ]
    )

    for label in ("LOW", "MODERATE", "HIGH"):
        bucket = result["by_confidence"][label]

        assert bucket["evaluation_count"] == 0
        assert bucket["mean_brier_score"] is None
        assert bucket["mean_log_loss"] is None
        assert bucket["directional_accuracy"] is None
        assert bucket["correct_predictions"] == 0
        assert bucket["incorrect_predictions"] == 0


# =========================================================
# Weakness detection
# =========================================================

def test_empty_groups_do_not_create_weaknesses():
    result = analyze_forecast_performance(
        [
            make_evaluation(
                confidence_score=20.0,
                actual_direction="BULLISH",
                predicted_direction="BULLISH",
            )
        ]
    )

    assert result["weaknesses"] == []


def test_low_overall_accuracy_creates_high_severity_weakness():
    records = make_records(
        4,
        actual_direction="BEARISH",
        predicted_direction="BULLISH",
    )

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if item["area"] == "directional_accuracy"
    ]

    assert len(matching) == 1
    assert matching[0]["severity"] == "HIGH"


def test_medium_overall_accuracy_creates_medium_weakness():
    records = [
        make_evaluation(
            forecast_id=f"BTC-MEDIUM-{index}",
            actual_direction="BULLISH",
            predicted_direction=(
                "BULLISH" if index < 5 else "BEARISH"
            ),
            confidence_score=20.0,
            directional_accuracy=(
                1.0 if index < 5 else 0.0
            ),
        )
        for index in range(9)
    ]

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if item["area"] == "directional_accuracy"
    ]

    assert len(matching) == 1
    assert matching[0]["severity"] == "MEDIUM"


def test_accuracy_at_sixty_percent_does_not_create_overall_weakness():
    records = [
        make_evaluation(
            forecast_id=f"BTC-60-{index}",
            actual_direction="BULLISH",
            predicted_direction=(
                "BULLISH" if index < 6 else "BEARISH"
            ),
            confidence_score=20.0,
            directional_accuracy=(
                1.0 if index < 6 else 0.0
            ),
        )
        for index in range(10)
    ]

    result = analyze_forecast_performance(records)

    overall_weaknesses = [
        item
        for item in result["weaknesses"]
        if item["area"] == "directional_accuracy"
    ]

    assert overall_weaknesses == []


def test_high_confidence_bucket_with_low_accuracy_is_detected():
    records = make_records(
        5,
        actual_direction="BEARISH",
        predicted_direction="BULLISH",
        confidence_score=70.0,
    )

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if (
            item["area"] == "confidence"
            and item["segment"] == "HIGH"
        )
    ]

    assert len(matching) == 1
    assert matching[0]["severity"] == "HIGH"


def test_very_high_confidence_bucket_with_medium_accuracy_is_detected():
    records = [
        make_evaluation(
            forecast_id=f"BTC-VHIGH-{index}",
            actual_direction="BULLISH",
            predicted_direction=(
                "BULLISH" if index < 4 else "BEARISH"
            ),
            confidence_score=90.0,
            directional_accuracy=(
                1.0 if index < 4 else 0.0
            ),
        )
        for index in range(7)
    ]

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if (
            item["area"] == "confidence"
            and item["segment"] == "VERY_HIGH"
        )
    ]

    assert len(matching) == 1
    assert matching[0]["severity"] == "MEDIUM"


def test_confidence_bucket_with_fewer_than_five_records_is_ignored():
    records = make_records(
        4,
        actual_direction="BEARISH",
        predicted_direction="BULLISH",
        confidence_score=70.0,
    )

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if item["area"] == "confidence"
    ]

    assert matching == []


def test_weak_horizon_with_at_least_five_records_is_detected():
    records = make_records(
        5,
        actual_direction="BEARISH",
        predicted_direction="BULLISH",
        confidence_score=20.0,
        horizon_days=30,
    )

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if (
            item["area"] == "forecast_horizon"
            and item["segment"] == "30"
        )
    ]

    assert len(matching) == 1
    assert matching[0]["severity"] == "HIGH"


def test_horizon_with_fewer_than_five_records_is_ignored():
    records = make_records(
        4,
        actual_direction="BEARISH",
        predicted_direction="BULLISH",
        confidence_score=20.0,
        horizon_days=30,
    )

    result = analyze_forecast_performance(records)

    matching = [
        item
        for item in result["weaknesses"]
        if item["area"] == "forecast_horizon"
    ]

    assert matching == []


# =========================================================
# Summary and grouping
# =========================================================

def test_summary_rounds_aggregate_metrics():
    records = [
        make_evaluation(
            forecast_id="BTC-ROUND-1",
            brier_score=0.123456789123,
            log_loss=0.987654321987,
            directional_accuracy=1.0,
        ),
        make_evaluation(
            forecast_id="BTC-ROUND-2",
            actual_direction="BEARISH",
            predicted_direction="BULLISH",
            brier_score=0.234567891234,
            log_loss=0.876543219876,
            directional_accuracy=0.0,
        ),
    ]

    result = analyze_forecast_performance(records)

    summary = result["summary"]

    assert summary["mean_brier_score"] == round(
        (0.123456789123 + 0.234567891234) / 2,
        10,
    )

    assert summary["mean_log_loss"] == round(
        (0.987654321987 + 0.876543219876) / 2,
        10,
    )


def test_group_keys_are_sorted():
    records = [
        make_evaluation(
            forecast_id="BTC-H30",
            horizon_days=30,
        ),
        make_evaluation(
            forecast_id="BTC-H7",
            horizon_days=7,
        ),
        make_evaluation(
            forecast_id="BTC-H14",
            horizon_days=14,
        ),
    ]

    result = analyze_forecast_performance(records)

    assert list(result["by_horizon_days"]) == [
        "14",
        "30",
        "7",
    ]


def test_forecast_ids_are_sorted_without_mutating_records():
    records = [
        make_evaluation(forecast_id="BTC-Z"),
        make_evaluation(forecast_id="BTC-A"),
        make_evaluation(forecast_id="BTC-M"),
    ]

    original = deepcopy(records)

    result = analyze_forecast_performance(records)

    assert result["forecast_ids"] == [
        "BTC-A",
        "BTC-M",
        "BTC-Z",
    ]

    assert records == original


# =========================================================
# Input immutability
# =========================================================

def test_nested_input_records_remain_unchanged_after_analysis():
    records = make_records(
        8,
        actual_direction="BEARISH",
        predicted_direction="BULLISH",
    )

    original = deepcopy(records)

    analyze_forecast_performance(records)

    assert records == original


def test_analysis_results_do_not_expose_input_dictionaries():
    record = make_evaluation()

    result = analyze_forecast_performance([record])

    result["metadata"]["asset"] = "CHANGED"
    result["forecast_ids"].append("INJECTED")

    assert record["forecast_id"] == "BTC-EDGE-001"


# =========================================================
# Result contract
# =========================================================

def test_result_contains_all_required_top_level_fields():
    result = analyze_forecast_performance(
        [make_evaluation()]
    )

    expected_fields = {
        "status",
        "analysis_type",
        "evaluation_count",
        "summary",
        "by_horizon_days",
        "by_confidence",
        "by_actual_direction",
        "weaknesses",
        "metadata",
        "forecast_ids",
    }

    assert set(result) == expected_fields


def test_summary_contains_all_required_fields():
    result = analyze_forecast_performance(
        [make_evaluation()]
    )

    expected_fields = {
        "evaluation_count",
        "mean_brier_score",
        "mean_log_loss",
        "directional_accuracy",
        "correct_predictions",
        "incorrect_predictions",
    }

    assert set(result["summary"]) == expected_fields


def test_analysis_metadata_preserves_method_contract():
    result = analyze_forecast_performance(
        [make_evaluation()]
    )

    metadata = result["metadata"]

    assert metadata["asset"] == "BTC"
    assert metadata["method"] == "historical_evaluation_analysis"
    assert metadata["input_records_modified"] is False
    assert metadata["statistical_calibration_assessed"] is False
    assert (
        metadata["minimum_group_size_for_weakness_detection"]
        == 5
    )


def test_invalid_directional_accuracy_consistency_is_rejected():
    record = make_evaluation(
        actual_direction="BULLISH",
        predicted_direction="BEARISH",
        directional_accuracy=1.0,
    )

    with pytest.raises(ForecastPerformanceError):
        analyze_forecast_performance([record])