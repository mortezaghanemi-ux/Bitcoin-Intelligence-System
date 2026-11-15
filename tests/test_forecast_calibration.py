"""Tests for the forecast calibration analyzer."""

import copy

import pytest

from forecast.calibration import (
    DEFAULT_BIN_COUNT,
    DEFAULT_MINIMUM_SAMPLE_SIZE,
    ForecastCalibrationError,
    analyze_forecast_calibration,
)


def make_evaluation(
    forecast_id="forecast-001",
    actual_direction="BULLISH",
    probabilities=None,
):
    if probabilities is None:
        probabilities = {
            "BULLISH": 70.0,
            "BEARISH": 20.0,
            "NEUTRAL": 10.0,
        }

    predicted_direction = max(
        probabilities,
        key=lambda direction: (
            probabilities[direction],
            direction == "NEUTRAL",
        ),
    )

    return {
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "actual_outcome": {
            "direction": actual_direction,
        },
        "predicted_outcome": {
            "probabilities": probabilities,
            "predicted_direction": predicted_direction,
            "confidence_score": 70.0,
        },
        "metadata": {
            "asset": "BTC",
            "horizon_days": 7,
        },
    }


def make_evaluations(count=30):
    records = []

    directions = [
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ]

    for index in range(count):
        direction = directions[index % 3]

        records.append(
            make_evaluation(
                forecast_id=f"forecast-{index:03d}",
                actual_direction=direction,
                probabilities={
                    "BULLISH": 60.0,
                    "BEARISH": 25.0,
                    "NEUTRAL": 15.0,
                },
            )
        )

    return records


def test_analyze_forecast_calibration_returns_success():
    records = make_evaluations()

    result = analyze_forecast_calibration(
        records
    )

    assert result["status"] == "SUCCESS"
    assert result["analysis_type"] == "FORECAST_CALIBRATION"
    assert result["sample_count"] == 30


def test_default_configuration_values():
    assert DEFAULT_BIN_COUNT == 10
    assert DEFAULT_MINIMUM_SAMPLE_SIZE == 30


def test_default_assessment_requires_thirty_records():
    result = analyze_forecast_calibration(
        make_evaluations(29)
    )

    assert (
        result["assessment_status"]
        == "INSUFFICIENT_SAMPLE_SIZE"
    )


def test_sufficient_sample_size_is_assessed():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    assert result["assessment_status"] == "ASSESSED"


def test_custom_minimum_sample_size():
    result = analyze_forecast_calibration(
        make_evaluations(5),
        minimum_sample_size=5,
    )

    assert result["assessment_status"] == "ASSESSED"


def test_all_three_directions_are_analyzed():
    result = analyze_forecast_calibration(
        make_evaluations()
    )

    assert set(result["by_direction"]) == {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }


def test_probability_bins_are_created():
    result = analyze_forecast_calibration(
        make_evaluations(),
        bin_count=5,
    )

    for direction in (
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ):
        bins = result["by_direction"][direction][
            "reliability_bins"
        ]

        assert len(bins) == 5


def test_reliability_bin_sample_counts():
    result = analyze_forecast_calibration(
        make_evaluations(30),
        bin_count=10,
    )

    bullish_bins = result["by_direction"]["BULLISH"][
        "reliability_bins"
    ]

    occupied = [
        item
        for item in bullish_bins
        if item["sample_count"] > 0
    ]

    assert len(occupied) == 1
    assert occupied[0]["sample_count"] == 30


def test_reliability_bin_probability_values():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    bullish_bins = result["by_direction"]["BULLISH"][
        "reliability_bins"
    ]

    occupied = [
        item
        for item in bullish_bins
        if item["sample_count"] > 0
    ]

    assert occupied[0]["mean_predicted_probability"] == 0.6
    assert occupied[0]["observed_frequency"] == pytest.approx(
        1 / 3
    )


def test_calibration_error_is_absolute_difference():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    bullish = result["by_direction"]["BULLISH"]

    expected = abs(
        0.6 - (10 / 30)
    )

    assert bullish["expected_calibration_error"] == pytest.approx(
        expected
    )


def test_empty_bins_have_none_metrics():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    bullish_bins = result["by_direction"]["BULLISH"][
        "reliability_bins"
    ]

    empty_bins = [
        item
        for item in bullish_bins
        if item["sample_count"] == 0
    ]

    assert empty_bins

    for item in empty_bins:
        assert item["mean_predicted_probability"] is None
        assert item["observed_frequency"] is None
        assert item["calibration_error"] is None


def test_overall_metrics_use_macro_average():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    class_errors = [
        result["by_direction"][direction][
            "expected_calibration_error"
        ]
        for direction in (
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        )
    ]

    expected = sum(class_errors) / 3

    assert result["metrics"][
        "expected_calibration_error"
    ] == pytest.approx(expected)


def test_class_actual_frequencies_sum_to_one():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    total = sum(
        result["by_direction"][direction]["actual_frequency"]
        for direction in (
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        )
    )

    assert total == pytest.approx(1.0)


def test_class_mean_probabilities_sum_to_one():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    total = sum(
        result["by_direction"][direction][
            "mean_predicted_probability"
        ]
        for direction in (
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        )
    )

    assert total == pytest.approx(1.0)


def test_probability_scale_is_normalized():
    result = analyze_forecast_calibration(
        make_evaluations(30)
    )

    assert result["metadata"]["probability_scale"] == "0_TO_1"

    for direction in (
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ):
        value = result["by_direction"][direction][
            "mean_predicted_probability"
        ]

        assert 0 <= value <= 1


def test_input_records_are_not_modified():
    records = make_evaluations()
    original = copy.deepcopy(records)

    analyze_forecast_calibration(records)

    assert records == original


def test_forecast_ids_are_sorted():
    records = [
        make_evaluation(forecast_id="forecast-c"),
        make_evaluation(forecast_id="forecast-a"),
        make_evaluation(forecast_id="forecast-b"),
    ]

    result = analyze_forecast_calibration(
        records
    )

    assert result["forecast_ids"] == [
        "forecast-a",
        "forecast-b",
        "forecast-c",
    ]


def test_duplicate_forecast_ids_are_rejected():
    records = [
        make_evaluation("duplicate"),
        make_evaluation("duplicate"),
    ]

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(records)


def test_empty_evaluations_are_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([])


def test_dictionary_input_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            make_evaluation()
        )


def test_string_input_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration("invalid")


@pytest.mark.parametrize(
    "bin_count",
    [
        0,
        1,
        101,
        -1,
        2.5,
        True,
        None,
    ],
)
def test_invalid_bin_count_is_rejected(bin_count):
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            make_evaluations(),
            bin_count=bin_count,
        )


@pytest.mark.parametrize(
    "minimum_sample_size",
    [
        0,
        -1,
        1000001,
        2.5,
        True,
        None,
    ],
)
def test_invalid_minimum_sample_size_is_rejected(
    minimum_sample_size,
):
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            make_evaluations(),
            minimum_sample_size=minimum_sample_size,
        )


def test_invalid_evaluation_status_is_rejected():
    record = make_evaluation()
    record["status"] = "FAILED"

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_invalid_forecast_id_is_rejected():
    record = make_evaluation()
    record["forecast_id"] = ""

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_missing_actual_outcome_is_rejected():
    record = make_evaluation()
    record["actual_outcome"] = None

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_invalid_actual_direction_is_rejected():
    record = make_evaluation()
    record["actual_outcome"]["direction"] = "UNKNOWN"

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_missing_predicted_outcome_is_rejected():
    record = make_evaluation()
    record["predicted_outcome"] = None

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_missing_metadata_is_rejected():
    record = make_evaluation()
    record["metadata"] = None

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_missing_probabilities_are_rejected():
    record = make_evaluation()
    record["predicted_outcome"]["probabilities"] = None

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_missing_probability_class_is_rejected():
    record = make_evaluation()
    record["predicted_outcome"]["probabilities"] = {
        "BULLISH": 70.0,
        "BEARISH": 30.0,
    }

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


@pytest.mark.parametrize(
    "value",
    [
        -1,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "70",
        None,
    ],
)
def test_invalid_probability_is_rejected(value):
    record = make_evaluation()

    record["predicted_outcome"]["probabilities"][
        "BULLISH"
    ] = value

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_probabilities_not_totaling_one_hundred_are_rejected():
    record = make_evaluation(
        probabilities={
            "BULLISH": 60.0,
            "BEARISH": 20.0,
            "NEUTRAL": 10.0,
        }
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_invalid_predicted_direction_is_rejected():
    record = make_evaluation()
    record["predicted_outcome"]["predicted_direction"] = "UNKNOWN"

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_predicted_direction_conflicting_with_probabilities_is_rejected():
    record = make_evaluation()

    record["predicted_outcome"]["predicted_direction"] = "BEARISH"

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([record])


def test_zero_probability_is_supported():
    record = make_evaluation(
        probabilities={
            "BULLISH": 100.0,
            "BEARISH": 0.0,
            "NEUTRAL": 0.0,
        }
    )

    result = analyze_forecast_calibration(
        [record],
        minimum_sample_size=1,
    )

    assert result["status"] == "SUCCESS"


def test_one_hundred_percent_probability_is_in_last_bin():
    record = make_evaluation(
        probabilities={
            "BULLISH": 100.0,
            "BEARISH": 0.0,
            "NEUTRAL": 0.0,
        }
    )

    result = analyze_forecast_calibration(
        [record],
        minimum_sample_size=1,
    )

    bullish_bins = result["by_direction"]["BULLISH"][
        "reliability_bins"
    ]

    assert bullish_bins[-1]["sample_count"] == 1


def test_single_record_can_be_analyzed():
    result = analyze_forecast_calibration(
        [make_evaluation()],
        minimum_sample_size=1,
    )

    assert result["sample_count"] == 1
    assert result["assessment_status"] == "ASSESSED"


def test_calibration_claim_is_never_made():
    result = analyze_forecast_calibration(
        make_evaluations()
    )

    assert (
        result["metadata"]["statistical_calibration_claimed"]
        is False
    )


def test_metadata_reports_analysis_method():
    result = analyze_forecast_calibration(
        make_evaluations()
    )

    assert (
        result["metadata"]["method"]
        == "multiclass_one_vs_rest_reliability"
    )


def test_non_dictionary_record_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            [None]
        )