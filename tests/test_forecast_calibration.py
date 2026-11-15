"""Tests for the forecast calibration analyzer."""

from copy import deepcopy

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
    """Build a valid forecast evaluation record."""

    if probabilities is None:
        probabilities = {
            "BULLISH": 0.60,
            "BEARISH": 0.25,
            "NEUTRAL": 0.15,
        }

    predicted_direction = max(
        probabilities,
        key=lambda direction: (
            probabilities[direction]
            if isinstance(probabilities[direction], (int, float))
            and not isinstance(probabilities[direction], bool)
            else float("-inf")
        ),
    )

    return {
        "status": "SUCCESS",
        "forecast_id": forecast_id,
        "actual_outcome": {
            "direction": actual_direction,
        },
        "predicted_outcome": {
            "probabilities": dict(probabilities),
            "predicted_direction": predicted_direction,
            "confidence_score": 0.60,
        },
        "metadata": {
            "asset": "BTC",
            "horizon_days": 7,
        },
    }


def make_evaluations(count=30):
    """Build multiple valid evaluation records."""

    directions = [
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ]

    evaluations = []

    for index in range(count):
        evaluations.append(
            make_evaluation(
                forecast_id=f"forecast-{index + 1:03d}",
                actual_direction=directions[index % 3],
            )
        )

    return evaluations


def test_default_configuration_constants():
    assert DEFAULT_BIN_COUNT == 10
    assert DEFAULT_MINIMUM_SAMPLE_SIZE == 30


def test_analyzer_returns_success_status():
    result = analyze_forecast_calibration(make_evaluations())

    assert result["status"] == "SUCCESS"


def test_analysis_type_is_correct():
    result = analyze_forecast_calibration(make_evaluations())

    assert result["analysis_type"] == "FORECAST_CALIBRATION"


def test_sample_count_is_correct():
    result = analyze_forecast_calibration(make_evaluations(30))

    assert result["sample_count"] == 30


def test_assessment_status_with_sufficient_sample_size():
    result = analyze_forecast_calibration(make_evaluations(30))

    assert result["assessment_status"] == "ASSESSED"


def test_assessment_status_with_insufficient_sample_size():
    result = analyze_forecast_calibration(make_evaluations(10))

    assert result["assessment_status"] == "INSUFFICIENT_SAMPLE_SIZE"


def test_all_directions_are_present():
    result = analyze_forecast_calibration(make_evaluations())

    assert set(result["by_direction"]) == {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }


def test_macro_average_metrics_are_present():
    result = analyze_forecast_calibration(make_evaluations())

    metrics = result["metrics"]

    assert "expected_calibration_error" in metrics
    assert "maximum_calibration_error" in metrics
    assert metrics["aggregation_method"] == "MACRO_AVERAGE"


def test_expected_calibration_error_is_valid():
    result = analyze_forecast_calibration(make_evaluations())

    ece = result["metrics"]["expected_calibration_error"]

    assert isinstance(ece, (int, float))
    assert 0.0 <= ece <= 1.0


def test_maximum_calibration_error_is_valid():
    result = analyze_forecast_calibration(make_evaluations())

    mce = result["metrics"]["maximum_calibration_error"]

    assert isinstance(mce, (int, float))
    assert 0.0 <= mce <= 1.0


def test_reliability_bins_are_created():
    result = analyze_forecast_calibration(
        make_evaluations(),
        bin_count=10,
    )

    for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
        bins = result["by_direction"][direction]["reliability_bins"]
        assert len(bins) == 10


def test_custom_bin_count_is_supported():
    result = analyze_forecast_calibration(
        make_evaluations(),
        bin_count=5,
    )

    for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
        bins = result["by_direction"][direction]["reliability_bins"]
        assert len(bins) == 5


def test_probability_scale_is_zero_to_one():
    result = analyze_forecast_calibration(make_evaluations())

    assert result["metadata"]["probability_scale"] == "0_TO_1"


def test_metadata_identifies_btc():
    result = analyze_forecast_calibration(make_evaluations())

    assert result["metadata"]["asset"] == "BTC"


def test_statistical_calibration_is_not_claimed():
    result = analyze_forecast_calibration(make_evaluations())

    assert (
        result["metadata"]["statistical_calibration_claimed"]
        is False
    )


def test_forecast_ids_are_sorted():
    evaluations = [
        make_evaluation(forecast_id="forecast-003"),
        make_evaluation(forecast_id="forecast-001"),
        make_evaluation(forecast_id="forecast-002"),
    ]

    result = analyze_forecast_calibration(evaluations)

    assert result["forecast_ids"] == [
        "forecast-001",
        "forecast-002",
        "forecast-003",
    ]


def test_probability_values_are_normalized():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.70,
            "BEARISH": 0.20,
            "NEUTRAL": 0.10,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "0.5",
        None,
    ],
)
def test_invalid_probability_values_are_rejected(invalid_probability):
    probabilities = {
        "BULLISH": 0.60,
        "BEARISH": 0.25,
        "NEUTRAL": 0.15,
    }

    probabilities["BULLISH"] = invalid_probability

    evaluation = make_evaluation(
        probabilities=probabilities,
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_probabilities_must_total_one():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.60,
            "BEARISH": 0.20,
            "NEUTRAL": 0.10,
        }
    )

    with pytest.raises(
        ForecastCalibrationError,
        match="Probabilities must total 1.0",
    ):
        analyze_forecast_calibration([evaluation])


def test_missing_probability_class_is_rejected():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.70,
            "BEARISH": 0.30,
        }
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_extra_probability_class_is_rejected():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.60,
            "BEARISH": 0.20,
            "NEUTRAL": 0.10,
            "UNKNOWN": 0.10,
        }
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_conflicting_predicted_direction_is_rejected():
    evaluation = make_evaluation()
    evaluation["predicted_outcome"]["predicted_direction"] = "BEARISH"

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


@pytest.mark.parametrize(
    "actual_direction",
    [
        "UNKNOWN",
        "bullish",
        "",
        None,
    ],
)
def test_invalid_actual_direction_is_rejected(actual_direction):
    evaluation = make_evaluation(
        actual_direction=actual_direction,
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_non_success_evaluation_is_rejected():
    evaluation = make_evaluation()
    evaluation["status"] = "FAILED"

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_missing_forecast_id_is_rejected():
    evaluation = make_evaluation()
    evaluation["forecast_id"] = ""

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_duplicate_forecast_ids_are_rejected():
    evaluations = [
        make_evaluation(forecast_id="duplicate-id"),
        make_evaluation(forecast_id="duplicate-id"),
    ]

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(evaluations)


def test_missing_actual_outcome_is_rejected():
    evaluation = make_evaluation()
    del evaluation["actual_outcome"]

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_missing_predicted_outcome_is_rejected():
    evaluation = make_evaluation()
    del evaluation["predicted_outcome"]

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_missing_metadata_is_rejected():
    evaluation = make_evaluation()
    del evaluation["metadata"]

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_empty_evaluations_are_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([])


@pytest.mark.parametrize(
    "invalid_bin_count",
    [0, 1, 1.5, True, "10", 101],
)
def test_invalid_bin_count_is_rejected(invalid_bin_count):
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            make_evaluations(),
            bin_count=invalid_bin_count,
        )


@pytest.mark.parametrize(
    "invalid_size",
    [0, -1, 1.5, True, "30", 1000001],
)
def test_invalid_minimum_sample_size_is_rejected(invalid_size):
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            make_evaluations(),
            minimum_sample_size=invalid_size,
        )


def test_zero_and_one_probabilities_are_supported():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 1.0,
            "BEARISH": 0.0,
            "NEUTRAL": 0.0,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_probability_one_is_included_in_last_bin():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 1.0,
            "BEARISH": 0.0,
            "NEUTRAL": 0.0,
        }
    )

    result = analyze_forecast_calibration(
        [evaluation],
        bin_count=10,
    )

    bullish_bins = result["by_direction"]["BULLISH"]["reliability_bins"]

    assert len(bullish_bins) == 10
    assert bullish_bins[-1]["sample_count"] == 1


def test_tied_maximum_probabilities_are_supported():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.40,
            "BEARISH": 0.40,
            "NEUTRAL": 0.20,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_generator_input_is_supported():
    evaluations = (
        make_evaluation(forecast_id=f"forecast-{index}")
        for index in range(30)
    )

    result = analyze_forecast_calibration(evaluations)

    assert result["sample_count"] == 30


def test_input_evaluations_are_not_modified():
    evaluations = make_evaluations()
    original = deepcopy(evaluations)

    analyze_forecast_calibration(evaluations)

    assert evaluations == original


def test_macro_average_ece_is_between_zero_and_one():
    result = analyze_forecast_calibration(make_evaluations())

    ece = result["metrics"]["expected_calibration_error"]

    assert 0.0 <= ece <= 1.0


def test_class_metrics_are_available_for_all_directions():
    result = analyze_forecast_calibration(make_evaluations())

    for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
        direction_result = result["by_direction"][direction]

        assert "expected_calibration_error" in direction_result
        assert "maximum_calibration_error" in direction_result
        assert "reliability_bins" in direction_result


def test_class_actual_frequencies_sum_to_one():
    result = analyze_forecast_calibration(make_evaluations())

    total = sum(
        result["by_direction"][direction]["actual_frequency"]
        for direction in ("BULLISH", "BEARISH", "NEUTRAL")
    )

    assert total == pytest.approx(1.0)


def test_class_mean_probabilities_sum_to_one():
    result = analyze_forecast_calibration(make_evaluations())

    total = sum(
        result["by_direction"][direction]["mean_predicted_probability"]
        for direction in ("BULLISH", "BEARISH", "NEUTRAL")
    )

    assert total == pytest.approx(1.0)


def test_non_dictionary_record_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([None])


def test_dictionary_input_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(make_evaluation())


def test_string_input_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration("invalid")