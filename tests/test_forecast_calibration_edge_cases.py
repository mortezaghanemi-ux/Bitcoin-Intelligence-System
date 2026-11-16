"""Advanced edge-case tests for forecast calibration."""

import math
from copy import deepcopy

import pytest

from forecast.calibration import (
    DEFAULT_BIN_COUNT,
    DEFAULT_MINIMUM_SAMPLE_SIZE,
    ForecastCalibrationError,
    VALID_DIRECTIONS,
    _analyze_class,
    _build_probability_bins,
    _calculate_class_metrics,
    _calculate_overall_metrics,
    _validate_direction,
    _validate_integer,
    _validate_probability,
    analyze_forecast_calibration,
)


_DEFAULT_METADATA = object()


def make_evaluation(
    forecast_id="forecast-001",
    actual_direction="BULLISH",
    probabilities=None,
    status="SUCCESS",
    metadata=_DEFAULT_METADATA,
):
    """Create a valid forecast evaluation."""

    if probabilities is None:
        probabilities = {
            "BULLISH": 0.60,
            "BEARISH": 0.25,
            "NEUTRAL": 0.15,
        }

    predicted_direction = max(
        probabilities,
        key=lambda direction: probabilities[direction],
    )

    if metadata is _DEFAULT_METADATA:
        metadata = {
            "asset": "BTC",
            "horizon_days": 7,
        }

    return {
        "status": status,
        "forecast_id": forecast_id,
        "actual_outcome": {
            "direction": actual_direction,
        },
        "predicted_outcome": {
            "probabilities": dict(probabilities),
            "predicted_direction": predicted_direction,
        },
        "metadata": metadata,
    }


def make_evaluations(count=30):
    """Create multiple unique valid evaluations."""

    directions = [
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ]

    return [
        make_evaluation(
            forecast_id=f"forecast-{index:04d}",
            actual_direction=directions[index % 3],
        )
        for index in range(count)
    ]


# ---------------------------------------------------------
# Probability validation edge cases
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        0.0,
        1.0,
        0.5,
        0.000001,
        0.999999,
    ],
)
def test_probability_accepts_valid_boundary_values(value):
    result = _validate_probability(value, "probability")

    assert result == float(value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "0.5",
        "",
        [],
        {},
        -0.000001,
        1.000001,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_probability_rejects_invalid_boundary_values(value):
    with pytest.raises(ForecastCalibrationError):
        _validate_probability(value, "probability")


def test_probability_accepts_large_finite_integer_within_range():
    result = _validate_probability(
        1,
        "probability",
    )

    assert result == 1.0


def test_probability_rejects_extremely_large_integer():
    with pytest.raises(ForecastCalibrationError):
        _validate_probability(
            10**10000,
            "probability",
        )


def test_probability_rejects_overflowing_float_conversion():
    class OverflowingNumber:
        def __float__(self):
            raise OverflowError("conversion overflow")

    with pytest.raises(ForecastCalibrationError):
        _validate_probability(
            OverflowingNumber(),
            "probability",
        )


# ---------------------------------------------------------
# Integer validation edge cases
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        2,
        10,
        100,
    ],
)
def test_integer_validator_accepts_valid_values(value):
    result = _validate_integer(
        value,
        "bin_count",
        2,
        100,
    )

    assert result == value


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        1,
        101,
        1.5,
        "10",
        None,
    ],
)
def test_integer_validator_rejects_invalid_values(value):
    with pytest.raises(ForecastCalibrationError):
        _validate_integer(
            value,
            "bin_count",
            2,
            100,
        )


def test_default_configuration_values_are_stable():
    assert DEFAULT_BIN_COUNT == 10
    assert DEFAULT_MINIMUM_SAMPLE_SIZE == 30


# ---------------------------------------------------------
# Direction validation edge cases
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "direction",
    [
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ],
)
def test_direction_validator_accepts_valid_directions(direction):
    assert _validate_direction(
        direction,
        "direction",
    ) == direction


@pytest.mark.parametrize(
    "direction",
    [
        "",
        "bullish",
        "Bearish",
        "NEUTRAL ",
        "UNKNOWN",
        None,
        1,
        True,
        [],
        {},
    ],
)
def test_direction_validator_rejects_invalid_directions(direction):
    with pytest.raises(ForecastCalibrationError):
        _validate_direction(
            direction,
            "direction",
        )


def test_valid_direction_set_contains_exactly_three_classes():
    assert VALID_DIRECTIONS == {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }


# ---------------------------------------------------------
# Evaluation record validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_status",
    [
        None,
        "",
        "FAILED",
        "SUCCESSFUL",
        "success",
        "PENDING",
        1,
    ],
)
def test_invalid_evaluation_status_is_rejected(invalid_status):
    evaluation = make_evaluation(status=invalid_status)

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


@pytest.mark.parametrize(
    "invalid_id",
    [
        None,
        "",
        "   ",
        123,
        [],
        {},
    ],
)
def test_invalid_forecast_ids_are_rejected(invalid_id):
    evaluation = make_evaluation()
    evaluation["forecast_id"] = invalid_id

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


def test_forecast_id_with_surrounding_spaces_is_preserved():
    evaluation = make_evaluation(
        forecast_id=" forecast-001 ",
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["forecast_ids"] == [
        " forecast-001 ",
    ]


@pytest.mark.parametrize(
    "invalid_actual",
    [
        None,
        [],
        "BULLISH",
        123,
    ],
)
def test_invalid_actual_outcome_container_is_rejected(invalid_actual):
    evaluation = make_evaluation()
    evaluation["actual_outcome"] = invalid_actual

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


@pytest.mark.parametrize(
    "invalid_predicted",
    [
        None,
        [],
        "BULLISH",
        123,
    ],
)
def test_invalid_predicted_outcome_container_is_rejected(
    invalid_predicted,
):
    evaluation = make_evaluation()
    evaluation["predicted_outcome"] = invalid_predicted

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        None,
        [],
        "metadata",
        123,
    ],
)
def test_invalid_metadata_container_is_rejected(invalid_metadata):
    evaluation = make_evaluation(
        metadata=invalid_metadata,
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


@pytest.mark.parametrize(
    "invalid_probabilities",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_invalid_probability_container_is_rejected(
    invalid_probabilities,
):
    evaluation = make_evaluation()
    evaluation["predicted_outcome"]["probabilities"] = (
        invalid_probabilities
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration([evaluation])


# ---------------------------------------------------------
# Probability distribution edge cases
# ---------------------------------------------------------


def test_probability_total_exactly_one_is_accepted():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.5,
            "BEARISH": 0.3,
            "NEUTRAL": 0.2,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_probability_total_within_tolerance_is_accepted():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.60005,
            "BEARISH": 0.25,
            "NEUTRAL": 0.15,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_probability_total_outside_tolerance_is_rejected():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.6002,
            "BEARISH": 0.25,
            "NEUTRAL": 0.15,
        }
    )

    with pytest.raises(
        ForecastCalibrationError,
        match="Probabilities must total 1.0",
    ):
        analyze_forecast_calibration([evaluation])


def test_all_probability_mass_on_bullish_is_supported():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 1.0,
            "BEARISH": 0.0,
            "NEUTRAL": 0.0,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_all_probability_mass_on_bearish_is_supported():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.0,
            "BEARISH": 1.0,
            "NEUTRAL": 0.0,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_all_probability_mass_on_neutral_is_supported():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.0,
            "BEARISH": 0.0,
            "NEUTRAL": 1.0,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_near_equal_probability_values_are_supported():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.33334,
            "BEARISH": 0.33333,
            "NEUTRAL": 0.33333,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_probability_direction_tie_is_accepted():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.4,
            "BEARISH": 0.4,
            "NEUTRAL": 0.2,
        }
    )

    result = analyze_forecast_calibration([evaluation])

    assert result["status"] == "SUCCESS"


def test_predicted_direction_with_non_maximum_probability_is_rejected():
    evaluation = make_evaluation(
        probabilities={
            "BULLISH": 0.6,
            "BEARISH": 0.25,
            "NEUTRAL": 0.15,
        }
    )

    evaluation["predicted_outcome"]["predicted_direction"] = "NEUTRAL"

    with pytest.raises(
        ForecastCalibrationError,
        match="Predicted direction conflicts with probabilities",
    ):
        analyze_forecast_calibration([evaluation])


# ---------------------------------------------------------
# Reliability bin edge cases
# ---------------------------------------------------------


def test_bin_count_two_is_supported():
    result = analyze_forecast_calibration(
        make_evaluations(),
        bin_count=2,
    )

    for direction in VALID_DIRECTIONS:
        assert len(
            result["by_direction"][direction]["reliability_bins"]
        ) == 2


def test_bin_count_one_hundred_is_supported():
    result = analyze_forecast_calibration(
        make_evaluations(),
        bin_count=100,
    )

    for direction in VALID_DIRECTIONS:
        assert len(
            result["by_direction"][direction]["reliability_bins"]
        ) == 100


def test_zero_probability_is_included_in_first_bin():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.0,
                "BEARISH": 0.5,
                "NEUTRAL": 0.5,
            },
        }
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        10,
    )

    assert bins[0]["sample_count"] == 1
    assert bins[-1]["sample_count"] == 0


def test_probability_one_is_included_in_last_bin():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 1.0,
                "BEARISH": 0.0,
                "NEUTRAL": 0.0,
            },
        }
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        10,
    )

    assert bins[-1]["sample_count"] == 1
    assert bins[0]["sample_count"] == 0


def test_probability_at_bin_boundary_is_assigned_to_upper_bin():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.5,
                "BEARISH": 0.3,
                "NEUTRAL": 0.2,
            },
        }
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        2,
    )

    assert bins[0]["sample_count"] == 0
    assert bins[1]["sample_count"] == 1


def test_empty_bins_return_none_metrics():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.6,
                "BEARISH": 0.25,
                "NEUTRAL": 0.15,
            },
        }
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        10,
    )

    empty_bins = [
        item
        for item in bins
        if item["sample_count"] == 0
    ]

    assert empty_bins

    for item in empty_bins:
        assert item["mean_predicted_probability"] is None
        assert item["observed_frequency"] is None
        assert item["calibration_error"] is None


def test_reliability_bins_have_contiguous_indexes():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.6,
                "BEARISH": 0.25,
                "NEUTRAL": 0.15,
            },
        }
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        10,
    )

    assert [
        item["bin_index"]
        for item in bins
    ] == list(range(10))


def test_reliability_bin_bounds_are_valid():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.6,
                "BEARISH": 0.25,
                "NEUTRAL": 0.15,
            },
        }
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        10,
    )

    assert bins[0]["lower_bound"] == 0.0
    assert bins[-1]["upper_bound"] == 1.0

    for item in bins:
        assert item["lower_bound"] < item["upper_bound"]


def test_reliability_bin_sample_counts_sum_to_record_count():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.1,
                "BEARISH": 0.6,
                "NEUTRAL": 0.3,
            },
        },
        {
            "actual_direction": "BEARISH",
            "probabilities": {
                "BULLISH": 0.4,
                "BEARISH": 0.4,
                "NEUTRAL": 0.2,
            },
        },
        {
            "actual_direction": "NEUTRAL",
            "probabilities": {
                "BULLISH": 0.0,
                "BEARISH": 0.0,
                "NEUTRAL": 1.0,
            },
        },
    ]

    bins = _build_probability_bins(
        records,
        "BULLISH",
        10,
    )

    assert sum(
        item["sample_count"]
        for item in bins
    ) == len(records)


# ---------------------------------------------------------
# Calibration metric edge cases
# ---------------------------------------------------------


def test_class_metrics_with_zero_sample_count():
    result = _calculate_class_metrics(
        [],
        0,
    )

    assert result == {
        "expected_calibration_error": None,
        "maximum_calibration_error": None,
        "occupied_bins": 0,
    }


def test_class_metrics_with_all_empty_bins():
    bins = [
        {
            "sample_count": 0,
            "calibration_error": None,
        }
        for _ in range(10)
    ]

    result = _calculate_class_metrics(
        bins,
        10,
    )

    assert result["expected_calibration_error"] == 0.0
    assert result["maximum_calibration_error"] == 0.0
    assert result["occupied_bins"] == 0


def test_class_metrics_with_single_occupied_bin():
    bins = [
        {
            "sample_count": 10,
            "calibration_error": 0.2,
        },
        {
            "sample_count": 0,
            "calibration_error": None,
        },
    ]

    result = _calculate_class_metrics(
        bins,
        10,
    )

    assert result["expected_calibration_error"] == 0.2
    assert result["maximum_calibration_error"] == 0.2
    assert result["occupied_bins"] == 1


def test_class_metrics_weight_errors_by_sample_count():
    bins = [
        {
            "sample_count": 3,
            "calibration_error": 0.1,
        },
        {
            "sample_count": 1,
            "calibration_error": 0.5,
        },
    ]

    result = _calculate_class_metrics(
        bins,
        4,
    )

    assert result["expected_calibration_error"] == pytest.approx(0.2)
    assert result["maximum_calibration_error"] == 0.5
    assert result["occupied_bins"] == 2


def test_overall_metrics_with_no_available_classes():
    class_results = {
        "BULLISH": {
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        },
        "BEARISH": {
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        },
        "NEUTRAL": {
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        },
    }

    result = _calculate_overall_metrics(class_results)

    assert result == {
        "expected_calibration_error": None,
        "maximum_calibration_error": None,
    }


def test_overall_metrics_ignore_unavailable_class_values():
    class_results = {
        "BULLISH": {
            "expected_calibration_error": 0.1,
            "maximum_calibration_error": 0.2,
        },
        "BEARISH": {
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        },
        "NEUTRAL": {
            "expected_calibration_error": 0.3,
            "maximum_calibration_error": 0.4,
        },
    }

    result = _calculate_overall_metrics(class_results)

    assert result["expected_calibration_error"] == 0.2
    assert result["maximum_calibration_error"] == 0.4


def test_overall_metrics_use_macro_average():
    class_results = {
        "BULLISH": {
            "expected_calibration_error": 0.1,
            "maximum_calibration_error": 0.2,
        },
        "BEARISH": {
            "expected_calibration_error": 0.2,
            "maximum_calibration_error": 0.3,
        },
        "NEUTRAL": {
            "expected_calibration_error": 0.6,
            "maximum_calibration_error": 0.7,
        },
    }

    result = _calculate_overall_metrics(class_results)

    assert result["expected_calibration_error"] == pytest.approx(0.3)
    assert result["maximum_calibration_error"] == 0.7


# ---------------------------------------------------------
# Analyzer integration edge cases
# ---------------------------------------------------------


def test_single_evaluation_is_supported():
    result = analyze_forecast_calibration(
        [make_evaluation()],
    )

    assert result["sample_count"] == 1
    assert result["assessment_status"] == "INSUFFICIENT_SAMPLE_SIZE"


def test_custom_minimum_sample_size_is_respected():
    result = analyze_forecast_calibration(
        make_evaluations(5),
        minimum_sample_size=5,
    )

    assert result["assessment_status"] == "ASSESSED"


def test_sample_one_below_minimum_is_insufficient():
    result = analyze_forecast_calibration(
        make_evaluations(29),
        minimum_sample_size=30,
    )

    assert result["assessment_status"] == "INSUFFICIENT_SAMPLE_SIZE"


def test_sample_exactly_at_minimum_is_assessed():
    result = analyze_forecast_calibration(
        make_evaluations(30),
        minimum_sample_size=30,
    )

    assert result["assessment_status"] == "ASSESSED"


def test_sample_above_minimum_is_assessed():
    result = analyze_forecast_calibration(
        make_evaluations(31),
        minimum_sample_size=30,
    )

    assert result["assessment_status"] == "ASSESSED"


def test_generator_is_consumed_once():
    evaluations = make_evaluations(5)
    generator = (item for item in evaluations)

    result = analyze_forecast_calibration(generator)

    assert result["sample_count"] == 5


def test_repeated_generator_input_is_not_required():
    evaluations = make_evaluations(3)
    generator = (item for item in evaluations)

    result = analyze_forecast_calibration(generator)

    assert len(result["forecast_ids"]) == 3


def test_input_records_remain_unchanged_after_analysis():
    evaluations = make_evaluations(30)
    original = deepcopy(evaluations)

    analyze_forecast_calibration(evaluations)

    assert evaluations == original


def test_nested_probability_dictionaries_remain_unchanged():
    evaluation = make_evaluation()
    original = deepcopy(evaluation)

    analyze_forecast_calibration([evaluation])

    assert evaluation == original


def test_forecast_ids_are_unique_in_successful_result():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    assert len(result["forecast_ids"]) == len(
        set(result["forecast_ids"])
    )


def test_forecast_ids_are_sorted_deterministically():
    evaluations = [
        make_evaluation(forecast_id="z-last"),
        make_evaluation(forecast_id="a-first"),
        make_evaluation(forecast_id="m-middle"),
    ]

    result = analyze_forecast_calibration(evaluations)

    assert result["forecast_ids"] == [
        "a-first",
        "m-middle",
        "z-last",
    ]


def test_all_class_results_have_expected_metadata():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    for direction in VALID_DIRECTIONS:
        class_result = result["by_direction"][direction]

        assert class_result["direction"] == direction
        assert class_result["sample_count"] == 30
        assert len(class_result["reliability_bins"]) == 10


def test_class_actual_frequencies_sum_to_one():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    total = sum(
        item["actual_frequency"]
        for item in result["by_direction"].values()
    )

    assert total == pytest.approx(1.0)


def test_class_mean_probabilities_sum_to_one():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    total = sum(
        item["mean_predicted_probability"]
        for item in result["by_direction"].values()
    )

    assert total == pytest.approx(1.0)


def test_all_reported_metrics_are_finite():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    for class_result in result["by_direction"].values():
        assert math.isfinite(
            class_result["actual_frequency"]
        )

        assert math.isfinite(
            class_result["mean_predicted_probability"]
        )

        assert math.isfinite(
            class_result["expected_calibration_error"]
        )

        assert math.isfinite(
            class_result["maximum_calibration_error"]
        )

    assert math.isfinite(
        result["metrics"]["expected_calibration_error"]
    )

    assert math.isfinite(
        result["metrics"]["maximum_calibration_error"]
    )


def test_calibration_errors_are_always_between_zero_and_one():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    for class_result in result["by_direction"].values():
        assert 0.0 <= class_result["expected_calibration_error"] <= 1.0
        assert 0.0 <= class_result["maximum_calibration_error"] <= 1.0

    assert 0.0 <= result["metrics"]["expected_calibration_error"] <= 1.0
    assert 0.0 <= result["metrics"]["maximum_calibration_error"] <= 1.0


def test_analysis_metadata_identifies_calibration_method():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    assert result["metadata"]["asset"] == "BTC"

    assert (
        result["metadata"]["method"]
        == "multiclass_one_vs_rest_reliability"
    )

    assert result["metadata"]["probability_scale"] == "0_TO_1"


def test_analysis_does_not_claim_statistical_calibration():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    assert (
        result["metadata"]["statistical_calibration_claimed"]
        is False
    )


def test_analysis_reports_input_records_were_not_modified():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    assert result["metadata"]["input_records_modified"] is False


def test_analysis_result_has_expected_top_level_structure():
    result = analyze_forecast_calibration(
        make_evaluations(30),
    )

    assert set(result) == {
        "status",
        "analysis_type",
        "assessment_status",
        "sample_count",
        "metrics",
        "by_direction",
        "metadata",
        "forecast_ids",
    }


def test_analysis_with_different_bin_counts_preserves_sample_count():
    evaluations = make_evaluations(30)

    result_five = analyze_forecast_calibration(
        evaluations,
        bin_count=5,
    )

    result_twenty = analyze_forecast_calibration(
        evaluations,
        bin_count=20,
    )

    assert result_five["sample_count"] == 30
    assert result_twenty["sample_count"] == 30

    assert result_five["metadata"]["bin_count"] == 5
    assert result_twenty["metadata"]["bin_count"] == 20


def test_analysis_with_one_million_minimum_sample_size_is_valid():
    result = analyze_forecast_calibration(
        make_evaluations(1),
        minimum_sample_size=1000000,
    )

    assert result["assessment_status"] == "INSUFFICIENT_SAMPLE_SIZE"


# ---------------------------------------------------------
# Class analysis edge cases
# ---------------------------------------------------------


def test_analyze_class_returns_expected_direction():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.6,
                "BEARISH": 0.25,
                "NEUTRAL": 0.15,
            },
        },
        {
            "actual_direction": "BEARISH",
            "probabilities": {
                "BULLISH": 0.5,
                "BEARISH": 0.3,
                "NEUTRAL": 0.2,
            },
        },
    ]

    result = _analyze_class(
        records,
        "BULLISH",
        5,
    )

    assert result["direction"] == "BULLISH"
    assert result["sample_count"] == 2


def test_analyze_class_actual_frequency_is_correct():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.6,
                "BEARISH": 0.25,
                "NEUTRAL": 0.15,
            },
        },
        {
            "actual_direction": "BEARISH",
            "probabilities": {
                "BULLISH": 0.5,
                "BEARISH": 0.3,
                "NEUTRAL": 0.2,
            },
        },
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.7,
                "BEARISH": 0.2,
                "NEUTRAL": 0.1,
            },
        },
    ]

    result = _analyze_class(
        records,
        "BULLISH",
        5,
    )

    assert result["actual_frequency"] == pytest.approx(2 / 3)


def test_analyze_class_mean_probability_is_correct():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.6,
                "BEARISH": 0.25,
                "NEUTRAL": 0.15,
            },
        },
        {
            "actual_direction": "BEARISH",
            "probabilities": {
                "BULLISH": 0.4,
                "BEARISH": 0.4,
                "NEUTRAL": 0.2,
            },
        },
    ]

    result = _analyze_class(
        records,
        "BULLISH",
        5,
    )

    assert result["mean_predicted_probability"] == pytest.approx(0.5)


def test_analyze_class_with_no_actual_occurrences():
    records = [
        {
            "actual_direction": "BEARISH",
            "probabilities": {
                "BULLISH": 0.2,
                "BEARISH": 0.5,
                "NEUTRAL": 0.3,
            },
        },
        {
            "actual_direction": "NEUTRAL",
            "probabilities": {
                "BULLISH": 0.1,
                "BEARISH": 0.4,
                "NEUTRAL": 0.5,
            },
        },
    ]

    result = _analyze_class(
        records,
        "BULLISH",
        5,
    )

    assert result["actual_frequency"] == 0.0
    assert result["sample_count"] == 2


def test_analyze_class_with_all_actual_occurrences():
    records = [
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.7,
                "BEARISH": 0.2,
                "NEUTRAL": 0.1,
            },
        },
        {
            "actual_direction": "BULLISH",
            "probabilities": {
                "BULLISH": 0.8,
                "BEARISH": 0.1,
                "NEUTRAL": 0.1,
            },
        },
    ]

    result = _analyze_class(
        records,
        "BULLISH",
        5,
    )

    assert result["actual_frequency"] == 1.0


# ---------------------------------------------------------
# Iterable input edge cases
# ---------------------------------------------------------


def test_tuple_input_is_supported():
    evaluations = tuple(make_evaluations(5))

    result = analyze_forecast_calibration(evaluations)

    assert result["sample_count"] == 5


def test_list_subclass_input_is_supported():
    class EvaluationList(list):
        pass

    evaluations = EvaluationList(make_evaluations(5))

    result = analyze_forecast_calibration(evaluations)

    assert result["sample_count"] == 5


def test_empty_generator_is_rejected():
    evaluations = (item for item in [])

    with pytest.raises(
        ForecastCalibrationError,
        match="At least one evaluation record is required",
    ):
        analyze_forecast_calibration(evaluations)


def test_invalid_record_after_valid_generator_record_is_rejected():
    evaluations = (
        item
        for item in [
            make_evaluation(),
            {"invalid": True},
        ]
    )

    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(evaluations)


def test_nested_dictionary_as_record_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(
            [{"forecast": make_evaluation()}]
        )


def test_bytes_input_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration(b"invalid")


def test_set_input_with_invalid_records_is_rejected():
    with pytest.raises(ForecastCalibrationError):
        analyze_forecast_calibration({1, 2, 3})