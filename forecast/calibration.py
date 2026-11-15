"""Bitcoin forecast probability calibration analyzer."""

from __future__ import annotations

import math

from typing import Any, Iterable


VALID_DIRECTIONS = {
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
}

DEFAULT_BIN_COUNT = 10
DEFAULT_MINIMUM_SAMPLE_SIZE = 30


class ForecastCalibrationError(ValueError):
    """Raised when forecast calibration analysis fails."""


def _validate_integer(
    value: Any,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Validate an integer configuration value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ForecastCalibrationError(
            f"{field_name} must be an integer between "
            f"{minimum} and {maximum}."
        )

    return value


def _validate_probability(
    value: Any,
    field_name: str,
) -> float:
    """Validate a probability expressed as a percentage."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 100
    ):
        raise ForecastCalibrationError(
            f"{field_name} must be between 0 and 100."
        )

    return float(value) / 100.0


def _validate_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one forecast evaluation."""

    if not isinstance(record, dict):
        raise ForecastCalibrationError(
            "Every evaluation must be a dictionary."
        )

    if record.get("status") != "SUCCESS":
        raise ForecastCalibrationError(
            "Only successful evaluations can be calibrated."
        )

    forecast_id = record.get("forecast_id")

    if (
        not isinstance(forecast_id, str)
        or not forecast_id.strip()
    ):
        raise ForecastCalibrationError(
            "Evaluation has an invalid forecast_id."
        )

    actual = record.get("actual_outcome")
    predicted = record.get("predicted_outcome")
    metadata = record.get("metadata")

    if not isinstance(actual, dict):
        raise ForecastCalibrationError(
            "actual_outcome must be a dictionary."
        )

    if not isinstance(predicted, dict):
        raise ForecastCalibrationError(
            "predicted_outcome must be a dictionary."
        )

    if not isinstance(metadata, dict):
        raise ForecastCalibrationError(
            "metadata must be a dictionary."
        )

    actual_direction = actual.get("direction")

    if actual_direction not in VALID_DIRECTIONS:
        raise ForecastCalibrationError(
            "Evaluation contains an invalid actual direction."
        )

    probabilities = predicted.get("probabilities")

    if not isinstance(probabilities, dict):
        raise ForecastCalibrationError(
            "Predicted probabilities must be a dictionary."
        )

    if set(probabilities) != VALID_DIRECTIONS:
        raise ForecastCalibrationError(
            "All three market probabilities are required."
        )

    normalized_probabilities = {
        direction: _validate_probability(
            probabilities[direction],
            f"probabilities.{direction}",
        )
        for direction in VALID_DIRECTIONS
    }

    probability_total = sum(
        normalized_probabilities.values()
    )

    if not math.isclose(
        probability_total,
        1.0,
        rel_tol=0.0,
        abs_tol=0.0001,
    ):
        raise ForecastCalibrationError(
            "Probabilities must total 100 percent."
        )

    predicted_direction = predicted.get(
        "predicted_direction"
    )

    if predicted_direction not in VALID_DIRECTIONS:
        raise ForecastCalibrationError(
            "Evaluation contains an invalid predicted direction."
        )

    if predicted_direction != max(
        VALID_DIRECTIONS,
        key=lambda direction: (
            normalized_probabilities[direction],
            direction == "NEUTRAL",
        ),
    ):
        raise ForecastCalibrationError(
            "Predicted direction conflicts with probabilities."
        )

    return {
        "forecast_id": forecast_id,
        "actual_direction": actual_direction,
        "predicted_direction": predicted_direction,
        "probabilities": normalized_probabilities,
    }


def _build_probability_bins(
    records: list[dict[str, Any]],
    direction: str,
    bin_count: int,
) -> list[dict[str, Any]]:
    """Build reliability bins for one market direction."""

    bins = []

    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count

        selected = []

        for record in records:
            probability = record["probabilities"][direction]

            is_last_bin = index == bin_count - 1

            if (
                lower <= probability < upper
                or (
                    is_last_bin
                    and probability == 1.0
                )
            ):
                selected.append(record)

        count = len(selected)

        if count == 0:
            bins.append({
                "bin_index": index,
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "sample_count": 0,
                "mean_predicted_probability": None,
                "observed_frequency": None,
                "calibration_error": None,
            })
            continue

        mean_probability = sum(
            record["probabilities"][direction]
            for record in selected
        ) / count

        observed_frequency = sum(
            record["actual_direction"] == direction
            for record in selected
        ) / count

        calibration_error = abs(
            mean_probability - observed_frequency
        )

        bins.append({
            "bin_index": index,
            "lower_bound": round(lower, 6),
            "upper_bound": round(upper, 6),
            "sample_count": count,
            "mean_predicted_probability": round(
                mean_probability,
                10,
            ),
            "observed_frequency": round(
                observed_frequency,
                10,
            ),
            "calibration_error": round(
                calibration_error,
                10,
            ),
        })

    return bins


def _calculate_class_metrics(
    bins: list[dict[str, Any]],
    sample_count: int,
) -> dict[str, Any]:
    """Calculate calibration metrics for one class."""

    if sample_count == 0:
        return {
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
            "occupied_bins": 0,
        }

    weighted_error = 0.0
    maximum_error = 0.0
    occupied_bins = 0

    for item in bins:
        count = item["sample_count"]

        if count == 0:
            continue

        error = item["calibration_error"]

        weighted_error += (
            count / sample_count
        ) * error

        maximum_error = max(
            maximum_error,
            error,
        )

        occupied_bins += 1

    return {
        "expected_calibration_error": round(
            weighted_error,
            10,
        ),
        "maximum_calibration_error": round(
            maximum_error,
            10,
        ),
        "occupied_bins": occupied_bins,
    }


def _analyze_class(
    records: list[dict[str, Any]],
    direction: str,
    bin_count: int,
) -> dict[str, Any]:
    """Analyze calibration for one market direction."""

    bins = _build_probability_bins(
        records,
        direction,
        bin_count,
    )

    metrics = _calculate_class_metrics(
        bins,
        len(records),
    )

    actual_count = sum(
        record["actual_direction"] == direction
        for record in records
    )

    predicted_probability_sum = sum(
        record["probabilities"][direction]
        for record in records
    )

    return {
        "direction": direction,
        "sample_count": len(records),
        "actual_frequency": round(
            actual_count / len(records),
            10,
        ),
        "mean_predicted_probability": round(
            predicted_probability_sum / len(records),
            10,
        ),
        "expected_calibration_error": (
            metrics["expected_calibration_error"]
        ),
        "maximum_calibration_error": (
            metrics["maximum_calibration_error"]
        ),
        "occupied_bins": metrics["occupied_bins"],
        "reliability_bins": bins,
    }


def _calculate_overall_metrics(
    class_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate macro-average calibration metrics."""

    ece_values = [
        result["expected_calibration_error"]
        for result in class_results.values()
        if result["expected_calibration_error"] is not None
    ]

    mce_values = [
        result["maximum_calibration_error"]
        for result in class_results.values()
        if result["maximum_calibration_error"] is not None
    ]

    if not ece_values:
        return {
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        }

    return {
        "expected_calibration_error": round(
            sum(ece_values) / len(ece_values),
            10,
        ),
        "maximum_calibration_error": round(
            max(mce_values),
            10,
        ),
    }


def analyze_forecast_calibration(
    evaluations: Iterable[dict[str, Any]],
    bin_count: int = DEFAULT_BIN_COUNT,
    minimum_sample_size: int = DEFAULT_MINIMUM_SAMPLE_SIZE,
) -> dict[str, Any]:
    """
    Analyze multiclass forecast probability calibration.

    Probabilities are interpreted as percentages in the
    evaluation records and normalized to the range 0..1.

    The analyzer calculates one-vs-rest reliability bins
    for BULLISH, BEARISH, and NEUTRAL outcomes.

    Expected calibration error (ECE) is calculated as the
    sample-weighted absolute difference between predicted
    probability and observed outcome frequency.

    Overall ECE is the macro-average of the three class ECEs.

    This module does not modify input records, retrain models,
    or claim that a model is statistically calibrated.
    """

    bin_count = _validate_integer(
        bin_count,
        "bin_count",
        2,
        100,
    )

    minimum_sample_size = _validate_integer(
        minimum_sample_size,
        "minimum_sample_size",
        1,
        1000000,
    )

    if isinstance(
        evaluations,
        (str, bytes, dict),
    ):
        raise ForecastCalibrationError(
            "evaluations must be an iterable of evaluation records."
        )

    try:
        source_records = list(evaluations)
    except TypeError as exc:
        raise ForecastCalibrationError(
            "evaluations must be iterable."
        ) from exc

    if not source_records:
        raise ForecastCalibrationError(
            "At least one evaluation record is required."
        )

    records = []
    seen_ids = set()

    for source_record in source_records:
        record = _validate_record(
            source_record
        )

        forecast_id = record["forecast_id"]

        if forecast_id in seen_ids:
            raise ForecastCalibrationError(
                "Duplicate forecast evaluations are not allowed."
            )

        seen_ids.add(forecast_id)
        records.append(record)

    class_results = {
        direction: _analyze_class(
            records,
            direction,
            bin_count,
        )
        for direction in sorted(VALID_DIRECTIONS)
    }

    overall_metrics = _calculate_overall_metrics(
        class_results
    )

    sample_count = len(records)

    if sample_count < minimum_sample_size:
        assessment_status = "INSUFFICIENT_SAMPLE_SIZE"
    else:
        assessment_status = "ASSESSED"

    return {
        "status": "SUCCESS",
        "analysis_type": "FORECAST_CALIBRATION",
        "assessment_status": assessment_status,
        "sample_count": sample_count,
        "metrics": {
            "expected_calibration_error": (
                overall_metrics["expected_calibration_error"]
            ),
            "maximum_calibration_error": (
                overall_metrics["maximum_calibration_error"]
            ),
            "aggregation_method": "MACRO_AVERAGE",
        },
        "by_direction": class_results,
        "metadata": {
            "asset": "BTC",
            "method": "multiclass_one_vs_rest_reliability",
            "bin_count": bin_count,
            "minimum_sample_size": minimum_sample_size,
            "input_records_modified": False,
            "statistical_calibration_claimed": False,
            "probability_scale": "0_TO_1",
        },
        "forecast_ids": sorted(seen_ids),
    }


if __name__ == "__main__":
    print("Bitcoin Forecast Calibration Analyzer loaded.")