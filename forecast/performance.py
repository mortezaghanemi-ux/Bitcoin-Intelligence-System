"""Bitcoin forecast performance analysis."""

from __future__ import annotations

import math

from typing import Any, Iterable


VALID_DIRECTIONS = {
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
}

CONFIDENCE_BUCKETS = (
    ("LOW", 0.0, 50.0),
    ("MODERATE", 50.0, 65.0),
    ("HIGH", 65.0, 80.0),
    ("VERY_HIGH", 80.0, 100.01),
)


class ForecastPerformanceError(ValueError):
    """Raised when forecast performance analysis fails."""


def _validate_number(
    value: Any,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ForecastPerformanceError(
            f"{field_name} must be a finite number."
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise ForecastPerformanceError(
            f"{field_name} is below its allowed minimum."
        )

    if maximum is not None and result > maximum:
        raise ForecastPerformanceError(
            f"{field_name} exceeds its allowed maximum."
        )

    return result


def _validate_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate one completed forecast evaluation."""

    if not isinstance(record, dict):
        raise ForecastPerformanceError(
            "Every evaluation must be a dictionary."
        )

    if record.get("status") != "SUCCESS":
        raise ForecastPerformanceError(
            "Only successful evaluations can be analyzed."
        )

    forecast_id = record.get("forecast_id")

    if (
        not isinstance(forecast_id, str)
        or not forecast_id.strip()
    ):
        raise ForecastPerformanceError(
            "Evaluation has an invalid forecast_id."
        )

    metrics = record.get("metrics")
    actual = record.get("actual_outcome")
    predicted = record.get("predicted_outcome")
    metadata = record.get("metadata")

    if not isinstance(metrics, dict):
        raise ForecastPerformanceError(
            "Evaluation metrics must be a dictionary."
        )

    if not isinstance(actual, dict):
        raise ForecastPerformanceError(
            "actual_outcome must be a dictionary."
        )

    if not isinstance(predicted, dict):
        raise ForecastPerformanceError(
            "predicted_outcome must be a dictionary."
        )

    if not isinstance(metadata, dict):
        raise ForecastPerformanceError(
            "Evaluation metadata must be a dictionary."
        )

    required_metrics = (
        "brier_score",
        "log_loss",
        "directional_accuracy",
    )

    if any(
        name not in metrics
        for name in required_metrics
    ):
        raise ForecastPerformanceError(
            "Evaluation is missing required metrics."
        )

    brier_score = _validate_number(
        metrics["brier_score"],
        "brier_score",
        0.0,
        2.0,
    )

    log_loss = _validate_number(
        metrics["log_loss"],
        "log_loss",
        0.0,
    )

    directional_accuracy = _validate_number(
        metrics["directional_accuracy"],
        "directional_accuracy",
        0.0,
        1.0,
    )

    if directional_accuracy not in (0.0, 1.0):
        raise ForecastPerformanceError(
            "directional_accuracy must be either 0 or 1."
        )

    actual_direction = actual.get("direction")

    if actual_direction not in VALID_DIRECTIONS:
        raise ForecastPerformanceError(
            "Evaluation contains an invalid actual direction."
        )

    predicted_direction = predicted.get(
        "predicted_direction"
    )

    if predicted_direction not in VALID_DIRECTIONS:
        raise ForecastPerformanceError(
            "Evaluation contains an invalid predicted direction."
        )

    confidence = _validate_number(
        predicted.get("confidence_score"),
        "confidence_score",
        0.0,
        100.0,
    )

    horizon_days = metadata.get("horizon_days")

    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or horizon_days <= 0
        or horizon_days > 365
    ):
        raise ForecastPerformanceError(
            "horizon_days must be an integer between 1 and 365."
        )

    if directional_accuracy != float(
        predicted_direction == actual_direction
    ):
        raise ForecastPerformanceError(
            "Directional accuracy conflicts with prediction directions."
        )

    return {
        "forecast_id": forecast_id,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "directional_accuracy": directional_accuracy,
        "actual_direction": actual_direction,
        "predicted_direction": predicted_direction,
        "confidence_score": confidence,
        "horizon_days": horizon_days,
    }


def _calculate_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aggregate performance metrics."""

    count = len(records)

    if count == 0:
        return {
            "evaluation_count": 0,
            "mean_brier_score": None,
            "mean_log_loss": None,
            "directional_accuracy": None,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
        }

    correct = sum(
        record["directional_accuracy"]
        for record in records
    )

    return {
        "evaluation_count": count,
        "mean_brier_score": round(
            sum(
                record["brier_score"]
                for record in records
            ) / count,
            10,
        ),
        "mean_log_loss": round(
            sum(
                record["log_loss"]
                for record in records
            ) / count,
            10,
        ),
        "directional_accuracy": round(
            correct / count,
            10,
        ),
        "correct_predictions": int(correct),
        "incorrect_predictions": count - int(correct),
    }


def _group_by(
    records: list[dict[str, Any]],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    """Calculate performance metrics for each group."""

    groups: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        key = str(record[field_name])

        groups.setdefault(
            key,
            [],
        ).append(record)

    return {
        key: _calculate_summary(groups[key])
        for key in sorted(groups)
    }


def _analyze_confidence(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Analyze prediction quality across confidence ranges."""

    result = {}

    for label, lower, upper in CONFIDENCE_BUCKETS:

        selected = [
            record
            for record in records
            if lower <= record["confidence_score"] < upper
        ]

        result[label] = _calculate_summary(
            selected
        )

        result[label]["confidence_range"] = {
            "minimum": lower,
            "maximum": (
                100.0 if upper > 100 else upper
            ),
            "maximum_inclusive": upper > 100,
        }

    return result


def _identify_weaknesses(
    summary: dict[str, Any],
    by_confidence: dict[str, dict[str, Any]],
    by_horizon: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify measurable areas of weak forecast performance."""

    weaknesses = []

    count = summary["evaluation_count"]

    if count == 0:
        return weaknesses

    accuracy = summary["directional_accuracy"]

    if accuracy < 0.50:
        weaknesses.append({
            "area": "directional_accuracy",
            "severity": "HIGH",
            "description": (
                "Overall directional accuracy is below 50 percent."
            ),
        })

    elif accuracy < 0.60:
        weaknesses.append({
            "area": "directional_accuracy",
            "severity": "MEDIUM",
            "description": (
                "Overall directional accuracy is below 60 percent."
            ),
        })

    for label, metrics in by_confidence.items():

        bucket_count = metrics["evaluation_count"]

        if bucket_count < 5:
            continue

        bucket_accuracy = metrics["directional_accuracy"]

        if bucket_accuracy < 0.50:
            weaknesses.append({
                "area": "confidence",
                "segment": label,
                "severity": "HIGH",
                "description": (
                    f"{label} confidence forecasts have "
                    "directional accuracy below 50 percent."
                ),
            })

        elif (
            label in ("HIGH", "VERY_HIGH")
            and bucket_accuracy < 0.60
        ):
            weaknesses.append({
                "area": "confidence",
                "segment": label,
                "severity": "MEDIUM",
                "description": (
                    f"{label} confidence forecasts are "
                    "less accurate than expected."
                ),
            })

    for horizon, metrics in by_horizon.items():

        if metrics["evaluation_count"] < 5:
            continue

        if metrics["directional_accuracy"] < 0.50:
            weaknesses.append({
                "area": "forecast_horizon",
                "segment": horizon,
                "severity": "HIGH",
                "description": (
                    f"Forecast horizon {horizon} days "
                    "has directional accuracy below 50 percent."
                ),
            })

    return weaknesses


def analyze_forecast_performance(
    evaluations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze historical forecast evaluation records.

    The analyzer:
    - Does not modify evaluation records.
    - Rejects duplicate forecast evaluations.
    - Calculates overall performance.
    - Compares performance by horizon.
    - Compares performance by confidence.
    - Compares performance by actual market direction.
    - Identifies measurable performance weaknesses.

    This analysis does not claim statistical calibration.
    """

    if isinstance(
        evaluations,
        (str, bytes, dict),
    ):
        raise ForecastPerformanceError(
            "evaluations must be an iterable of evaluation records."
        )

    try:
        source_records = list(evaluations)
    except TypeError as exc:
        raise ForecastPerformanceError(
            "evaluations must be iterable."
        ) from exc

    if not source_records:
        raise ForecastPerformanceError(
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
            raise ForecastPerformanceError(
                "Duplicate forecast evaluations are not allowed."
            )

        seen_ids.add(forecast_id)
        records.append(record)

    summary = _calculate_summary(
        records
    )

    by_horizon = _group_by(
        records,
        "horizon_days",
    )

    by_actual_direction = _group_by(
        records,
        "actual_direction",
    )

    by_confidence = _analyze_confidence(
        records
    )

    weaknesses = _identify_weaknesses(
        summary,
        by_confidence,
        by_horizon,
    )

    return {
        "status": "SUCCESS",
        "analysis_type": "FORECAST_PERFORMANCE",
        "evaluation_count": len(records),
        "summary": summary,
        "by_horizon_days": by_horizon,
        "by_confidence": by_confidence,
        "by_actual_direction": by_actual_direction,
        "weaknesses": weaknesses,
        "metadata": {
            "asset": "BTC",
            "method": "historical_evaluation_analysis",
            "input_records_modified": False,
            "statistical_calibration_assessed": False,
            "minimum_group_size_for_weakness_detection": 5,
        },
        "forecast_ids": sorted(seen_ids),
    }


if __name__ == "__main__":
    print("Bitcoin Forecast Performance Analyzer loaded.")