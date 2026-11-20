
"""Forecast performance report builder and storage adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from forecast.performance import (
    ForecastPerformanceError,
    analyze_forecast_performance,
)
from reports.engine import (
    DEFAULT_REPORT_DIRECTORY,
    ReportEngine,
    ReportEngineError,
)


class ForecastPerformanceReportError(ValueError):
    """Raised when a forecast performance report cannot be built."""


def _validate_analysis(
    analysis: Any,
) -> dict[str, Any]:
    """Validate the output of the forecast performance analyzer."""

    if not isinstance(analysis, dict):
        raise ForecastPerformanceReportError(
            "Forecast performance analysis must be a dictionary."
        )

    if analysis.get("status") != "SUCCESS":
        raise ForecastPerformanceReportError(
            "Only successful performance analyses can be reported."
        )

    if analysis.get("analysis_type") != "FORECAST_PERFORMANCE":
        raise ForecastPerformanceReportError(
            "Invalid forecast performance analysis type."
        )

    required_fields = (
        "evaluation_count",
        "summary",
        "by_horizon_days",
        "by_confidence",
        "by_actual_direction",
        "weaknesses",
        "metadata",
        "forecast_ids",
    )

    missing = [
        field
        for field in required_fields
        if field not in analysis
    ]

    if missing:
        raise ForecastPerformanceReportError(
            "Performance analysis is missing fields: "
            + ", ".join(missing)
        )

    count = analysis["evaluation_count"]

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ForecastPerformanceReportError(
            "evaluation_count must be a positive integer."
        )

    summary = analysis["summary"]

    if not isinstance(summary, dict):
        raise ForecastPerformanceReportError(
            "Performance summary must be a dictionary."
        )

    if summary.get("evaluation_count") != count:
        raise ForecastPerformanceReportError(
            "Summary evaluation count does not match analysis."
        )

    for field in (
        "mean_brier_score",
        "mean_log_loss",
        "directional_accuracy",
        "correct_predictions",
        "incorrect_predictions",
    ):
        if field not in summary:
            raise ForecastPerformanceReportError(
                f"Performance summary is missing {field}."
            )

    for field in (
        "by_horizon_days",
        "by_confidence",
        "by_actual_direction",
    ):
        if not isinstance(analysis[field], dict):
            raise ForecastPerformanceReportError(
                f"{field} must be a dictionary."
            )

    if not isinstance(analysis["weaknesses"], list):
        raise ForecastPerformanceReportError(
            "weaknesses must be a list."
        )

    if not isinstance(analysis["metadata"], dict):
        raise ForecastPerformanceReportError(
            "Performance metadata must be a dictionary."
        )

    forecast_ids = analysis["forecast_ids"]

    if (
        not isinstance(forecast_ids, list)
        or len(forecast_ids) != count
        or any(
            not isinstance(item, str) or not item.strip()
            for item in forecast_ids
        )
        or len(set(forecast_ids)) != len(forecast_ids)
    ):
        raise ForecastPerformanceReportError(
            "forecast_ids must contain one unique ID per evaluation."
        )

    return analysis


def build_forecast_performance_report(
    evaluations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a standardized report from completed forecast evaluations.

    Only successful historical evaluation records are accepted.
    The original evaluation records are not modified.

    This function does not generate forecasts, evaluate incomplete
    forecasts, or perform statistical calibration.
    """

    try:
        analysis = analyze_forecast_performance(
            evaluations
        )

    except ForecastPerformanceError as exc:
        raise ForecastPerformanceReportError(
            f"Unable to analyze forecast performance: {exc}"
        ) from exc

    validated = _validate_analysis(
        analysis
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    payload = {
        "analysis": validated,
        "interpretation_notice": (
            "Performance metrics describe historical forecast outcomes. "
            "They do not guarantee future accuracy. "
            "Directional accuracy and confidence-group results "
            "must not be interpreted as proof of statistical calibration."
        ),
    }

    metadata = {
        "asset": "BTC",
        "analysis_type": "FORECAST_PERFORMANCE",
        "evaluation_count": validated["evaluation_count"],
        "forecast_ids": validated["forecast_ids"],
        "method": "historical_evaluation_analysis",
        "statistical_calibration_assessed": False,
        "input_records_modified": False,
    }

    return {
        "report_type": "forecast_performance",
        "generated_at": generated_at,
        "source": "forecast.performance",
        "metadata": metadata,
        "payload": payload,
    }


def generate_forecast_performance_report(
    evaluations: Iterable[dict[str, Any]],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> dict[str, Any]:
    """
    Build and persist a forecast performance report.

    Reports are stored through the central ReportEngine.
    Existing historical reports are never overwritten.
    """

    report_document = build_forecast_performance_report(
        evaluations
    )

    engine = ReportEngine(
        output_directory=output_directory
    )

    try:
        return engine.generate(
            report_type=report_document["report_type"],
            payload=report_document["payload"],
            generated_at=report_document["generated_at"],
            source=report_document["source"],
            metadata=report_document["metadata"],
        )

    except ReportEngineError as exc:
        raise ForecastPerformanceReportError(
            f"Unable to save forecast performance report: {exc}"
        ) from exc


if __name__ == "__main__":
    print("Bitcoin Forecast Performance Report module loaded.")
