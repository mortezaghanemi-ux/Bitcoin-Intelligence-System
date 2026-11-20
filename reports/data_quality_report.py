
"""Data quality report generation for the Bitcoin Intelligence System."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from reports.engine import ReportEngine, ReportEngineError


class DataQualityReportError(ValueError):
    """Raised when data quality report input is invalid."""


def _parse_utc_timestamp(value: Any) -> datetime:
    """Parse and validate a timezone-aware UTC timestamp."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataQualityReportError(
                "Invalid validation timestamp."
            ) from exc

    if not isinstance(value, datetime):
        raise DataQualityReportError(
            "Validation timestamp must be a datetime or ISO string."
        )

    if value.tzinfo is None:
        raise DataQualityReportError(
            "Validation timestamp must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataQualityReportError(
            "Invalid validation timestamp timezone."
        ) from exc

    if offset is None or offset.total_seconds() != 0:
        raise DataQualityReportError(
            "Validation timestamp must use UTC."
        )

    return value.astimezone(timezone.utc)


def _validate_count(name: str, value: Any) -> int:
    """Validate a non-negative integer count."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise DataQualityReportError(
            f"{name} must be a non-negative integer."
        )

    return value


def _validate_dataframe(
    name: str,
    value: Any,
) -> pd.DataFrame:
    """Validate a DataFrame returned by the validation pipeline."""

    if not isinstance(value, pd.DataFrame):
        raise DataQualityReportError(
            f"{name} must be a pandas DataFrame."
        )

    return value


def _extract_error_counts(
    rejected: pd.DataFrame,
) -> dict[str, int]:
    """Count validation error codes from rejected candle records."""

    if rejected.empty:
        return {}

    if "validation_errors" not in rejected.columns:
        raise DataQualityReportError(
            "Rejected candles must contain validation_errors."
        )

    counts: Counter[str] = Counter()

    for value in rejected["validation_errors"]:
        if not isinstance(value, str) or not value.strip():
            continue

        for error in value.split(";"):
            normalized = error.strip()

            if normalized:
                counts[normalized] += 1

    return dict(sorted(counts.items()))


def _normalize_source_reports(
    reports: Any,
) -> list[dict[str, Any]]:
    """Validate and normalize source-specific validation reports."""

    if not isinstance(reports, list):
        raise DataQualityReportError(
            "Pipeline reports must be a list."
        )

    normalized = []
    seen_sources = set()

    required_fields = {
        "report_id",
        "dataset",
        "source",
        "validation_timestamp",
        "records_checked",
        "valid_records",
        "rejected_records",
        "warnings",
        "errors",
        "status",
    }

    for report in reports:
        if not isinstance(report, dict):
            raise DataQualityReportError(
                "Every source report must be a dictionary."
            )

        if set(report) != required_fields:
            raise DataQualityReportError(
                "A source report has an incompatible schema."
            )

        source = report["source"]

        if not isinstance(source, str) or not source.strip():
            raise DataQualityReportError(
                "Source report must contain a valid source."
            )

        source = source.strip()

        if source in seen_sources:
            raise DataQualityReportError(
                "Duplicate source reports are not allowed."
            )

        seen_sources.add(source)

        if report["dataset"] != "candles":
            raise DataQualityReportError(
                "Unsupported source report dataset."
            )

        timestamp = _parse_utc_timestamp(
            report["validation_timestamp"]
        )

        checked = _validate_count(
            "records_checked",
            int(report["records_checked"])
            if isinstance(report["records_checked"], str)
            and report["records_checked"].isdigit()
            else report["records_checked"],
        )

        valid = _validate_count(
            "valid_records",
            int(report["valid_records"])
            if isinstance(report["valid_records"], str)
            and report["valid_records"].isdigit()
            else report["valid_records"],
        )

        rejected = _validate_count(
            "rejected_records",
            int(report["rejected_records"])
            if isinstance(report["rejected_records"], str)
            and report["rejected_records"].isdigit()
            else report["rejected_records"],
        )

        if valid + rejected != checked:
            raise DataQualityReportError(
                "Source report record counts are inconsistent."
            )

        if report["status"] not in {"PASS", "FAIL", "WARNING"}:
            raise DataQualityReportError(
                "Source report contains an invalid status."
            )

        normalized.append(
            {
                "report_id": report["report_id"],
                "dataset": "candles",
                "source": source,
                "validation_timestamp": timestamp.isoformat(),
                "records_checked": checked,
                "valid_records": valid,
                "rejected_records": rejected,
                "status": report["status"],
            }
        )

    return sorted(
        normalized,
        key=lambda item: item["source"],
    )


def build_data_quality_report(
    validation_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a normalized data quality report.

    The input must be the result returned by
    validation.pipeline.run_candle_validation().
    """

    if not isinstance(validation_result, dict):
        raise DataQualityReportError(
            "Validation result must be a dictionary."
        )

    required_fields = {
        "accepted",
        "rejected",
        "validation",
        "reports",
        "status",
    }

    missing = required_fields - set(validation_result)

    if missing:
        raise DataQualityReportError(
            "Validation result is missing fields: "
            + ", ".join(sorted(missing))
        )

    accepted = _validate_dataframe(
        "accepted",
        validation_result["accepted"],
    )

    rejected = _validate_dataframe(
        "rejected",
        validation_result["rejected"],
    )

    validation = validation_result["validation"]

    if not isinstance(validation, dict):
        raise DataQualityReportError(
            "Validation summary must be a dictionary."
        )

    validation_fields = {
        "dataset",
        "validation_timestamp",
        "records_checked",
        "valid_records",
        "rejected_records",
        "status",
        "errors",
    }

    if not validation_fields.issubset(validation):
        raise DataQualityReportError(
            "Validation summary has an incompatible schema."
        )

    if validation["dataset"] != "candles":
        raise DataQualityReportError(
            "Only candle validation results are supported."
        )

    timestamp = _parse_utc_timestamp(
        validation["validation_timestamp"]
    )

    records_checked = _validate_count(
        "records_checked",
        validation["records_checked"],
    )

    valid_records = _validate_count(
        "valid_records",
        validation["valid_records"],
    )

    rejected_records = _validate_count(
        "rejected_records",
        validation["rejected_records"],
    )

    error_count = _validate_count(
        "errors",
        validation["errors"],
    )

    if valid_records + rejected_records != records_checked:
        raise DataQualityReportError(
            "Validation record counts are inconsistent."
        )

    if len(accepted) != valid_records:
        raise DataQualityReportError(
            "Accepted DataFrame size does not match validation summary."
        )

    if len(rejected) != rejected_records:
        raise DataQualityReportError(
            "Rejected DataFrame size does not match validation summary."
        )

    if records_checked != valid_records + rejected_records:
        raise DataQualityReportError(
            "Total validation record count is inconsistent."
        )

    expected_status = (
        "PASS"
        if rejected_records == 0
        else "FAIL"
    )

    if validation["status"] != expected_status:
        raise DataQualityReportError(
            "Validation summary status is inconsistent."
        )

    if validation_result["status"] != expected_status:
        raise DataQualityReportError(
            "Pipeline status is inconsistent with validation results."
        )

    source_reports = _normalize_source_reports(
        validation_result["reports"]
    )

    source_checked = sum(
        item["records_checked"]
        for item in source_reports
    )

    source_valid = sum(
        item["valid_records"]
        for item in source_reports
    )

    source_rejected = sum(
        item["rejected_records"]
        for item in source_reports
    )

    if source_checked != records_checked:
        raise DataQualityReportError(
            "Source report totals do not match validation totals."
        )

    if source_valid != valid_records:
        raise DataQualityReportError(
            "Source report valid counts do not match validation totals."
        )

    if source_rejected != rejected_records:
        raise DataQualityReportError(
            "Source report rejected counts do not match validation totals."
        )

    error_counts = _extract_error_counts(rejected)

    observed_error_count = sum(
        error_counts.values()
    )

    if observed_error_count != error_count:
        raise DataQualityReportError(
            "Rejected candle error counts do not match validation summary."
        )

    quality_pct = (
        round(valid_records / records_checked * 100, 4)
        if records_checked
        else 100.0
    )

    rejection_pct = (
        round(rejected_records / records_checked * 100, 4)
        if records_checked
        else 0.0
    )

    return {
        "dataset": "candles",
        "validation_timestamp": timestamp.isoformat(),
        "status": expected_status,
        "summary": {
            "records_checked": records_checked,
            "valid_records": valid_records,
            "rejected_records": rejected_records,
            "validation_errors": error_count,
            "quality_pct": quality_pct,
            "rejection_pct": rejection_pct,
        },
        "error_counts": error_counts,
        "source_reports": source_reports,
        "sources_checked": len(source_reports),
        "limitations": [
            "This report reflects the supplied candle validation run only.",
            "It does not independently verify the external data provider.",
            "It does not measure forecast accuracy or trading performance.",
        ],
    }


def create_data_quality_report(
    validation_result: dict[str, Any],
    *,
    output_directory=None,
) -> dict[str, Any]:
    """
    Build and persist a data quality report.

    No market data is modified and no validation is rerun.
    """

    payload = build_data_quality_report(
        validation_result
    )

    engine = (
        ReportEngine(output_directory)
        if output_directory is not None
        else ReportEngine()
    )

    try:
        return engine.generate(
            report_type="data_quality",
            payload=payload,
            generated_at=payload["validation_timestamp"],
            metadata={
                "dataset": "candles",
                "validation_status": payload["status"],
                "sources_checked": payload["sources_checked"],
            },
        )

    except ReportEngineError:
        raise

    except (TypeError, ValueError, OSError) as exc:
        raise DataQualityReportError(
            f"Unable to create data quality report: {exc}"
        ) from exc


if __name__ == "__main__":
    print("Bitcoin Intelligence Data Quality Report module loaded.")
