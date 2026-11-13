"""Candle validation pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from validation.candles import validate_candles
from validation.report import record_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_reports.csv"
)


class ValidationPipelineError(RuntimeError):
    """Raised when the candle validation pipeline fails."""


def _is_utc_datetime(value):
    """Return True only for timezone-aware UTC datetimes."""

    if not isinstance(value, datetime):
        return False

    try:
        return (
            value.tzinfo is not None
            and value.utcoffset() == timedelta(0)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _normalize_source(value):
    """Convert source values into safe, consistent labels."""

    try:
        if pd.isna(value):
            return "UNKNOWN"
    except (TypeError, ValueError):
        pass

    normalized = str(value).strip()

    return normalized or "UNKNOWN"


def run_candle_validation(
    data: pd.DataFrame,
    *,
    report_path=DEFAULT_REPORT_PATH,
    validation_time=None,
    max_age_minutes=15,
):
    """Validate candles and persist source-specific reports."""

    # Input validation

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame"
        )

    if data.empty:
        raise ValidationPipelineError(
            "Cannot validate an empty candle dataset."
        )

    # Validation timestamp

    checked_at = (
        validation_time
        if validation_time is not None
        else datetime.now(timezone.utc)
    )

    if not _is_utc_datetime(checked_at):
        raise ValueError(
            "validation_time must be a timezone-aware UTC datetime"
        )

    checked_at = checked_at.astimezone(timezone.utc)

    try:

        # Validate candle records

        accepted, rejected, validation_result = validate_candles(
            data,
            validation_time=checked_at,
            max_age_minutes=max_age_minutes,
            require_closed=True,
        )

        # Normalize original source labels

        report_sources = data["source"].map(
            _normalize_source
        )

        accepted_sources = accepted["source"].map(
            _normalize_source
        )

        rejected_sources = rejected["source"].map(
            _normalize_source
        )

        report_rows = []

        # Create one report per source

        for source in sorted(report_sources.unique()):

            source_mask = report_sources == source

            accepted_mask = accepted_sources == source

            rejected_mask = rejected_sources == source

            source_accepted = accepted.loc[
                accepted_mask
            ]

            source_rejected = rejected.loc[
                rejected_mask
            ]

            errors = set()

            if not source_rejected.empty:

                for message in source_rejected[
                    "validation_errors"
                ].dropna():

                    errors.update(
                        item
                        for item in str(message).split(";")
                        if item
                    )

            report = record_report(
                output_path=report_path,
                dataset="candles",
                source=source,
                records_checked=int(source_mask.sum()),
                valid_records=len(source_accepted),
                rejected_records=len(source_rejected),
                errors=sorted(errors),
                validation_timestamp=checked_at,
            )

            report_rows.append(report)

    except Exception as exc:

        raise ValidationPipelineError(
            f"Candle validation pipeline failed: {exc}"
        ) from exc

    return {
        "accepted": accepted,
        "rejected": rejected,
        "validation": validation_result,
        "reports": report_rows,
        "status": (
            "PASS"
            if rejected.empty
            else "FAIL"
        ),
    }