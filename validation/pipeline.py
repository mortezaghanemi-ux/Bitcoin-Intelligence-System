"""Candle validation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
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
    pass


def run_candle_validation(
    data: pd.DataFrame,
    *,
    report_path=DEFAULT_REPORT_PATH,
    validation_time=None,
    max_age_minutes=15,
):
    """Validate candles and persist source-specific reports."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValidationPipelineError(
            "Cannot validate an empty candle dataset."
        )

    checked_at = validation_time or datetime.now(timezone.utc)

    if (
        checked_at.tzinfo is None
        or checked_at.utcoffset().total_seconds() != 0
    ):
        raise ValueError("validation_time must be UTC")

    try:
        accepted, rejected, validation_result = validate_candles(
            data,
            validation_time=checked_at,
            max_age_minutes=max_age_minutes,
            require_closed=True,
        )

        report_sources = data["source"].fillna("").astype(str)

        report_sources = report_sources.map(
            lambda value: value.strip() or "UNKNOWN"
        )

        report_rows = []

        for source in sorted(report_sources.unique()):

            source_mask = report_sources == source

            source_accepted = accepted[
                accepted["source"].fillna("").astype(str).str.strip()
                .replace("", "UNKNOWN") == source
            ]

            source_rejected = rejected[
                rejected["source"].fillna("").astype(str).str.strip()
                .replace("", "UNKNOWN") == source
            ]

            errors = set()

            if not source_rejected.empty:

                for message in source_rejected[
                    "validation_errors"
                ].dropna():

                    errors.update(
                        item
                        for item in message.split(";")
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