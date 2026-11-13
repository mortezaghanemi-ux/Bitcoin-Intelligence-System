"""Append-only storage for data validation reports."""

from __future__ import annotations

import csv
import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_COLUMNS = (
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
)

VALID_STATUSES = {"PASS", "FAIL", "WARNING"}


class ValidationReportError(ValueError):
    pass


def _utc_timestamp(value=None):

    timestamp = value or datetime.now(timezone.utc)

    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset().total_seconds() != 0
    ):
        raise ValidationReportError(
            "validation_timestamp must be timezone-aware UTC"
        )

    return timestamp.astimezone(timezone.utc).isoformat()


def _normalize_messages(value, field):

    if value is None:
        items = []

    elif isinstance(value, (str, bytes)):
        raise ValidationReportError(
            f"{field} must be a sequence, not a string"
        )

    else:
        items = list(value)

    if any(
        not isinstance(item, str) or not item.strip()
        for item in items
    ):
        raise ValidationReportError(
            f"{field} must contain non-empty strings"
        )

    return json.dumps(
        items,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_report(
    *,
    dataset,
    source,
    records_checked,
    valid_records,
    rejected_records,
    warnings=None,
    errors=None,
    validation_timestamp=None,
):

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValidationReportError("dataset is required")

    if not isinstance(source, str) or not source.strip():
        raise ValidationReportError("source is required")

    counts = {
        "records_checked": records_checked,
        "valid_records": valid_records,
        "rejected_records": rejected_records,
    }

    for name, count in counts.items():

        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValidationReportError(
                f"{name} must be a non-negative integer"
            )

    if valid_records + rejected_records != records_checked:
        raise ValidationReportError(
            "valid_records + rejected_records must equal records_checked"
        )

    warning_json = _normalize_messages(warnings, "warnings")
    error_json = _normalize_messages(errors, "errors")

    warning_items = json.loads(warning_json)
    error_items = json.loads(error_json)

    status = (
        "FAIL"
        if rejected_records or error_items
        else "WARNING"
        if warning_items
        else "PASS"
    )

    timestamp = _utc_timestamp(validation_timestamp)

    report_id = "|".join(
        (dataset.strip(), source.strip(), timestamp)
    )

    return {
        "report_id": report_id,
        "dataset": dataset.strip(),
        "source": source.strip(),
        "validation_timestamp": timestamp,
        "records_checked": str(records_checked),
        "valid_records": str(valid_records),
        "rejected_records": str(rejected_records),
        "warnings": warning_json,
        "errors": error_json,
        "status": status,
    }


def save_report(report, output_path):

    if (
        not isinstance(report, dict)
        or set(report) != set(REPORT_COLUMNS)
    ):
        raise ValidationReportError(
            "report does not match the required schema"
        )

    if report["status"] not in VALID_STATUSES:
        raise ValidationReportError("invalid report status")

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_rows = []

    if destination.exists():

        with destination.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            if tuple(reader.fieldnames or ()) != REPORT_COLUMNS:
                raise ValidationReportError(
                    "existing report file has an incompatible schema"
                )

            existing_rows = list(reader)

    if any(
        row.get("report_id") == report["report_id"]
        for row in existing_rows
    ):
        raise ValidationReportError(
            "duplicate validation report"
        )

    rows = existing_rows + [
        {
            column: report[column]
            for column in REPORT_COLUMNS
        }
    ]

    temp_name = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:

            temp_name = temp_file.name

            writer = csv.DictWriter(
                temp_file,
                fieldnames=REPORT_COLUMNS,
            )

            writer.writeheader()
            writer.writerows(rows)

            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_name, destination)

    except OSError:

        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

        raise

    return destination


def record_report(
    *,
    output_path,
    dataset,
    source,
    records_checked,
    valid_records,
    rejected_records,
    warnings=None,
    errors=None,
    validation_timestamp=None,
):

    report = build_report(
        dataset=dataset,
        source=source,
        records_checked=records_checked,
        valid_records=valid_records,
        rejected_records=rejected_records,
        warnings=warnings,
        errors=errors,
        validation_timestamp=validation_timestamp,
    )

    save_report(report, output_path)

    return report