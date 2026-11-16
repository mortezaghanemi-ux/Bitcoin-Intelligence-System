"""Append-only storage for data validation reports."""

from __future__ import annotations

import csv
import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable


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
    """Raised when a validation report is invalid."""


def _utc_timestamp(value=None):
    """Return a validated UTC timestamp in ISO format."""

    if value is None:
        timestamp = datetime.now(timezone.utc)
    else:
        timestamp = value

    if not isinstance(timestamp, datetime):
        raise ValidationReportError(
            "validation_timestamp must be a datetime"
        )

    try:
        if timestamp.tzinfo is None:
            raise ValidationReportError(
                "validation_timestamp must be timezone-aware UTC"
            )

        offset = timestamp.utcoffset()

    except ValidationReportError:
        raise

    except Exception as exc:
        raise ValidationReportError(
            "Invalid validation_timestamp timezone"
        ) from exc

    if offset is None or offset.total_seconds() != 0:
        raise ValidationReportError(
            "validation_timestamp must be timezone-aware UTC"
        )

    try:
        return timestamp.astimezone(timezone.utc).isoformat()

    except (ValueError, OverflowError) as exc:
        raise ValidationReportError(
            "Invalid validation_timestamp"
        ) from exc


def _normalize_messages(value, field):
    """Validate and serialize warning/error messages."""

    if value is None:
        items = []

    elif isinstance(value, (str, bytes, dict)):
        raise ValidationReportError(
            f"{field} must be a sequence, not a string or mapping"
        )

    elif isinstance(value, Iterable):
        try:
            items = list(value)
        except Exception as exc:
            raise ValidationReportError(
                f"Invalid {field} sequence"
            ) from exc

    else:
        raise ValidationReportError(
            f"{field} must be a sequence"
        )

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


def _validate_count(name, value):
    """Validate a non-negative integer count."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValidationReportError(
            f"{name} must be a non-negative integer"
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
    """Build a normalized validation report."""

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValidationReportError(
            "dataset is required"
        )

    if not isinstance(source, str) or not source.strip():
        raise ValidationReportError(
            "source is required"
        )

    _validate_count(
        "records_checked",
        records_checked,
    )

    _validate_count(
        "valid_records",
        valid_records,
    )

    _validate_count(
        "rejected_records",
        rejected_records,
    )

    if valid_records + rejected_records != records_checked:
        raise ValidationReportError(
            "valid_records + rejected_records must equal records_checked"
        )

    warning_json = _normalize_messages(
        warnings,
        "warnings",
    )

    error_json = _normalize_messages(
        errors,
        "errors",
    )

    warning_items = json.loads(warning_json)
    error_items = json.loads(error_json)

    if rejected_records or error_items:
        status = "FAIL"

    elif warning_items:
        status = "WARNING"

    else:
        status = "PASS"

    timestamp = _utc_timestamp(
        validation_timestamp
    )

    normalized_dataset = dataset.strip()
    normalized_source = source.strip()

    report_id = "|".join(
        (
            normalized_dataset,
            normalized_source,
            timestamp,
        )
    )

    return {
        "report_id": report_id,
        "dataset": normalized_dataset,
        "source": normalized_source,
        "validation_timestamp": timestamp,
        "records_checked": str(records_checked),
        "valid_records": str(valid_records),
        "rejected_records": str(rejected_records),
        "warnings": warning_json,
        "errors": error_json,
        "status": status,
    }


def _validate_report_schema(report):
    """Validate a report before writing it."""

    if (
        not isinstance(report, dict)
        or set(report) != set(REPORT_COLUMNS)
    ):
        raise ValidationReportError(
            "report does not match the required schema"
        )

    for field in (
        "report_id",
        "dataset",
        "source",
        "validation_timestamp",
        "warnings",
        "errors",
        "status",
    ):
        if (
            not isinstance(report[field], str)
            or not report[field].strip()
        ):
            raise ValidationReportError(
                f"{field} must be a non-empty string"
            )

    if report["status"] not in VALID_STATUSES:
        raise ValidationReportError(
            "invalid report status"
        )

    counts = {}

    for field in (
        "records_checked",
        "valid_records",
        "rejected_records",
    ):
        value = report[field]

        if (
            not isinstance(value, str)
            or not value.isdigit()
        ):
            raise ValidationReportError(
                f"{field} must contain a non-negative integer"
            )

        counts[field] = int(value)

    if (
        counts["valid_records"]
        + counts["rejected_records"]
        != counts["records_checked"]
    ):
        raise ValidationReportError(
            "report record counts are inconsistent"
        )

    timestamp = _utc_timestamp(
        datetime.fromisoformat(
            report["validation_timestamp"]
        )
    )

    if timestamp != report["validation_timestamp"]:
        raise ValidationReportError(
            "validation_timestamp must use normalized UTC format"
        )

    expected_id = "|".join(
        (
            report["dataset"],
            report["source"],
            timestamp,
        )
    )

    if report["report_id"] != expected_id:
        raise ValidationReportError(
            "report_id does not match report identity"
        )

    try:
        warnings = json.loads(report["warnings"])
        errors = json.loads(report["errors"])

    except (TypeError, ValueError) as exc:
        raise ValidationReportError(
            "warnings and errors must contain valid JSON"
        ) from exc

    if (
        not isinstance(warnings, list)
        or not isinstance(errors, list)
    ):
        raise ValidationReportError(
            "warnings and errors must be JSON lists"
        )

    for field, messages in (
        ("warnings", warnings),
        ("errors", errors),
    ):
        if any(
            not isinstance(item, str) or not item.strip()
            for item in messages
        ):
            raise ValidationReportError(
                f"{field} must contain non-empty strings"
            )

    expected_status = (
        "FAIL"
        if counts["rejected_records"] or errors
        else "WARNING"
        if warnings
        else "PASS"
    )

    if report["status"] != expected_status:
        raise ValidationReportError(
            "report status is inconsistent with its contents"
        )


def _read_existing_reports(destination):
    """Read and validate existing append-only report history."""

    if not destination.exists():
        return []

    try:
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

            rows = list(reader)

    except ValidationReportError:
        raise

    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValidationReportError(
            "Unable to read existing report file"
        ) from exc

    existing_ids = set()

    for row in rows:

        if set(row) != set(REPORT_COLUMNS):
            raise ValidationReportError(
                "existing report file contains an invalid row"
            )

        _validate_report_schema(row)

        report_id = row["report_id"]

        if report_id in existing_ids:
            raise ValidationReportError(
                "existing report file contains duplicate report IDs"
            )

        existing_ids.add(report_id)

    return rows


def save_report(report, output_path):
    """Atomically append a report without overwriting history."""

    _validate_report_schema(report)

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_rows = _read_existing_reports(
        destination
    )

    if any(
        row["report_id"] == report["report_id"]
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
                extrasaction="raise",
            )

            writer.writeheader()
            writer.writerows(rows)

            temp_file.flush()
            os.fsync(
                temp_file.fileno()
            )

        os.replace(
            temp_name,
            destination,
        )

        temp_name = None

    except Exception:

        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass

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
    """Build, persist, and return a validation report."""

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

    save_report(
        report,
        output_path,
    )

    return report