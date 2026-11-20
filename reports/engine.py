
"""Central report generation and append-only report storage."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reports"
)

REPORT_SCHEMA_VERSION = 1

REPORT_TYPES = {
    "market",
    "forecast",
    "trading",
    "data_quality",
    "forecast_performance",
    "trading_performance",
}


class ReportEngineError(RuntimeError):
    """Raised when report generation or storage fails."""


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _parse_utc_timestamp(value: Any) -> datetime:
    """Validate and normalize a timezone-aware UTC timestamp."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReportEngineError(
                "Invalid report timestamp."
            ) from exc

    if not isinstance(value, datetime):
        raise ReportEngineError(
            "Report timestamp must be a datetime or ISO string."
        )

    if value.tzinfo is None:
        raise ReportEngineError(
            "Report timestamp must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()
    except (ValueError, OverflowError) as exc:
        raise ReportEngineError(
            "Invalid report timestamp timezone."
        ) from exc

    if offset is None or offset.total_seconds() != 0:
        raise ReportEngineError(
            "Report timestamp must use UTC."
        )

    return value.astimezone(timezone.utc)


def _normalize_report_type(report_type: str) -> str:
    """Validate a report type and prevent unsafe file names."""

    if not isinstance(report_type, str):
        raise ReportEngineError(
            "report_type must be a string."
        )

    normalized = report_type.strip().lower()

    if not re.fullmatch(r"[a-z][a-z0-9_]{0,49}", normalized):
        raise ReportEngineError(
            "report_type contains unsupported characters."
        )

    if normalized not in REPORT_TYPES:
        raise ReportEngineError(
            f"Unsupported report type: {normalized}."
        )

    return normalized


def _json_default(value: Any) -> Any:
    """Convert supported Python values into JSON-compatible values."""

    if isinstance(value, datetime):
        timestamp = _parse_utc_timestamp(value)
        return timestamp.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    raise TypeError(
        f"Unsupported report value type: {type(value).__name__}"
    )


def _serialize_report(report: dict[str, Any]) -> str:
    """Serialize a report using strict JSON rules."""

    try:
        return json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportEngineError(
            f"Report contains unsupported or invalid data: {exc}"
        ) from exc


def _validate_payload(payload: Any) -> dict[str, Any]:
    """Validate the report payload."""

    if not isinstance(payload, dict):
        raise ReportEngineError(
            "Report payload must be a dictionary."
        )

    if not payload:
        raise ReportEngineError(
            "Report payload cannot be empty."
        )

    if any(
        not isinstance(key, str) or not key.strip()
        for key in payload
    ):
        raise ReportEngineError(
            "Every report payload key must be a non-empty string."
        )

    # Verify JSON compatibility before creating a report.
    _serialize_report(payload)

    return dict(payload)


def build_report(
    *,
    report_type: str,
    payload: dict[str, Any],
    generated_at: datetime | str | None = None,
    source: str = "Bitcoin-Intelligence-System",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a standardized report envelope.

    The payload must contain data that has already passed
    the relevant validation and analysis pipelines.
    """

    normalized_type = _normalize_report_type(report_type)
    normalized_payload = _validate_payload(payload)

    if generated_at is None:
        timestamp = _utc_now()
    else:
        timestamp = _parse_utc_timestamp(generated_at)

    if not isinstance(source, str) or not source.strip():
        raise ReportEngineError(
            "Report source must be a non-empty string."
        )

    if metadata is None:
        normalized_metadata = {}
    elif isinstance(metadata, dict):
        normalized_metadata = dict(metadata)
    else:
        raise ReportEngineError(
            "Report metadata must be a dictionary."
        )

    _serialize_report(normalized_metadata)

    report_id = (
        f"{normalized_type}-"
        f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{uuid.uuid4().hex[:12]}"
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "report_type": normalized_type,
        "status": "SUCCESS",
        "generated_at": timestamp.isoformat(),
        "source": source.strip(),
        "metadata": normalized_metadata,
        "payload": normalized_payload,
    }

    _serialize_report(report)

    return report


def _validate_report_document(
    report: Any,
) -> dict[str, Any]:
    """Validate a stored report document."""

    if not isinstance(report, dict):
        raise ReportEngineError(
            "Stored report must be a dictionary."
        )

    required_fields = {
        "schema_version",
        "report_id",
        "report_type",
        "status",
        "generated_at",
        "source",
        "metadata",
        "payload",
    }

    if set(report) != required_fields:
        raise ReportEngineError(
            "Stored report has an incompatible schema."
        )

    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ReportEngineError(
            "Unsupported report schema version."
        )

    if report["status"] != "SUCCESS":
        raise ReportEngineError(
            "Stored report has an invalid status."
        )

    if (
        not isinstance(report["report_id"], str)
        or not report["report_id"].strip()
    ):
        raise ReportEngineError(
            "Stored report has an invalid report_id."
        )

    normalized_type = _normalize_report_type(
        report["report_type"]
    )

    if normalized_type != report["report_type"]:
        raise ReportEngineError(
            "Stored report type is not normalized."
        )

    timestamp = _parse_utc_timestamp(
        report["generated_at"]
    )

    if timestamp.isoformat() != report["generated_at"]:
        raise ReportEngineError(
            "Stored report timestamp is not normalized UTC."
        )

    if (
        not isinstance(report["source"], str)
        or not report["source"].strip()
    ):
        raise ReportEngineError(
            "Stored report has an invalid source."
        )

    if not isinstance(report["metadata"], dict):
        raise ReportEngineError(
            "Stored report metadata must be a dictionary."
        )

    if not isinstance(report["payload"], dict):
        raise ReportEngineError(
            "Stored report payload must be a dictionary."
        )

    _serialize_report(report)

    return report


def save_report(
    report: dict[str, Any],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> Path:
    """
    Save a report as a new JSON file.

    Existing reports are never overwritten.
    The file is written atomically.
    """

    validated_report = _validate_report_document(report)

    try:
        destination_directory = Path(output_directory)
        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not destination_directory.is_dir():
            raise ReportEngineError(
                "Report output path is not a directory."
            )

    except ReportEngineError:
        raise

    except (OSError, TypeError, ValueError) as exc:
        raise ReportEngineError(
            f"Unable to prepare report directory: {exc}"
        ) from exc

    report_type = validated_report["report_type"]
    report_id = validated_report["report_id"]

    filename = f"{report_type}_{report_id}.json"
    destination = destination_directory / filename

    serialized = _serialize_report(validated_report)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination_directory,
            prefix=".report_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            temporary_file.write(serialized)
            temporary_file.write("\n")
            temporary_file.flush()

            os.fsync(temporary_file.fileno())

        # Hard-link creation fails if the destination already exists.
        # This prevents accidental replacement of historical reports.
        os.link(
            temporary_path,
            destination,
        )

        temporary_path.unlink()
        temporary_path = None

        return destination

    except FileExistsError as exc:
        raise ReportEngineError(
            "A report with this ID already exists."
        ) from exc

    except (OSError, TypeError, ValueError) as exc:
        raise ReportEngineError(
            f"Unable to save report: {exc}"
        ) from exc

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_report(
    report_path: str | Path,
) -> dict[str, Any]:
    """Load and validate one previously saved report."""

    try:
        path = Path(report_path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ReportEngineError(
            f"Unable to read report: {exc}"
        ) from exc

    return _validate_report_document(report)


def list_reports(
    report_type: str | None = None,
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> list[dict[str, Any]]:
    """
    Load report history without modifying stored files.

    Reports are returned in chronological order.
    """

    normalized_type = (
        _normalize_report_type(report_type)
        if report_type is not None
        else None
    )

    directory = Path(output_directory)

    if not directory.exists():
        return []

    if not directory.is_dir():
        raise ReportEngineError(
            "Report history path is not a directory."
        )

    pattern = (
        f"{normalized_type}_*.json"
        if normalized_type
        else "*.json"
    )

    try:
        report_paths = sorted(directory.glob(pattern))
    except OSError as exc:
        raise ReportEngineError(
            f"Unable to list report history: {exc}"
        ) from exc

    reports = []
    seen_ids = set()

    for report_path in report_paths:
        report = load_report(report_path)

        if report["report_id"] in seen_ids:
            raise ReportEngineError(
                "Duplicate report IDs found in report history."
            )

        seen_ids.add(report["report_id"])
        reports.append(report)

    reports.sort(
        key=lambda item: (
            item["generated_at"],
            item["report_id"],
        )
    )

    return reports


class ReportEngine:
    """Central interface for generating and retrieving reports."""

    def __init__(
        self,
        output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    ):
        self.output_directory = Path(output_directory)

    def generate(
        self,
        *,
        report_type: str,
        payload: dict[str, Any],
        generated_at: datetime | str | None = None,
        source: str = "Bitcoin-Intelligence-System",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build and persist a new report."""

        report = build_report(
            report_type=report_type,
            payload=payload,
            generated_at=generated_at,
            source=source,
            metadata=metadata,
        )

        saved_path = save_report(
            report,
            self.output_directory,
        )

        result = dict(report)
        result["storage"] = {
            "saved": True,
            "path": str(saved_path),
        }

        return result

    def get_history(
        self,
        report_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return previously saved reports."""

        return list_reports(
            report_type=report_type,
            output_directory=self.output_directory,
        )


def generate_report(
    *,
    report_type: str,
    payload: dict[str, Any],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    generated_at: datetime | str | None = None,
    source: str = "Bitcoin-Intelligence-System",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function for generating one report."""

    engine = ReportEngine(
        output_directory=output_directory
    )

    return engine.generate(
        report_type=report_type,
        payload=payload,
        generated_at=generated_at,
        source=source,
        metadata=metadata,
    )


if __name__ == "__main__":
    print("Bitcoin Intelligence Report Engine loaded.")
