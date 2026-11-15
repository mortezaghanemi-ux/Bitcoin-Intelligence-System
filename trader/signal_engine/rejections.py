"""Append-only storage for rejected trading signals.

Only explicitly rejected NO_TRADE signals are stored.
Signals requiring manual review are never treated as rejections.
Historical rejection records are never overwritten or modified.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REJECTIONS_PATH = (
    PROJECT_ROOT / "data" / "trader" / "signal_rejections.jsonl"
)


class SignalRejectionError(RuntimeError):
    """Raised when signal rejection operations fail."""


def _validate_signal_id(value: Any) -> str:
    """Validate and normalize a SHA-256 signal identifier."""

    if not isinstance(value, str):
        raise SignalRejectionError(
            "signal_id must be a string."
        )

    normalized = value.strip().lower()

    if len(normalized) != 64:
        raise SignalRejectionError(
            "signal_id must contain exactly 64 hexadecimal characters."
        )

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise SignalRejectionError(
            "signal_id must be a valid SHA-256 hexadecimal value."
        ) from exc

    return normalized


def _validate_timestamp(value: Any) -> str:
    """Validate and normalize an ISO-8601 UTC timestamp."""

    if not isinstance(value, str) or not value.strip():
        raise SignalRejectionError(
            "analysis_timestamp must be a non-empty ISO-8601 string."
        )

    try:
        timestamp = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SignalRejectionError(
            "analysis_timestamp is invalid."
        ) from exc

    if timestamp.tzinfo is None:
        raise SignalRejectionError(
            "analysis_timestamp must be timezone-aware."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise SignalRejectionError(
            "analysis_timestamp must use UTC."
        )

    return timestamp.astimezone(timezone.utc).isoformat()


def _normalize_rejection(
    signal: dict[str, Any],
) -> dict[str, Any]:
    """Validate and create an immutable rejection snapshot."""

    if not isinstance(signal, dict):
        raise SignalRejectionError(
            "Signal must be a dictionary."
        )

    if "signal_id" not in signal:
        raise SignalRejectionError(
            "Signal is missing signal_id."
        )

    if "analysis_timestamp" not in signal:
        raise SignalRejectionError(
            "Signal is missing analysis_timestamp."
        )

    normalized = dict(signal)

    normalized["signal_id"] = _validate_signal_id(
        signal["signal_id"]
    )

    normalized["analysis_timestamp"] = _validate_timestamp(
        signal["analysis_timestamp"]
    )

    if normalized.get("status") != "REJECTED":
        raise SignalRejectionError(
            "Only signals with REJECTED status can be stored."
        )

    if normalized.get("signal") != "NO_TRADE":
        raise SignalRejectionError(
            "Only NO_TRADE signals can be stored as rejections."
        )

    rejection_reasons = normalized.get("rejection_reasons")

    if (
        not isinstance(rejection_reasons, list)
        or not rejection_reasons
    ):
        raise SignalRejectionError(
            "Rejected signals must contain rejection_reasons."
        )

    if not all(
        isinstance(reason, str) and bool(reason.strip())
        for reason in rejection_reasons
    ):
        raise SignalRejectionError(
            "Every rejection reason must be a non-empty string."
        )

    execution = normalized.get("execution")

    if not isinstance(execution, dict):
        raise SignalRejectionError(
            "Signal execution metadata is missing."
        )

    if (
        execution.get("manual_review_only") is not True
        or execution.get("auto_trading") is not False
        or execution.get("exchange_connection") is not False
        or execution.get("order_execution") is not False
    ):
        raise SignalRejectionError(
            "Signal must explicitly disable automatic and exchange execution."
        )

    try:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

        snapshot = json.loads(encoded)

    except (TypeError, ValueError, OverflowError) as exc:
        raise SignalRejectionError(
            "Signal contains non-JSON-compatible values."
        ) from exc

    return snapshot


def _read_json_lines(
    path: Path,
) -> list[dict[str, Any]]:
    """Read existing rejection records and validate their structure."""

    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise SignalRejectionError(
                        f"Invalid JSON at rejection line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise SignalRejectionError(
                        f"Rejection line {line_number} is not an object."
                    )

                try:
                    normalized = _normalize_rejection(record)

                except SignalRejectionError as exc:
                    raise SignalRejectionError(
                        f"Invalid rejection record at line {line_number}: {exc}"
                    ) from exc

                records.append(normalized)

    except OSError as exc:
        raise SignalRejectionError(
            f"Unable to read rejection history: {exc}"
        ) from exc

    return records


class SignalRejectionStore:
    """Persistent append-only storage for rejected trading signals."""

    def __init__(
        self,
        rejection_path: str | Path | None = None,
    ) -> None:

        self.rejection_path = Path(
            rejection_path or DEFAULT_REJECTIONS_PATH
        )

    def read_all(self) -> list[dict[str, Any]]:
        """Return all stored rejection records."""

        records = _read_json_lines(
            self.rejection_path
        )

        return json.loads(
            json.dumps(records)
        )

    def append(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one rejected signal without modifying existing records.

        Duplicate signal identifiers are not stored again.
        """

        normalized = _normalize_rejection(signal)

        existing_records = _read_json_lines(
            self.rejection_path
        )

        for record in existing_records:

            if record["signal_id"] == normalized["signal_id"]:

                duplicate = json.loads(
                    json.dumps(record)
                )

                duplicate["duplicate"] = True

                return duplicate

        self.rejection_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

        try:
            with self.rejection_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as file:

                file.write(encoded + "\n")
                file.flush()
                os.fsync(file.fileno())

        except OSError as exc:
            raise SignalRejectionError(
                f"Unable to append rejection history: {exc}"
            ) from exc

        return json.loads(
            json.dumps(normalized)
        )

    def get(
        self,
        signal_id: str,
    ) -> dict[str, Any] | None:
        """Find a rejected signal by its identifier."""

        normalized_id = _validate_signal_id(
            signal_id
        )

        for record in self.read_all():

            if record["signal_id"] == normalized_id:
                return record

        return None

    def contains(
        self,
        signal_id: str,
    ) -> bool:
        """Check whether a rejected signal already exists."""

        return self.get(signal_id) is not None

    def count(self) -> int:
        """Return the total number of rejected signals."""

        return len(self.read_all())

    def count_by_reason(self) -> dict[str, int]:
        """Count rejection records grouped by rejection reason."""

        counts: dict[str, int] = {}

        for record in self.read_all():

            for reason in record["rejection_reasons"]:

                counts[reason] = counts.get(reason, 0) + 1

        return counts

    def verify_integrity(self) -> dict[str, Any]:
        """Check rejection history for duplicate identifiers."""

        records = self.read_all()

        identifiers = [
            record["signal_id"]
            for record in records
        ]

        duplicate_ids = sorted({
            identifier
            for identifier in identifiers
            if identifiers.count(identifier) > 1
        })

        return {
            "valid": not duplicate_ids,
            "total_records": len(records),
            "unique_records": len(set(identifiers)),
            "duplicate_signal_ids": duplicate_ids,
            "storage_format": "JSONL",
            "append_only": True,
            "rejected_signals_only": True,
        }


def record_rejection(
    signal: dict[str, Any],
    rejection_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience function for recording one rejected signal."""

    store = SignalRejectionStore(rejection_path)

    return store.append(signal)


def load_rejections(
    rejection_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience function for loading rejection history."""

    store = SignalRejectionStore(rejection_path)

    return store.read_all()


if __name__ == "__main__":

    store = SignalRejectionStore()

    print(
        json.dumps(
            {
                "rejection_path": str(store.rejection_path),
                "total_rejections": store.count(),
                "rejection_reasons": store.count_by_reason(),
                "integrity": store.verify_integrity(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )