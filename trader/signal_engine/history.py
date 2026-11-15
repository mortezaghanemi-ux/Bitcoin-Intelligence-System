"""Append-only trading signal history storage.

Stores generated and rejected trading signals in JSON Lines format.
Historical records are never overwritten or modified.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "trader" / "signal_history.jsonl"
)


class SignalHistoryError(RuntimeError):
    """Raised when signal history operations fail."""


def _validate_signal_id(value: Any) -> str:
    """Validate a SHA-256 signal identifier."""

    if not isinstance(value, str):
        raise SignalHistoryError(
            "signal_id must be a string."
        )

    normalized = value.strip().lower()

    if len(normalized) != 64:
        raise SignalHistoryError(
            "signal_id must contain exactly 64 hexadecimal characters."
        )

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise SignalHistoryError(
            "signal_id must be a valid SHA-256 hexadecimal value."
        ) from exc

    return normalized


def _validate_timestamp(value: Any) -> str:
    """Validate and normalize an ISO-8601 UTC timestamp."""

    if not isinstance(value, str) or not value.strip():
        raise SignalHistoryError(
            "analysis_timestamp must be a non-empty ISO-8601 string."
        )

    try:
        timestamp = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SignalHistoryError(
            "analysis_timestamp is invalid."
        ) from exc

    if timestamp.tzinfo is None:
        raise SignalHistoryError(
            "analysis_timestamp must be timezone-aware."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise SignalHistoryError(
            "analysis_timestamp must use UTC."
        )

    return timestamp.astimezone(timezone.utc).isoformat()


def _normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Validate and create an immutable JSON-compatible snapshot."""

    if not isinstance(signal, dict):
        raise SignalHistoryError(
            "Signal must be a dictionary."
        )

    if "signal_id" not in signal:
        raise SignalHistoryError(
            "Signal is missing signal_id."
        )

    if "analysis_timestamp" not in signal:
        raise SignalHistoryError(
            "Signal is missing analysis_timestamp."
        )

    normalized = dict(signal)

    normalized["signal_id"] = _validate_signal_id(
        signal["signal_id"]
    )

    normalized["analysis_timestamp"] = _validate_timestamp(
        signal["analysis_timestamp"]
    )

    if normalized.get("status") not in (
        "REJECTED",
        "REQUIRES_MANUAL_REVIEW",
    ):
        raise SignalHistoryError(
            "Signal status must be REJECTED or REQUIRES_MANUAL_REVIEW."
        )

    if normalized.get("signal") not in (
        "LONG",
        "SHORT",
        "NO_TRADE",
    ):
        raise SignalHistoryError(
            "Signal type must be LONG, SHORT or NO_TRADE."
        )

    if normalized.get("signal") == "NO_TRADE":
        if normalized["status"] != "REJECTED":
            raise SignalHistoryError(
                "NO_TRADE signals must have REJECTED status."
            )

    execution = normalized.get("execution")

    if not isinstance(execution, dict):
        raise SignalHistoryError(
            "Signal execution metadata is missing."
        )

    if (
        execution.get("manual_review_only") is not True
        or execution.get("auto_trading") is not False
        or execution.get("exchange_connection") is not False
        or execution.get("order_execution") is not False
    ):
        raise SignalHistoryError(
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
        raise SignalHistoryError(
            "Signal contains non-JSON-compatible values."
        ) from exc

    return snapshot


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read and validate all existing history records."""

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
                    raise SignalHistoryError(
                        f"Invalid JSON at history line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise SignalHistoryError(
                        f"History line {line_number} is not an object."
                    )

                if "signal_id" not in record:
                    raise SignalHistoryError(
                        f"History line {line_number} has no signal_id."
                    )

                try:
                    record["signal_id"] = _validate_signal_id(
                        record["signal_id"]
                    )
                except SignalHistoryError as exc:
                    raise SignalHistoryError(
                        f"Invalid signal_id at history line {line_number}."
                    ) from exc

                records.append(record)

    except OSError as exc:
        raise SignalHistoryError(
            f"Unable to read signal history: {exc}"
        ) from exc

    return records


class SignalHistoryStore:
    """Persistent append-only storage for trading signals."""

    def __init__(
        self,
        history_path: str | Path | None = None,
    ) -> None:

        self.history_path = Path(
            history_path or DEFAULT_HISTORY_PATH
        )

    def read_all(self) -> list[dict[str, Any]]:
        """Return all stored signal records."""

        records = _read_json_lines(self.history_path)

        return json.loads(
            json.dumps(records)
        )

    def append(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        """Append a new signal without modifying existing records.

        Duplicate signal identifiers are not stored again.
        """

        normalized = _normalize_signal(signal)

        existing_records = _read_json_lines(
            self.history_path
        )

        for record in existing_records:
            if record["signal_id"] == normalized["signal_id"]:
                duplicate = json.loads(
                    json.dumps(record)
                )
                duplicate["duplicate"] = True
                return duplicate

        self.history_path.parent.mkdir(
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
            with self.history_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as file:

                file.write(encoded + "\n")
                file.flush()
                os.fsync(file.fileno())

        except OSError as exc:
            raise SignalHistoryError(
                f"Unable to append signal history: {exc}"
            ) from exc

        return json.loads(
            json.dumps(normalized)
        )

    def get(
        self,
        signal_id: str,
    ) -> dict[str, Any] | None:
        """Find a signal by its unique identifier."""

        normalized_id = _validate_signal_id(signal_id)

        for record in self.read_all():
            if record["signal_id"] == normalized_id:
                return record

        return None

    def contains(
        self,
        signal_id: str,
    ) -> bool:
        """Check whether a signal identifier already exists."""

        return self.get(signal_id) is not None

    def count(self) -> int:
        """Return the number of stored signal records."""

        return len(self.read_all())

    def count_by_status(self) -> dict[str, int]:
        """Return signal counts grouped by status."""

        counts = {
            "REJECTED": 0,
            "REQUIRES_MANUAL_REVIEW": 0,
        }

        for record in self.read_all():
            status = record.get("status")

            if status in counts:
                counts[status] += 1

        return counts

    def verify_integrity(self) -> dict[str, Any]:
        """Check history records for duplicate identifiers and corruption."""

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
        }


def save_signal(
    signal: dict[str, Any],
    history_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience function for storing one trading signal."""

    store = SignalHistoryStore(history_path)

    return store.append(signal)


def load_signal_history(
    history_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience function for loading signal history."""

    store = SignalHistoryStore(history_path)

    return store.read_all()


if __name__ == "__main__":
    store = SignalHistoryStore()

    print(
        json.dumps(
            {
                "history_path": str(store.history_path),
                "total_records": store.count(),
                "status_counts": store.count_by_status(),
                "integrity": store.verify_integrity(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )