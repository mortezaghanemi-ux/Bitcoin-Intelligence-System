"""Mempool.space Bitcoin on-chain data collector."""

from __future__ import annotations

import json
import math

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://mempool.space/api"

ENDPOINTS = {
    "mempool": "/mempool",
    "fees": "/v1/fees/recommended",
    "tip_height": "/blocks/tip/height",
    "difficulty_adjustment": "/v1/difficulty-adjustment",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "onchain"
    / "mempool_space"
)

DEFAULT_TIMEOUT = 15.0


class MempoolSpaceCollectorError(RuntimeError):
    """Raised when Mempool.space collection fails."""


def _normalize_utc(
    value: datetime | None,
) -> datetime:
    """Validate and normalize a UTC datetime."""

    if value is None:
        return datetime.now(timezone.utc)

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
    ):
        raise ValueError(
            "collection_time must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError(
            "Invalid collection_time timezone."
        ) from exc

    if (
        offset is None
        or offset.total_seconds() != 0
    ):
        raise ValueError(
            "collection_time must use UTC."
        )

    return value.astimezone(timezone.utc)


def _iso_utc(
    value: datetime,
) -> str:
    """Convert datetime to ISO-8601 UTC."""

    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_timeout(
    timeout: float,
) -> float:
    """Validate HTTP timeout."""

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout must be positive and finite."
        )

    return float(timeout)


def _validate_json_value(
    value: Any,
    path: str = "response",
) -> None:
    """Reject unsupported or non-finite JSON values."""

    if value is None or isinstance(
        value,
        (str, bool, int),
    ):
        return

    if isinstance(value, float):

        if not math.isfinite(value):
            raise MempoolSpaceCollectorError(
                f"Non-finite numeric value at {path}."
            )

        return

    if isinstance(value, list):

        for index, item in enumerate(value):

            _validate_json_value(
                item,
                f"{path}[{index}]",
            )

        return

    if isinstance(value, dict):

        for key, item in value.items():

            if not isinstance(key, str):
                raise MempoolSpaceCollectorError(
                    f"Invalid JSON object key at {path}."
                )

            _validate_json_value(
                item,
                f"{path}.{key}",
            )

        return

    raise MempoolSpaceCollectorError(
        f"Unsupported response value at {path}."
    )


def _validate_number(
    value: Any,
    field: str,
    *,
    minimum: float = 0,
) -> float:
    """Validate a finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise MempoolSpaceCollectorError(
            f"Invalid numeric field: {field}."
        )

    return float(value)


def _fetch_json(
    session: requests.Session,
    endpoint: str,
    timeout: float,
) -> Any:
    """Fetch and validate one public API response."""

    url = BASE_URL + endpoint

    try:
        response = session.get(
            url,
            headers={
                "accept": "application/json",
            },
            timeout=timeout,
        )

        response.raise_for_status()

        payload = response.json()

    except requests.RequestException as exc:
        raise MempoolSpaceCollectorError(
            f"Mempool.space request failed "
            f"for {endpoint}: {exc}"
        ) from exc

    except ValueError as exc:
        raise MempoolSpaceCollectorError(
            f"Mempool.space returned invalid JSON "
            f"for {endpoint}."
        ) from exc

    _validate_json_value(payload)

    return payload


def _validate_mempool(
    payload: Any,
) -> dict[str, Any]:
    """Validate current mempool statistics."""

    if not isinstance(payload, dict):
        raise MempoolSpaceCollectorError(
            "Mempool response must be a JSON object."
        )

    required = (
        "count",
        "vsize",
        "total_fee",
    )

    for field in required:

        if field not in payload:
            raise MempoolSpaceCollectorError(
                f"Mempool response is missing {field}."
            )

    count = _validate_number(
        payload["count"],
        "mempool.count",
    )

    vsize = _validate_number(
        payload["vsize"],
        "mempool.vsize",
    )

    total_fee = _validate_number(
        payload["total_fee"],
        "mempool.total_fee",
    )

    if not count.is_integer():
        raise MempoolSpaceCollectorError(
            "Mempool transaction count must be an integer."
        )

    return {
        "count": int(count),
        "vsize": vsize,
        "total_fee": total_fee,
        "raw": payload,
    }


def _validate_fees(
    payload: Any,
) -> dict[str, Any]:
    """Validate recommended transaction fee rates."""

    if not isinstance(payload, dict):
        raise MempoolSpaceCollectorError(
            "Fee response must be a JSON object."
        )

    required = (
        "fastestFee",
        "halfHourFee",
        "hourFee",
        "economyFee",
        "minimumFee",
    )

    fees = {}

    for field in required:

        if field not in payload:
            raise MempoolSpaceCollectorError(
                f"Fee response is missing {field}."
            )

        fees[field] = _validate_number(
            payload[field],
            f"fees.{field}",
        )

    return {
        **fees,
        "unit": "sat/vB",
        "raw": payload,
    }


def _validate_tip_height(
    payload: Any,
) -> int:
    """Validate latest Bitcoin block height."""

    if isinstance(payload, str):

        if not payload.isdigit():
            raise MempoolSpaceCollectorError(
                "Invalid block tip height."
            )

        height = int(payload)

    elif (
        isinstance(payload, int)
        and not isinstance(payload, bool)
    ):
        height = payload

    else:
        raise MempoolSpaceCollectorError(
            "Invalid block tip height."
        )

    if height <= 0:
        raise MempoolSpaceCollectorError(
            "Block tip height must be positive."
        )

    return height


def _validate_difficulty_adjustment(
    payload: Any,
) -> dict[str, Any]:
    """Validate difficulty adjustment response."""

    if not isinstance(payload, dict):
        raise MempoolSpaceCollectorError(
            "Difficulty adjustment must be a JSON object."
        )

    if not payload:
        raise MempoolSpaceCollectorError(
            "Difficulty adjustment response is empty."
        )

    return payload


def _load_existing_timestamps(
    path: Path,
) -> set[str]:
    """Load existing collection timestamps."""

    existing = set()

    if not path.exists():
        return existing

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise MempoolSpaceCollectorError(
                        f"Invalid JSONL at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise MempoolSpaceCollectorError(
                        f"Invalid record at line {line_number}."
                    )

                timestamp = record.get(
                    "collected_at"
                )

                if (
                    not isinstance(timestamp, str)
                    or not timestamp
                ):
                    raise MempoolSpaceCollectorError(
                        f"Missing collection timestamp "
                        f"at line {line_number}."
                    )

                existing.add(timestamp)

    except OSError as exc:
        raise MempoolSpaceCollectorError(
            f"Cannot read existing on-chain history: {exc}"
        ) from exc

    return existing


def collect_mempool_space(
    *,
    output_dir: str | Path | None = None,
    collection_time: datetime | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any] | None:
    """
    Collect Bitcoin on-chain network statistics.

    Public API endpoints are used.

    No API credentials are required.

    Collected datasets:

    - Mempool statistics
    - Recommended transaction fees
    - Latest block height
    - Difficulty adjustment information

    Raw provider responses are preserved.

    Existing observations are never overwritten.

    Duplicate collection timestamps are skipped.
    """

    request_timeout = _validate_timeout(
        timeout
    )

    checked_at = _normalize_utc(
        collection_time
    )

    collected_at = _iso_utc(
        checked_at
    )

    destination = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    output_file = (
        destination
        / "mempool_space_BTC.jsonl"
    )

    existing = _load_existing_timestamps(
        output_file
    )

    if collected_at in existing:
        return None

    try:
        with requests.Session() as session:

            mempool_payload = _fetch_json(
                session,
                ENDPOINTS["mempool"],
                request_timeout,
            )

            fees_payload = _fetch_json(
                session,
                ENDPOINTS["fees"],
                request_timeout,
            )

            tip_payload = _fetch_json(
                session,
                ENDPOINTS["tip_height"],
                request_timeout,
            )

            difficulty_payload = _fetch_json(
                session,
                ENDPOINTS["difficulty_adjustment"],
                request_timeout,
            )

    except MempoolSpaceCollectorError:
        raise

    except requests.RequestException as exc:
        raise MempoolSpaceCollectorError(
            f"On-chain collection failed: {exc}"
        ) from exc

    mempool = _validate_mempool(
        mempool_payload
    )

    fees = _validate_fees(
        fees_payload
    )

    tip_height = _validate_tip_height(
        tip_payload
    )

    difficulty = _validate_difficulty_adjustment(
        difficulty_payload
    )

    record = {
        "schema_version": 1,
        "source": "mempool_space",
        "asset": "BTC",
        "timestamp_basis": "collection_observation",
        "collected_at": collected_at,
        "provider_timestamp": None,
        "mempool": mempool,
        "fees": fees,
        "blockchain": {
            "tip_height": tip_height,
        },
        "difficulty_adjustment": difficulty,
    }

    try:
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

            file.flush()

    except OSError as exc:
        raise MempoolSpaceCollectorError(
            f"Cannot save on-chain raw data: {exc}"
        ) from exc

    return record