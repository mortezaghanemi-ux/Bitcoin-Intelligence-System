"""FRED economic data collector."""

from __future__ import annotations

import json
import math
import os
import re

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "macro"
    / "fred"
)

DEFAULT_TIMEOUT = 20.0
DEFAULT_PAGE_SIZE = 100000


class FREDCollectorError(RuntimeError):
    """Raised when FRED collection fails."""


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


def _validate_api_key(
    api_key: str | None,
) -> str:
    """Resolve and validate the FRED API key."""

    credential = (
        api_key
        if api_key is not None
        else os.getenv("FRED_API_KEY")
    )

    if (
        not isinstance(credential, str)
        or not credential.strip()
    ):
        raise ValueError(
            "FRED_API_KEY is required. "
            "Set it as an environment variable."
        )

    credential = credential.strip()

    if not re.fullmatch(
        r"[A-Za-z0-9]{32}",
        credential,
    ):
        raise ValueError(
            "FRED API key must contain 32 alphanumeric characters."
        )

    return credential


def _validate_series_ids(
    series_ids: list[str] | tuple[str, ...],
) -> list[str]:
    """Validate requested FRED series identifiers."""

    if not isinstance(
        series_ids,
        (list, tuple),
    ):
        raise ValueError(
            "series_ids must be a list or tuple."
        )

    if not series_ids:
        raise ValueError(
            "At least one FRED series ID is required."
        )

    normalized = []

    for series_id in series_ids:

        if (
            not isinstance(series_id, str)
            or not series_id.strip()
        ):
            raise ValueError(
                "Every series ID must be a non-empty string."
            )

        value = series_id.strip().upper()

        if not re.fullmatch(
            r"[A-Z0-9_]+",
            value,
        ):
            raise ValueError(
                f"Invalid FRED series ID: {value}."
            )

        if value not in normalized:
            normalized.append(value)

    return normalized


def _validate_date(
    value: date | str | None,
    field: str,
) -> str | None:
    """Validate a FRED observation date."""

    if value is None:
        return None

    if isinstance(value, datetime):
        raise ValueError(
            f"{field} must be a date, not datetime."
        )

    if isinstance(value, date):
        return value.isoformat()

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be YYYY-MM-DD."
        )

    try:
        parsed = date.fromisoformat(value)

    except ValueError as exc:
        raise ValueError(
            f"{field} must be a valid YYYY-MM-DD date."
        ) from exc

    if parsed.isoformat() != value:
        raise ValueError(
            f"{field} must use YYYY-MM-DD format."
        )

    return value


def _validate_page_size(
    value: int,
) -> int:
    """Validate FRED pagination size."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= DEFAULT_PAGE_SIZE
    ):
        raise ValueError(
            "page_size must be between 1 and 100000."
        )

    return value


def _load_existing_keys(
    path: Path,
) -> set[tuple[str, str, str, str]]:
    """Load existing observation identities."""

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
                    raise FREDCollectorError(
                        f"Invalid JSONL at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise FREDCollectorError(
                        f"Invalid record at line {line_number}."
                    )

                series_id = record.get("series_id")
                observation_date = record.get(
                    "observation_date"
                )
                realtime_start = record.get(
                    "realtime_start"
                )
                realtime_end = record.get(
                    "realtime_end"
                )

                if not all(
                    isinstance(item, str)
                    and item
                    for item in (
                        series_id,
                        observation_date,
                        realtime_start,
                        realtime_end,
                    )
                ):
                    raise FREDCollectorError(
                        f"Missing observation identity "
                        f"at line {line_number}."
                    )

                existing.add(
                    (
                        series_id,
                        observation_date,
                        realtime_start,
                        realtime_end,
                    )
                )

    except OSError as exc:
        raise FREDCollectorError(
            f"Cannot read existing FRED history: {exc}"
        ) from exc

    return existing


def _parse_observation(
    observation: Any,
    series_id: str,
    collection_time: datetime,
) -> dict[str, Any]:
    """Validate and normalize one FRED observation."""

    if not isinstance(observation, dict):
        raise FREDCollectorError(
            "FRED observation must be a JSON object."
        )

    observation_date = observation.get("date")

    if not isinstance(observation_date, str):
        raise FREDCollectorError(
            "FRED observation has no valid date."
        )

    try:
        parsed_date = date.fromisoformat(
            observation_date
        )

    except ValueError as exc:
        raise FREDCollectorError(
            "FRED observation contains an invalid date."
        ) from exc

    if parsed_date.isoformat() != observation_date:
        raise FREDCollectorError(
            "FRED observation date must use YYYY-MM-DD."
        )

    realtime_start = observation.get(
        "realtime_start"
    )

    realtime_end = observation.get(
        "realtime_end"
    )

    for field, value in (
        ("realtime_start", realtime_start),
        ("realtime_end", realtime_end),
    ):

        if not isinstance(value, str):
            raise FREDCollectorError(
                f"Invalid FRED {field}."
            )

        try:
            parsed = date.fromisoformat(value)

        except ValueError as exc:
            raise FREDCollectorError(
                f"Invalid FRED {field} date."
            ) from exc

        if parsed.isoformat() != value:
            raise FREDCollectorError(
                f"Invalid FRED {field} format."
            )

    raw_value = observation.get("value")

    if not isinstance(raw_value, str):
        raise FREDCollectorError(
            "FRED observation value must be a string."
        )

    if raw_value == ".":
        numeric_value = None
        value_status = "MISSING"

    else:

        try:
            numeric_value = float(raw_value)

        except ValueError as exc:
            raise FREDCollectorError(
                "FRED observation contains an invalid value."
            ) from exc

        if not math.isfinite(numeric_value):
            raise FREDCollectorError(
                "FRED observation contains a non-finite value."
            )

        value_status = "VALID"

    return {
        "schema_version": 1,
        "source": "fred",
        "series_id": series_id,
        "observation_date": observation_date,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "value": numeric_value,
        "raw_value": raw_value,
        "value_status": value_status,
        "collected_at": _iso_utc(
            collection_time
        ),
        "provider_timestamp": None,
    }


def _fetch_observation_page(
    session: requests.Session,
    *,
    api_key: str,
    series_id: str,
    observation_start: str | None,
    observation_end: str | None,
    limit: int,
    offset: int,
    timeout: float,
) -> dict[str, Any]:
    """Fetch one page of FRED observations."""

    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "sort_order": "asc",
        "limit": limit,
        "offset": offset,
    }

    if observation_start is not None:
        params["observation_start"] = observation_start

    if observation_end is not None:
        params["observation_end"] = observation_end

    try:
        response = session.get(
            API_URL,
            params=params,
            headers={
                "accept": "application/json",
            },
            timeout=timeout,
        )

        response.raise_for_status()

        payload = response.json()

    except requests.RequestException as exc:
        # Do not include the exception text because the
        # request URL may contain the private API key.
        raise FREDCollectorError(
            f"FRED request failed for series {series_id}."
        ) from exc

    except ValueError as exc:
        raise FREDCollectorError(
            "FRED returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise FREDCollectorError(
            "FRED response must be a JSON object."
        )

    if (
        "error_code" in payload
        or "error_message" in payload
    ):
        raise FREDCollectorError(
            f"FRED API rejected series {series_id}."
        )

    observations = payload.get(
        "observations"
    )

    if not isinstance(observations, list):
        raise FREDCollectorError(
            "FRED response has no valid observations list."
        )

    count = payload.get("count")

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise FREDCollectorError(
            "FRED response has an invalid observation count."
        )

    return {
        "count": count,
        "observations": observations,
    }


def _fetch_series_observations(
    session: requests.Session,
    *,
    api_key: str,
    series_id: str,
    observation_start: str | None,
    observation_end: str | None,
    page_size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """Fetch all available observation pages."""

    observations = []
    offset = 0
    total_count = None

    while True:

        page = _fetch_observation_page(
            session,
            api_key=api_key,
            series_id=series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            limit=page_size,
            offset=offset,
            timeout=timeout,
        )

        if total_count is None:
            total_count = page["count"]

        elif page["count"] != total_count:
            raise FREDCollectorError(
                f"FRED observation count changed "
                f"during collection of {series_id}."
            )

        current = page["observations"]

        if len(current) > page_size:
            raise FREDCollectorError(
                "FRED returned more observations than requested."
            )

        observations.extend(current)

        offset += len(current)

        if offset >= total_count:
            break

        if not current:
            raise FREDCollectorError(
                f"FRED pagination stopped unexpectedly "
                f"for {series_id}."
            )

    return observations


def collect_fred_observations(
    series_ids: list[str] | tuple[str, ...],
    *,
    observation_start: date | str | None = None,
    observation_end: date | str | None = None,
    output_dir: str | Path | None = None,
    collection_time: datetime | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    page_size: int = DEFAULT_PAGE_SIZE,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Collect economic observations from one or more FRED series.

    The FRED API key is read from FRED_API_KEY unless
    api_key is explicitly supplied.

    API credentials are never written to output files.

    Missing values are preserved as None with raw_value ".".

    Historical observations are append-only.

    Revised observation vintages are preserved separately.
    """

    credential = _validate_api_key(
        api_key
    )

    normalized_series = _validate_series_ids(
        series_ids
    )

    start_date = _validate_date(
        observation_start,
        "observation_start",
    )

    end_date = _validate_date(
        observation_end,
        "observation_end",
    )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise ValueError(
            "observation_start cannot be after observation_end."
        )

    request_timeout = _validate_timeout(
        timeout
    )

    checked_at = _normalize_utc(
        collection_time
    )

    requested_page_size = _validate_page_size(
        page_size
    )

    destination = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    output_file = (
        destination
        / "fred_observations.jsonl"
    )

    existing = _load_existing_keys(
        output_file
    )

    records = []

    try:
        with requests.Session() as session:

            for series_id in normalized_series:

                observations = _fetch_series_observations(
                    session,
                    api_key=credential,
                    series_id=series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                    page_size=requested_page_size,
                    timeout=request_timeout,
                )

                for observation in observations:

                    record = _parse_observation(
                        observation,
                        series_id,
                        checked_at,
                    )

                    key = (
                        record["series_id"],
                        record["observation_date"],
                        record["realtime_start"],
                        record["realtime_end"],
                    )

                    if key in existing:
                        continue

                    existing.add(key)

                    records.append(record)

    except FREDCollectorError:
        raise

    except requests.RequestException as exc:
        raise FREDCollectorError(
            "FRED collection failed."
        ) from exc

    if not records:
        return []

    try:
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open(
            "a",
            encoding="utf-8",
        ) as file:

            for record in records:

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
        raise FREDCollectorError(
            f"Cannot save FRED raw data: {exc}"
        ) from exc

    return records