"""CoinGecko Bitcoin market-data collector."""

from __future__ import annotations

import json
import math
import os

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "market"
    / "coingecko"
)

DEFAULT_TIMEOUT = 15.0


class CoinGeckoCollectorError(RuntimeError):
    """Raised when CoinGecko collection fails."""


def _normalize_utc(
    value: datetime | None,
) -> datetime:
    """Validate and normalize a UTC datetime."""

    if value is None:
        return datetime.now(timezone.utc)

    if not isinstance(value, datetime):
        raise ValueError(
            "collection_time must be a datetime."
        )

    if value.tzinfo is None:
        raise ValueError(
            "collection_time must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid collection_time timezone."
        ) from exc

    if offset is None or offset.total_seconds() != 0:
        raise ValueError(
            "collection_time must use UTC."
        )

    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    """Convert datetime to ISO-8601 UTC."""

    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_days(days: int) -> int:
    """Validate CoinGecko history window."""

    allowed = {
        1,
        7,
        14,
        30,
        90,
        180,
        365,
    }

    if (
        isinstance(days, bool)
        or not isinstance(days, int)
        or days not in allowed
    ):
        raise ValueError(
            "days must be one of "
            "1, 7, 14, 30, 90, 180, 365."
        )

    return days


def _validate_currency(
    currency: str,
) -> str:
    """Validate quote currency."""

    if (
        not isinstance(currency, str)
        or not currency.strip()
        or not currency.strip().isalnum()
    ):
        raise ValueError(
            "Invalid quote currency."
        )

    return currency.strip().lower()


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


def _load_existing_keys(
    path: Path,
) -> set[tuple[str, str]]:
    """Load existing metric/timestamp keys."""

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
                    raise CoinGeckoCollectorError(
                        f"Invalid JSONL at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise CoinGeckoCollectorError(
                        f"Invalid record at line {line_number}."
                    )

                metric = record.get("metric")

                timestamp = record.get(
                    "provider_timestamp"
                )

                if (
                    not isinstance(metric, str)
                    or not isinstance(timestamp, str)
                ):
                    raise CoinGeckoCollectorError(
                        f"Invalid record fields at line {line_number}."
                    )

                existing.add(
                    (metric, timestamp)
                )

    except OSError as exc:
        raise CoinGeckoCollectorError(
            "Cannot read existing CoinGecko history."
        ) from None

    return existing


def _parse_series(
    payload: dict[str, Any],
    metric_name: str,
    collection_time: datetime,
) -> list[dict[str, Any]]:
    """Parse and validate one CoinGecko data series."""

    series = payload.get(metric_name)

    if not isinstance(series, list):
        raise CoinGeckoCollectorError(
            f"Missing or invalid {metric_name} series."
        )

    records = []

    for index, point in enumerate(series):

        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
        ):
            raise CoinGeckoCollectorError(
                f"Invalid {metric_name} point at index {index}."
            )

        timestamp_ms, raw_value = point

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, (int, float))
            or not math.isfinite(timestamp_ms)
            or timestamp_ms < 0
        ):
            raise CoinGeckoCollectorError(
                f"Invalid timestamp in {metric_name}."
            )

        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(raw_value)
            or raw_value < 0
        ):
            raise CoinGeckoCollectorError(
                f"Invalid value in {metric_name}."
            )

        try:
            observed_at = datetime.fromtimestamp(
                timestamp_ms / 1000.0,
                tz=timezone.utc,
            )

        except (
            ValueError,
            OverflowError,
            OSError,
        ):
            raise CoinGeckoCollectorError(
                f"Timestamp out of range in {metric_name}."
            ) from None

        if observed_at > collection_time:
            raise CoinGeckoCollectorError(
                "Future observation detected."
            )

        records.append(
            {
                "schema_version": 1,
                "source": "coingecko",
                "asset": "BTC",
                "metric": metric_name,
                "provider_timestamp": _iso_utc(
                    observed_at
                ),
                "collected_at": _iso_utc(
                    collection_time
                ),
                "value": float(raw_value),
            }
        )

    return records


def collect_coingecko_market_chart(
    *,
    vs_currency: str = "usd",
    days: int = 1,
    output_dir: str | Path | None = None,
    collection_time: datetime | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Collect Bitcoin price, market capitalization and volume.

    API credentials are optional.

    Credentials are read from:
        COINGECKO_API_KEY

    Raw observations are appended to JSONL.
    Existing observations are never overwritten.
    Duplicate metric/timestamp records are prevented.
    """

    currency = _validate_currency(
        vs_currency
    )

    history_days = _validate_days(
        days
    )

    request_timeout = _validate_timeout(
        timeout
    )

    checked_at = _normalize_utc(
        collection_time
    )

    credential = (
        api_key
        if api_key is not None
        else os.getenv("COINGECKO_API_KEY")
    )

    if credential is not None:

        if (
            not isinstance(credential, str)
            or not credential.strip()
        ):
            raise ValueError(
                "api_key must be a non-empty string."
            )

    headers = {
        "accept": "application/json",
    }

    if credential:
        headers["x-cg-demo-api-key"] = (
            credential.strip()
        )

    # Never expose request exception details.
    # HTTP libraries may include sensitive headers
    # or API credentials in exception messages.
    try:
        response = requests.get(
            API_URL,
            params={
                "vs_currency": currency,
                "days": history_days,
            },
            headers=headers,
            timeout=request_timeout,
        )

        response.raise_for_status()

        payload = response.json()

    except requests.RequestException:
        raise CoinGeckoCollectorError(
            "CoinGecko request failed."
        ) from None

    except ValueError:
        raise CoinGeckoCollectorError(
            "CoinGecko returned invalid JSON."
        ) from None

    if not isinstance(payload, dict):
        raise CoinGeckoCollectorError(
            "CoinGecko response must be a JSON object."
        )

    destination = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    output_file = (
        destination
        / f"coingecko_BTC_{currency}_{history_days}d.jsonl"
    )

    existing = _load_existing_keys(
        output_file
    )

    all_records = []

    for metric_name in (
        "prices",
        "market_caps",
        "total_volumes",
    ):

        metric_records = _parse_series(
            payload,
            metric_name,
            checked_at,
        )

        for record in metric_records:

            record["quote_currency"] = currency

            key = (
                record["metric"],
                record["provider_timestamp"],
            )

            if key in existing:
                continue

            existing.add(key)

            all_records.append(record)

    if not all_records:
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

            for record in all_records:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

            file.flush()

    except OSError:
        raise CoinGeckoCollectorError(
            "Cannot save CoinGecko raw data."
        ) from None

    return all_records