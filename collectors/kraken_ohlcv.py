"""Kraken public OHLCV data collector."""

from __future__ import annotations

import json
import math
import re

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = "https://api.kraken.com/0/public/OHLC"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "market"
    / "ohlcv"
)

TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


class KrakenCollectorError(Exception):
    """Raised when Kraken collection fails."""


def _utc_datetime(value: datetime | None) -> datetime:
    """Return a validated UTC datetime."""

    if value is None:
        return datetime.now(timezone.utc)

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError(
            "collection_time must be timezone-aware UTC."
        )

    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    """Return an ISO-8601 UTC timestamp."""

    return value.isoformat().replace("+00:00", "Z")


def _validate_pair(pair: str) -> str:
    """Validate a Kraken trading pair."""

    if (
        not isinstance(pair, str)
        or not re.fullmatch(r"[A-Za-z0-9]+", pair)
    ):
        raise ValueError("Invalid trading pair.")

    return pair.upper()


def _resolve_candles(result: dict) -> list:
    """Find the candle array in a Kraken response."""

    candidates = [
        value
        for key, value in result.items()
        if key != "last" and isinstance(value, list)
    ]

    if len(candidates) != 1:
        raise KrakenCollectorError(
            "Unexpected Kraken OHLC response structure."
        )

    return candidates[0]


def _load_existing_timestamps(path: Path) -> set[str]:
    """Read existing candle timestamps for duplicate prevention."""

    existing = set()

    if not path.exists():
        return existing

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise KrakenCollectorError(
                        f"Invalid JSONL record at line {line_number}."
                    ) from exc

                timestamp = record.get("open_time")

                if not isinstance(timestamp, str):
                    raise KrakenCollectorError(
                        f"Missing open_time at line {line_number}."
                    )

                existing.add(timestamp)

    except OSError as exc:
        raise KrakenCollectorError(
            f"Cannot read existing candle history: {exc}"
        ) from exc

    return existing


def _validate_candle(candle: Any) -> float:
    """Validate the raw Kraken candle structure."""

    if not isinstance(candle, (list, tuple)) or len(candle) != 8:
        raise KrakenCollectorError(
            "Invalid Kraken candle structure."
        )

    try:
        timestamp = float(candle[0])

    except (TypeError, ValueError, OverflowError) as exc:
        raise KrakenCollectorError(
            "Invalid candle timestamp."
        ) from exc

    if not math.isfinite(timestamp) or timestamp < 0:
        raise KrakenCollectorError(
            "Candle timestamp must be finite and non-negative."
        )

    return timestamp


def collect_kraken_ohlcv(
    pair: str = "XBTUSD",
    timeframe: str = "5m",
    *,
    output_dir: str | Path | None = None,
    collection_time: datetime | None = None,
    timeout: float = 15,
) -> list[dict[str, Any]]:
    """
    Collect closed OHLCV candles from Kraken.

    Original provider candle values are preserved.

    Returns only newly stored closed candles.
    """

    pair = _validate_pair(pair)

    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout must be a positive finite number."
        )

    checked_at = _utc_datetime(collection_time)

    interval = TIMEFRAME_MINUTES[timeframe]

    try:
        response = requests.get(
            API_URL,
            params={
                "pair": pair,
                "interval": interval,
            },
            timeout=timeout,
        )

        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise KrakenCollectorError(
            f"Kraken request failed: {exc}"
        ) from exc

    except ValueError as exc:
        raise KrakenCollectorError(
            "Kraken returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise KrakenCollectorError(
            "Kraken response must be a JSON object."
        )

    errors = payload.get("error")

    if not isinstance(errors, list):
        raise KrakenCollectorError(
            "Invalid Kraken error field."
        )

    if errors:
        raise KrakenCollectorError(
            "Kraken API error: " + "; ".join(map(str, errors))
        )

    result = payload.get("result")

    if not isinstance(result, dict):
        raise KrakenCollectorError(
            "Kraken response does not contain valid results."
        )

    candles = _resolve_candles(result)

    if not candles:
        return []

    # Kraken always includes the currently forming candle
    # as the final entry. It must never be used as a
    # completed candle.
    closed_candles = candles[:-1]

    destination = Path(
        output_dir
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    filename = (
        f"kraken_{pair}_{timeframe}.jsonl"
    )

    output_file = destination / filename

    existing = _load_existing_timestamps(
        output_file
    )

    new_records = []

    interval_seconds = interval * 60

    for candle in closed_candles:

        open_epoch = _validate_candle(candle)

        try:
            open_time = datetime.fromtimestamp(
                open_epoch,
                tz=timezone.utc,
            )

        except (ValueError, OverflowError, OSError) as exc:
            raise KrakenCollectorError(
                "Candle timestamp is outside the supported range."
            ) from exc

        close_time = (
            open_time
            + timedelta(seconds=interval_seconds)
        )

        open_iso = _iso_utc(open_time)

        if open_iso in existing:
            continue

        # Never store a candle that has not yet closed.
        if close_time > checked_at:
            continue

        record = {
            "schema_version": 1,
            "source": "kraken",
            "pair": pair,
            "timeframe": timeframe,
            "open_time": open_iso,
            "close_time": _iso_utc(close_time),
            "collected_at": _iso_utc(checked_at),
            "provider_candle": list(candle),
        }

        new_records.append(record)
        existing.add(open_iso)

    if not new_records:
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

            for record in new_records:
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
        raise KrakenCollectorError(
            f"Cannot save raw candle data: {exc}"
        ) from exc

    return new_records