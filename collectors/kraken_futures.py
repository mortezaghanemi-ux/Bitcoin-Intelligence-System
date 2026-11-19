"""Kraken Futures public market-data collector."""

from __future__ import annotations

import json
import math

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = (
    "https://futures.kraken.com/derivatives/api/v3/tickers"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "derivatives"
    / "kraken_futures"
)

DEFAULT_TIMEOUT = 15.0

MAX_PROVIDER_AGE_MINUTES = 15

NUMERIC_FIELDS = (
    "markPrice",
    "bid",
    "bidSize",
    "ask",
    "askSize",
    "vol24h",
    "volumeQuote",
    "openInterest",
    "open24h",
    "indexPrice",
    "last",
    "lastSize",
    "fundingRate",
    "fundingRatePrediction",
)


class KrakenFuturesCollectorError(RuntimeError):
    """Raised when Kraken Futures collection fails."""


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
    ):
        raise ValueError(
            "Invalid collection_time timezone."
        ) from None

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


def _validate_symbol(
    symbol: str,
) -> str:
    """Validate an optional futures symbol."""

    if (
        not isinstance(symbol, str)
        or not symbol.strip()
    ):
        raise ValueError(
            "symbol must be a non-empty string."
        )

    return symbol.strip().upper()


def _is_bitcoin_instrument(
    ticker: dict[str, Any],
) -> bool:
    """Identify Bitcoin futures instruments."""

    symbol = str(
        ticker.get("symbol", "")
    ).upper()

    pair = str(
        ticker.get("pair", "")
    ).upper()

    return (
        "XBT" in symbol
        or "BTC" in symbol
        or "XBT" in pair
        or "BTC" in pair
    )


def _parse_provider_time(
    value: Any,
) -> datetime:
    """Parse the provider's ISO-8601 timestamp."""

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise KrakenFuturesCollectorError(
            "Ticker is missing lastTime."
        )

    try:
        parsed = datetime.fromisoformat(
            value.strip().replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        raise KrakenFuturesCollectorError(
            "Invalid ticker lastTime timestamp."
        ) from None

    if parsed.tzinfo is None:
        raise KrakenFuturesCollectorError(
            "Ticker lastTime must include timezone."
        )

    return parsed.astimezone(timezone.utc)


def _validate_numeric_fields(
    ticker: dict[str, Any],
) -> dict[str, float | None]:
    """Validate available numeric ticker values."""

    values = {}

    for field in NUMERIC_FIELDS:

        raw_value = ticker.get(field)

        if raw_value is None:
            values[field] = None
            continue

        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(raw_value)
        ):
            raise KrakenFuturesCollectorError(
                f"Invalid numeric field: {field}."
            )

        values[field] = float(raw_value)

    return values


def _load_existing_keys(
    path: Path,
) -> set[tuple[str, str]]:
    """Load existing symbol/timestamp keys."""

    existing = set()

    if not path.exists():
        return existing

    read_failed = False

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

                invalid_json = False
                record = None

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    invalid_json = True

                if invalid_json:
                    raise KrakenFuturesCollectorError(
                        f"Invalid JSONL at line {line_number}."
                    )

                if not isinstance(record, dict):
                    raise KrakenFuturesCollectorError(
                        f"Invalid record at line {line_number}."
                    )

                symbol = record.get("symbol")

                timestamp = record.get(
                    "provider_timestamp"
                )

                if (
                    not isinstance(symbol, str)
                    or not isinstance(timestamp, str)
                ):
                    raise KrakenFuturesCollectorError(
                        f"Missing record identity at line {line_number}."
                    )

                existing.add(
                    (symbol, timestamp)
                )

    except OSError:
        read_failed = True

    if read_failed:
        raise KrakenFuturesCollectorError(
            "Cannot read existing futures history."
        )

    return existing


def _parse_ticker(
    ticker: dict[str, Any],
    collection_time: datetime,
) -> dict[str, Any]:
    """Validate and normalize one Bitcoin futures ticker."""

    symbol = ticker.get("symbol")

    pair = ticker.get("pair")

    if (
        not isinstance(symbol, str)
        or not symbol.strip()
    ):
        raise KrakenFuturesCollectorError(
            "Ticker has an invalid symbol."
        )

    if (
        not isinstance(pair, str)
        or not pair.strip()
    ):
        raise KrakenFuturesCollectorError(
            "Ticker has an invalid pair."
        )

    provider_time = _parse_provider_time(
        ticker.get("lastTime")
    )

    if provider_time > collection_time:
        raise KrakenFuturesCollectorError(
            "Ticker timestamp is in the future."
        )

    maximum_age = timedelta(
        minutes=MAX_PROVIDER_AGE_MINUTES
    )

    if collection_time - provider_time > maximum_age:
        raise KrakenFuturesCollectorError(
            "Ticker data is stale."
        )

    numeric_values = _validate_numeric_fields(
        ticker
    )

    mark_price = numeric_values["markPrice"]

    last_price = numeric_values["last"]

    if (
        mark_price is not None
        and mark_price <= 0
    ):
        raise KrakenFuturesCollectorError(
            "Mark price must be greater than zero."
        )

    if (
        last_price is not None
        and last_price <= 0
    ):
        raise KrakenFuturesCollectorError(
            "Last price must be greater than zero."
        )

    bid = numeric_values["bid"]

    ask = numeric_values["ask"]

    if (
        bid is not None
        and ask is not None
        and bid > ask
    ):
        raise KrakenFuturesCollectorError(
            "Ticker bid price cannot exceed ask price."
        )

    suspended = ticker.get(
        "suspended",
        False,
    )

    if not isinstance(suspended, bool):
        raise KrakenFuturesCollectorError(
            "Invalid suspended status."
        )

    tag = ticker.get("tag")

    if (
        tag is not None
        and not isinstance(tag, str)
    ):
        raise KrakenFuturesCollectorError(
            "Invalid futures contract tag."
        )

    return {
        "schema_version": 1,
        "source": "kraken_futures",
        "asset": "BTC",
        "symbol": symbol.strip(),
        "pair": pair.strip(),
        "contract_type": tag,
        "provider_timestamp": _iso_utc(
            provider_time
        ),
        "collected_at": _iso_utc(
            collection_time
        ),
        "suspended": suspended,
        **numeric_values,
    }


def _fetch_payload(
    timeout: float,
) -> dict[str, Any]:
    """Fetch the public Kraken Futures API response safely."""

    request_failed = False

    invalid_json = False

    payload = None

    try:
        response = requests.get(
            API_URL,
            headers={
                "accept": "application/json",
            },
            timeout=timeout,
        )

        response.raise_for_status()

    except requests.RequestException:
        request_failed = True

    if request_failed:
        raise KrakenFuturesCollectorError(
            "Kraken Futures request failed."
        )

    try:
        payload = response.json()

    except ValueError:
        invalid_json = True

    if invalid_json:
        raise KrakenFuturesCollectorError(
            "Kraken Futures returned invalid JSON."
        )

    if not isinstance(payload, dict):
        raise KrakenFuturesCollectorError(
            "Kraken Futures response must be a JSON object."
        )

    return payload


def collect_kraken_futures(
    *,
    symbol: str | None = None,
    output_dir: str | Path | None = None,
    collection_time: datetime | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """
    Collect public Bitcoin futures ticker information.

    No private API credentials are used.

    Only Bitcoin futures instruments are stored.

    Future and stale provider observations are rejected.

    Existing raw observations are preserved.

    Duplicate symbol/timestamp records are skipped.
    """

    request_timeout = _validate_timeout(
        timeout
    )

    checked_at = _normalize_utc(
        collection_time
    )

    requested_symbol = (
        _validate_symbol(symbol)
        if symbol is not None
        else None
    )

    payload = _fetch_payload(
        request_timeout
    )

    if payload.get("result") not in (
        None,
        "success",
    ):
        raise KrakenFuturesCollectorError(
            "Kraken Futures API returned an unsuccessful result."
        )

    tickers = payload.get("tickers")

    if not isinstance(tickers, list):
        raise KrakenFuturesCollectorError(
            "Kraken Futures response has no valid tickers list."
        )

    destination = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    output_file = (
        destination
        / "kraken_futures_BTC.jsonl"
    )

    existing = _load_existing_keys(
        output_file
    )

    records = []

    for ticker in tickers:

        if not isinstance(ticker, dict):
            raise KrakenFuturesCollectorError(
                "Invalid ticker structure in response."
            )

        if not _is_bitcoin_instrument(ticker):
            continue

        ticker_symbol = ticker.get("symbol")

        if (
            requested_symbol is not None
            and str(ticker_symbol).upper()
            != requested_symbol
        ):
            continue

        record = _parse_ticker(
            ticker,
            checked_at,
        )

        key = (
            record["symbol"],
            record["provider_timestamp"],
        )

        if key in existing:
            continue

        existing.add(key)

        records.append(record)

    if not records:
        return []

    save_failed = False

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

    except OSError:
        save_failed = True

    if save_failed:
        raise KrakenFuturesCollectorError(
            "Cannot save Kraken Futures data."
        )

    return records