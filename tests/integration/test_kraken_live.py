"""Live integration tests for the Kraken OHLCV API.

These tests make real HTTP requests to Kraken.
They are skipped during normal unit-test runs.

To enable:
    RUN_KRAKEN_LIVE_TESTS=1 pytest -q tests/integration/test_kraken_live.py
"""

from __future__ import annotations

import json
import math
import os

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from collectors.kraken_ohlcv import (
    API_URL,
    collect_kraken_ohlcv,
)


RUN_LIVE_TESTS = (
    os.getenv("RUN_KRAKEN_LIVE_TESTS", "").strip() == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_LIVE_TESTS,
    reason=(
        "Live Kraken integration tests are disabled. "
        "Set RUN_KRAKEN_LIVE_TESTS=1 to enable them."
    ),
)


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp."""

    assert isinstance(value, str)
    assert value.endswith("Z")

    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)

    return parsed.astimezone(timezone.utc)


def finite_number(value: object, field: str) -> float:
    """Convert a provider value into a finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        pytest.fail(
            f"Kraken returned an invalid {field} value: {value!r}"
        )

    assert math.isfinite(number), (
        f"Kraken returned a non-finite {field} value."
    )

    return number


def test_live_kraken_ohlcv_api_returns_valid_closed_candles(
    tmp_path: Path,
):
    """Fetch and validate real closed XBTUSD 5-minute candles."""

    records = collect_kraken_ohlcv(
        pair="XBTUSD",
        timeframe="5m",
        output_dir=tmp_path,
        collection_time=None,
        timeout=20,
    )

    # The temporary output directory is isolated for this test.
    # A successful live request should provide closed candles.
    assert records, (
        "Kraken API returned no new closed candles. "
        "Check API availability and the returned OHLC data."
    )

    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    assert output_file.is_file(), (
        "The collector did not create the expected JSONL file."
    )

    stored_lines = [
        line
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    stored_records = [
        json.loads(line)
        for line in stored_lines
    ]

    assert stored_records == records, (
        "Returned records do not match the persisted JSONL records."
    )

    assert len(stored_records) == len(records)

    seen_open_times = set()

    for record in records:

        assert record["schema_version"] == 1
        assert record["source"] == "kraken"
        assert record["pair"] == "XBTUSD"
        assert record["timeframe"] == "5m"

        open_time = parse_utc_timestamp(
            record["open_time"]
        )

        close_time = parse_utc_timestamp(
            record["close_time"]
        )

        collected_at = parse_utc_timestamp(
            record["collected_at"]
        )

        # Every candle must represent exactly five minutes.
        assert close_time - open_time == timedelta(
            minutes=5
        )

        # Only fully closed candles may be stored.
        assert close_time <= collected_at

        # Duplicate candle timestamps are not allowed.
        assert record["open_time"] not in seen_open_times
        seen_open_times.add(record["open_time"])

        open_price = finite_number(
            record["open"],
            "open",
        )

        high_price = finite_number(
            record["high"],
            "high",
        )

        low_price = finite_number(
            record["low"],
            "low",
        )

        close_price = finite_number(
            record["close"],
            "close",
        )

        volume = finite_number(
            record["volume"],
            "volume",
        )

        # OHLC consistency checks.
        assert high_price >= max(
            open_price,
            close_price,
            low_price,
        )

        assert low_price <= min(
            open_price,
            close_price,
            high_price,
        )

        assert volume >= 0

        # The original Kraken candle must be preserved.
        provider_candle = record["provider_candle"]

        assert isinstance(provider_candle, list)
        assert len(provider_candle) == 8

        assert finite_number(
            provider_candle[0],
            "provider timestamp",
        ) >= 0

        assert finite_number(
            provider_candle[1],
            "provider open",
        ) == open_price

        assert finite_number(
            provider_candle[2],
            "provider high",
        ) == high_price

        assert finite_number(
            provider_candle[3],
            "provider low",
        ) == low_price

        assert finite_number(
            provider_candle[4],
            "provider close",
        ) == close_price

        assert finite_number(
            provider_candle[6],
            "provider volume",
        ) == volume

    # Records must be ordered chronologically.
    timestamps = [
        parse_utc_timestamp(record["open_time"])
        for record in records
    ]

    assert timestamps == sorted(timestamps)

    # The collector must use the configured public Kraken endpoint.
    assert API_URL == (
        "https://api.kraken.com/0/public/OHLC"
    )