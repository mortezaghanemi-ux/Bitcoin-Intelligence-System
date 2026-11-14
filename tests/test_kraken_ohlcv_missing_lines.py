"""Tests for remaining Kraken OHLCV collector branches."""

import json
from datetime import datetime, timezone

import pytest

import collectors.kraken_ohlcv as kraken


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def make_candle(timestamp):
    return [
        timestamp,
        "100",
        "110",
        "90",
        "105",
        "102",
        "12.5",
        100,
    ]


def make_payload(candles):
    return {
        "error": [],
        "result": {
            "XXBTZUSD": candles,
            "last": 123456,
        },
    }


def mock_kraken(monkeypatch, payload):
    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload),
    )


def test_uses_current_utc_time_when_collection_time_is_omitted(
    tmp_path,
    monkeypatch,
):
    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(monkeypatch, make_payload(candles))

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
    )

    assert len(records) == 1

    collected_at = records[0]["collected_at"]

    assert collected_at.endswith("Z")

    parsed_time = datetime.fromisoformat(
        collected_at.replace("Z", "+00:00")
    )

    assert parsed_time.tzinfo == timezone.utc


def test_skips_blank_lines_in_existing_history(
    tmp_path,
    monkeypatch,
):
    output_file = tmp_path / "kraken_XBTUSD_5m.jsonl"

    existing_record = {
        "open_time": "2026-01-01T00:00:00Z",
        "source": "kraken",
    }

    output_file.write_text(
        "\n" + json.dumps(existing_record) + "\n\n",
        encoding="utf-8",
    )

    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
        make_candle(1767226200),
    ]

    mock_kraken(monkeypatch, make_payload(candles))

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=datetime(
            2026, 1, 1, 0, 11,
            tzinfo=timezone.utc,
        ),
    )

    assert len(records) == 1

    assert (
        records[0]["open_time"]
        == "2026-01-01T00:05:00Z"
    )

    saved_lines = [
        line
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(saved_lines) == 2


def test_wraps_existing_history_read_error(
    tmp_path,
    monkeypatch,
):
    output_file = tmp_path / "kraken_XBTUSD_5m.jsonl"

    output_file.mkdir()

    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(monkeypatch, make_payload(candles))

    with pytest.raises(
        kraken.KrakenCollectorError,
        match="Cannot read existing candle history",
    ):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=datetime(
                2026, 1, 1, 0, 6,
                tzinfo=timezone.utc,
            ),
        )