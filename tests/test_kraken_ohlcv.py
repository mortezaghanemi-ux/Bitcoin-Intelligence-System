"""Tests for the Kraken OHLCV collector."""

import json

from datetime import datetime, timezone

import pytest
import requests

import collectors.kraken_ohlcv as kraken


class FakeResponse:
    """Mock HTTP response."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def make_candle(timestamp, price="100"):
    """Create a valid Kraken OHLC candle."""

    return [
        timestamp,
        price,
        "110",
        "90",
        "105",
        "102",
        "12.5",
        100,
    ]


def make_payload(candles):
    """Create a valid Kraken API response."""

    return {
        "error": [],
        "result": {
            "XXBTZUSD": candles,
            "last": 123456,
        },
    }


@pytest.fixture
def collection_time():
    return datetime(
        2026,
        1,
        1,
        0,
        6,
        tzinfo=timezone.utc,
    )


@pytest.fixture
def candle_data():
    first_timestamp = 1767225600
    second_timestamp = 1767225900

    return [
        make_candle(first_timestamp),
        make_candle(second_timestamp),
    ]


def test_collects_closed_candle(
    tmp_path,
    monkeypatch,
    collection_time,
    candle_data,
):
    """Only closed candles should be collected."""

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(candle_data)
        ),
    )

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert len(records) == 1

    record = records[0]

    assert record["source"] == "kraken"
    assert record["pair"] == "XBTUSD"
    assert record["timeframe"] == "5m"

    assert record["open"] == "100"
    assert record["high"] == "110"
    assert record["low"] == "90"
    assert record["close"] == "105"
    assert record["volume"] == "12.5"

    assert record["schema_version"] == 1


def test_saves_jsonl_file(
    tmp_path,
    monkeypatch,
    collection_time,
    candle_data,
):
    """Collected candles should be saved as JSONL."""

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(candle_data)
        ),
    )

    kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    assert output_file.exists()

    records = [
        json.loads(line)
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 1
    assert records[0]["source"] == "kraken"


def test_does_not_store_duplicate_candles(
    tmp_path,
    monkeypatch,
    collection_time,
    candle_data,
):
    """Existing candle timestamps must not be duplicated."""

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(candle_data)
        ),
    )

    first_result = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    second_result = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert len(first_result) == 1
    assert second_result == []


def test_uses_utc_timestamps(
    tmp_path,
    monkeypatch,
    collection_time,
    candle_data,
):
    """Stored timestamps must use UTC notation."""

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(candle_data)
        ),
    )

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert records[0]["open_time"].endswith("Z")
    assert records[0]["close_time"].endswith("Z")
    assert records[0]["collected_at"].endswith("Z")


def test_uses_correct_kraken_request_parameters(
    tmp_path,
    monkeypatch,
    collection_time,
    candle_data,
):
    """The collector must request the correct pair and interval."""

    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout

        return FakeResponse(
            make_payload(candle_data)
        )

    monkeypatch.setattr(
        kraken.requests,
        "get",
        fake_get,
    )

    kraken.collect_kraken_ohlcv(
        pair="XBTUSD",
        timeframe="15m",
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert captured["url"] == kraken.API_URL
    assert captured["params"]["pair"] == "XBTUSD"
    assert captured["params"]["interval"] == 15
    assert captured["timeout"] == 15


def test_rejects_invalid_pair():
    """Invalid trading pairs must be rejected."""

    with pytest.raises(ValueError):
        kraken.collect_kraken_ohlcv(
            pair="BTC/USD"
        )


def test_rejects_unsupported_timeframe():
    """Unsupported timeframes must be rejected."""

    with pytest.raises(ValueError):
        kraken.collect_kraken_ohlcv(
            timeframe="2m"
        )


def test_rejects_invalid_timeout():
    """Timeout must be a positive finite number."""

    with pytest.raises(ValueError):
        kraken.collect_kraken_ohlcv(
            timeout=0
        )


def test_rejects_non_utc_collection_time():
    """Collection time must be timezone-aware UTC."""

    local_time = datetime(
        2026,
        1,
        1,
        0,
        0,
    )

    with pytest.raises(ValueError):
        kraken.collect_kraken_ohlcv(
            collection_time=local_time
        )


def test_handles_kraken_api_errors(
    tmp_path,
    monkeypatch,
    collection_time,
):
    """Kraken API errors must raise collector errors."""

    payload = {
        "error": ["EGeneral:Temporary error"],
        "result": {},
    }

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_handles_network_errors(
    tmp_path,
    monkeypatch,
    collection_time,
):
    """Network errors must be handled safely."""

    def fake_get(*args, **kwargs):
        raise requests.ConnectionError(
            "Connection failed"
        )

    monkeypatch.setattr(
        kraken.requests,
        "get",
        fake_get,
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_invalid_candle_structure(
    tmp_path,
    monkeypatch,
    collection_time,
):
    """Malformed candle rows must be rejected."""

    candles = [
        [1767225600, "100", "110"],
        make_candle(1767225900),
    ]

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(candles)
        ),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_invalid_existing_jsonl(
    tmp_path,
    monkeypatch,
    collection_time,
    candle_data,
):
    """Corrupted existing JSONL must not be ignored."""

    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    tmp_path.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(candle_data)
        ),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )