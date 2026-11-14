"""Edge case tests for the Kraken OHLCV collector."""

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


def make_candle(
    timestamp=1767225600,
    open_price="100",
    high="110",
    low="90",
    close="105",
    volume="12.5",
):
    """Create a Kraken OHLC candle."""

    return [
        timestamp,
        open_price,
        high,
        low,
        close,
        "102",
        volume,
        100,
    ]


def make_payload(candles):
    """Create a valid Kraken response."""

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


def mock_kraken(monkeypatch, payload):
    """Replace the real HTTP request."""

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload),
    )


def test_rejects_non_dictionary_response(
    tmp_path,
    monkeypatch,
    collection_time,
):
    mock_kraken(monkeypatch, [])

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_missing_error_field(
    tmp_path,
    monkeypatch,
    collection_time,
):
    mock_kraken(
        monkeypatch,
        {
            "result": {},
        },
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_invalid_error_field(
    tmp_path,
    monkeypatch,
    collection_time,
):
    mock_kraken(
        monkeypatch,
        {
            "error": "invalid",
            "result": {},
        },
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_non_dictionary_result(
    tmp_path,
    monkeypatch,
    collection_time,
):
    mock_kraken(
        monkeypatch,
        {
            "error": [],
            "result": [],
        },
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_multiple_candle_arrays(
    tmp_path,
    monkeypatch,
    collection_time,
):
    mock_kraken(
        monkeypatch,
        {
            "error": [],
            "result": {
                "XXBTZUSD": [],
                "XETHZUSD": [],
                "last": 123,
            },
        },
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "invalid",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        -1,
    ],
)
def test_rejects_invalid_candle_timestamps(
    tmp_path,
    monkeypatch,
    collection_time,
    timestamp,
):
    candles = [
        make_candle(timestamp),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_timestamp_outside_supported_range(
    tmp_path,
    monkeypatch,
    collection_time,
):
    candles = [
        make_candle(1e30),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_does_not_save_unclosed_candle(
    tmp_path,
    monkeypatch,
    collection_time,
):
    candles = [
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert records == []

    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    assert not output_file.exists()


def test_returns_empty_when_api_has_no_candles(
    tmp_path,
    monkeypatch,
    collection_time,
):
    mock_kraken(
        monkeypatch,
        make_payload([]),
    )

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert records == []


def test_does_not_overwrite_existing_history(
    tmp_path,
    monkeypatch,
    collection_time,
):
    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    existing_record = {
        "open_time": "2025-12-31T23:55:00Z",
        "source": "kraken",
    }

    output_file.write_text(
        json.dumps(existing_record) + "\n",
        encoding="utf-8",
    )

    # First candle: already exists in history.
    # Second candle: new and closed.
    # Third candle: currently forming and must be ignored.
    candles = [
        make_candle(1767225300),
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert len(records) == 1

    saved_lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(saved_lines) == 2

    original = json.loads(saved_lines[0])

    assert original == existing_record


def test_rejects_existing_jsonl_non_dictionary_record(
    tmp_path,
    monkeypatch,
    collection_time,
):
    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    output_file.write_text(
        json.dumps(["invalid", "record"]) + "\n",
        encoding="utf-8",
    )

    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_existing_record_without_open_time(
    tmp_path,
    monkeypatch,
    collection_time,
):
    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    output_file.write_text(
        json.dumps({"source": "kraken"}) + "\n",
        encoding="utf-8",
    )

    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_invalid_json_existing_history(
    tmp_path,
    monkeypatch,
    collection_time,
):
    output_file = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    output_file.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_wraps_output_directory_creation_error(
    tmp_path,
    monkeypatch,
    collection_time,
):
    output_dir = tmp_path / "not_a_directory"

    output_dir.write_text(
        "This is a file.",
        encoding="utf-8",
    )

    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=output_dir,
            collection_time=collection_time,
        )


def test_preserves_original_provider_candle(
    tmp_path,
    monkeypatch,
    collection_time,
):
    original_candle = make_candle(
        timestamp=1767225600,
    )

    candles = [
        original_candle,
        make_candle(1767225900),
    ]

    mock_kraken(
        monkeypatch,
        make_payload(candles),
    )

    records = kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert len(records) == 1

    assert records[0]["provider_candle"] == original_candle


def test_does_not_modify_api_payload(
    tmp_path,
    monkeypatch,
    collection_time,
):
    candles = [
        make_candle(1767225600),
        make_candle(1767225900),
    ]

    payload = make_payload(candles)

    original_payload = json.loads(
        json.dumps(payload)
    )

    mock_kraken(
        monkeypatch,
        payload,
    )

    kraken.collect_kraken_ohlcv(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert payload == original_payload


def test_handles_http_status_error(
    tmp_path,
    monkeypatch,
    collection_time,
):
    class FailedResponse:
        def raise_for_status(self):
            raise requests.HTTPError(
                "HTTP 500"
            )

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: FailedResponse(),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_handles_invalid_json_response(
    tmp_path,
    monkeypatch,
    collection_time,
):
    class InvalidJSONResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("Invalid JSON")

    monkeypatch.setattr(
        kraken.requests,
        "get",
        lambda *args, **kwargs: InvalidJSONResponse(),
    )

    with pytest.raises(kraken.KrakenCollectorError):
        kraken.collect_kraken_ohlcv(
            output_dir=tmp_path,
            collection_time=collection_time,
        )