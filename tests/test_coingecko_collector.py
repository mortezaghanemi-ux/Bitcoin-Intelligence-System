"""Tests for the CoinGecko Bitcoin market-data collector."""

import json

from datetime import datetime, timedelta, timezone

import pytest
import requests

import collectors.coingecko as coingecko


API_KEY = "TEST_COINGECKO_KEY_123456"

COLLECTION_TIME = datetime(
    2026, 11, 19, 10, 30, tzinfo=timezone.utc
)

OBSERVATION_TIME = datetime(
    2026, 11, 18, 10, 0, tzinfo=timezone.utc
)

OBSERVATION_TIMESTAMP_MS = int(
    OBSERVATION_TIME.timestamp() * 1000
)


class FakeResponse:
    """Mock HTTP response."""

    def __init__(
        self,
        payload=None,
        status_error=None,
        json_error=None,
    ):
        self.payload = payload
        self.status_error = status_error
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def make_payload(
    timestamp=OBSERVATION_TIMESTAMP_MS,
    price=95000.0,
    market_cap=1800000000000.0,
    volume=25000000000.0,
):
    """Create a valid CoinGecko market-chart response."""

    return {
        "prices": [[timestamp, price]],
        "market_caps": [[timestamp, market_cap]],
        "total_volumes": [[timestamp, volume]],
    }


def mock_response(monkeypatch, payload):
    """Replace the real CoinGecko HTTP request."""

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse(payload)

    monkeypatch.setattr(
        coingecko.requests,
        "get",
        fake_get,
    )

    return captured


def collect(
    output_dir,
    *,
    collection_time=COLLECTION_TIME,
    **kwargs,
):
    """Call the collector with stable test defaults."""

    return coingecko.collect_coingecko_market_chart(
        output_dir=output_dir,
        collection_time=collection_time,
        **kwargs,
    )


# ==================================================
# Successful collection
# ==================================================

def test_collects_all_market_metrics(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    records = collect(tmp_path)

    assert len(records) == 3
    assert {
        record["metric"] for record in records
    } == {
        "prices",
        "market_caps",
        "total_volumes",
    }


def test_returns_expected_record_structure(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    records = collect(tmp_path)

    record = next(
        item for item in records
        if item["metric"] == "prices"
    )

    assert record["schema_version"] == 1
    assert record["source"] == "coingecko"
    assert record["asset"] == "BTC"
    assert record["metric"] == "prices"
    assert record["value"] == 95000.0
    assert record["quote_currency"] == "usd"
    assert record["provider_timestamp"] == "2026-11-18T10:00:00Z"
    assert record["collected_at"] == "2026-11-19T10:30:00Z"


def test_collects_custom_quote_currency(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    records = collect(tmp_path, vs_currency="eur")

    assert len(records) == 3
    assert all(r["quote_currency"] == "eur" for r in records)
    assert captured["kwargs"]["params"]["vs_currency"] == "eur"


def test_normalizes_quote_currency(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    records = collect(tmp_path, vs_currency=" USD ")

    assert len(records) == 3
    assert all(r["quote_currency"] == "usd" for r in records)
    assert captured["kwargs"]["params"]["vs_currency"] == "usd"


@pytest.mark.parametrize("days", [1, 7, 14, 30, 90, 180, 365])
def test_accepts_supported_history_windows(tmp_path, monkeypatch, days):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path, days=days)

    assert captured["kwargs"]["params"]["days"] == days


# ==================================================
# HTTP request validation
# ==================================================

def test_uses_correct_api_url(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    assert captured["url"] == coingecko.API_URL


def test_uses_default_request_parameters(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    assert captured["kwargs"]["params"] == {
        "vs_currency": "usd",
        "days": 1,
    }


def test_uses_default_timeout(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    assert captured["kwargs"]["timeout"] == coingecko.DEFAULT_TIMEOUT


def test_uses_custom_timeout(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path, timeout=25)

    assert captured["kwargs"]["timeout"] == 25.0


def test_sends_json_accept_header(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    assert captured["kwargs"]["headers"] == {
        "accept": "application/json",
    }


def test_sends_optional_api_key(tmp_path, monkeypatch):
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path, api_key=API_KEY)

    assert captured["kwargs"]["headers"]["x-cg-demo-api-key"] == API_KEY


def test_reads_api_key_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", API_KEY)
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    assert captured["kwargs"]["headers"]["x-cg-demo-api-key"] == API_KEY


def test_explicit_api_key_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", "ENVIRONMENT_KEY")
    captured = mock_response(monkeypatch, make_payload())

    collect(tmp_path, api_key=API_KEY)

    assert captured["kwargs"]["headers"]["x-cg-demo-api-key"] == API_KEY


def test_no_api_key_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    captured = mock_response(monkeypatch, make_payload())

    records = collect(tmp_path)

    assert len(records) == 3
    assert "x-cg-demo-api-key" not in captured["kwargs"]["headers"]


# ==================================================
# Input validation
# ==================================================

@pytest.mark.parametrize(
    "currency",
    ["", " ", None, 123, "usd/eur", "usd-eur", "usd eur"],
)
def test_rejects_invalid_quote_currency(tmp_path, currency):
    with pytest.raises(ValueError):
        collect(tmp_path, vs_currency=currency)


@pytest.mark.parametrize(
    "days",
    [0, -1, 2, 3, 5, 31, 999, True, False, 1.5, "7", None],
)
def test_rejects_invalid_history_window(tmp_path, days):
    with pytest.raises(ValueError):
        collect(tmp_path, days=days)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("inf"), float("-inf"), float("nan"), True, False, "10", None],
)
def test_rejects_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError):
        collect(tmp_path, timeout=timeout)


@pytest.mark.parametrize("api_key", ["", " ", 123])
def test_rejects_invalid_explicit_api_key(tmp_path, api_key):
    with pytest.raises(ValueError):
        collect(tmp_path, api_key=api_key)


def test_none_api_key_uses_environment_or_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    captured = mock_response(monkeypatch, make_payload())

    records = collect(tmp_path, api_key=None)

    assert len(records) == 3
    assert "x-cg-demo-api-key" not in captured["kwargs"]["headers"]


def test_rejects_naive_collection_time(tmp_path):
    naive_time = datetime(2026, 11, 19, 10, 30)

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        collect(tmp_path, collection_time=naive_time)


def test_rejects_non_utc_collection_time(tmp_path):
    non_utc_time = datetime(
        2026, 11, 19, 13, 30,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ValueError, match="must use UTC"):
        collect(tmp_path, collection_time=non_utc_time)


def test_rejects_non_datetime_collection_time(tmp_path):
    with pytest.raises(ValueError):
        collect(tmp_path, collection_time="2026-11-19T10:30:00Z")


def test_normalizes_utc_collection_time(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    records = collect(
        tmp_path,
        collection_time=datetime(
            2026, 11, 19, 10, 30,
            tzinfo=timezone.utc,
        ),
    )

    assert all(
        r["collected_at"] == "2026-11-19T10:30:00Z"
        for r in records
    )


# ==================================================
# Provider response validation
# ==================================================

@pytest.mark.parametrize("payload", [None, [], "invalid", 123])
def test_rejects_non_dictionary_response(tmp_path, monkeypatch, payload):
    mock_response(monkeypatch, payload)

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="JSON object",
    ):
        collect(tmp_path)


@pytest.mark.parametrize(
    "metric",
    ["prices", "market_caps", "total_volumes"],
)
def test_rejects_missing_metric_series(tmp_path, monkeypatch, metric):
    payload = make_payload()
    del payload[metric]
    mock_response(monkeypatch, payload)

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Missing or invalid",
    ):
        collect(tmp_path)


@pytest.mark.parametrize("invalid_series", [None, {}, "invalid", 123])
def test_rejects_invalid_metric_series(tmp_path, monkeypatch, invalid_series):
    payload = make_payload()
    payload["prices"] = invalid_series
    mock_response(monkeypatch, payload)

    with pytest.raises(coingecko.CoinGeckoCollectorError):
        collect(tmp_path)


@pytest.mark.parametrize(
    "invalid_point",
    [None, [], [1], [1, 2, 3], "invalid", {}],
)
def test_rejects_invalid_data_points(tmp_path, monkeypatch, invalid_point):
    payload = make_payload()
    payload["prices"] = [invalid_point]
    mock_response(monkeypatch, payload)

    with pytest.raises(coingecko.CoinGeckoCollectorError):
        collect(tmp_path)


@pytest.mark.parametrize(
    "timestamp",
    [-1, None, True, False, "invalid", float("nan"), float("inf"), float("-inf")],
)
def test_rejects_invalid_provider_timestamps(tmp_path, monkeypatch, timestamp):
    payload = make_payload()
    payload["prices"] = [[timestamp, 95000.0]]
    mock_response(monkeypatch, payload)

    with pytest.raises(coingecko.CoinGeckoCollectorError):
        collect(tmp_path)


def test_rejects_future_observation(tmp_path, monkeypatch):
    future_timestamp = int(
        datetime(2026, 11, 20, tzinfo=timezone.utc).timestamp() * 1000
    )
    mock_response(
        monkeypatch,
        make_payload(timestamp=future_timestamp),
    )

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Future observation",
    ):
        collect(tmp_path)


def test_rejects_timestamp_out_of_range(tmp_path, monkeypatch):
    payload = make_payload()
    payload["prices"] = [[1e30, 95000.0]]
    mock_response(monkeypatch, payload)

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Timestamp out of range",
    ):
        collect(tmp_path)


@pytest.mark.parametrize(
    "value",
    [-1, None, True, False, "95000", float("nan"), float("inf"), float("-inf")],
)
def test_rejects_invalid_market_values(tmp_path, monkeypatch, value):
    payload = make_payload()
    payload["prices"] = [[OBSERVATION_TIMESTAMP_MS, value]]
    mock_response(monkeypatch, payload)

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Invalid value",
    ):
        collect(tmp_path)


def test_accepts_zero_market_value(tmp_path, monkeypatch):
    mock_response(
        monkeypatch,
        make_payload(price=0, market_cap=0, volume=0),
    )

    records = collect(tmp_path)

    assert len(records) == 3
    assert all(r["value"] == 0.0 for r in records)


# ==================================================
# Duplicate prevention and append-only history
# ==================================================

def test_does_not_save_duplicate_observations(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    first = collect(tmp_path)
    second = collect(tmp_path)

    assert len(first) == 3
    assert second == []


def test_prevents_duplicate_points_in_same_response(tmp_path, monkeypatch):
    payload = make_payload()
    payload["prices"] = [
        [OBSERVATION_TIMESTAMP_MS, 95000.0],
        [OBSERVATION_TIMESTAMP_MS, 95100.0],
    ]
    mock_response(monkeypatch, payload)

    records = collect(tmp_path)

    price_records = [
        r for r in records if r["metric"] == "prices"
    ]

    assert len(price_records) == 1
    assert price_records[0]["value"] == 95000.0


def test_preserves_different_observation_timestamps(tmp_path, monkeypatch):
    second_timestamp = OBSERVATION_TIMESTAMP_MS + 60 * 60 * 1000
    payload = make_payload()
    payload["prices"] = [
        [OBSERVATION_TIMESTAMP_MS, 95000.0],
        [second_timestamp, 95100.0],
    ]
    mock_response(monkeypatch, payload)

    records = collect(tmp_path)

    price_records = [
        r for r in records if r["metric"] == "prices"
    ]

    assert len(price_records) == 2


def test_saves_jsonl_file(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"

    assert output_file.exists()

    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_uses_correct_output_filename(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    collect(tmp_path, vs_currency="eur", days=7)

    assert (
        tmp_path / "coingecko_BTC_eur_7d.jsonl"
    ).exists()


def test_appends_only_new_observations(tmp_path, monkeypatch):
    """A second collection adds only genuinely new metric timestamps."""

    first_timestamp = OBSERVATION_TIMESTAMP_MS
    second_timestamp = first_timestamp + 60 * 60 * 1000

    first_payload = make_payload(
        timestamp=first_timestamp,
        price=95000.0,
        market_cap=1800000000000.0,
        volume=25000000000.0,
    )

    mock_response(monkeypatch, first_payload)

    first_records = collect(tmp_path)

    assert len(first_records) == 3

    second_payload = make_payload(
        timestamp=second_timestamp,
        price=95100.0,
        market_cap=1801000000000.0,
        volume=25100000000.0,
    )

    mock_response(monkeypatch, second_payload)

    second_records = collect(tmp_path)

    assert len(second_records) == 3

    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"
    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 6

    stored_records = [json.loads(line) for line in lines]

    for metric in ("prices", "market_caps", "total_volumes"):
        metric_records = [
            r for r in stored_records
            if r["metric"] == metric
        ]
        assert len(metric_records) == 2

        timestamps = {
            r["provider_timestamp"] for r in metric_records
        }
        assert len(timestamps) == 2


def test_existing_duplicate_metrics_are_not_appended(tmp_path, monkeypatch):
    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"

    original_records = [
        {
            "metric": metric,
            "provider_timestamp": "2026-11-18T10:00:00Z",
        }
        for metric in ("prices", "market_caps", "total_volumes")
    ]

    output_file.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in original_records
        ),
        encoding="utf-8",
    )

    mock_response(monkeypatch, make_payload())

    records = collect(tmp_path)

    assert records == []

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_preserves_existing_history(tmp_path, monkeypatch):
    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"

    original_record = {
        "metric": "prices",
        "provider_timestamp": "2026-11-17T10:00:00Z",
    }

    output_file.write_text(
        json.dumps(original_record) + "\n",
        encoding="utf-8",
    )

    mock_response(monkeypatch, make_payload())

    records = collect(tmp_path)

    assert len(records) == 3

    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 4
    assert json.loads(lines[0]) == original_record


def test_rejects_invalid_existing_jsonl(tmp_path, monkeypatch):
    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"
    output_file.write_text("{invalid json}\n", encoding="utf-8")

    mock_response(monkeypatch, make_payload())

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Invalid JSONL",
    ):
        collect(tmp_path)


def test_rejects_non_dictionary_existing_record(tmp_path, monkeypatch):
    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"
    output_file.write_text(
        json.dumps(["invalid"]) + "\n",
        encoding="utf-8",
    )

    mock_response(monkeypatch, make_payload())

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Invalid record",
    ):
        collect(tmp_path)


def test_rejects_existing_record_without_identity(tmp_path, monkeypatch):
    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"
    output_file.write_text(
        json.dumps({"source": "coingecko"}) + "\n",
        encoding="utf-8",
    )

    mock_response(monkeypatch, make_payload())

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Invalid record fields",
    ):
        collect(tmp_path)


def test_empty_response_does_not_create_output_file(tmp_path, monkeypatch):
    mock_response(
        monkeypatch,
        {
            "prices": [],
            "market_caps": [],
            "total_volumes": [],
        },
    )

    records = collect(tmp_path)

    assert records == []
    assert not (
        tmp_path / "coingecko_BTC_usd_1d.jsonl"
    ).exists()


def test_output_contains_valid_jsonl(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    collect(tmp_path)

    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"

    for line in output_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert isinstance(record, dict)
        assert record["source"] == "coingecko"
        assert record["asset"] == "BTC"


def test_api_key_is_not_saved_to_output(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    collect(tmp_path, api_key=API_KEY)

    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"
    content = output_file.read_text(encoding="utf-8")

    assert API_KEY not in content
    assert "api_key" not in content


# ==================================================
# HTTP and API errors
# ==================================================

def test_handles_http_error(tmp_path, monkeypatch):
    class FailedResponse:
        def raise_for_status(self):
            raise requests.HTTPError("HTTP 500")

    monkeypatch.setattr(
        coingecko.requests,
        "get",
        lambda *args, **kwargs: FailedResponse(),
    )

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="request failed",
    ):
        collect(tmp_path)


def test_handles_connection_error(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("Connection failed")

    monkeypatch.setattr(
        coingecko.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="request failed",
    ):
        collect(tmp_path)


def test_handles_timeout_error(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(
        coingecko.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="request failed",
    ):
        collect(tmp_path)


def test_handles_invalid_json_response(tmp_path, monkeypatch):
    class InvalidJSONResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("Invalid JSON")

    monkeypatch.setattr(
        coingecko.requests,
        "get",
        lambda *args, **kwargs: InvalidJSONResponse(),
    )

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="invalid JSON",
    ):
        collect(tmp_path)


def test_handles_request_error_without_exposing_key(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError(
            f"Failed with key {API_KEY}"
        )

    monkeypatch.setattr(
        coingecko.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
    ) as error:
        collect(tmp_path, api_key=API_KEY)

    assert API_KEY not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


# ==================================================
# File-system errors
# ==================================================

def test_handles_output_directory_creation_error(tmp_path, monkeypatch):
    output_dir = tmp_path / "not_a_directory"
    output_dir.write_text("This path is a file.", encoding="utf-8")

    mock_response(monkeypatch, make_payload())

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Cannot save",
    ):
        collect(output_dir)


def test_handles_output_file_write_error(tmp_path, monkeypatch):
    mock_response(monkeypatch, make_payload())

    output_file = tmp_path / "coingecko_BTC_usd_1d.jsonl"
    output_file.mkdir()

    with pytest.raises(
        coingecko.CoinGeckoCollectorError,
        match="Cannot save",
    ):
        collect(tmp_path)