"""Tests for the Kraken Futures market-data collector."""

import json

from datetime import datetime, timedelta, timezone

import pytest
import requests

import collectors.kraken_futures as kraken_futures


COLLECTION_TIME = datetime(
    2026, 11, 19, 10, 30,
    tzinfo=timezone.utc,
)

PROVIDER_TIME = datetime(
    2026, 11, 19, 10, 25,
    tzinfo=timezone.utc,
)

PROVIDER_TIMESTAMP = "2026-11-19T10:25:00Z"


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


def make_ticker(**overrides):
    """Create a valid Bitcoin futures ticker."""

    ticker = {
        "symbol": "PF_XBTUSD",
        "pair": "XBT:USD",
        "lastTime": PROVIDER_TIMESTAMP,
        "markPrice": 95000.0,
        "bid": 94990.0,
        "bidSize": 2.5,
        "ask": 95010.0,
        "askSize": 3.0,
        "vol24h": 1250.0,
        "volumeQuote": 118750000.0,
        "openInterest": 850.0,
        "open24h": 94000.0,
        "indexPrice": 94995.0,
        "last": 95000.0,
        "lastSize": 0.5,
        "fundingRate": 0.0001,
        "fundingRatePrediction": 0.00012,
        "suspended": False,
        "tag": "perpetual",
    }

    ticker.update(overrides)

    return ticker


def make_payload(tickers=None, result="success"):
    """Create a valid Kraken Futures response."""

    if tickers is None:
        tickers = [make_ticker()]

    return {
        "result": result,
        "tickers": tickers,
    }


def mock_response(monkeypatch, payload):
    """Mock the Kraken Futures HTTP request."""

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs

        return FakeResponse(payload)

    monkeypatch.setattr(
        kraken_futures.requests,
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
    """Collect with stable test defaults."""

    return kraken_futures.collect_kraken_futures(
        output_dir=output_dir,
        collection_time=collection_time,
        **kwargs,
    )


def read_records(output_dir):
    """Read stored JSONL records."""

    path = (
        output_dir
        / "kraken_futures_BTC.jsonl"
    )

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# ==================================================
# Successful collection
# ==================================================

def test_collects_bitcoin_futures_ticker(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    records = collect(tmp_path)

    assert len(records) == 1
    assert records[0]["symbol"] == "PF_XBTUSD"


def test_returns_expected_record_structure(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    records = collect(tmp_path)

    record = records[0]

    assert record["schema_version"] == 1
    assert record["source"] == "kraken_futures"
    assert record["asset"] == "BTC"
    assert record["symbol"] == "PF_XBTUSD"
    assert record["pair"] == "XBT:USD"
    assert record["contract_type"] == "perpetual"
    assert record["provider_timestamp"] == PROVIDER_TIMESTAMP
    assert record["collected_at"] == "2026-11-19T10:30:00Z"
    assert record["suspended"] is False


def test_converts_numeric_values_to_float(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        markPrice=95000,
        bid=94990,
        ask=95010,
        fundingRate=0,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert records[0]["markPrice"] == 95000.0
    assert records[0]["bid"] == 94990.0
    assert records[0]["ask"] == 95010.0
    assert records[0]["fundingRate"] == 0.0


def test_accepts_missing_optional_numeric_fields(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        fundingRate=None,
        fundingRatePrediction=None,
        openInterest=None,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert records[0]["fundingRate"] is None
    assert records[0]["fundingRatePrediction"] is None
    assert records[0]["openInterest"] is None


def test_uses_correct_api_url(
    tmp_path,
    monkeypatch,
):
    captured = mock_response(
        monkeypatch,
        make_payload(),
    )

    collect(tmp_path)

    assert captured["url"] == kraken_futures.API_URL


def test_uses_default_timeout(
    tmp_path,
    monkeypatch,
):
    captured = mock_response(
        monkeypatch,
        make_payload(),
    )

    collect(tmp_path)

    assert (
        captured["kwargs"]["timeout"]
        == kraken_futures.DEFAULT_TIMEOUT
    )


def test_uses_custom_timeout(
    tmp_path,
    monkeypatch,
):
    captured = mock_response(
        monkeypatch,
        make_payload(),
    )

    collect(tmp_path, timeout=25)

    assert captured["kwargs"]["timeout"] == 25.0


def test_sends_json_accept_header(
    tmp_path,
    monkeypatch,
):
    captured = mock_response(
        monkeypatch,
        make_payload(),
    )

    collect(tmp_path)

    assert captured["kwargs"]["headers"] == {
        "accept": "application/json",
    }


def test_does_not_require_private_api_credentials(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(
        "KRAKEN_API_KEY",
        raising=False,
    )

    monkeypatch.delenv(
        "KRAKEN_API_SECRET",
        raising=False,
    )

    mock_response(
        monkeypatch,
        make_payload(),
    )

    records = collect(tmp_path)

    assert len(records) == 1


# ==================================================
# Input validation
# ==================================================

@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        False,
        "10",
        None,
    ],
)
def test_rejects_invalid_timeout(
    tmp_path,
    timeout,
):
    with pytest.raises(ValueError):
        collect(
            tmp_path,
            timeout=timeout,
        )


def test_rejects_naive_collection_time(
    tmp_path,
):
    naive_time = datetime(
        2026, 11, 19, 10, 30,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware UTC",
    ):
        collect(
            tmp_path,
            collection_time=naive_time,
        )


def test_rejects_non_utc_collection_time(
    tmp_path,
):
    non_utc = datetime(
        2026,
        11,
        19,
        13,
        30,
        tzinfo=timezone(
            timedelta(hours=3),
        ),
    )

    with pytest.raises(
        ValueError,
        match="must use UTC",
    ):
        collect(
            tmp_path,
            collection_time=non_utc,
        )


def test_rejects_non_datetime_collection_time(
    tmp_path,
):
    with pytest.raises(ValueError):
        collect(
            tmp_path,
            collection_time="2026-11-19T10:30:00Z",
        )


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        " ",
        None,
    ],
)
def test_rejects_invalid_requested_symbol(
    tmp_path,
    symbol,
):
    if symbol is None:
        return

    with pytest.raises(ValueError):
        collect(
            tmp_path,
            symbol=symbol,
        )


def test_normalizes_requested_symbol(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    records = collect(
        tmp_path,
        symbol=" pf_xbtusd ",
    )

    assert len(records) == 1
    assert records[0]["symbol"] == "PF_XBTUSD"


# ==================================================
# Bitcoin instrument filtering
# ==================================================

def test_filters_non_bitcoin_instruments(
    tmp_path,
    monkeypatch,
):
    tickers = [
        make_ticker(),
        make_ticker(
            symbol="PF_ETHUSD",
            pair="ETH:USD",
        ),
        make_ticker(
            symbol="PF_SOLUSD",
            pair="SOL:USD",
        ),
    ]

    mock_response(
        monkeypatch,
        make_payload(tickers),
    )

    records = collect(tmp_path)

    assert len(records) == 1
    assert records[0]["asset"] == "BTC"


def test_accepts_btc_symbol(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        symbol="PF_BTCUSD",
        pair="BTC:USD",
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert len(records) == 1


def test_accepts_bitcoin_pair_when_symbol_does_not_contain_btc(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        symbol="CUSTOM_CONTRACT",
        pair="XBT:USD",
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert len(records) == 1


def test_requested_symbol_filters_other_bitcoin_contracts(
    tmp_path,
    monkeypatch,
):
    tickers = [
        make_ticker(),
        make_ticker(
            symbol="PF_XBTUSDT",
            pair="XBT:USDT",
        ),
    ]

    mock_response(
        monkeypatch,
        make_payload(tickers),
    )

    records = collect(
        tmp_path,
        symbol="PF_XBTUSD",
    )

    assert len(records) == 1
    assert records[0]["symbol"] == "PF_XBTUSD"


def test_non_matching_requested_symbol_returns_empty_list(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    records = collect(
        tmp_path,
        symbol="PF_XBTUSDT",
    )

    assert records == []


# ==================================================
# Provider response validation
# ==================================================

@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_rejects_non_dictionary_response(
    tmp_path,
    monkeypatch,
    payload,
):
    mock_response(
        monkeypatch,
        payload,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="JSON object",
    ):
        collect(tmp_path)


@pytest.mark.parametrize(
    "result",
    [
        "error",
        "failure",
        "failed",
    ],
)
def test_rejects_unsuccessful_api_result(
    tmp_path,
    monkeypatch,
    result,
):
    mock_response(
        monkeypatch,
        make_payload(result=result),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="unsuccessful result",
    ):
        collect(tmp_path)


@pytest.mark.parametrize(
    "tickers",
    [
        None,
        {},
        "invalid",
        123,
    ],
)
def test_rejects_invalid_tickers_container(
    tmp_path,
    monkeypatch,
    tickers,
):
    mock_response(
        monkeypatch,
        {
            "result": "success",
            "tickers": tickers,
        },
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="tickers list",
    ):
        collect(tmp_path)


def test_rejects_non_dictionary_ticker(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload([None]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="ticker structure",
    ):
        collect(tmp_path)


def test_rejects_missing_symbol(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker()
    ticker["symbol"] = ""

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="symbol",
    ):
        collect(tmp_path)


def test_rejects_missing_pair(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker()
    ticker["pair"] = ""

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="pair",
    ):
        collect(tmp_path)


# ==================================================
# Timestamp validation
# ==================================================

@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        "invalid",
        123,
    ],
)
def test_rejects_invalid_provider_timestamp(
    tmp_path,
    monkeypatch,
    timestamp,
):
    ticker = make_ticker(
        lastTime=timestamp,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
    ):
        collect(tmp_path)


def test_rejects_provider_timestamp_without_timezone(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        lastTime="2026-11-19T10:25:00",
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="include timezone",
    ):
        collect(tmp_path)


def test_rejects_future_provider_timestamp(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        lastTime="2026-11-20T10:25:00Z",
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="future",
    ):
        collect(tmp_path)


def test_normalizes_provider_timestamp_to_utc(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        lastTime="2026-11-19T13:25:00+03:00",
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert (
        records[0]["provider_timestamp"]
        == PROVIDER_TIMESTAMP
    )


# ==================================================
# Numeric validation
# ==================================================

@pytest.mark.parametrize(
    "field,value",
    [
        ("markPrice", True),
        ("markPrice", False),
        ("markPrice", "95000"),
        ("markPrice", float("nan")),
        ("markPrice", float("inf")),
        ("markPrice", float("-inf")),
        ("fundingRate", True),
        ("fundingRate", "0.001"),
        ("fundingRate", float("nan")),
        ("openInterest", float("inf")),
        ("vol24h", float("-inf")),
    ],
)
def test_rejects_invalid_numeric_fields(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    ticker = make_ticker(
        **{field: value},
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Invalid numeric field",
    ):
        collect(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "markPrice",
        "last",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_rejects_non_positive_market_prices(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    ticker = make_ticker(
        **{field: value},
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="greater than zero",
    ):
        collect(tmp_path)


def test_rejects_bid_greater_than_ask(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        bid=95100.0,
        ask=95000.0,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="bid price cannot exceed ask",
    ):
        collect(tmp_path)


def test_accepts_equal_bid_and_ask(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        bid=95000.0,
        ask=95000.0,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert len(records) == 1


# ==================================================
# Contract metadata validation
# ==================================================

@pytest.mark.parametrize(
    "suspended",
    [
        "false",
        0,
        1,
        None,
    ],
)
def test_rejects_invalid_suspended_status(
    tmp_path,
    monkeypatch,
    suspended,
):
    ticker = make_ticker(
        suspended=suspended,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="suspended status",
    ):
        collect(tmp_path)


def test_accepts_missing_suspended_status_as_false(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker()
    del ticker["suspended"]

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert records[0]["suspended"] is False


def test_rejects_invalid_contract_tag(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker(
        tag=123,
    )

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="contract tag",
    ):
        collect(tmp_path)


def test_accepts_missing_contract_tag(
    tmp_path,
    monkeypatch,
):
    ticker = make_ticker()
    del ticker["tag"]

    mock_response(
        monkeypatch,
        make_payload([ticker]),
    )

    records = collect(tmp_path)

    assert records[0]["contract_type"] is None


# ==================================================
# Duplicate prevention and append-only history
# ==================================================

def test_does_not_save_duplicate_observations(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    first = collect(tmp_path)
    second = collect(tmp_path)

    assert len(first) == 1
    assert second == []

    assert len(read_records(tmp_path)) == 1


def test_prevents_duplicates_in_same_response(
    tmp_path,
    monkeypatch,
):
    tickers = [
        make_ticker(),
        make_ticker(),
    ]

    mock_response(
        monkeypatch,
        make_payload(tickers),
    )

    records = collect(tmp_path)

    assert len(records) == 1


def test_accepts_new_provider_timestamp(
    tmp_path,
    monkeypatch,
):
    first_ticker = make_ticker()

    mock_response(
        monkeypatch,
        make_payload([first_ticker]),
    )

    first = collect(tmp_path)

    second_ticker = make_ticker(
        lastTime="2026-11-19T10:26:00Z",
        markPrice=95100.0,
        last=95100.0,
    )

    mock_response(
        monkeypatch,
        make_payload([second_ticker]),
    )

    second = collect(tmp_path)

    assert len(first) == 1
    assert len(second) == 1

    stored = read_records(tmp_path)

    assert len(stored) == 2


def test_duplicate_identity_uses_symbol_and_timestamp(
    tmp_path,
    monkeypatch,
):
    tickers = [
        make_ticker(
            symbol="PF_XBTUSD",
            pair="XBT:USD",
        ),
        make_ticker(
            symbol="PF_XBTUSDT",
            pair="XBT:USDT",
        ),
    ]

    mock_response(
        monkeypatch,
        make_payload(tickers),
    )

    records = collect(tmp_path)

    assert len(records) == 2


def test_preserves_existing_history(
    tmp_path,
    monkeypatch,
):
    output_file = (
        tmp_path / "kraken_futures_BTC.jsonl"
    )

    original = {
        "symbol": "PF_XBTUSD",
        "provider_timestamp": "2026-11-18T10:00:00Z",
        "custom_history_marker": "preserve-me",
    }

    output_file.write_text(
        json.dumps(original) + "\n",
        encoding="utf-8",
    )

    mock_response(
        monkeypatch,
        make_payload(),
    )

    records = collect(tmp_path)

    assert len(records) == 1

    stored = read_records(tmp_path)

    assert len(stored) == 2
    assert stored[0] == original


def test_uses_correct_output_filename(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    collect(tmp_path)

    assert (
        tmp_path / "kraken_futures_BTC.jsonl"
    ).exists()


def test_saves_valid_jsonl(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    collect(tmp_path)

    records = read_records(tmp_path)

    assert len(records) == 1
    assert records[0]["source"] == "kraken_futures"
    assert records[0]["asset"] == "BTC"


def test_empty_bitcoin_result_does_not_create_file(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(
            [
                make_ticker(
                    symbol="PF_ETHUSD",
                    pair="ETH:USD",
                )
            ]
        ),
    )

    records = collect(tmp_path)

    assert records == []

    assert not (
        tmp_path / "kraken_futures_BTC.jsonl"
    ).exists()


def test_empty_tickers_do_not_create_file(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload([]),
    )

    records = collect(tmp_path)

    assert records == []

    assert not (
        tmp_path / "kraken_futures_BTC.jsonl"
    ).exists()


# ==================================================
# Existing history validation
# ==================================================

def test_rejects_invalid_existing_jsonl(
    tmp_path,
    monkeypatch,
):
    output_file = (
        tmp_path / "kraken_futures_BTC.jsonl"
    )

    output_file.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    mock_response(
        monkeypatch,
        make_payload(),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Invalid JSONL",
    ):
        collect(tmp_path)


def test_rejects_non_dictionary_existing_record(
    tmp_path,
    monkeypatch,
):
    output_file = (
        tmp_path / "kraken_futures_BTC.jsonl"
    )

    output_file.write_text(
        json.dumps(["invalid"]) + "\n",
        encoding="utf-8",
    )

    mock_response(
        monkeypatch,
        make_payload(),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Invalid record",
    ):
        collect(tmp_path)


def test_rejects_existing_record_without_identity(
    tmp_path,
    monkeypatch,
):
    output_file = (
        tmp_path / "kraken_futures_BTC.jsonl"
    )

    output_file.write_text(
        json.dumps({
            "source": "kraken_futures",
        }) + "\n",
        encoding="utf-8",
    )

    mock_response(
        monkeypatch,
        make_payload(),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Missing record identity",
    ):
        collect(tmp_path)


# ==================================================
# HTTP and API errors
# ==================================================

def test_handles_http_error(
    tmp_path,
    monkeypatch,
):
    def fake_get(*args, **kwargs):
        return FakeResponse(
            status_error=requests.HTTPError(
                "HTTP 500",
            )
        )

    monkeypatch.setattr(
        kraken_futures.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="request failed",
    ):
        collect(tmp_path)


def test_handles_connection_error(
    tmp_path,
    monkeypatch,
):
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError(
            "Connection failed",
        )

    monkeypatch.setattr(
        kraken_futures.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="request failed",
    ):
        collect(tmp_path)


def test_handles_timeout_error(
    tmp_path,
    monkeypatch,
):
    def fake_get(*args, **kwargs):
        raise requests.Timeout(
            "Request timed out",
        )

    monkeypatch.setattr(
        kraken_futures.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="request failed",
    ):
        collect(tmp_path)


def test_handles_invalid_json_response(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        None,
    )

    def fake_get(*args, **kwargs):
        return FakeResponse(
            json_error=ValueError(
                "Invalid JSON",
            )
        )

    monkeypatch.setattr(
        kraken_futures.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="invalid JSON",
    ):
        collect(tmp_path)


def test_handles_output_directory_creation_error(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    def fail_mkdir(*args, **kwargs):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(
        kraken_futures.Path,
        "mkdir",
        fail_mkdir,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Cannot save",
    ):
        collect(tmp_path / "new-directory")


def test_handles_output_write_error(
    tmp_path,
    monkeypatch,
):
    mock_response(
        monkeypatch,
        make_payload(),
    )

    original_open = kraken_futures.Path.open

    def fail_output_open(path, *args, **kwargs):
        if path.name == "kraken_futures_BTC.jsonl":
            raise OSError("Disk write failed")

        return original_open(
            path,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        kraken_futures.Path,
        "open",
        fail_output_open,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Cannot save",
    ):
        collect(tmp_path)


def test_handles_existing_history_read_error(
    tmp_path,
    monkeypatch,
):
    output_file = (
        tmp_path / "kraken_futures_BTC.jsonl"
    )

    output_file.write_text(
        "",
        encoding="utf-8",
    )

    original_open = kraken_futures.Path.open

    def fail_history_read(path, mode="r", *args, **kwargs):
        if (
            path.name == "kraken_futures_BTC.jsonl"
            and mode == "r"
        ):
            raise OSError("Read failed")

        return original_open(
            path,
            mode,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        kraken_futures.Path,
        "open",
        fail_history_read,
    )

    mock_response(
        monkeypatch,
        make_payload(),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Cannot read existing futures history",
    ):
        collect(tmp_path)