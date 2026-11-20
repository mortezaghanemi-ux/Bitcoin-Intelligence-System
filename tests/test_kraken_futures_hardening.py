
"""Security and stale-data hardening tests for Kraken Futures."""

import json

from datetime import datetime, timedelta, timezone

import pytest
import requests

import collectors.kraken_futures as kraken_futures


COLLECTION_TIME = datetime(
    2026,
    11,
    19,
    10,
    30,
    tzinfo=timezone.utc,
)

SECRET_MARKER = "SENSITIVE_TEST_SECRET_987654"


class FakeResponse:
    """Minimal mocked HTTP response."""

    def __init__(
        self,
        payload=None,
        json_error=None,
    ):
        self.payload = payload
        self.json_error = json_error

    def raise_for_status(self):
        return None

    def json(self):
        if self.json_error is not None:
            raise self.json_error

        return self.payload


def make_ticker(provider_time):
    """Build a valid Bitcoin futures ticker."""

    return {
        "symbol": "PF_XBTUSD",
        "pair": "XBT:USD",
        "lastTime": provider_time,
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


def install_response(monkeypatch, response):
    """Install a mocked Kraken Futures response."""

    def fake_get(*args, **kwargs):
        return response

    monkeypatch.setattr(
        kraken_futures.requests,
        "get",
        fake_get,
    )


def collect(output_dir, monkeypatch, ticker):
    """Collect one ticker using a fixed UTC collection time."""

    install_response(
        monkeypatch,
        FakeResponse(
            {
                "result": "success",
                "tickers": [ticker],
            }
        ),
    )

    return kraken_futures.collect_kraken_futures(
        output_dir=output_dir,
        collection_time=COLLECTION_TIME,
    )


# ==================================================
# Stale provider data
# ==================================================

def test_rejects_provider_data_older_than_15_minutes(
    tmp_path,
    monkeypatch,
):
    provider_time = (
        COLLECTION_TIME - timedelta(minutes=15, seconds=1)
    )

    ticker = make_ticker(
        provider_time.isoformat()
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="stale",
    ):
        collect(
            tmp_path,
            monkeypatch,
            ticker,
        )

    output_file = (
        tmp_path / "kraken_futures_BTC.jsonl"
    )

    assert not output_file.exists()


def test_accepts_provider_data_exactly_15_minutes_old(
    tmp_path,
    monkeypatch,
):
    provider_time = (
        COLLECTION_TIME - timedelta(minutes=15)
    )

    ticker = make_ticker(
        provider_time.isoformat()
    )

    records = collect(
        tmp_path,
        monkeypatch,
        ticker,
    )

    assert len(records) == 1

    assert records[0]["provider_timestamp"] == (
        "2026-11-19T10:15:00Z"
    )


def test_stale_threshold_matches_collector_configuration():
    assert (
        kraken_futures.MAX_PROVIDER_AGE_MINUTES
        == 15
    )


# ==================================================
# Network error information protection
# ==================================================

def test_request_error_does_not_expose_sensitive_details(
    tmp_path,
    monkeypatch,
):
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError(
            f"Request failed: {SECRET_MARKER}"
        )

    monkeypatch.setattr(
        kraken_futures.requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="request failed",
    ) as error:
        kraken_futures.collect_kraken_futures(
            output_dir=tmp_path,
            collection_time=COLLECTION_TIME,
        )

    assert SECRET_MARKER not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_invalid_json_error_does_not_expose_sensitive_details(
    tmp_path,
    monkeypatch,
):
    install_response(
        monkeypatch,
        FakeResponse(
            json_error=ValueError(
                f"Invalid response: {SECRET_MARKER}"
            )
        ),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="invalid JSON",
    ) as error:
        kraken_futures.collect_kraken_futures(
            output_dir=tmp_path,
            collection_time=COLLECTION_TIME,
        )

    assert SECRET_MARKER not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


# ==================================================
# Existing history read error protection
# ==================================================

def test_history_read_error_does_not_expose_sensitive_details(
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

    def failing_open(
        path,
        mode="r",
        *args,
        **kwargs,
    ):
        if (
            path.name == "kraken_futures_BTC.jsonl"
            and mode == "r"
        ):
            raise OSError(
                f"History read failed: {SECRET_MARKER}"
            )

        return original_open(
            path,
            mode,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        kraken_futures.Path,
        "open",
        failing_open,
    )

    install_response(
        monkeypatch,
        FakeResponse(
            {
                "result": "success",
                "tickers": [
                    make_ticker(
                        "2026-11-19T10:25:00Z"
                    )
                ],
            }
        ),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Cannot read existing futures history",
    ) as error:
        kraken_futures.collect_kraken_futures(
            output_dir=tmp_path,
            collection_time=COLLECTION_TIME,
        )

    assert SECRET_MARKER not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


# ==================================================
# Output save error protection
# ==================================================

def test_output_write_error_does_not_expose_sensitive_details(
    tmp_path,
    monkeypatch,
):
    original_open = kraken_futures.Path.open

    def failing_open(
        path,
        mode="r",
        *args,
        **kwargs,
    ):
        if (
            path.name == "kraken_futures_BTC.jsonl"
            and mode == "a"
        ):
            raise OSError(
                f"Disk write failed: {SECRET_MARKER}"
            )

        return original_open(
            path,
            mode,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        kraken_futures.Path,
        "open",
        failing_open,
    )

    install_response(
        monkeypatch,
        FakeResponse(
            {
                "result": "success",
                "tickers": [
                    make_ticker(
                        "2026-11-19T10:25:00Z"
                    )
                ],
            }
        ),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="Cannot save Kraken Futures data",
    ) as error:
        kraken_futures.collect_kraken_futures(
            output_dir=tmp_path,
            collection_time=COLLECTION_TIME,
        )

    assert SECRET_MARKER not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
