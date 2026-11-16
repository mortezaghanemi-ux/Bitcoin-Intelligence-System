"""Tests for deterministic trading signal identity."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.signal_engine.engine import (
    SUPPORTED_TIMEFRAMES,
    _build_signal_identity,
    _validate_input_data,
)


UTC = timezone.utc
ANALYSIS_TIME = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}

CANDLE_COUNT = 40


@pytest.fixture
def market_data():
    data = {}

    for timeframe in SUPPORTED_TIMEFRAMES:
        interval = TIMEFRAME_SECONDS[timeframe]

        start = ANALYSIS_TIME - timedelta(
            seconds=interval * (CANDLE_COUNT + 2)
        )

        rows = []

        for index in range(CANDLE_COUNT):
            open_time = start + timedelta(
                seconds=index * interval
            )
            close_time = open_time + timedelta(
                seconds=interval
            )

            opened = 95000.0 + index * 10.0
            closed = opened + 5.0

            rows.append({
                "open_time": open_time,
                "close_time": close_time,
                "open": opened,
                "high": closed + 10.0,
                "low": opened - 10.0,
                "close": closed,
                "volume": 100.0 + index,
                "source": "kraken",
            })

        data[timeframe] = pd.DataFrame(rows)

    return data


def build_identity(data):
    validated = _validate_input_data(
        data=data,
        analysis_time=pd.Timestamp(ANALYSIS_TIME),
        minimum_candles=CANDLE_COUNT,
    )

    return _build_signal_identity(validated)


def test_identical_market_data_produces_identical_identity(market_data):
    first = build_identity(market_data)
    second = build_identity(market_data)

    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize(
    "column,change",
    [
        ("open", 0.25),
        ("high", 0.25),
        ("low", -0.25),
        ("close", 0.25),
        ("volume", 50.0),
    ],
)
def test_historical_candle_changes_identity(
    market_data,
    column,
    change,
):
    original_identity = build_identity(market_data)

    changed_data = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in market_data.items()
    }

    historical_index = 5

    changed_data["5m"].loc[
        historical_index,
        column,
    ] += change

    changed_identity = build_identity(changed_data)

    assert changed_identity != original_identity


def test_historical_source_change_changes_identity(market_data):
    original_identity = build_identity(market_data)

    changed_data = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in market_data.items()
    }

    changed_data["5m"].loc[5, "source"] = "coingecko"

    assert build_identity(changed_data) != original_identity


@pytest.mark.parametrize(
    "timeframe",
    SUPPORTED_TIMEFRAMES,
)
def test_historical_change_in_every_timeframe_changes_identity(
    market_data,
    timeframe,
):
    original_identity = build_identity(market_data)

    changed_data = {
        key: candles.copy(deep=True)
        for key, candles in market_data.items()
    }

    changed_data[timeframe].loc[5, "close"] += 0.25

    changed_identity = build_identity(changed_data)

    assert changed_identity != original_identity


def test_latest_candle_change_changes_identity(market_data):
    original_identity = build_identity(market_data)

    changed_data = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in market_data.items()
    }

    last_index = len(changed_data["5m"]) - 1

    changed_data["5m"].loc[last_index, "close"] += 0.25

    assert build_identity(changed_data) != original_identity


def test_change_in_one_timeframe_changes_identity(market_data):
    original_identity = build_identity(market_data)

    changed_data = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in market_data.items()
    }

    changed_data["1h"].loc[10, "volume"] += 50.0

    assert build_identity(changed_data) != original_identity


def test_original_market_data_is_not_modified(market_data):
    original = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in market_data.items()
    }

    build_identity(market_data)

    for timeframe in SUPPORTED_TIMEFRAMES:
        pd.testing.assert_frame_equal(
            market_data[timeframe],
            original[timeframe],
        )


def test_identity_is_sha256_hex_digest(market_data):
    identity = build_identity(market_data)

    assert len(identity) == 64
    assert all(
        character in "0123456789abcdef"
        for character in identity
    )