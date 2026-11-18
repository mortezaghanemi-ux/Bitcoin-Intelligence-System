"""Tests for trading pipeline candle continuity validation."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import trader.pipeline as pipeline_module

from trader.pipeline import (
    TradingPipeline,
    TradingPipelineError,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026,
    11,
    15,
    12,
    0,
    tzinfo=UTC,
)

TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


def make_config():
    return {
        "trader": {
            "trader": {
                "enabled": True,
                "execution": {
                    "manual_review_only": True,
                    "auto_trading": False,
                    "exchange_connection": False,
                    "exchange_api_keys_required": False,
                },
                "timeframes": list(TIMEFRAMES),
                "signals": {
                    "confidence": {
                        "minimum_threshold": 60,
                    }
                },
                "position": {
                    "minimum_risk_reward": 1.5,
                },
                "risk": {
                    "maximum_leverage": 2,
                },
            }
        },
        "data_quality": {
            "freshness": {
                "enabled": True,
                "reject_stale_data": True,
                "max_age_minutes": {
                    "candles": 15,
                },
            }
        },
    }


# ---------------------------------------------------------
# Candle factories
# ---------------------------------------------------------


def make_candles(timeframe, count=4):
    interval_minutes = TIMEFRAMES[timeframe]

    latest_close = pd.Timestamp(CHECK_TIME)

    start = latest_close - pd.Timedelta(
        minutes=interval_minutes * count
    )

    rows = []

    for index in range(count):
        opened = start + pd.Timedelta(
            minutes=interval_minutes * index
        )

        closed = opened + pd.Timedelta(
            minutes=interval_minutes
        )

        price = 95000 + index * 10

        rows.append({
            "open_time": opened,
            "close_time": closed,
            "open": price,
            "high": price + 20,
            "low": price - 20,
            "close": price + 10,
            "volume": 100 + index,
            "source": "kraken",
        })

    return pd.DataFrame(rows)


def make_market_data():
    return {
        timeframe: make_candles(timeframe)
        for timeframe in TIMEFRAMES
    }


def make_signal():
    return {
        "signal_id": "continuity-test-signal",
        "analysis_timestamp": CHECK_TIME.isoformat(),
        "signal": "LONG",
        "status": "REQUIRES_MANUAL_REVIEW",
        "data_quality": {
            "future_data_used": False,
            "closed_candles_only": True,
        },
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
    }


# ---------------------------------------------------------
# Fake storage
# ---------------------------------------------------------


class FakeHistoryStore:

    def __init__(self):
        self.records = {}

    def read_all(self):
        return list(self.records.values())

    def get(self, signal_id):
        return self.records.get(signal_id)

    def append(self, signal):
        signal_id = signal["signal_id"]

        if signal_id in self.records:
            return {
                **self.records[signal_id],
                "duplicate": True,
            }

        self.records[signal_id] = dict(signal)

        return dict(signal)


class FakeRejectionStore:

    def __init__(self):
        self.records = {}

    def read_all(self):
        return list(self.records.values())

    def get(self, signal_id):
        return self.records.get(signal_id)

    def contains(self, signal_id):
        return signal_id in self.records

    def append(self, signal):
        signal_id = signal["signal_id"]

        if signal_id in self.records:
            return {
                **self.records[signal_id],
                "duplicate": True,
            }

        self.records[signal_id] = dict(signal)

        return dict(signal)


# ---------------------------------------------------------
# Fake signal engine
# ---------------------------------------------------------


class FakeSignalEngine:

    def __init__(self):
        self.calls = 0

    def generate(
        self,
        data,
        *,
        analysis_time=None,
        account_equity=None,
        leverage=1.0,
    ):
        self.calls += 1

        return make_signal()


# ---------------------------------------------------------
# Fake candle manager
# ---------------------------------------------------------


class FakeCandleManager:

    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.candles = pd.DataFrame()

    def ingest(
        self,
        data,
        *,
        validation_time=None,
        max_age_minutes=None,
    ):
        self.candles = data.copy(deep=True)

        return {
            "status": "PASS",
            "accepted": self.candles,
            "rejected": pd.DataFrame(),
            "report": {
                "status": "PASS",
            },
        }

    def get_candles(self):
        return self.candles.copy(deep=True)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


@pytest.fixture
def pipeline(monkeypatch):
    history_store = FakeHistoryStore()
    rejection_store = FakeRejectionStore()
    engine = FakeSignalEngine()

    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        lambda **kwargs: make_config(),
    )

    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    instance = TradingPipeline(
        history_store=history_store,
        rejection_store=rejection_store,
        engine=engine,
    )

    return instance


@pytest.fixture
def continuity_validator():
    instance = TradingPipeline.__new__(
        TradingPipeline
    )

    return instance


# ---------------------------------------------------------
# Direct continuity validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_accepts_contiguous_candles(
    continuity_validator,
    timeframe,
):
    candles = make_candles(timeframe)

    continuity_validator._validate_candle_continuity(
        timeframe,
        candles,
    )


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_rejects_missing_candle_interval(
    continuity_validator,
    timeframe,
):
    candles = make_candles(timeframe)

    candles = candles.drop(
        index=1
    ).reset_index(drop=True)

    with pytest.raises(
        TradingPipelineError,
        match="missing interval",
    ):
        continuity_validator._validate_candle_continuity(
            timeframe,
            candles,
        )


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_rejects_misaligned_candle_timestamp(
    continuity_validator,
    timeframe,
):
    candles = make_candles(timeframe)

    candles.loc[1, "open_time"] = (
        candles.loc[1, "open_time"]
        + timedelta(seconds=30)
    )

    candles.loc[1, "close_time"] = (
        candles.loc[1, "close_time"]
        + timedelta(seconds=30)
    )

    with pytest.raises(
        TradingPipelineError,
        match="not aligned",
    ):
        continuity_validator._validate_candle_continuity(
            timeframe,
            candles,
        )


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_rejects_invalid_candle_duration(
    continuity_validator,
    timeframe,
):
    candles = make_candles(timeframe)

    candles.loc[1, "close_time"] = (
        candles.loc[1, "close_time"]
        + timedelta(seconds=30)
    )

    with pytest.raises(
        TradingPipelineError,
        match="invalid open/close duration",
    ):
        continuity_validator._validate_candle_continuity(
            timeframe,
            candles,
        )


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_accepts_single_candle(
    continuity_validator,
    timeframe,
):
    candles = make_candles(
        timeframe,
        count=1,
    )

    continuity_validator._validate_candle_continuity(
        timeframe,
        candles,
    )


# ---------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_pipeline_rejects_gap_before_signal_generation(
    pipeline,
    timeframe,
):
    data = make_market_data()

    data[timeframe] = (
        data[timeframe]
        .drop(index=1)
        .reset_index(drop=True)
    )

    with pytest.raises(
        TradingPipelineError,
        match="missing interval",
    ):
        pipeline.run(
            data,
            analysis_time=CHECK_TIME,
            account_equity=10000,
            leverage=1,
        )

    assert pipeline.engine.calls == 0

    assert pipeline.history_store.records == {}

    assert pipeline.rejection_store.records == {}


def test_pipeline_accepts_contiguous_market_data(
    pipeline,
):
    result = pipeline.run(
        make_market_data(),
        analysis_time=CHECK_TIME,
        account_equity=10000,
        leverage=1,
    )

    assert result["pipeline_status"] == "SUCCESS"

    assert pipeline.engine.calls == 1

    assert result["validated_timeframes"] == list(
        TIMEFRAMES
    )

    assert result["persistence"]["status"] == "SAVED"


def test_pipeline_does_not_persist_signal_after_gap(
    pipeline,
):
    data = make_market_data()

    data["4h"] = (
        data["4h"]
        .drop(index=1)
        .reset_index(drop=True)
    )

    with pytest.raises(TradingPipelineError):
        pipeline.run(
            data,
            analysis_time=CHECK_TIME,
            account_equity=10000,
            leverage=1,
        )

    assert pipeline.engine.calls == 0

    assert len(
        pipeline.history_store.records
    ) == 0

    assert len(
        pipeline.rejection_store.records
    ) == 0


def test_pipeline_does_not_modify_original_candles(
    pipeline,
):
    data = make_market_data()

    original = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in data.items()
    }

    pipeline.run(
        data,
        analysis_time=CHECK_TIME,
        account_equity=10000,
        leverage=1,
    )

    for timeframe in TIMEFRAMES:
        pd.testing.assert_frame_equal(
            data[timeframe],
            original[timeframe],
        )


def test_pipeline_preserves_manual_execution_safety(
    pipeline,
):
    result = pipeline.run(
        make_market_data(),
        analysis_time=CHECK_TIME,
        account_equity=10000,
        leverage=1,
    )

    assert result["execution"] == {
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
        "order_execution": False,
    }