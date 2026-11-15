"""Tests for the Bitcoin trading pipeline."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import trader.pipeline as pipeline_module

from trader.pipeline import (
    TradingPipeline,
    TradingPipelineError,
    TradingPipelinePersistenceError,
)


UTC = timezone.utc
CHECK_TIME = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

TIMEFRAMES = ("5m", "15m", "1h", "4h")


# ---------------------------------------------------------
# Test fixtures
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


def make_candles(timeframe="5m", count=40):
    intervals = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
    }

    minutes = intervals[timeframe]

    start = CHECK_TIME - timedelta(
        minutes=minutes * (count + 1)
    )

    rows = []

    for index in range(count):

        opened = start + timedelta(
            minutes=minutes * index
        )

        closed = opened + timedelta(
            minutes=minutes
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


def make_signal(
    signal="LONG",
    status="REQUIRES_MANUAL_REVIEW",
):
    result = {
        "signal_id": "a" * 64,
        "analysis_timestamp": CHECK_TIME.isoformat(),
        "signal": signal,
        "status": status,
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

    if signal == "NO_TRADE":
        result["rejection_reasons"] = [
            "insufficient_market_confirmation"
        ]

    return result


class FakeHistoryStore:

    def __init__(self):
        self.records = {}

    def read_all(self):
        return list(self.records.values())

    def get(self, signal_id):
        return self.records.get(signal_id)

    def append(self, signal):
        self.records[signal["signal_id"]] = dict(signal)
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
        self.records[signal["signal_id"]] = dict(signal)
        return dict(signal)


class FakeEngine:

    def __init__(self, signal=None):
        self.signal = signal or make_signal()
        self.calls = []

    def generate(
        self,
        data,
        *,
        analysis_time=None,
        account_equity=None,
        leverage=1.0,
    ):
        self.calls.append({
            "data": data,
            "analysis_time": analysis_time,
            "account_equity": account_equity,
            "leverage": leverage,
        })

        return dict(self.signal)


class FakeCandleManager:

    datasets = {}

    def __init__(self, timeframe):
        self.timeframe = timeframe
        self._candles = pd.DataFrame()

    def ingest(
        self,
        data,
        *,
        validation_time=None,
        max_age_minutes=None,
    ):
        self._candles = data.copy(deep=True)

        return {
            "status": "PASS",
            "accepted": self._candles,
            "rejected": pd.DataFrame(),
            "report": {
                "status": "PASS",
            },
        }

    def get_candles(self):
        return self._candles.copy(deep=True)


@pytest.fixture
def stores():
    return FakeHistoryStore(), FakeRejectionStore()


@pytest.fixture
def pipeline(monkeypatch, stores):
    history, rejections = stores
    engine = FakeEngine()

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
        history_store=history,
        rejection_store=rejections,
        engine=engine,
    )

    return instance


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


def test_pipeline_initializes_successfully(pipeline):
    assert pipeline.timeframes == TIMEFRAMES
    assert pipeline.trader_config["enabled"] is True


def test_pipeline_rejects_disabled_trader(monkeypatch, stores):
    config = make_config()
    config["trader"]["trader"]["enabled"] = False

    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        lambda **kwargs: config,
    )

    history, rejections = stores

    with pytest.raises(TradingPipelineError):
        TradingPipeline(
            history_store=history,
            rejection_store=rejections,
            engine=FakeEngine(),
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("manual_review_only", False),
        ("auto_trading", True),
        ("exchange_connection", True),
        ("exchange_api_keys_required", True),
    ],
)
def test_pipeline_rejects_unsafe_execution_config(
    monkeypatch,
    stores,
    key,
    value,
):
    config = make_config()

    config["trader"]["trader"]["execution"][key] = value

    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        lambda **kwargs: config,
    )

    history, rejections = stores

    with pytest.raises(TradingPipelineError):
        TradingPipeline(
            history_store=history,
            rejection_store=rejections,
            engine=FakeEngine(),
        )


def test_pipeline_rejects_unsupported_timeframes(
    monkeypatch,
    stores,
):
    config = make_config()

    config["trader"]["trader"]["timeframes"] = [
        "5m",
        "15m",
        "1h",
    ]

    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        lambda **kwargs: config,
    )

    history, rejections = stores

    with pytest.raises(TradingPipelineError):
        TradingPipeline(
            history_store=history,
            rejection_store=rejections,
            engine=FakeEngine(),
        )


def test_pipeline_wraps_configuration_errors(monkeypatch, stores):
    def failing_loader(**kwargs):
        raise ValueError("invalid configuration")

    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        failing_loader,
    )

    history, rejections = stores

    with pytest.raises(TradingPipelineError):
        TradingPipeline(
            history_store=history,
            rejection_store=rejections,
            engine=FakeEngine(),
        )


# ---------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------


def test_default_analysis_time_is_utc():
    result = TradingPipeline._normalize_analysis_time(None)

    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)


def test_aware_utc_analysis_time_is_accepted():
    result = TradingPipeline._normalize_analysis_time(
        CHECK_TIME
    )

    assert result == CHECK_TIME


def test_naive_analysis_time_is_rejected():
    with pytest.raises(TradingPipelineError):
        TradingPipeline._normalize_analysis_time(
            datetime(2026, 11, 15, 12, 0)
        )


def test_non_utc_analysis_time_is_rejected():
    non_utc = datetime(
        2026,
        11,
        15,
        14,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(TradingPipelineError):
        TradingPipeline._normalize_analysis_time(
            non_utc
        )


def test_invalid_analysis_time_type_is_rejected():
    with pytest.raises(TradingPipelineError):
        TradingPipeline._normalize_analysis_time(
            "2026-11-15T12:00:00Z"
        )


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------


def test_pipeline_rejects_non_dictionary_data(pipeline):
    with pytest.raises(TradingPipelineError):
        pipeline._validate_input([])


def test_pipeline_rejects_missing_timeframe(pipeline):
    data = make_market_data()
    del data["4h"]

    with pytest.raises(TradingPipelineError):
        pipeline._validate_input(data)


def test_pipeline_rejects_unknown_timeframe(pipeline):
    data = make_market_data()
    data["1d"] = make_candles("1h")

    with pytest.raises(TradingPipelineError):
        pipeline._validate_input(data)


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_pipeline_rejects_non_dataframe_input(
    pipeline,
    timeframe,
):
    data = make_market_data()
    data[timeframe] = []

    with pytest.raises(TradingPipelineError):
        pipeline._validate_input(data)


@pytest.mark.parametrize(
    "timeframe",
    TIMEFRAMES,
)
def test_pipeline_rejects_empty_dataframe(
    pipeline,
    timeframe,
):
    data = make_market_data()
    data[timeframe] = pd.DataFrame()

    with pytest.raises(TradingPipelineError):
        pipeline._validate_input(data)


# ---------------------------------------------------------
# Freshness validation
# ---------------------------------------------------------


def test_fresh_candles_are_accepted(pipeline):
    candles = make_candles("5m")

    pipeline._validate_freshness(
        "5m",
        candles,
        CHECK_TIME,
    )


def test_future_candle_is_rejected(pipeline):
    candles = make_candles("5m")

    candles.loc[
        candles.index[-1],
        "close_time",
    ] = CHECK_TIME + timedelta(minutes=1)

    with pytest.raises(TradingPipelineError):
        pipeline._validate_freshness(
            "5m",
            candles,
            CHECK_TIME,
        )


def test_stale_candle_is_rejected(pipeline):
    candles = make_candles("5m")

    candles.loc[
        candles.index[-1],
        "close_time",
    ] = CHECK_TIME - timedelta(hours=2)

    with pytest.raises(TradingPipelineError):
        pipeline._validate_freshness(
            "5m",
            candles,
            CHECK_TIME,
        )


def test_naive_candle_timestamp_is_rejected(pipeline):
    candles = make_candles("5m")

    candles["close_time"] = (
        candles["close_time"].dt.tz_localize(None)
    )

    with pytest.raises(TradingPipelineError):
        pipeline._validate_freshness(
            "5m",
            candles,
            CHECK_TIME,
        )


def test_empty_candles_are_rejected_by_freshness(pipeline):
    with pytest.raises(TradingPipelineError):
        pipeline._validate_freshness(
            "5m",
            pd.DataFrame(),
            CHECK_TIME,
        )


# ---------------------------------------------------------
# Candle preparation
# ---------------------------------------------------------


def test_prepare_candles_returns_all_timeframes(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    prepared = pipeline._prepare_candles(
        make_market_data(),
        CHECK_TIME,
    )

    assert set(prepared) == set(TIMEFRAMES)

    for timeframe in TIMEFRAMES:
        assert isinstance(
            prepared[timeframe],
            pd.DataFrame,
        )

        assert not prepared[timeframe].empty


def test_prepare_candles_rejects_failed_validation(
    pipeline,
    monkeypatch,
):
    class FailedManager:

        def __init__(self, timeframe):
            pass

        def ingest(self, data, **kwargs):
            return {
                "status": "FAIL",
                "report": {
                    "errors": 1,
                },
                "rejected": pd.DataFrame(),
            }

    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FailedManager,
    )

    with pytest.raises(TradingPipelineError):
        pipeline._prepare_candles(
            make_market_data(),
            CHECK_TIME,
        )


def test_prepare_candles_rejects_rejected_records(
    pipeline,
    monkeypatch,
):
    class RejectedManager:

        def __init__(self, timeframe):
            self.timeframe = timeframe

        def ingest(self, data, **kwargs):
            return {
                "status": "PASS",
                "rejected": pd.DataFrame({
                    "validation_errors": [
                        "invalid_record"
                    ]
                }),
                "report": {},
            }

        def get_candles(self):
            return make_candles(self.timeframe)

    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        RejectedManager,
    )

    with pytest.raises(TradingPipelineError):
        pipeline._prepare_candles(
            make_market_data(),
            CHECK_TIME,
        )


# ---------------------------------------------------------
# Signal persistence
# ---------------------------------------------------------


def test_long_signal_is_saved_to_history(
    pipeline,
    stores,
):
    history, rejections = stores
    signal = make_signal()

    result = pipeline._persist_signal(signal)

    assert result["status"] == "SAVED"
    assert result["history_saved"] is True
    assert result["rejection_saved"] is False

    assert signal["signal_id"] in history.records
    assert rejections.records == {}


def test_no_trade_signal_is_saved_as_rejection(
    pipeline,
    stores,
):
    history, rejections = stores

    signal = make_signal(
        signal="NO_TRADE",
        status="REJECTED",
    )

    result = pipeline._persist_signal(signal)

    assert result["status"] == "SAVED"
    assert result["history_saved"] is True
    assert result["rejection_saved"] is True

    assert signal["signal_id"] in history.records
    assert signal["signal_id"] in rejections.records


def test_duplicate_signal_is_not_saved_again(
    pipeline,
    stores,
):
    history, rejections = stores
    signal = make_signal()

    history.records[signal["signal_id"]] = dict(signal)

    result = pipeline._persist_signal(signal)

    assert result["status"] == "DUPLICATE"
    assert len(history.records) == 1


def test_invalid_signal_identifier_is_rejected(pipeline):
    signal = make_signal()
    signal["signal_id"] = ""

    with pytest.raises(TradingPipelineError):
        pipeline._persist_signal(signal)


def test_history_persistence_failure_is_reported(pipeline):
    class FailingHistory:

        def get(self, signal_id):
            return None

        def append(self, signal):
            raise OSError("disk unavailable")

    pipeline.history_store = FailingHistory()

    with pytest.raises(TradingPipelinePersistenceError):
        pipeline._persist_signal(make_signal())


def test_rejection_persistence_failure_is_reported(pipeline):
    class FailingRejections:

        def append(self, signal):
            raise OSError("disk unavailable")

    pipeline.rejection_store = FailingRejections()

    signal = make_signal(
        signal="NO_TRADE",
        status="REJECTED",
    )

    with pytest.raises(TradingPipelinePersistenceError):
        pipeline._persist_signal(signal)


# ---------------------------------------------------------
# Complete pipeline execution
# ---------------------------------------------------------


def test_complete_pipeline_run_succeeds(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    result = pipeline.run(
        make_market_data(),
        analysis_time=CHECK_TIME,
        account_equity=10000,
        leverage=1,
    )

    assert result["pipeline_status"] == "SUCCESS"

    assert result["validated_timeframes"] == list(
        TIMEFRAMES
    )

    assert result["persistence"]["status"] == "SAVED"

    assert result["execution"] == {
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
        "order_execution": False,
    }


def test_pipeline_passes_risk_parameters_to_engine(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    pipeline.run(
        make_market_data(),
        analysis_time=CHECK_TIME,
        account_equity=5000,
        leverage=2,
    )

    call = pipeline.engine.calls[-1]

    assert call["analysis_time"] == CHECK_TIME
    assert call["account_equity"] == 5000
    assert call["leverage"] == 2


def test_pipeline_rejects_engine_future_data_flag(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    pipeline.engine.signal["data_quality"][
        "future_data_used"
    ] = True

    with pytest.raises(TradingPipelineError):
        pipeline.run(
            make_market_data(),
            analysis_time=CHECK_TIME,
        )


def test_pipeline_rejects_engine_without_closed_candle_flag(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    pipeline.engine.signal["data_quality"][
        "closed_candles_only"
    ] = False

    with pytest.raises(TradingPipelineError):
        pipeline.run(
            make_market_data(),
            analysis_time=CHECK_TIME,
        )


def test_pipeline_rejects_unsafe_engine_output(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    pipeline.engine.signal["execution"][
        "auto_trading"
    ] = True

    with pytest.raises(TradingPipelineError):
        pipeline.run(
            make_market_data(),
            analysis_time=CHECK_TIME,
        )


def test_pipeline_rejects_non_dictionary_engine_output(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    pipeline.engine.generate = lambda *args, **kwargs: None

    with pytest.raises(TradingPipelineError):
        pipeline.run(
            make_market_data(),
            analysis_time=CHECK_TIME,
        )


def test_pipeline_rejects_engine_error(
    pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "CandleManager",
        FakeCandleManager,
    )

    def failing_generate(*args, **kwargs):
        raise pipeline_module.TradingSignalEngineError(
            "engine failure"
        )

    pipeline.engine.generate = failing_generate

    with pytest.raises(TradingPipelineError):
        pipeline.run(
            make_market_data(),
            analysis_time=CHECK_TIME,
        )