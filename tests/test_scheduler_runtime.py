
"""Integration-focused tests for the scheduler application runtime."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from config.loader import load_config
from scheduler.runtime import (
    SchedulerRuntime,
    SchedulerRuntimeError,
    SUPPORTED_TIMEFRAMES,
)


CHECK_TIME = datetime(
    2026, 11, 20, 12, 0,
    tzinfo=timezone.utc,
)


def make_runtime(tmp_path):
    """Create a scheduler runtime using the real project configuration."""

    return SchedulerRuntime(
        config=load_config(),
        history_path=tmp_path / "scheduler_history.jsonl",
    )


def make_history(
    timestamp=CHECK_TIME,
    *,
    stale_timeframes=(),
):
    """Build minimal historical candle frames for freshness tests."""

    histories = {}

    for timeframe in SUPPORTED_TIMEFRAMES:
        age_minutes = 1

        if timeframe in stale_timeframes:
            age_minutes = 500

        close_time = timestamp - timedelta(
            minutes=age_minutes,
        )

        histories[timeframe] = pd.DataFrame(
            {
                "close_time": pd.to_datetime(
                    [close_time],
                    utc=True,
                ),
            }
        )

    return histories


def prepare_validated_runtime(
    runtime,
    *,
    timestamp=CHECK_TIME,
    stale_timeframes=(),
):
    """Prepare a runtime with historical data already validated."""

    histories = make_history(
        timestamp,
        stale_timeframes=stale_timeframes,
    )

    runtime._reset_cycle(timestamp)

    runtime._cycle_state["validation_passed"] = True
    runtime._cycle_state["historical_candles"] = histories

    runtime._cycle_state["history_freshness"] = {
        timeframe: runtime._is_history_fresh(
            timeframe,
            histories[timeframe],
            timestamp,
        )
        for timeframe in SUPPORTED_TIMEFRAMES
    }

    return histories


def test_runtime_initializes_with_project_configuration(tmp_path):
    runtime = make_runtime(tmp_path)

    assert runtime.pair == "XBTUSD"
    assert runtime.schedule_config["timezone"] == "UTC"
    assert runtime.engine is not None


def test_runtime_rejects_leverage_above_two(tmp_path):
    with pytest.raises(
        SchedulerRuntimeError,
        match="Leverage cannot exceed 2",
    ):
        SchedulerRuntime(
            config=load_config(),
            history_path=tmp_path / "history.jsonl",
            leverage=3,
        )


def test_runtime_rejects_non_utc_timestamp(tmp_path):
    runtime = make_runtime(tmp_path)

    invalid_time = datetime(
        2026, 11, 20, 12, 0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(
        SchedulerRuntimeError,
        match="UTC",
    ):
        runtime.run_once(now=invalid_time)


def test_runtime_requires_validation_before_analysis(tmp_path):
    runtime = make_runtime(tmp_path)

    runtime._reset_cycle(CHECK_TIME)

    assert runtime._analysis_is_ready() is False


def test_runtime_accepts_no_new_data_when_history_is_fresh(
    tmp_path,
    monkeypatch,
):
    runtime = make_runtime(tmp_path)

    histories = make_history()

    runtime._reset_cycle(CHECK_TIME)

    runtime._cycle_state["candle_results"] = {
        timeframe: {
            "status": "NO_NEW_DATA",
            "accepted_records": 0,
            "rejected_records": 0,
        }
        for timeframe in SUPPORTED_TIMEFRAMES
    }

    monkeypatch.setattr(
        runtime,
        "_load_shared_history",
        lambda: histories,
    )

    result = runtime._task_validation()

    assert result["status"] == "SUCCESS"
    assert result["has_new_data"] is False
    assert result["validated_timeframes"] == []

    assert set(
        result["no_new_data_timeframes"]
    ) == set(SUPPORTED_TIMEFRAMES)

    assert all(
        result["history_freshness"].values()
    )

    assert runtime._analysis_is_ready() is True


def test_runtime_rejects_stale_one_hour_history(
    tmp_path,
    monkeypatch,
):
    runtime = make_runtime(tmp_path)

    histories = make_history(
        stale_timeframes=("1h",),
    )

    runtime._reset_cycle(CHECK_TIME)

    runtime._cycle_state["candle_results"] = {
        timeframe: {
            "status": "NO_NEW_DATA",
            "accepted_records": 0,
            "rejected_records": 0,
        }
        for timeframe in SUPPORTED_TIMEFRAMES
    }

    monkeypatch.setattr(
        runtime,
        "_load_shared_history",
        lambda: histories,
    )

    result = runtime._task_validation()

    assert result["status"] == "SUCCESS"
    assert result["history_freshness"]["1h"] is False
    assert runtime._analysis_is_ready() is False


def test_runtime_rejects_missing_timeframe_history(
    tmp_path,
    monkeypatch,
):
    runtime = make_runtime(tmp_path)

    incomplete_history = {
        timeframe: pd.DataFrame(
            {
                "close_time": pd.to_datetime(
                    [CHECK_TIME - timedelta(minutes=1)],
                    utc=True,
                ),
            }
        )
        for timeframe in SUPPORTED_TIMEFRAMES
        if timeframe != "4h"
    }

    monkeypatch.setattr(
        "trader.data_loader.load_trading_candles",
        lambda **kwargs: incomplete_history,
    )

    with pytest.raises(
        SchedulerRuntimeError,
        match="incomplete timeframe collection",
    ):
        runtime._load_shared_history()


def test_runtime_market_analysis_uses_full_persisted_history(
    tmp_path,
    monkeypatch,
):
    runtime = make_runtime(tmp_path)

    histories = prepare_validated_runtime(runtime)

    runtime._cycle_state["historical_candles"]["1h"] = (
        pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        CHECK_TIME - timedelta(hours=3),
                        CHECK_TIME - timedelta(hours=2),
                        CHECK_TIME - timedelta(hours=1),
                    ],
                    utc=True,
                ),
                "close_time": pd.to_datetime(
                    [
                        CHECK_TIME - timedelta(hours=2),
                        CHECK_TIME - timedelta(hours=1),
                        CHECK_TIME,
                    ],
                    utc=True,
                ),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [10.0, 11.0, 12.0],
                "source": ["kraken", "kraken", "kraken"],
            }
        )
    )

    runtime._cycle_state["history_freshness"]["1h"] = True

    captured = {}

    def fake_analysis(*, data, analysis_time):
        captured["data"] = data.copy(deep=True)
        captured["analysis_time"] = analysis_time

        return {
            "trend": "BULLISH",
        }

    def fake_report(*, market_analysis):
        return {
            "report_type": "MARKET_ANALYSIS",
            "generated_at": CHECK_TIME.isoformat(),
        }

    import market_analysis.engine as analysis_module
    import reports.market_report as report_module

    monkeypatch.setattr(
        analysis_module,
        "analyze_market",
        fake_analysis,
    )

    monkeypatch.setattr(
        report_module,
        "generate_market_report",
        fake_report,
    )

    result = runtime._task_market_analysis()

    assert result["status"] == "SUCCESS"
    assert result["input_mode"] == "FULL_PERSISTED_HISTORY"
    assert result["input_candles"] == 3
    assert len(captured["data"]) == 3
    assert captured["analysis_time"] == CHECK_TIME

    assert runtime._cycle_state["market_analysis"] == {
        "trend": "BULLISH",
    }

    assert runtime._cycle_state["market_analysis_passed"] is True


def test_runtime_skips_market_analysis_when_history_is_stale(
    tmp_path,
):
    runtime = make_runtime(tmp_path)

    prepare_validated_runtime(
        runtime,
        stale_timeframes=("1h",),
    )

    result = runtime._task_market_analysis()

    assert result["status"] == "SKIPPED"
    assert result["reason"] == "NO_FRESH_VALIDATED_1H_HISTORY"


def test_runtime_passes_shared_history_to_trader(
    tmp_path,
    monkeypatch,
):
    runtime = make_runtime(tmp_path)

    histories = prepare_validated_runtime(runtime)

    runtime._cycle_state["market_analysis_passed"] = True

    import main

    captured = {}

    def fake_run_trader(**kwargs):
        captured.update(kwargs)

        return {
            "runtime_status": "SUCCESS",
            "signal": "NO_TRADE",
            "data_mode": "SHARED_HISTORY",
            "analysis_timestamp": CHECK_TIME.isoformat(),
            "candle_counts": {
                timeframe: len(data)
                for timeframe, data
                in kwargs["candle_data"].items()
            },
            "execution": {
                "manual_review_only": True,
                "auto_trading": False,
                "exchange_connection": False,
                "order_execution": False,
            },
        }

    monkeypatch.setattr(
        main,
        "run_trader",
        fake_run_trader,
    )

    result = runtime._task_trader()

    assert result["status"] == "SUCCESS"
    assert result["manual_review_only"] is True
    assert result["auto_trading"] is False
    assert result["exchange_connection"] is False
    assert result["order_execution"] is False

    assert captured["pair"] == runtime.pair
    assert captured["leverage"] == runtime.leverage
    assert captured["analysis_time"] == CHECK_TIME

    assert set(
        captured["candle_data"]
    ) == set(SUPPORTED_TIMEFRAMES)

    for timeframe in SUPPORTED_TIMEFRAMES:
        assert captured["candle_data"][timeframe].equals(
            histories[timeframe]
        )

        assert (
            captured["candle_data"][timeframe]
            is not histories[timeframe]
        )


def test_runtime_skips_trader_without_market_analysis(
    tmp_path,
):
    runtime = make_runtime(tmp_path)

    prepare_validated_runtime(runtime)

    runtime._cycle_state["market_analysis_passed"] = False

    result = runtime._task_trader()

    assert result["status"] == "SKIPPED"
    assert result["reason"] == "MARKET_ANALYSIS_NOT_COMPLETED"


def test_runtime_rejects_unsafe_trading_result(
    tmp_path,
    monkeypatch,
):
    runtime = make_runtime(tmp_path)

    prepare_validated_runtime(runtime)

    runtime._cycle_state["market_analysis_passed"] = True

    import main

    def unsafe_run_trader(**kwargs):
        return {
            "runtime_status": "SUCCESS",
            "signal": "NO_TRADE",
            "execution": {
                "manual_review_only": True,
                "auto_trading": True,
                "exchange_connection": False,
                "order_execution": False,
            },
        }

    monkeypatch.setattr(
        main,
        "run_trader",
        unsafe_run_trader,
    )

    with pytest.raises(
        SchedulerRuntimeError,
        match="Trading safety verification failed",
    ):
        runtime._task_trader()


def test_runtime_resets_cycle_state_between_runs(tmp_path):
    runtime = make_runtime(tmp_path)

    runtime._reset_cycle(CHECK_TIME)

    runtime._cycle_state["market_analysis"] = {
        "trend": "BULLISH",
    }

    next_time = CHECK_TIME + timedelta(minutes=5)

    runtime._reset_cycle(next_time)

    assert runtime._cycle_state["timestamp"] == next_time
    assert runtime._cycle_state["market_analysis"] is None
    assert runtime._cycle_state["market_analysis_passed"] is False
    assert runtime._cycle_state["historical_candles"] == {}
    assert runtime._cycle_state["candle_results"] == {}
