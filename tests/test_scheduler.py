
"""Tests for the UTC scheduler engine."""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from scheduler.engine import (
    SchedulerEngine,
    SchedulerError,
)


NOW = datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)


def make_config():
    """Return a complete, isolated scheduler configuration."""

    return {
        "timezone": "UTC",
        "market_data": {
            "enabled": True,
            "interval_minutes": 5,
        },
        "candles": {
            "enabled": False,
            "timeframes": ["5m", "15m", "1h", "4h"],
            "collect_closed_candles_only": True,
        },
        "derivatives": {
            "enabled": False,
            "interval_minutes": 5,
        },
        "etf_flows": {
            "enabled": False,
            "interval_minutes": 60,
        },
        "onchain": {
            "enabled": False,
            "interval_minutes": 60,
        },
        "macro": {
            "enabled": False,
            "interval_minutes": 1440,
        },
        "validation": {
            "enabled": True,
            "run_after_collection": True,
            "run_before_every_engine": True,
        },
        "market_analysis": {
            "enabled": True,
            "run_after_validation": True,
        },
        "forecast": {
            "enabled": True,
            "interval_hours": 6,
            "run_after_validation": True,
        },
        "trader": {
            "enabled": True,
            "run_after_data_validation": True,
            "run_after_market_analysis": True,
            "require_closed_candles": True,
        },
        "execution": {
            "prevent_overlapping_runs": True,
            "sequential_dependencies": True,
            "stop_on_critical_error": True,
            "allow_retry_on_failure": True,
            "maximum_retries": 2,
        },
        "history": {
            "enabled": True,
            "record_execution_status": True,
            "record_execution_timestamps": True,
            "prevent_duplicate_runs": True,
            "preserve_execution_history": True,
        },
    }


def make_tasks(events=None):
    """Return callbacks for the enabled scheduler tasks."""

    events = events if events is not None else []

    def callback(name):
        def run():
            events.append(name)
            return {"task": name, "completed": True}

        return run

    return {
        name: callback(name)
        for name in (
            "market_data",
            "candles",
            "derivatives",
            "etf_flows",
            "onchain",
            "macro",
            "validation",
            "market_analysis",
            "forecast",
            "trader",
        )
    }


def make_engine(
    tmp_path,
    config=None,
    tasks=None,
    clock=None,
):
    """Create an engine with isolated history storage."""

    return SchedulerEngine(
        config=config or make_config(),
        tasks=tasks if tasks is not None else make_tasks(),
        history_path=tmp_path / "scheduler_history.jsonl",
        clock=clock or (lambda: NOW),
    )


def test_scheduler_runs_tasks_in_dependency_order(tmp_path):
    events = []
    engine = make_engine(
        tmp_path,
        tasks=make_tasks(events),
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "SUCCESS"
    assert events == [
        "market_data",
        "validation",
        "market_analysis",
        "forecast",
        "trader",
    ]
    assert result["task_count"] == 5
    assert result["successful_tasks"] == 5
    assert result["failed_tasks"] == 0


def test_scheduler_returns_utc_execution_metadata(tmp_path):
    engine = make_engine(tmp_path)

    result = engine.run_cycle(now=NOW)

    assert result["timestamp"].endswith("+00:00")
    assert result["execution"]["timezone"] == "UTC"
    assert result["execution"]["sequential"] is True
    assert result["execution"]["automatic_trading"] is False
    assert result["execution"]["order_execution"] is False


def test_scheduler_does_not_run_disabled_collection_tasks(tmp_path):
    config = make_config()
    config["market_data"]["enabled"] = False

    engine = make_engine(
        tmp_path,
        config=config,
        tasks=make_tasks(),
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "SUCCESS"
    assert result["tasks"] == []


def test_scheduler_force_runs_enabled_tasks(tmp_path):
    config = make_config()
    config["market_data"]["enabled"] = False

    events = []
    engine = make_engine(
        tmp_path,
        config=config,
        tasks=make_tasks(events),
    )

    result = engine.run_cycle(
        now=NOW,
        force=True,
    )

    assert result["status"] == "SUCCESS"
    assert events == [
        "validation",
        "market_analysis",
        "forecast",
        "trader",
    ]


def test_scheduler_rejects_non_boolean_force(tmp_path):
    engine = make_engine(tmp_path)

    with pytest.raises(SchedulerError):
        engine.run_cycle(
            now=NOW,
            force=1,
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-11-20T08:00:00+00:00",
        None,
        123,
    ],
)
def test_scheduler_rejects_invalid_timestamps(tmp_path, timestamp):
    engine = make_engine(tmp_path)

    with pytest.raises(SchedulerError):
        engine.run_cycle(now=timestamp)


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 11, 20, 8, 0),
        datetime(
            2026,
            11,
            20,
            8,
            0,
            tzinfo=timezone(timedelta(hours=3)),
        ),
    ],
)
def test_scheduler_rejects_non_utc_timestamps(tmp_path, timestamp):
    engine = make_engine(tmp_path)

    with pytest.raises(SchedulerError):
        engine.run_cycle(now=timestamp)


def test_scheduler_rejects_invalid_timezone(tmp_path):
    config = make_config()
    config["timezone"] = "Europe/Berlin"

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        float("inf"),
        float("nan"),
        "5",
    ],
)
def test_scheduler_rejects_invalid_market_interval(tmp_path, value):
    config = make_config()
    config["market_data"]["interval_minutes"] = value

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        1.5,
        "2",
    ],
)
def test_scheduler_rejects_invalid_retry_count(tmp_path, value):
    config = make_config()
    config["execution"]["maximum_retries"] = value

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


@pytest.mark.parametrize(
    "value",
    [
        "yes",
        1,
        None,
    ],
)
def test_scheduler_rejects_invalid_execution_boolean(tmp_path, value):
    config = make_config()
    config["execution"]["stop_on_critical_error"] = value

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


def test_scheduler_rejects_invalid_history_configuration(tmp_path):
    config = make_config()
    config["history"]["enabled"] = "true"

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


@pytest.mark.parametrize(
    "timeframes",
    [
        None,
        [],
        ["5m", ""],
        ["5m", "5m"],
        [5, "15m"],
    ],
)
def test_scheduler_rejects_invalid_candle_timeframes(
    tmp_path,
    timeframes,
):
    config = make_config()
    config["candles"]["enabled"] = True
    config["candles"]["timeframes"] = timeframes

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


def test_scheduler_rejects_missing_callback(tmp_path):
    engine = make_engine(
        tmp_path,
        tasks={},
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "FAILED"
    assert result["failed_tasks"] == 1
    assert result["tasks"][0]["task"] == "market_data"
    assert result["tasks"][0]["status"] == "FAILED"


def test_scheduler_retries_failed_task(tmp_path):
    attempts = []

    def unstable_task():
        attempts.append(1)

        if len(attempts) < 3:
            raise RuntimeError("Temporary failure")

        return {"completed": True}

    config = make_config()
    config["market_analysis"]["enabled"] = False
    config["forecast"]["enabled"] = False
    config["trader"]["enabled"] = False

    tasks = make_tasks()
    tasks["market_data"] = unstable_task

    engine = make_engine(
        tmp_path,
        config=config,
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "SUCCESS"
    assert len(attempts) == 3
    assert result["tasks"][0]["attempt"] == 3
    assert result["tasks"][0]["status"] == "SUCCESS"


def test_scheduler_does_not_retry_when_disabled(tmp_path):
    attempts = []

    def failing_task():
        attempts.append(1)
        raise RuntimeError("Failure")

    config = make_config()
    config["execution"]["allow_retry_on_failure"] = False

    tasks = make_tasks()
    tasks["market_data"] = failing_task

    engine = make_engine(
        tmp_path,
        config=config,
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "FAILED"
    assert len(attempts) == 1


def test_scheduler_stops_after_critical_collection_failure(tmp_path):
    events = []

    def failing_collection():
        events.append("market_data")
        raise RuntimeError("Collection failed")

    tasks = make_tasks(events)
    tasks["market_data"] = failing_collection

    config = make_config()
    config["execution"]["allow_retry_on_failure"] = False

    engine = make_engine(
        tmp_path,
        config=config,
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "FAILED"
    assert events == ["market_data"]
    assert result["failed_tasks"] == 1


def test_scheduler_records_execution_history(tmp_path):
    history_path = tmp_path / "scheduler_history.jsonl"
    engine = SchedulerEngine(
        config=make_config(),
        tasks=make_tasks(),
        history_path=history_path,
        clock=lambda: NOW,
    )

    engine.run_cycle(now=NOW)

    records = engine.get_history()

    assert len(records) == 5
    assert records[0]["task"] == "market_data"
    assert records[0]["status"] == "SUCCESS"
    assert records[0]["started_at"].endswith("+00:00")
    assert records[0]["finished_at"].endswith("+00:00")


def test_scheduler_returns_empty_history_when_file_missing(tmp_path):
    engine = make_engine(tmp_path)

    assert engine.get_history() == []


def test_scheduler_rejects_corrupted_history(tmp_path):
    history_path = tmp_path / "scheduler_history.jsonl"
    history_path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    engine = make_engine(tmp_path)
    engine.history_path = history_path

    with pytest.raises(SchedulerError):
        engine.get_history()


def test_scheduler_does_not_write_history_when_disabled(tmp_path):
    config = make_config()
    config["history"]["enabled"] = False

    history_path = tmp_path / "disabled_history.jsonl"

    engine = SchedulerEngine(
        config=config,
        tasks=make_tasks(),
        history_path=history_path,
        clock=lambda: NOW,
    )

    engine.run_cycle(now=NOW)

    assert not history_path.exists()
    assert engine.get_history() == []


def test_scheduler_sanitizes_non_json_callback_result(tmp_path):
    config = make_config()
    config["market_analysis"]["enabled"] = False
    config["forecast"]["enabled"] = False
    config["trader"]["enabled"] = False

    tasks = make_tasks()
    tasks["market_data"] = lambda: object()

    engine = make_engine(
        tmp_path,
        config=config,
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "SUCCESS"
    recorded_result = result["tasks"][0]["result"]
    assert recorded_result["recorded"] is False
    assert recorded_result["result_type"] == "object"


def test_scheduler_skips_overlapping_cycle(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocking_task():
        entered.set()
        release.wait(timeout=3)
        return {"completed": True}

    config = make_config()
    config["market_analysis"]["enabled"] = False
    config["forecast"]["enabled"] = False
    config["trader"]["enabled"] = False

    tasks = make_tasks()
    tasks["market_data"] = blocking_task

    engine = make_engine(
        tmp_path,
        config=config,
        tasks=tasks,
    )

    first_result = []

    worker = threading.Thread(
        target=lambda: first_result.append(
            engine.run_cycle(now=NOW)
        )
    )

    worker.start()

    try:
        assert entered.wait(timeout=2)

        second_result = engine.run_cycle(
            now=NOW + timedelta(seconds=1)
        )

        assert second_result["status"] == "SKIPPED"
        assert second_result["reason"] == "RUN_ALREADY_IN_PROGRESS"

    finally:
        release.set()
        worker.join(timeout=3)

    assert not worker.is_alive()
    assert first_result[0]["status"] == "SUCCESS"


def test_scheduler_history_is_valid_jsonl(tmp_path):
    history_path = tmp_path / "scheduler_history.jsonl"

    engine = SchedulerEngine(
        config=make_config(),
        tasks=make_tasks(),
        history_path=history_path,
        clock=lambda: NOW,
    )

    engine.run_cycle(now=NOW)

    lines = history_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines

    for line in lines:
        record = json.loads(line)
        assert isinstance(record, dict)
        assert record["status"] in {"SUCCESS", "FAILED"}
