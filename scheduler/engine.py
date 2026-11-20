
"""UTC-based scheduler and sequential pipeline orchestration."""

from __future__ import annotations

import json
import math
import threading

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scheduler_history.jsonl"
)

TaskCallable = Callable[[], Any]


class SchedulerError(RuntimeError):
    """Raised when scheduler configuration or execution is invalid."""


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _normalize_timestamp(value: datetime) -> datetime:
    """Validate and normalize a timezone-aware UTC timestamp."""

    if not isinstance(value, datetime):
        raise SchedulerError(
            "Scheduler timestamp must be a datetime."
        )

    if value.tzinfo is None:
        raise SchedulerError(
            "Scheduler timestamp must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchedulerError(
            "Invalid scheduler timestamp."
        ) from exc

    if offset != timedelta(0):
        raise SchedulerError(
            "Scheduler timestamp must use UTC."
        )

    return value.astimezone(timezone.utc)


def _require_mapping(
    value: Any,
    name: str,
) -> Mapping:
    """Require a mapping configuration."""

    if not isinstance(value, Mapping):
        raise SchedulerError(
            f"{name} must be a mapping."
        )

    return value


def _require_bool(
    value: Any,
    name: str,
) -> bool:
    """Require an actual boolean."""

    if not isinstance(value, bool):
        raise SchedulerError(
            f"{name} must be a boolean."
        )

    return value


def _require_positive_interval(
    value: Any,
    name: str,
) -> float:
    """Validate a positive finite scheduling interval."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SchedulerError(
            f"{name} must be a positive finite number."
        )

    return float(value)


class SchedulerEngine:
    """
    Sequential UTC scheduler.

    Tasks are injected by the application. This module does not
    create exchange connections or execute trading orders.

    Supported task names:
        market_data
        candles
        derivatives
        etf_flows
        onchain
        macro
        validation
        market_analysis
        forecast
        trader
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        tasks: Mapping[str, TaskCallable] | None = None,
        history_path: str | Path = DEFAULT_HISTORY_PATH,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.config = self._extract_schedule_config(config)

        self.tasks = self._validate_tasks(tasks or {})

        self.history_path = Path(history_path)
        self.clock = clock

        self._lock = threading.Lock()
        self._last_runs: dict[str, datetime] = {}

        self._validate_configuration()

    @staticmethod
    def _extract_schedule_config(
        config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Accept either the full project config or schedules section."""

        if config is None:
            from config.loader import load_config

            config = load_config()

        root = _require_mapping(
            config,
            "config",
        )

        if "schedules" in root:
            schedules_file = _require_mapping(
                root["schedules"],
                "config.schedules",
            )

            schedule_config = schedules_file.get(
                "schedules"
            )
        else:
            schedule_config = root

        return _require_mapping(
            schedule_config,
            "schedules",
        )

    @staticmethod
    def _validate_tasks(
        tasks: Mapping[str, TaskCallable],
    ) -> dict[str, TaskCallable]:
        """Validate injected task callbacks."""

        if not isinstance(tasks, Mapping):
            raise SchedulerError(
                "tasks must be a mapping."
            )

        validated = {}

        for name, callback in tasks.items():

            if not isinstance(name, str) or not name.strip():
                raise SchedulerError(
                    "Every task name must be a non-empty string."
                )

            if not callable(callback):
                raise SchedulerError(
                    f"Task '{name}' must be callable."
                )

            validated[name] = callback

        return validated

    def _validate_configuration(self) -> None:
        """Validate scheduler configuration and safety rules."""

        if self.config.get("timezone") != "UTC":
            raise SchedulerError(
                "Scheduler timezone must be UTC."
            )

        execution = _require_mapping(
            self.config.get("execution"),
            "schedules.execution",
        )

        history = _require_mapping(
            self.config.get("history"),
            "schedules.history",
        )

        for key in (
            "prevent_overlapping_runs",
            "sequential_dependencies",
            "stop_on_critical_error",
            "allow_retry_on_failure",
        ):
            _require_bool(
                execution.get(key),
                f"execution.{key}",
            )

        retries = execution.get("maximum_retries")

        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or retries < 0
        ):
            raise SchedulerError(
                "execution.maximum_retries must be a "
                "non-negative integer."
            )

        for key in (
            "enabled",
            "record_execution_status",
            "record_execution_timestamps",
            "prevent_duplicate_runs",
            "preserve_execution_history",
        ):
            _require_bool(
                history.get(key),
                f"history.{key}",
            )

        for section in (
            "market_data",
            "derivatives",
            "etf_flows",
            "onchain",
            "macro",
            "candles",
            "validation",
            "market_analysis",
            "forecast",
            "trader",
        ):
            if section in self.config:
                _require_mapping(
                    self.config[section],
                    f"schedules.{section}",
                )

        for section in (
            "market_data",
            "derivatives",
            "etf_flows",
            "onchain",
            "macro",
        ):
            settings = self.config.get(section)

            if settings is None:
                continue

            if "enabled" in settings:
                _require_bool(
                    settings["enabled"],
                    f"{section}.enabled",
                )

            if settings.get("enabled") is True:
                _require_positive_interval(
                    settings.get("interval_minutes"),
                    f"{section}.interval_minutes",
                )

        forecast = self.config.get("forecast", {})

        if forecast.get("enabled") is True:
            _require_positive_interval(
                forecast.get("interval_hours"),
                "forecast.interval_hours",
            )

        candles = self.config.get("candles", {})

        if candles.get("enabled") is True:
            timeframes = candles.get("timeframes")

            if (
                not isinstance(timeframes, list)
                or not timeframes
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in timeframes
                )
            ):
                raise SchedulerError(
                    "candles.timeframes must be a non-empty list."
                )

            if len(timeframes) != len(set(timeframes)):
                raise SchedulerError(
                    "Duplicate candle timeframes are not allowed."
                )

            _require_bool(
                candles.get("collect_closed_candles_only"),
                "candles.collect_closed_candles_only",
            )

    def _interval_for(self, task_name: str) -> timedelta | None:
        """Return the configured interval for a periodic task."""

        if task_name == "candles":
            section = self.config.get("market_data", {})
            value = section.get("interval_minutes")
            unit = "minutes"

        elif task_name == "forecast":
            section = self.config.get("forecast", {})
            value = section.get("interval_hours")
            unit = "hours"

        else:
            section = self.config.get(task_name, {})
            value = section.get("interval_minutes")
            unit = "minutes"

        if value is None:
            return None

        amount = _require_positive_interval(
            value,
            f"{task_name} interval",
        )

        return timedelta(
            **{unit: amount}
        )

    def _is_enabled(self, task_name: str) -> bool:
        """Check whether a task is enabled."""

        section = self.config.get(task_name)

        if not isinstance(section, Mapping):
            return False

        return section.get("enabled") is True

    def _is_due(
        self,
        task_name: str,
        now: datetime,
    ) -> bool:
        """Determine whether a periodic task is due."""

        if not self._is_enabled(task_name):
            return False

        interval = self._interval_for(task_name)

        if interval is None:
            return False

        previous = self._last_runs.get(task_name)

        if previous is None:
            return True

        return now - previous >= interval

    def _record_history(
        self,
        record: dict[str, Any],
    ) -> None:
        """Append one execution record to JSONL history."""

        history_config = self.config.get("history", {})

        if history_config.get("enabled") is not True:
            return

        if history_config.get(
            "preserve_execution_history"
        ) is not True:
            return

        try:
            self.history_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.history_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                )
                file.write("\n")

        except (OSError, TypeError, ValueError) as exc:
            raise SchedulerError(
                f"Unable to write scheduler history: {exc}"
            ) from exc

    def _execute_task(
        self,
        task_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Execute one registered task with configured retries."""

        callback = self.tasks.get(task_name)

        if callback is None:
            raise SchedulerError(
                f"No callback registered for enabled task '{task_name}'."
            )

        execution = self.config["execution"]

        retries_allowed = (
            execution.get("allow_retry_on_failure") is True
        )

        maximum_retries = (
            execution.get("maximum_retries", 0)
            if retries_allowed
            else 0
        )

        started_at = _normalize_timestamp(
            self.clock()
        )

        last_error = None

        for attempt in range(maximum_retries + 1):
            try:
                result = callback()

                finished_at = _normalize_timestamp(
                    self.clock()
                )

                record = {
                    "task": task_name,
                    "status": "SUCCESS",
                    "attempt": attempt + 1,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "result": self._safe_result(result),
                }

                self._record_history(record)

                self._last_runs[task_name] = timestamp

                return record

            except Exception as exc:
                last_error = exc

        finished_at = _normalize_timestamp(
            self.clock()
        )

        record = {
            "task": task_name,
            "status": "FAILED",
            "attempt": maximum_retries + 1,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "error": (
                f"{type(last_error).__name__}: {last_error}"
            ),
        }

        self._record_history(record)

        return record

    @staticmethod
    def _safe_result(value: Any) -> Any:
        """Convert callback output to JSON-safe history data."""

        try:
            json.dumps(
                value,
                allow_nan=False,
            )
            return value

        except (TypeError, ValueError, OverflowError):
            return {
                "result_type": type(value).__name__,
                "recorded": False,
            }

    def run_cycle(
        self,
        now: datetime | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Execute all currently due tasks in dependency order.

        Collection -> validation -> market analysis ->
        forecast and trader.

        Missing callbacks are reported as failures when their
        corresponding enabled task is due.
        """

        timestamp = _normalize_timestamp(
            now if now is not None else self.clock()
        )

        if not isinstance(force, bool):
            raise SchedulerError(
                "force must be a boolean."
            )

        execution = self.config["execution"]

        prevent_overlap = (
            execution.get("prevent_overlapping_runs") is True
        )

        acquired = (
            self._lock.acquire(blocking=False)
            if prevent_overlap
            else self._lock.acquire()
        )

        if not acquired:
            return {
                "status": "SKIPPED",
                "reason": "RUN_ALREADY_IN_PROGRESS",
                "timestamp": timestamp.isoformat(),
                "tasks": [],
            }

        try:
            return self._run_cycle_locked(
                timestamp,
                force,
            )

        finally:
            self._lock.release()

    def _run_cycle_locked(
        self,
        timestamp: datetime,
        force: bool,
    ) -> dict[str, Any]:
        """Run one sequential scheduling cycle."""

        records = []
        collection_names = (
            "market_data",
            "candles",
            "derivatives",
            "etf_flows",
            "onchain",
            "macro",
        )

        due_collections = [
            name
            for name in collection_names
            if self._is_enabled(name)
            and (
                force
                or self._is_due(name, timestamp)
            )
        ]

        for name in due_collections:
            record = self._execute_task(
                name,
                timestamp,
            )
            records.append(record)

            if record["status"] == "FAILED":
                if self.config["execution"].get(
                    "stop_on_critical_error"
                ) is True:
                    return self._cycle_result(
                        timestamp,
                        records,
                        "FAILED",
                    )

        collection_succeeded = bool(due_collections) and all(
            record["status"] == "SUCCESS"
            for record in records
        )

        validation_config = self.config.get(
            "validation",
            {},
        )

        should_validate = (
            validation_config.get("enabled") is True
            and (
                force
                or (
                    collection_succeeded
                    and validation_config.get(
                        "run_after_collection"
                    ) is True
                )
            )
        )

        if should_validate:
            record = self._execute_task(
                "validation",
                timestamp,
            )
            records.append(record)

            if record["status"] == "FAILED":
                return self._cycle_result(
                    timestamp,
                    records,
                    "FAILED",
                )

        validation_succeeded = (
            any(
                item["task"] == "validation"
                and item["status"] == "SUCCESS"
                for item in records
            )
        )

        if validation_succeeded:
            market_config = self.config.get(
                "market_analysis",
                {},
            )

            if market_config.get("enabled") is True:
                record = self._execute_task(
                    "market_analysis",
                    timestamp,
                )
                records.append(record)

                if record["status"] == "FAILED":
                    return self._cycle_result(
                        timestamp,
                        records,
                        "FAILED",
                    )

        market_succeeded = (
            any(
                item["task"] == "market_analysis"
                and item["status"] == "SUCCESS"
                for item in records
            )
        )

        if validation_succeeded:
            forecast_config = self.config.get(
                "forecast",
                {},
            )

            if (
                forecast_config.get("enabled") is True
                and (
                    force
                    or self._is_due("forecast", timestamp)
                )
            ):
                record = self._execute_task(
                    "forecast",
                    timestamp,
                )
                records.append(record)

        trader_config = self.config.get(
            "trader",
            {},
        )

        trader_ready = (
            validation_succeeded
            and (
                not trader_config.get(
                    "run_after_market_analysis",
                    True,
                )
                or market_succeeded
            )
        )

        if (
            trader_config.get("enabled") is True
            and trader_ready
        ):
            record = self._execute_task(
                "trader",
                timestamp,
            )
            records.append(record)

        failed = any(
            item["status"] == "FAILED"
            for item in records
        )

        return self._cycle_result(
            timestamp,
            records,
            "FAILED" if failed else "SUCCESS",
        )

    @staticmethod
    def _cycle_result(
        timestamp: datetime,
        records: list[dict[str, Any]],
        status: str,
    ) -> dict[str, Any]:
        """Build a standardized cycle result."""

        return {
            "status": status,
            "timestamp": timestamp.isoformat(),
            "task_count": len(records),
            "successful_tasks": sum(
                item["status"] == "SUCCESS"
                for item in records
            ),
            "failed_tasks": sum(
                item["status"] == "FAILED"
                for item in records
            ),
            "tasks": records,
            "execution": {
                "timezone": "UTC",
                "sequential": True,
                "automatic_trading": False,
                "order_execution": False,
            },
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Read scheduler execution history."""

        if not self.history_path.exists():
            return []

        records = []

        try:
            with self.history_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SchedulerError(
                            f"Invalid history JSON at line {line_number}."
                        ) from exc

                    if not isinstance(record, dict):
                        raise SchedulerError(
                            f"Invalid history record at line {line_number}."
                        )

                    records.append(record)

        except OSError as exc:
            raise SchedulerError(
                f"Unable to read scheduler history: {exc}"
            ) from exc

        return records


if __name__ == "__main__":
    print(
        "SchedulerEngine module loaded. "
        "Register application task callbacks before running."
    )
