
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

SUCCESS_STATUSES = {"SUCCESS", "PASS"}
SKIPPED_STATUSES = {"SKIPPED", "NO_NEW_DATA"}
FAILED_STATUSES = {"FAILED", "ERROR"}


# Supplementary data sources. The candle collection that every signal
# depends on is never optional.
OPTIONAL_COLLECTION_CHOICES = (
    "derivatives",
    "etf_flows",
    "onchain",
    "macro",
)


class SchedulerError(RuntimeError):
    """Raised when scheduler configuration or execution is invalid."""


class _UnsetTimestamp:
    """Internal sentinel for an omitted scheduler timestamp."""


_UNSET_TIMESTAMP = _UnsetTimestamp()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_timestamp(value: datetime) -> datetime:
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


def _parse_history_timestamp(value: Any) -> datetime | None:
    """Parse a persisted UTC timestamp without accepting naive values."""

    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(value)
        return _normalize_timestamp(parsed)
    except (ValueError, TypeError, SchedulerError):
        return None


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise SchedulerError(
            f"{name} must be a mapping."
        )
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchedulerError(
            f"{name} must be a boolean."
        )
    return value


def _require_positive_interval(
    value: Any,
    name: str,
) -> float:
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

    Collection -> validation -> market analysis ->
    forecast and manual-review-only trading.

    This engine never connects to an exchange or executes orders.
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

        if not callable(clock):
            raise SchedulerError(
                "clock must be callable."
            )

        self.clock = clock

        self._lock = threading.Lock()
        self._last_runs: dict[str, datetime] = {}

        # A corrupted or unreadable history file must not prevent
        # scheduler initialization. Keep the original error so
        # future history writes can be blocked safely.
        self._history_restore_error: SchedulerError | None = None

        self._validate_configuration()

        self._restore_last_runs()

    @staticmethod
    def _extract_schedule_config(
        config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if config is None:
            from config.loader import load_config
            config = load_config()

        root = _require_mapping(config, "config")

        if "schedules" in root:
            schedules_file = _require_mapping(
                root["schedules"],
                "config.schedules",
            )
            schedule_config = schedules_file.get("schedules")
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

        optional = execution.get("optional_collections", [])

        if (
            not isinstance(optional, list)
            or any(
                item not in OPTIONAL_COLLECTION_CHOICES
                for item in optional
            )
        ):
            raise SchedulerError(
                "execution.optional_collections must be a list "
                "containing only: "
                + ", ".join(OPTIONAL_COLLECTION_CHOICES)
                + "."
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

        sections = (
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
        )

        for section in sections:
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

        validation = self.config.get("validation", {})

        for key in (
            "enabled",
            "run_after_collection",
            "run_before_every_engine",
        ):
            if key in validation:
                _require_bool(
                    validation[key],
                    f"validation.{key}",
                )

        market_analysis = self.config.get(
            "market_analysis",
            {},
        )

        for key in ("enabled", "run_after_validation"):
            if key in market_analysis:
                _require_bool(
                    market_analysis[key],
                    f"market_analysis.{key}",
                )

        if "enabled" in forecast:
            _require_bool(
                forecast["enabled"],
                "forecast.enabled",
            )

        if "run_after_validation" in forecast:
            _require_bool(
                forecast["run_after_validation"],
                "forecast.run_after_validation",
            )

        trader = self.config.get("trader", {})

        for key in (
            "enabled",
            "run_after_data_validation",
            "run_after_market_analysis",
            "require_closed_candles",
        ):
            if key in trader:
                _require_bool(
                    trader[key],
                    f"trader.{key}",
                )

    def _restore_last_runs(self) -> None:
        """
        Restore successful task timestamps from persistent history.

        Corrupted or unreadable history must not prevent scheduler
        initialization. The file is never modified or deleted here.

        New records use scheduled_at. Older records without that
        field use finished_at for backward compatibility.

        Failed executions are deliberately not restored.
        """

        history_config = self.config.get("history", {})

        if history_config.get("enabled") is not True:
            return

        if history_config.get(
            "preserve_execution_history"
        ) is not True:
            return

        if not self.history_path.exists():
            return

        try:
            records = self.get_history()
        except SchedulerError as exc:
            self._history_restore_error = exc
            return

        for record in records:
            task_name = record.get("task")
            status = record.get("status")

            if (
                not isinstance(task_name, str)
                or not task_name.strip()
            ):
                continue

            if status not in SUCCESS_STATUSES | SKIPPED_STATUSES:
                continue

            timestamp = _parse_history_timestamp(
                record.get("scheduled_at")
            )

            if timestamp is None:
                timestamp = _parse_history_timestamp(
                    record.get("finished_at")
                )

            if timestamp is None:
                continue

            previous = self._last_runs.get(task_name)

            if previous is None or timestamp > previous:
                self._last_runs[task_name] = timestamp

    def _interval_for(
        self,
        task_name: str,
    ) -> timedelta | None:
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

        return timedelta(**{unit: amount})

    def _is_enabled(self, task_name: str) -> bool:
        section = self.config.get(task_name)

        return (
            isinstance(section, Mapping)
            and section.get("enabled") is True
        )

    def _duplicate_prevention_enabled(self) -> bool:
        history_config = self.config.get("history", {})

        return (
            history_config.get("enabled") is True
            and history_config.get(
                "prevent_duplicate_runs"
            ) is True
        )

    def _has_already_run(
        self,
        task_name: str,
        timestamp: datetime,
    ) -> bool:
        """Return True when the task already completed at this timestamp."""

        if not self._duplicate_prevention_enabled():
            return False

        previous = self._last_runs.get(task_name)

        return previous is not None and previous == timestamp

    def _is_due(
        self,
        task_name: str,
        now: datetime,
    ) -> bool:
        if not self._is_enabled(task_name):
            return False

        if self._has_already_run(task_name, now):
            return False

        interval = self._interval_for(task_name)

        if interval is None:
            return False

        previous = self._last_runs.get(task_name)

        if previous is None:
            return True

        elapsed = now - previous

        return elapsed >= interval

    def _record_history(
        self,
        record: dict[str, Any],
    ) -> None:
        history_config = self.config.get("history", {})

        if history_config.get("enabled") is not True:
            return

        if history_config.get(
            "preserve_execution_history"
        ) is not True:
            return

        # Never append to a history file that was found corrupted
        # or unreadable during scheduler initialization.
        if self._history_restore_error is not None:
            raise SchedulerError(
                "Cannot append to corrupted or unreadable scheduler "
                "history. Repair the history file before writing "
                "new records."
            ) from self._history_restore_error

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

    def _persist_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Persist a task record without repeating its callback.

        A history failure is reported in the returned record.
        It never causes the original task callback to run again.
        """

        try:
            self._record_history(record)

        except SchedulerError as exc:
            record["history_error"] = str(exc)

            original_status = record.get("status", "FAILED")
            record["callback_status"] = original_status
            record["status"] = "FAILED"

            if "error" not in record:
                record["error"] = (
                    "Task callback completed, but execution "
                    "history could not be saved."
                )

        return record

    @staticmethod
    def _safe_result(value: Any) -> Any:
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

    @staticmethod
    def _result_status(result: Any) -> str:
        """
        Classify callback outcomes.

        Generic callback results without a status field remain
        successful for backward compatibility.
        """

        if not isinstance(result, Mapping):
            return "SUCCESS"

        status = result.get("status")

        if not isinstance(status, str):
            return "SUCCESS"

        normalized = status.strip().upper()

        if normalized in FAILED_STATUSES:
            return "FAILED"

        if normalized in SKIPPED_STATUSES:
            return "SKIPPED"

        if normalized in SUCCESS_STATUSES:
            return "SUCCESS"

        raise SchedulerError(
            f"Unsupported callback status: {status!r}."
        )

    def _execute_task(
        self,
        task_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        if self._has_already_run(task_name, timestamp):
            return {
                "task": task_name,
                "status": "SKIPPED",
                "attempt": 0,
                "scheduled_at": timestamp.isoformat(),
                "reason": "DUPLICATE_RUN",
            }

        callback = self.tasks.get(task_name)

        if callback is None:
            record = {
                "task": task_name,
                "status": "FAILED",
                "attempt": 0,
                "scheduled_at": timestamp.isoformat(),
                "error": (
                    f"No callback registered for enabled task "
                    f"'{task_name}'."
                ),
            }

            return self._persist_record(record)

        execution = self.config["execution"]

        retries_allowed = (
            execution.get("allow_retry_on_failure") is True
        )

        maximum_retries = (
            execution.get("maximum_retries", 0)
            if retries_allowed
            else 0
        )

        started_at = _normalize_timestamp(self.clock())
        last_error = None

        for attempt in range(maximum_retries + 1):
            try:
                result = callback()
                task_status = self._result_status(result)

                if task_status == "FAILED":
                    raise SchedulerError(
                        f"Task '{task_name}' returned FAILED."
                    )

            except Exception as exc:
                last_error = exc
                continue

            finished_at = _normalize_timestamp(self.clock())

            record = {
                "task": task_name,
                "status": task_status,
                "attempt": attempt + 1,
                "scheduled_at": timestamp.isoformat(),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "result": self._safe_result(result),
            }

            self._last_runs[task_name] = timestamp

            return self._persist_record(record)

        finished_at = _normalize_timestamp(self.clock())

        record = {
            "task": task_name,
            "status": "FAILED",
            "attempt": maximum_retries + 1,
            "scheduled_at": timestamp.isoformat(),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "error": (
                f"{type(last_error).__name__}: {last_error}"
            ),
        }

        return self._persist_record(record)

    def run_cycle(
        self,
        now: datetime | None | _UnsetTimestamp = _UNSET_TIMESTAMP,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Run one scheduler cycle.

        If now is omitted, the injected clock supplies the timestamp.
        If now is explicitly None or is not a valid UTC datetime,
        SchedulerError is raised.
        """

        if isinstance(now, _UnsetTimestamp):
            timestamp = _normalize_timestamp(self.clock())
        else:
            timestamp = _normalize_timestamp(now)

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
        records: list[dict[str, Any]] = []

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
            and not self._has_already_run(name, timestamp)
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

            if (
                record["status"] == "FAILED"
                and name not in self._optional_collections()
                and self.config["execution"].get(
                    "stop_on_critical_error"
                ) is True
            ):
                return self._cycle_result(
                    timestamp,
                    records,
                    "FAILED",
                )

        collection_records = [
            item
            for item in records
            if item["task"] in ("market_data", "candles")
        ]

        candle_collection_succeeded = any(
            item["status"] == "SUCCESS"
            for item in collection_records
        )

        collection_failures = any(
            item["status"] == "FAILED"
            for item in records
            if item["task"] in collection_names
            and item["task"] not in self._optional_collections()
        )

        collection_succeeded = (
            bool(due_collections)
            and candle_collection_succeeded
            and not collection_failures
        )

        validation_config = self.config.get(
            "validation",
            {},
        )

        validation_enabled = (
            validation_config.get("enabled") is True
        )

        run_after_collection = (
            validation_config.get(
                "run_after_collection"
            ) is True
        )

        run_before_every_engine = (
            validation_config.get(
                "run_before_every_engine"
            ) is True
        )

        engine_requested = (
            self._is_enabled("market_analysis")
            or (
                self._is_enabled("forecast")
                and (
                    force
                    or self._is_due("forecast", timestamp)
                )
            )
            or self._is_enabled("trader")
        )

        should_validate = (
            validation_enabled
            and not self._has_already_run(
                "validation",
                timestamp,
            )
            and (
                force
                or (
                    collection_succeeded
                    and (
                        run_after_collection
                        or (
                            run_before_every_engine
                            and engine_requested
                        )
                    )
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
            self._has_already_run(
                "validation",
                timestamp,
            )
            or any(
                item["task"] == "validation"
                and item["status"] == "SUCCESS"
                for item in records
            )
        )

        if not validation_succeeded:
            return self._cycle_result(
                timestamp,
                records,
                "FAILED"
                if self._has_critical_failure(records)
                else "SUCCESS",
            )

        market_config = self.config.get(
            "market_analysis",
            {},
        )

        market_analysis_succeeded = False

        if (
            market_config.get("enabled") is True
            and market_config.get(
                "run_after_validation",
                True,
            ) is True
        ):
            if self._has_already_run(
                "market_analysis",
                timestamp,
            ):
                market_analysis_succeeded = True
            else:
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

                market_analysis_succeeded = (
                    record["status"] == "SUCCESS"
                )

        forecast_config = self.config.get(
            "forecast",
            {},
        )

        forecast_ready = (
            forecast_config.get("enabled") is True
            and not self._has_already_run(
                "forecast",
                timestamp,
            )
            and (
                force
                or self._is_due("forecast", timestamp)
            )
            and forecast_config.get(
                "run_after_validation",
                True,
            ) is True
        )

        if forecast_ready:
            record = self._execute_task(
                "forecast",
                timestamp,
            )
            records.append(record)

            if record["status"] == "FAILED":
                return self._cycle_result(
                    timestamp,
                    records,
                    "FAILED",
                )

        trader_config = self.config.get(
            "trader",
            {},
        )

        trader_requires_validation = (
            trader_config.get(
                "run_after_data_validation",
                True,
            ) is True
        )

        trader_requires_analysis = (
            trader_config.get(
                "run_after_market_analysis",
                True,
            ) is True
        )

        trader_ready = (
            trader_config.get("enabled") is True
            and trader_requires_validation
            and (
                not trader_requires_analysis
                or market_analysis_succeeded
            )
            and not self._has_already_run(
                "trader",
                timestamp,
            )
        )

        if trader_ready:
            record = self._execute_task(
                "trader",
                timestamp,
            )
            records.append(record)

        failed = self._has_critical_failure(records)

        return self._cycle_result(
            timestamp,
            records,
            "FAILED" if failed else "SUCCESS",
        )

    def _optional_collections(self) -> frozenset[str]:
        """Names of supplementary sources that may fail safely."""

        return frozenset(
            self.config["execution"].get(
                "optional_collections",
                [],
            )
        )

    def _has_critical_failure(
        self,
        records: list[dict[str, Any]],
    ) -> bool:
        optional = self._optional_collections()

        return any(
            item["status"] == "FAILED"
            and item["task"] not in optional
            for item in records
        )

    def _cycle_result(
        self,
        timestamp: datetime,
        records: list[dict[str, Any]],
        status: str,
    ) -> dict[str, Any]:
        optional = self._optional_collections()

        return {
            "status": status,
            "degraded_sources": [
                item["task"]
                for item in records
                if item["status"] == "FAILED"
                and item["task"] in optional
            ],
            "timestamp": timestamp.isoformat(),
            "task_count": len(records),
            "successful_tasks": sum(
                item["status"] == "SUCCESS"
                for item in records
            ),
            "skipped_tasks": sum(
                item["status"] == "SKIPPED"
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
        """
        Read scheduler history.

        Corrupted JSONL records are reported to the caller.
        This method never silently skips malformed records.
        """

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
                            f"Invalid history JSON at line "
                            f"{line_number}."
                        ) from exc

                    if not isinstance(record, dict):
                        raise SchedulerError(
                            f"Invalid history record at line "
                            f"{line_number}."
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
