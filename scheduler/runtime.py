
"""Application runtime for the Bitcoin Intelligence System scheduler.

Connects project collectors, candle validation, market analysis,
forecasting and manual-review-only trading to SchedulerEngine.

Safety and data policies:
- All runtime timestamps use UTC.
- Raw collector data remains append-only.
- Only validated closed candles reach analysis engines.
- NO_NEW_DATA is a normal collection outcome.
- Analysis requires fresh validated 1h candles.
- Forecast reuses the market analysis from the same cycle.
- Trading remains manual-review-only.
- No exchange connection or order execution is permitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.loader import ConfigurationError, load_config
from scheduler.engine import SchedulerEngine, SchedulerError


DEFAULT_PAIR = "XBTUSD"
DEFAULT_POLL_INTERVAL_SECONDS = 30.0

SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h", "4h")

LOGGER = logging.getLogger("bitcoin_intelligence.scheduler")


class SchedulerRuntimeError(RuntimeError):
    """Raised when scheduler runtime initialization or execution fails."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(value: datetime | None) -> datetime:
    timestamp = _utc_now() if value is None else value

    if not isinstance(timestamp, datetime):
        raise SchedulerRuntimeError(
            "Runtime timestamp must be a datetime."
        )

    if timestamp.tzinfo is None:
        raise SchedulerRuntimeError(
            "Runtime timestamp must be timezone-aware."
        )

    try:
        offset = timestamp.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchedulerRuntimeError(
            "Runtime timestamp is invalid."
        ) from exc

    if offset is None or offset.total_seconds() != 0:
        raise SchedulerRuntimeError(
            "Runtime timestamp must use UTC."
        )

    return timestamp.astimezone(timezone.utc)


def _validate_pair(value: str) -> str:
    if not isinstance(value, str):
        raise SchedulerRuntimeError(
            "Trading pair must be a string."
        )

    normalized = value.strip().upper()

    if (
        not normalized
        or not normalized.isalnum()
        or len(normalized) > 20
    ):
        raise SchedulerRuntimeError(
            "Invalid trading pair."
        )

    return normalized


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SchedulerRuntimeError(
            f"{name} must be a positive finite number."
        )

    return float(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchedulerRuntimeError(
            f"{name} must be a mapping."
        )

    return value


def _success_result(
    result: Any,
    name: str,
    accepted_statuses: tuple[str, ...] = ("SUCCESS",),
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise SchedulerRuntimeError(
            f"{name} returned an invalid result."
        )

    if result.get("status") not in accepted_statuses:
        raise SchedulerRuntimeError(
            f"{name} failed with status {result.get('status')!r}."
        )

    return result


class SchedulerRuntime:
    """Connect application workflows to SchedulerEngine."""

    def __init__(
        self,
        *,
        pair: str = DEFAULT_PAIR,
        leverage: float = 1.0,
        account_equity: float | None = None,
        config: Mapping[str, Any] | None = None,
        history_path: str | Path | None = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.pair = _validate_pair(pair)

        self.leverage = _positive_number(
            leverage,
            "leverage",
        )

        if self.leverage > 2:
            raise SchedulerRuntimeError(
                "Leverage cannot exceed 2."
            )

        self.account_equity = (
            None
            if account_equity is None
            else _positive_number(
                account_equity,
                "account_equity",
            )
        )

        self.poll_interval_seconds = _positive_number(
            poll_interval_seconds,
            "poll_interval_seconds",
        )

        self.config = self._load_config(config)
        self.schedule_config = self._get_schedule_config()
        self.source_config = self._get_source_config()

        self._validate_safety()
        self._validate_configuration()

        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cycle_state: dict[str, Any] = {}

        engine_kwargs: dict[str, Any] = {
            "config": self.config,
            "tasks": self._build_callbacks(),
        }

        if history_path is not None:
            engine_kwargs["history_path"] = Path(history_path)

        try:
            self.engine = SchedulerEngine(**engine_kwargs)
        except SchedulerError as exc:
            raise SchedulerRuntimeError(
                f"Unable to initialize scheduler engine: {exc}"
            ) from exc

    @staticmethod
    def _load_config(
        config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if config is not None:
            return _mapping(
                config,
                "project configuration",
            )

        try:
            loaded = load_config()
        except ConfigurationError as exc:
            raise SchedulerRuntimeError(
                f"Unable to load configuration: {exc}"
            ) from exc

        return _mapping(
            loaded,
            "project configuration",
        )

    def _get_schedule_config(self) -> Mapping[str, Any]:
        schedules_file = _mapping(
            self.config.get("schedules"),
            "config.schedules",
        )

        return _mapping(
            schedules_file.get("schedules"),
            "config.schedules.schedules",
        )

    def _get_source_config(self) -> Mapping[str, Any]:
        source_file = _mapping(
            self.config.get("data_sources"),
            "config.data_sources",
        )

        return _mapping(
            source_file.get("sources"),
            "config.data_sources.sources",
        )

    def _validate_safety(self) -> None:
        # Actual config/trader.yaml structure:
        # trader:
        #   execution: ...
        #   risk: ...

        trader_file = _mapping(
            self.config.get("trader"),
            "config.trader",
        )

        trader = _mapping(
            trader_file.get("trader"),
            "config.trader.trader",
        )

        execution = _mapping(
            trader.get("execution"),
            "config.trader.trader.execution",
        )

        required_safety = {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "exchange_api_keys_required": False,
        }

        for key, expected in required_safety.items():
            if execution.get(key) is not expected:
                raise SchedulerRuntimeError(
                    f"Unsafe trading configuration: {key}."
                )

        risk = _mapping(
            trader.get("risk"),
            "config.trader.trader.risk",
        )

        maximum_leverage = risk.get("maximum_leverage")

        configured_maximum = _positive_number(
            maximum_leverage,
            "maximum_leverage",
        )

        if configured_maximum > 2:
            configured_maximum = 2.0

        if self.leverage > configured_maximum:
            raise SchedulerRuntimeError(
                "Requested leverage exceeds the configured maximum."
            )

    def _validate_configuration(self) -> None:
        if self.schedule_config.get("timezone") != "UTC":
            raise SchedulerRuntimeError(
                "Scheduler timezone must be UTC."
            )

        candles = _mapping(
            self.schedule_config.get("candles"),
            "schedules.candles",
        )

        timeframes = candles.get("timeframes")

        if (
            not isinstance(timeframes, list)
            or not timeframes
            or any(
                not isinstance(item, str)
                or item not in SUPPORTED_TIMEFRAMES
                for item in timeframes
            )
        ):
            raise SchedulerRuntimeError(
                "Configured candle timeframes are invalid."
            )

        if len(timeframes) != len(set(timeframes)):
            raise SchedulerRuntimeError(
                "Duplicate candle timeframes are not allowed."
            )

        if candles.get("collect_closed_candles_only") is not True:
            raise SchedulerRuntimeError(
                "Only closed candles may be collected."
            )

    def _source_enabled(self, name: str) -> bool:
        source = self.source_config.get(name)

        return (
            isinstance(source, Mapping)
            and source.get("enabled") is True
        )

    def _provider_name(self, name: str) -> str | None:
        source = self.source_config.get(name)

        if not isinstance(source, Mapping):
            return None

        provider = source.get("primary")

        if not isinstance(provider, str) or not provider.strip():
            return None

        return provider.strip()

    def _provider_enabled(self, name: str) -> bool:
        providers = self.source_config.get("providers")

        if not isinstance(providers, Mapping):
            return False

        provider = providers.get(name)

        return (
            isinstance(provider, Mapping)
            and provider.get("enabled") is True
        )

    def _reset_cycle(self, timestamp: datetime) -> None:
        with self._state_lock:
            self._cycle_state = {
                "timestamp": timestamp,
                "candle_results": {},
                "accepted_candles": {},
                "collection_failures": [],
                "validation_passed": False,
                "fresh_timeframes": [],
                "market_analysis": None,
                "market_analysis_passed": False,
                "forecast": None,
            }

    def _record_candle_result(
        self,
        timeframe: str,
        result: Mapping[str, Any],
    ) -> None:
        accepted = result.get("accepted")
        rejected = result.get("rejected")

        accepted_count = (
            len(accepted)
            if accepted is not None
            else 0
        )

        rejected_count = (
            len(rejected)
            if rejected is not None
            else 0
        )

        with self._state_lock:
            self._cycle_state["candle_results"][timeframe] = {
                "status": result.get("status"),
                "accepted_records": accepted_count,
                "rejected_records": rejected_count,
            }

            if result.get("status") not in (
                "PASS",
                "NO_NEW_DATA",
            ):
                self._cycle_state["collection_failures"].append(
                    timeframe
                )

    def _collect_one_timeframe(
        self,
        timeframe: str,
    ) -> dict[str, Any]:
        from collectors.pipeline import run_kraken_pipeline

        timestamp = self._cycle_state["timestamp"]

        try:
            result = run_kraken_pipeline(
                pair=self.pair,
                timeframe=timeframe,
                collection_time=timestamp,
            )
        except Exception as exc:
            with self._state_lock:
                self._cycle_state["collection_failures"].append(
                    timeframe
                )

            raise SchedulerRuntimeError(
                f"Kraken collection failed for {timeframe}: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise SchedulerRuntimeError(
                f"Invalid Kraken pipeline result for {timeframe}."
            )

        status = result.get("status")

        if status == "NO_NEW_DATA":
            self._record_candle_result(
                timeframe,
                result,
            )

            with self._state_lock:
                self._cycle_state["accepted_candles"][timeframe] = None

            return {
                "status": "NO_NEW_DATA",
                "timeframe": timeframe,
                "collected_records": 0,
                "accepted_records": 0,
                "rejected_records": 0,
            }

        if status != "PASS":
            self._record_candle_result(
                timeframe,
                result,
            )

            raise SchedulerRuntimeError(
                f"Kraken pipeline failed for {timeframe}: {status!r}."
            )

        accepted = result.get("accepted")

        if accepted is None or not hasattr(accepted, "empty"):
            self._record_candle_result(
                timeframe,
                result,
            )

            raise SchedulerRuntimeError(
                f"Invalid accepted candle data for {timeframe}."
            )

        if accepted.empty:
            self._record_candle_result(
                timeframe,
                result,
            )

            raise SchedulerRuntimeError(
                f"Pipeline reported PASS but returned no accepted "
                f"candles for {timeframe}."
            )

        accepted_copy = accepted.copy(deep=True)

        self._record_candle_result(
            timeframe,
            result,
        )

        with self._state_lock:
            self._cycle_state["accepted_candles"][timeframe] = (
                accepted_copy
            )

        return {
            "status": "SUCCESS",
            "timeframe": timeframe,
            "collected_records": result.get(
                "collected_records",
                len(accepted_copy),
            ),
            "accepted_records": len(accepted_copy),
            "rejected_records": len(
                result.get("rejected", [])
            ),
        }

    def _task_market_data(self) -> dict[str, Any]:
        if not self._source_enabled("market"):
            return {
                "status": "SKIPPED",
                "reason": "MARKET_SOURCE_DISABLED",
            }

        provider = self._provider_name("market")

        if provider != "kraken":
            raise SchedulerRuntimeError(
                "Unsupported primary market provider."
            )

        if not self._provider_enabled("kraken"):
            raise SchedulerRuntimeError(
                "Kraken provider is disabled."
            )

        result = self._collect_one_timeframe("5m")

        return {
            **result,
            "provider": provider,
            "pair": self.pair,
        }

    def _task_candles(self) -> dict[str, Any]:
        if not self._source_enabled("market"):
            return {
                "status": "SKIPPED",
                "reason": "MARKET_SOURCE_DISABLED",
            }

        provider = self._provider_name("market")

        if provider != "kraken":
            raise SchedulerRuntimeError(
                "Unsupported candle provider."
            )

        if not self._provider_enabled("kraken"):
            raise SchedulerRuntimeError(
                "Kraken provider is disabled."
            )

        candles = _mapping(
            self.schedule_config.get("candles"),
            "schedules.candles",
        )

        results = []

        for timeframe in candles["timeframes"]:
            # The 5m timeframe is collected by market_data.
            if timeframe == "5m":
                continue

            results.append(
                self._collect_one_timeframe(timeframe)
            )

        return {
            "status": "SUCCESS",
            "provider": provider,
            "pair": self.pair,
            "timeframes": results,
        }

    def _task_derivatives(self) -> dict[str, Any]:
        if not self._source_enabled("derivatives"):
            return {
                "status": "SKIPPED",
                "reason": "DERIVATIVES_SOURCE_DISABLED",
            }

        provider = self._provider_name("derivatives")

        if provider != "kraken_futures":
            raise SchedulerRuntimeError(
                "Unsupported derivatives provider."
            )

        if not self._provider_enabled("kraken_futures"):
            raise SchedulerRuntimeError(
                "Kraken Futures provider is disabled."
            )

        from collectors.kraken_futures import (
            collect_kraken_futures,
        )

        records = collect_kraken_futures(
            collection_time=self._cycle_state["timestamp"],
        )

        if not isinstance(records, list):
            raise SchedulerRuntimeError(
                "Invalid Kraken Futures collector result."
            )

        return {
            "status": "SUCCESS",
            "provider": provider,
            "record_count": len(records),
        }

    def _task_etf_flows(self) -> dict[str, Any]:
        if not self._source_enabled("etf"):
            return {
                "status": "SKIPPED",
                "reason": "ETF_SOURCE_DISABLED",
            }

        provider = self._provider_name("etf")

        if provider is None:
            return {
                "status": "SKIPPED",
                "reason": "ETF_PROVIDER_NOT_CONFIGURED",
            }

        if provider != "farside":
            raise SchedulerRuntimeError(
                f"Unsupported ETF provider: {provider}."
            )

        if not os.environ.get("BITCOIN_ETF_SOURCE_URL"):
            return {
                "status": "SKIPPED",
                "reason": "ETF_SOURCE_URL_NOT_CONFIGURED",
            }

        from collectors.etf_flows import (
            collect_bitcoin_etf_flows,
        )

        result = collect_bitcoin_etf_flows()

        if not isinstance(result, dict):
            raise SchedulerRuntimeError(
                "Invalid ETF collector result."
            )

        if result.get("status") != "SUCCESS":
            raise SchedulerRuntimeError(
                "ETF collection did not complete successfully."
            )

        return {
            "status": "SUCCESS",
            "provider": provider,
            "record_count": result.get("record_count", 0),
        }

    def _task_onchain(self) -> dict[str, Any]:
        if not self._source_enabled("onchain"):
            return {
                "status": "SKIPPED",
                "reason": "ONCHAIN_SOURCE_DISABLED",
            }

        provider = self._provider_name("onchain")

        if provider != "mempool_space":
            raise SchedulerRuntimeError(
                "Unsupported on-chain provider."
            )

        if not self._provider_enabled("mempool_space"):
            raise SchedulerRuntimeError(
                "mempool.space provider is disabled."
            )

        from collectors.mempool_space import (
            collect_mempool_space,
        )

        result = collect_mempool_space(
            collection_time=self._cycle_state["timestamp"],
        )

        if result is not None and not isinstance(result, Mapping):
            raise SchedulerRuntimeError(
                "Invalid on-chain collector result."
            )

        return {
            "status": "SUCCESS",
            "provider": provider,
            "new_observation": result is not None,
        }

    def _task_macro(self) -> dict[str, Any]:
        if not self._source_enabled("macro"):
            return {
                "status": "SKIPPED",
                "reason": "MACRO_SOURCE_DISABLED",
            }

        provider = self._provider_name("macro")

        if provider != "fred":
            raise SchedulerRuntimeError(
                "Unsupported macro provider."
            )

        if not self._provider_enabled("fred"):
            raise SchedulerRuntimeError(
                "FRED provider is disabled."
            )

        from collectors.macro_pipeline import run_fred_pipeline

        result = run_fred_pipeline(
            collection_time=self._cycle_state["timestamp"],
        )

        result = _success_result(
            result,
            "FRED pipeline",
            accepted_statuses=(
                "SUCCESS",
                "NO_NEW_DATA",
            ),
        )

        return {
            "status": "SUCCESS",
            "provider": provider,
            "collected_records": result.get(
                "collected_records",
                0,
            ),
            "message": result.get("message"),
        }

    def _task_validation(self) -> dict[str, Any]:
        with self._state_lock:
            failures = list(
                self._cycle_state["collection_failures"]
            )

            candle_results = dict(
                self._cycle_state["candle_results"]
            )

        if failures:
            raise SchedulerRuntimeError(
                "Candle collection failed for: "
                + ", ".join(sorted(set(failures)))
            )

        if not candle_results:
            raise SchedulerRuntimeError(
                "No candle collection results are available."
            )

        fresh_timeframes = []
        no_new_data_timeframes = []
        accepted_total = 0
        rejected_total = 0

        for timeframe, result in candle_results.items():
            status = result.get("status")

            if status == "NO_NEW_DATA":
                no_new_data_timeframes.append(timeframe)
                continue

            if status != "PASS":
                raise SchedulerRuntimeError(
                    f"Unexpected pipeline status for {timeframe}: "
                    f"{status!r}."
                )

            accepted_count = result.get(
                "accepted_records",
                0,
            )

            if accepted_count <= 0:
                raise SchedulerRuntimeError(
                    f"No accepted candles for {timeframe}."
                )

            fresh_timeframes.append(timeframe)
            accepted_total += accepted_count
            rejected_total += result.get(
                "rejected_records",
                0,
            )

        with self._state_lock:
            self._cycle_state["validation_passed"] = True
            self._cycle_state["fresh_timeframes"] = (
                fresh_timeframes
            )

        return {
            "status": "SUCCESS",
            "has_new_data": bool(fresh_timeframes),
            "validated_timeframes": fresh_timeframes,
            "no_new_data_timeframes": no_new_data_timeframes,
            "accepted_records": accepted_total,
            "rejected_records": rejected_total,
        }

    def _analysis_is_ready(self) -> bool:
        with self._state_lock:
            return (
                self._cycle_state.get("validation_passed") is True
                and "1h"
                in self._cycle_state.get(
                    "fresh_timeframes",
                    [],
                )
            )

    def _task_market_analysis(self) -> dict[str, Any]:
        if not self._analysis_is_ready():
            return {
                "status": "SKIPPED",
                "reason": "NO_NEW_VALIDATED_1H_CANDLES",
            }

        with self._state_lock:
            candles = self._cycle_state[
                "accepted_candles"
            ].get("1h")

            timestamp = self._cycle_state["timestamp"]

        if candles is None or candles.empty:
            raise SchedulerRuntimeError(
                "Fresh validated 1h candles are unavailable."
            )

        from market_analysis.engine import analyze_market
        from reports.market_report import generate_market_report

        try:
            analysis = analyze_market(
                data=candles.copy(deep=True),
                analysis_time=timestamp,
            )

            report = generate_market_report(
                market_analysis=analysis,
            )

        except Exception as exc:
            raise SchedulerRuntimeError(
                f"Market analysis workflow failed: {exc}"
            ) from exc

        if not isinstance(analysis, dict):
            raise SchedulerRuntimeError(
                "Market analysis engine returned an invalid result."
            )

        if not isinstance(report, dict):
            raise SchedulerRuntimeError(
                "Market report generator returned an invalid result."
            )

        with self._state_lock:
            self._cycle_state["market_analysis"] = analysis
            self._cycle_state["market_analysis_passed"] = True

        return {
            "status": "SUCCESS",
            "pair": self.pair,
            "timeframe": "1h",
            "report_type": report.get("report_type"),
            "generated_at": report.get("generated_at"),
        }

    def _task_forecast(self) -> dict[str, Any]:
        if not self._analysis_is_ready():
            return {
                "status": "SKIPPED",
                "reason": "NO_NEW_VALIDATED_1H_CANDLES",
            }

        with self._state_lock:
            analysis = self._cycle_state.get("market_analysis")
            timestamp = self._cycle_state["timestamp"]

        if not isinstance(analysis, dict):
            return {
                "status": "SKIPPED",
                "reason": "MARKET_ANALYSIS_NOT_AVAILABLE",
            }

        from forecast.engine import generate_forecast
        from reports.forecast_report import generate_forecast_report

        try:
            forecast = generate_forecast(
                market_analysis=analysis,
                forecast_time=timestamp,
            )

            report = generate_forecast_report(
                forecast
            )

        except Exception as exc:
            raise SchedulerRuntimeError(
                f"Forecast workflow failed: {exc}"
            ) from exc

        if not isinstance(forecast, dict):
            raise SchedulerRuntimeError(
                "Forecast engine returned an invalid result."
            )

        if not isinstance(report, dict):
            raise SchedulerRuntimeError(
                "Forecast report generator returned an invalid result."
            )

        with self._state_lock:
            self._cycle_state["forecast"] = forecast

        return {
            "status": "SUCCESS",
            "pair": self.pair,
            "forecast_id": forecast.get("forecast_id"),
            "report_type": report.get("report_type"),
            "generated_at": report.get("generated_at"),
        }

    def _task_trader(self) -> dict[str, Any]:
        if not self._analysis_is_ready():
            return {
                "status": "SKIPPED",
                "reason": "NO_NEW_VALIDATED_1H_CANDLES",
            }

        trader_schedule = _mapping(
            self.schedule_config.get("trader"),
            "schedules.trader",
        )

        with self._state_lock:
            market_analysis_passed = self._cycle_state.get(
                "market_analysis_passed"
            )

        if (
            trader_schedule.get("run_after_market_analysis") is True
            and not market_analysis_passed
        ):
            return {
                "status": "SKIPPED",
                "reason": "MARKET_ANALYSIS_NOT_COMPLETED",
            }

        from main import run_trader

        try:
            result = run_trader(
                pair=self.pair,
                leverage=self.leverage,
                account_equity=self.account_equity,
            )
        except Exception as exc:
            raise SchedulerRuntimeError(
                f"Trading runtime failed: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise SchedulerRuntimeError(
                "Trading runtime returned an invalid result."
            )

        if result.get("runtime_status") != "SUCCESS":
            raise SchedulerRuntimeError(
                "Trading runtime did not report SUCCESS."
            )

        execution = _mapping(
            result.get("execution"),
            "trading execution metadata",
        )

        required_safety = {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        }

        for key, expected in required_safety.items():
            if execution.get(key) is not expected:
                raise SchedulerRuntimeError(
                    f"Trading safety verification failed: {key}."
                )

        return {
            "status": "SUCCESS",
            "pair": self.pair,
            "signal": result.get("signal"),
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        }

    def _build_callbacks(self) -> dict[str, Any]:
        return {
            "market_data": self._task_market_data,
            "candles": self._task_candles,
            "derivatives": self._task_derivatives,
            "etf_flows": self._task_etf_flows,
            "onchain": self._task_onchain,
            "macro": self._task_macro,
            "validation": self._task_validation,
            "market_analysis": self._task_market_analysis,
            "forecast": self._task_forecast,
            "trader": self._task_trader,
        }

    def run_once(
        self,
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(force, bool):
            raise SchedulerRuntimeError(
                "force must be a boolean."
            )

        timestamp = _normalize_utc(now)
        self._reset_cycle(timestamp)

        try:
            result = self.engine.run_cycle(
                now=timestamp,
                force=force,
            )
        except SchedulerError as exc:
            raise SchedulerRuntimeError(
                f"Scheduler cycle failed: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise SchedulerRuntimeError(
                "Scheduler engine returned an invalid result."
            )

        result = dict(result)

        with self._state_lock:
            candle_results = dict(
                self._cycle_state["candle_results"]
            )

            fresh_timeframes = list(
                self._cycle_state["fresh_timeframes"]
            )

        result["runtime"] = {
            "pair": self.pair,
            "timestamp": timestamp.isoformat(),
            "timezone": "UTC",
            "manual_review_only": True,
            "automatic_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        }

        result["candle_collection"] = candle_results
        result["fresh_timeframes"] = fresh_timeframes

        return result

    def run_forever(
        self,
        *,
        stop_event: threading.Event | None = None,
        force_first_cycle: bool = False,
    ) -> None:
        if not isinstance(force_first_cycle, bool):
            raise SchedulerRuntimeError(
                "force_first_cycle must be a boolean."
            )

        external_stop = (
            self._stop_event
            if stop_event is None
            else stop_event
        )

        if not isinstance(external_stop, threading.Event):
            raise SchedulerRuntimeError(
                "stop_event must be a threading.Event."
            )

        first_cycle = True

        LOGGER.info(
            "Scheduler runtime started for %s.",
            self.pair,
        )

        try:
            while not external_stop.is_set():
                force = first_cycle and force_first_cycle
                first_cycle = False

                try:
                    result = self.run_once(force=force)

                    LOGGER.info(
                        "Scheduler cycle status: %s",
                        result.get("status"),
                    )

                except SchedulerRuntimeError:
                    LOGGER.exception(
                        "Scheduler cycle failed."
                    )

                external_stop.wait(
                    self.poll_interval_seconds
                )

        except KeyboardInterrupt:
            LOGGER.info("Scheduler shutdown requested.")

        finally:
            LOGGER.info("Scheduler runtime stopped.")

    def stop(self) -> None:
        self._stop_event.set()

    def get_history(self) -> list[dict[str, Any]]:
        try:
            return self.engine.get_history()
        except SchedulerError as exc:
            raise SchedulerRuntimeError(
                f"Unable to read scheduler history: {exc}"
            ) from exc


def create_scheduler_runtime(
    *,
    pair: str = DEFAULT_PAIR,
    leverage: float = 1.0,
    account_equity: float | None = None,
    config: Mapping[str, Any] | None = None,
    history_path: str | Path | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> SchedulerRuntime:
    return SchedulerRuntime(
        pair=pair,
        leverage=leverage,
        account_equity=account_equity,
        config=config,
        history_path=history_path,
        poll_interval_seconds=poll_interval_seconds,
    )


def run_scheduler_once(
    *,
    pair: str = DEFAULT_PAIR,
    leverage: float = 1.0,
    account_equity: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    runtime = create_scheduler_runtime(
        pair=pair,
        leverage=leverage,
        account_equity=account_equity,
    )

    return runtime.run_once(force=force)


def run_scheduler_forever(
    *,
    pair: str = DEFAULT_PAIR,
    leverage: float = 1.0,
    account_equity: float | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    force_first_cycle: bool = False,
) -> None:
    runtime = create_scheduler_runtime(
        pair=pair,
        leverage=leverage,
        account_equity=account_equity,
        poll_interval_seconds=poll_interval_seconds,
    )

    runtime.run_forever(
        force_first_cycle=force_first_cycle,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Bitcoin Intelligence System scheduler."
    )

    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--account-equity", type=float, default=None)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scheduler cycle and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force enabled tasks in a one-time cycle.",
    )
    parser.add_argument(
        "--force-first-cycle",
        action="store_true",
        help="Force enabled tasks on the first continuous cycle.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format=(
            "%(asctime)s UTC "
            "%(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    try:
        if arguments.once:
            result = run_scheduler_once(
                pair=arguments.pair,
                leverage=arguments.leverage,
                account_equity=arguments.account_equity,
                force=arguments.force,
            )

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                    default=str,
                )
            )

            return (
                0
                if result.get("status") in ("SUCCESS", "SKIPPED")
                else 1
            )

        run_scheduler_forever(
            pair=arguments.pair,
            leverage=arguments.leverage,
            account_equity=arguments.account_equity,
            poll_interval_seconds=arguments.poll_interval,
            force_first_cycle=arguments.force_first_cycle,
        )

        return 0

    except (
        SchedulerRuntimeError,
        ValueError,
        TypeError,
    ) as exc:
        LOGGER.error(
            "Scheduler runtime failed: %s",
            exc,
        )
        return 1

    except KeyboardInterrupt:
        LOGGER.info("Scheduler runtime interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
