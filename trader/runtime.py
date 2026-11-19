"""End-to-end runtime for Kraken-based Bitcoin trading signals.

This runtime coordinates:
1. Kraken public OHLCV collection.
2. Candle validation and raw-data persistence.
3. Loading complete historical candle datasets.
4. Trading pipeline execution.
5. Signal and rejection persistence.

No private exchange API is used.
No orders are submitted.
Automatic trading remains disabled.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from collectors.kraken_ohlcv import TIMEFRAME_MINUTES
from collectors.pipeline import (
    DEFAULT_REPORT_PATH,
    KrakenPipelineError,
    run_kraken_pipeline,
)
from config.loader import (
    ConfigurationError,
    load_config,
)
from trader.data_loader import (
    DEFAULT_DATA_DIR,
    TRADING_TIMEFRAMES,
    TradingDataLoaderError,
    load_trading_candles,
)
from trader.pipeline import (
    TradingPipelineError,
    run_trading_pipeline,
)


class TradingRuntimeError(RuntimeError):
    """Raised when the complete trading runtime cannot finish safely."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SIGNAL_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "trader"
    / "signal_history.jsonl"
)

DEFAULT_SIGNAL_REJECTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "trader"
    / "signal_rejections.jsonl"
)


def _normalize_pair(pair: str) -> str:
    """Validate and normalize a Kraken trading pair."""

    if (
        not isinstance(pair, str)
        or not pair
        or not pair.isalnum()
    ):
        raise ValueError("Invalid trading pair.")

    return pair.upper()


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> datetime:
    """Return a timezone-aware UTC analysis timestamp."""

    if analysis_time is None:
        return datetime.now(timezone.utc)

    if not isinstance(analysis_time, datetime):
        raise ValueError(
            "analysis_time must be a datetime instance."
        )

    if analysis_time.tzinfo is None:
        raise ValueError(
            "analysis_time must be timezone-aware UTC."
        )

    try:
        offset = analysis_time.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid analysis_time."
        ) from exc

    if offset != timedelta(0):
        raise ValueError(
            "analysis_time must use UTC."
        )

    return analysis_time.astimezone(timezone.utc)


def _validate_positive_number(
    value: Any,
    *,
    name: str,
) -> float:
    """Validate a positive finite numeric parameter."""

    import math

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    return float(value)


def _resolve_paths(
    data_dir: str | Path | None,
    report_path: str | Path | None,
    history_path: str | Path | None,
    rejection_path: str | Path | None,
) -> dict[str, Path]:
    """Resolve all runtime storage paths."""

    try:
        resolved = {
            "data": Path(
                data_dir if data_dir is not None else DEFAULT_DATA_DIR
            ).expanduser().resolve(),

            "report": Path(
                report_path
                if report_path is not None
                else DEFAULT_REPORT_PATH
            ).expanduser().resolve(),

            "history": Path(
                history_path
                if history_path is not None
                else DEFAULT_SIGNAL_HISTORY_PATH
            ).expanduser().resolve(),

            "rejections": Path(
                rejection_path
                if rejection_path is not None
                else DEFAULT_SIGNAL_REJECTION_PATH
            ).expanduser().resolve(),
        }

    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise TradingRuntimeError(
            f"Invalid runtime storage path: {exc}"
        ) from exc

    return resolved


def _load_runtime_configuration() -> dict[str, Any]:
    """Load configuration and verify runtime safety settings."""

    try:
        config = load_config()

    except ConfigurationError as exc:
        raise TradingRuntimeError(
            f"Unable to load project configuration: {exc}"
        ) from exc

    trader = config["trader"].get(
        "trader",
        {},
    )

    if trader.get("enabled") is not True:
        raise TradingRuntimeError(
            "Trading runtime is disabled in configuration."
        )

    execution = trader.get(
        "execution",
        {},
    )

    required_safety = {
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
        "exchange_api_keys_required": False,
    }

    for key, expected in required_safety.items():

        if execution.get(key) is not expected:
            raise TradingRuntimeError(
                "Unsafe execution configuration: "
                f"{key} must be {expected}."
            )

    configured_timeframes = trader.get(
        "timeframes",
        [],
    )

    if (
        not isinstance(configured_timeframes, list)
        or len(configured_timeframes) != len(TRADING_TIMEFRAMES)
        or set(configured_timeframes) != set(TRADING_TIMEFRAMES)
    ):
        raise TradingRuntimeError(
            "Configured trading timeframes do not match "
            "the runtime-supported timeframes."
        )

    quality = config.get(
        "data_quality",
        {},
    )

    freshness = quality.get(
        "freshness",
        {},
    )

    if (
        freshness.get("enabled") is not True
        or freshness.get("reject_stale_data") is not True
    ):
        raise TradingRuntimeError(
            "Candle freshness validation must remain enabled."
        )

    limits = freshness.get(
        "max_age_minutes",
        {},
    )

    grace_minutes = limits.get("candles")

    grace_minutes = _validate_positive_number(
        grace_minutes,
        name="data_quality.freshness.max_age_minutes.candles",
    )

    return {
        "trader": trader,
        "freshness_grace_minutes": grace_minutes,
    }


def _collect_all_timeframes(
    *,
    pair: str,
    data_dir: Path,
    report_path: Path,
    collection_time: datetime,
    timeout: float,
    grace_minutes: float,
) -> dict[str, dict[str, Any]]:
    """Collect and validate Kraken candles for every trading timeframe."""

    results: dict[str, dict[str, Any]] = {}

    for timeframe in TRADING_TIMEFRAMES:

        interval_minutes = TIMEFRAME_MINUTES.get(
            timeframe
        )

        if interval_minutes is None:
            raise TradingRuntimeError(
                f"Unsupported runtime timeframe: {timeframe}"
            )

        # Each timeframe receives its own age allowance.
        # This prevents valid 1h and 4h candles from being
        # incorrectly rejected by the 5m freshness window.
        maximum_age = (
            interval_minutes + grace_minutes
        )

        try:
            result = run_kraken_pipeline(
                pair=pair,
                timeframe=timeframe,
                output_dir=data_dir,
                report_path=report_path,
                collection_time=collection_time,
                max_age_minutes=maximum_age,
                timeout=timeout,
            )

        except (
            KrakenPipelineError,
            ValueError,
            OSError,
        ) as exc:
            raise TradingRuntimeError(
                f"Kraken collection failed for {timeframe}: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise TradingRuntimeError(
                f"Invalid Kraken pipeline response for {timeframe}."
            )

        status = result.get("status")

        if status not in ("PASS", "NO_NEW_DATA"):
            raise TradingRuntimeError(
                f"Kraken validation did not pass for {timeframe}. "
                f"Status: {status}."
            )

        accepted = result.get("accepted")

        if status == "PASS":

            if accepted is None or accepted.empty:
                raise TradingRuntimeError(
                    f"{timeframe} returned PASS without accepted candles."
                )

        results[timeframe] = {
            "status": status,
            "pair": result.get("pair"),
            "timeframe": timeframe,
            "collected_records": result.get(
                "collected_records",
                0,
            ),
            "accepted_records": (
                len(accepted)
                if accepted is not None
                else 0
            ),
            "message": result.get("message"),
            "maximum_age_minutes": maximum_age,
        }

    return results


def run_trading_runtime(
    pair: str = "XBTUSD",
    *,
    analysis_time: datetime | None = None,
    data_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    history_path: str | Path | None = None,
    rejection_path: str | Path | None = None,
    account_equity: float | None = None,
    leverage: float = 1.0,
    timeout: float = 15,
) -> dict[str, Any]:
    """Run the complete Kraken-to-trading-signal workflow.

    The runtime:
    - Collects newly closed Kraken candles.
    - Validates every timeframe.
    - Loads complete persisted candle histories.
    - Generates and persists a trading signal.
    - Returns a structured execution report.

    It never uses private Kraken endpoints and never executes orders.
    """

    normalized_pair = _normalize_pair(pair)

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    normalized_leverage = _validate_positive_number(
        leverage,
        name="leverage",
    )

    if normalized_leverage > 2:
        raise ValueError(
            "leverage cannot exceed the project maximum of 2."
        )

    normalized_timeout = _validate_positive_number(
        timeout,
        name="timeout",
    )

    if account_equity is not None:

        account_equity = _validate_positive_number(
            account_equity,
            name="account_equity",
        )

    paths = _resolve_paths(
        data_dir,
        report_path,
        history_path,
        rejection_path,
    )

    configuration = _load_runtime_configuration()

    try:
        paths["data"].mkdir(
            parents=True,
            exist_ok=True,
        )

        paths["report"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths["history"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths["rejections"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:
        raise TradingRuntimeError(
            f"Unable to prepare runtime storage: {exc}"
        ) from exc

    # --------------------------------------------------
    # STEP 1: Collect and validate all Kraken timeframes.
    # --------------------------------------------------

    collection_results = _collect_all_timeframes(
        pair=normalized_pair,
        data_dir=paths["data"],
        report_path=paths["report"],
        collection_time=timestamp,
        timeout=normalized_timeout,
        grace_minutes=configuration[
            "freshness_grace_minutes"
        ],
    )

    # --------------------------------------------------
    # STEP 2: Load complete persisted candle histories.
    # --------------------------------------------------

    try:
        candle_data = load_trading_candles(
            pair=normalized_pair,
            data_dir=paths["data"],
        )

    except (
        TradingDataLoaderError,
        ValueError,
        OSError,
    ) as exc:
        raise TradingRuntimeError(
            f"Unable to load complete Kraken candle history: {exc}"
        ) from exc

    candle_counts = {}

    for timeframe in TRADING_TIMEFRAMES:

        candles = candle_data.get(timeframe)

        if candles is None or candles.empty:
            raise TradingRuntimeError(
                f"No historical candles available for {timeframe}."
            )

        candle_counts[timeframe] = len(candles)

    # --------------------------------------------------
    # STEP 3: Generate and persist the trading signal.
    # --------------------------------------------------

    try:
        trading_result = run_trading_pipeline(
            candle_data,
            analysis_time=timestamp,
            account_equity=account_equity,
            leverage=normalized_leverage,
            history_path=paths["history"],
            rejection_path=paths["rejections"],
        )

    except TradingPipelineError as exc:
        raise TradingRuntimeError(
            f"Trading signal pipeline failed: {exc}"
        ) from exc

    except (ValueError, TypeError, OSError) as exc:
        raise TradingRuntimeError(
            f"Unexpected trading runtime failure: {exc}"
        ) from exc

    if not isinstance(trading_result, dict):
        raise TradingRuntimeError(
            "Trading pipeline returned an invalid result."
        )

    if trading_result.get("pipeline_status") != "SUCCESS":
        raise TradingRuntimeError(
            "Trading pipeline did not report SUCCESS."
        )

    signal = trading_result.get("signal")

    if not isinstance(signal, dict):
        raise TradingRuntimeError(
            "Trading pipeline returned an invalid signal."
        )

    # --------------------------------------------------
    # STEP 4: Verify execution safety.
    # --------------------------------------------------

    execution = trading_result.get(
        "execution",
        {},
    )

    if (
        execution.get("manual_review_only") is not True
        or execution.get("auto_trading") is not False
        or execution.get("exchange_connection") is not False
        or execution.get("order_execution") is not False
    ):
        raise TradingRuntimeError(
            "Runtime execution safety verification failed."
        )

    # --------------------------------------------------
    # STEP 5: Return the complete runtime report.
    # --------------------------------------------------

    return {
        "runtime_status": "SUCCESS",
        "pair": normalized_pair,
        "analysis_timestamp": timestamp.isoformat(),
        "collection": collection_results,
        "candle_counts": candle_counts,
        "signal": signal,
        "persistence": trading_result.get(
            "persistence",
            {},
        ),
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
        "storage": {
            "raw_candles": str(paths["data"]),
            "validation_reports": str(paths["report"]),
            "signal_history": str(paths["history"]),
            "signal_rejections": str(paths["rejections"]),
        },
    }


if __name__ == "__main__":

    try:
        result = run_trading_runtime()

        print("TRADING RUNTIME: SUCCESS")

        print(
            f"Pair: {result['pair']}"
        )

        print(
            f"Analysis time: {result['analysis_timestamp']}"
        )

        print(
            f"Signal: {result['signal'].get('signal')}"
        )

        print(
            f"Signal status: {result['signal'].get('status')}"
        )

        print(
            "Execution: MANUAL REVIEW ONLY"
        )

        print(
            "Automatic trading: DISABLED"
        )

    except TradingRuntimeError as exc:

        print(
            f"TRADING RUNTIME FAILED: {exc}"
        )

    except (ValueError, TypeError) as exc:

        print(
            f"INVALID RUNTIME PARAMETERS: {exc}"
        )