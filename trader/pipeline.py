"""Trading pipeline orchestration.

Coordinates candle validation, freshness checks, signal generation
and append-only signal persistence.

This pipeline never connects to an exchange or executes orders.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.loader import load_config
from trader.candles.manager import (
    CandleManager,
    CandleManagerError,
)
from trader.signal_engine.engine import (
    TradingSignalEngine,
    TradingSignalEngineError,
    SUPPORTED_TIMEFRAMES,
)
from trader.signal_engine.history import (
    SignalHistoryStore,
    SignalHistoryError,
)
from trader.signal_engine.rejections import (
    SignalRejectionStore,
    SignalRejectionError,
)


class TradingPipelineError(RuntimeError):
    """Raised when the trading pipeline cannot complete safely."""


class TradingPipelinePersistenceError(TradingPipelineError):
    """Raised when signal persistence is incomplete."""


class TradingPipeline:
    """Coordinate validated candles and persistent signal generation."""

    TIMEFRAME_MINUTES = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
    }

    REQUIRED_EXECUTION_SETTINGS = {
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
        "exchange_api_keys_required": False,
    }

    def __init__(
        self,
        *,
        history_path: str | Path | None = None,
        rejection_path: str | Path | None = None,
        config_dir: str | Path | None = None,
        engine: TradingSignalEngine | None = None,
        history_store: SignalHistoryStore | None = None,
        rejection_store: SignalRejectionStore | None = None,
    ) -> None:

        try:
            self.config = load_config(
                config_dir=(
                    config_dir
                    if config_dir is not None
                    else self._default_config_dir()
                )
            )

        except Exception as exc:
            raise TradingPipelineError(
                f"Unable to load project configuration: {exc}"
            ) from exc

        trader_config = self.config["trader"].get(
            "trader",
            {},
        )

        if trader_config.get("enabled") is not True:
            raise TradingPipelineError(
                "Trading pipeline is disabled in configuration."
            )

        execution = trader_config.get(
            "execution",
            {},
        )

        for setting, expected in (
            self.REQUIRED_EXECUTION_SETTINGS.items()
        ):

            if execution.get(setting) is not expected:
                raise TradingPipelineError(
                    "Unsafe trading execution configuration detected: "
                    f"{setting} must be {expected}."
                )

        self.trader_config = trader_config

        self.timeframes = tuple(
            trader_config.get("timeframes", [])
        )

        if (
            len(self.timeframes) != len(set(self.timeframes))
            or set(self.timeframes) != set(SUPPORTED_TIMEFRAMES)
        ):
            raise TradingPipelineError(
                "Configured trader timeframes do not match "
                "the supported engine timeframes."
            )

        self.history_store = (
            history_store
            if history_store is not None
            else SignalHistoryStore(history_path)
        )

        self.rejection_store = (
            rejection_store
            if rejection_store is not None
            else SignalRejectionStore(rejection_path)
        )

        self.engine = (
            engine
            if engine is not None
            else self._build_engine()
        )

        self._validate_stores()

    @staticmethod
    def _default_config_dir() -> Path:
        """Return the project's configuration directory."""

        return (
            Path(__file__).resolve().parents[1]
            / "config"
        )

    def _build_engine(self) -> TradingSignalEngine:
        """Build the signal engine from project configuration."""

        position = self.trader_config.get(
            "position",
            {},
        )

        risk = self.trader_config.get(
            "risk",
            {},
        )

        confidence = self.trader_config.get(
            "signals",
            {},
        ).get(
            "confidence",
            {},
        )

        return TradingSignalEngine(
            minimum_candles=30,
            minimum_confidence=float(
                confidence.get("minimum_threshold", 60)
            ),
            minimum_risk_reward=float(
                position.get("minimum_risk_reward", 1.5)
            ),
            maximum_leverage=float(
                risk.get("maximum_leverage", 2)
            ),
        )

    def _validate_stores(self) -> None:
        """Verify that existing persistent records are readable and unique."""

        try:
            self.history_store.read_all()
            self.rejection_store.read_all()

            history_integrity = (
                self.history_store.verify_integrity()
            )

            rejection_integrity = (
                self.rejection_store.verify_integrity()
            )

            if not isinstance(history_integrity, dict):
                raise TradingPipelineError(
                    "Invalid signal history integrity report."
                )

            if not isinstance(rejection_integrity, dict):
                raise TradingPipelineError(
                    "Invalid rejection history integrity report."
                )

            if history_integrity.get("valid") is not True:
                raise TradingPipelineError(
                    "Signal history contains duplicate or invalid records."
                )

            if rejection_integrity.get("valid") is not True:
                raise TradingPipelineError(
                    "Rejection history contains duplicate or invalid records."
                )

        except TradingPipelineError:
            raise

        except (
            SignalHistoryError,
            SignalRejectionError,
            OSError,
        ) as exc:
            raise TradingPipelineError(
                f"Signal storage integrity check failed: {exc}"
            ) from exc

    @staticmethod
    def _normalize_analysis_time(
        analysis_time: datetime | None,
    ) -> datetime:
        """Validate and normalize the analysis timestamp."""

        if analysis_time is None:
            return datetime.now(timezone.utc)

        if not isinstance(analysis_time, datetime):
            raise TradingPipelineError(
                "analysis_time must be a datetime instance."
            )

        if analysis_time.tzinfo is None:
            raise TradingPipelineError(
                "analysis_time must be timezone-aware UTC."
            )

        if analysis_time.utcoffset() != timedelta(0):
            raise TradingPipelineError(
                "analysis_time must use UTC."
            )

        return analysis_time.astimezone(timezone.utc)

    def _validate_input(
        self,
        data: dict[str, pd.DataFrame],
    ) -> None:
        """Validate the four timeframe datasets."""

        if not isinstance(data, dict):
            raise TradingPipelineError(
                "data must be a dictionary."
            )

        if set(data) != set(self.timeframes):
            missing = sorted(
                set(self.timeframes) - set(data)
            )

            unknown = sorted(
                set(data) - set(self.timeframes)
            )

            raise TradingPipelineError(
                "Invalid timeframe collection. "
                f"Missing: {missing}; unsupported: {unknown}."
            )

        for timeframe in self.timeframes:

            if not isinstance(
                data[timeframe],
                pd.DataFrame,
            ):
                raise TradingPipelineError(
                    f"{timeframe} data must be a DataFrame."
                )

            if data[timeframe].empty:
                raise TradingPipelineError(
                    f"{timeframe} candle data is empty."
                )

    @staticmethod
    def _select_single_source(
        candles: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """Select one deterministic source without mixing candle histories."""

        if candles.empty:
            raise TradingPipelineError(
                f"No accepted candles available for {timeframe}."
            )

        if "source" not in candles.columns:
            raise TradingPipelineError(
                f"{timeframe} candle data has no source column."
            )

        sources = candles["source"].dropna().astype(str)

        sources = sources.loc[
            sources.str.strip() != ""
        ]

        if sources.empty:
            raise TradingPipelineError(
                f"{timeframe} has no valid candle source."
            )

        source_counts = sources.value_counts()

        maximum_count = source_counts.max()

        candidates = sorted(
            source_counts.loc[
                source_counts == maximum_count
            ].index.tolist()
        )

        selected_source = candidates[0]

        selected = candles.loc[
            candles["source"].astype(str) == selected_source
        ].copy(deep=True)

        selected = selected.sort_values(
            "open_time",
            kind="stable",
        ).reset_index(drop=True)

        if selected.empty:
            raise TradingPipelineError(
                f"Unable to select a valid source for {timeframe}."
            )

        if selected["open_time"].duplicated().any():
            raise TradingPipelineError(
                f"{timeframe} contains duplicate candle timestamps "
                f"for source {selected_source}."
            )

        if selected["close_time"].duplicated().any():
            raise TradingPipelineError(
                f"{timeframe} contains duplicate candle close times "
                f"for source {selected_source}."
            )

        return selected

    def _validate_freshness(
        self,
        timeframe: str,
        candles: pd.DataFrame,
        analysis_time: datetime,
    ) -> None:
        """Reject a timeframe whose latest closed candle is stale."""

        if candles.empty:
            raise TradingPipelineError(
                f"No accepted candles available for {timeframe}."
            )

        try:
            close_times = pd.to_datetime(
                candles["close_time"],
                errors="raise",
                utc=True,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise TradingPipelineError(
                f"{timeframe} contains invalid close timestamps."
            ) from exc

        if close_times.isna().any():
            raise TradingPipelineError(
                f"{timeframe} contains missing close timestamps."
            )

        latest_close = close_times.max().to_pydatetime()

        if latest_close.tzinfo is None:
            raise TradingPipelineError(
                f"{timeframe} latest candle has no timezone."
            )

        if latest_close.utcoffset() != timedelta(0):
            raise TradingPipelineError(
                f"{timeframe} latest candle is not UTC."
            )

        age = analysis_time - latest_close

        if age < timedelta(0):
            raise TradingPipelineError(
                f"{timeframe} contains a future candle."
            )

        quality = self.config["data_quality"]

        freshness = quality.get(
            "freshness",
            {},
        )

        if freshness.get("enabled") is not True:
            return

        if freshness.get("reject_stale_data") is not True:
            raise TradingPipelineError(
                "Stale-data rejection must remain enabled."
            )

        limits = freshness.get(
            "max_age_minutes",
            {},
        )

        configured_grace = float(
            limits.get("candles", 15)
        )

        if (
            pd.isna(configured_grace)
            or configured_grace < 0
            or not pd.notna(configured_grace)
        ):
            raise TradingPipelineError(
                "Invalid candle freshness configuration."
            )

        timeframe_minutes = self.TIMEFRAME_MINUTES[
            timeframe
        ]

        maximum_age = timedelta(
            minutes=timeframe_minutes + configured_grace
        )

        if age > maximum_age:
            raise TradingPipelineError(
                f"{timeframe} candles are stale. "
                f"Latest close age: {age}. "
                f"Maximum allowed age: {maximum_age}."
            )

    def _prepare_candles(
        self,
        data: dict[str, pd.DataFrame],
        analysis_time: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Validate, isolate and prepare candles for signal generation."""

        prepared: dict[str, pd.DataFrame] = {}

        for timeframe in self.timeframes:

            manager = CandleManager(timeframe)

            try:
                result = manager.ingest(
                    data[timeframe].copy(deep=True),
                    validation_time=analysis_time,
                )

            except (
                CandleManagerError,
                ValueError,
                TypeError,
            ) as exc:
                raise TradingPipelineError(
                    f"Candle validation failed for {timeframe}: {exc}"
                ) from exc

            if result.get("status") != "PASS":
                report = result.get("report", {})

                raise TradingPipelineError(
                    f"{timeframe} candle validation failed. "
                    f"Report: {report}"
                )

            rejected = result.get("rejected")

            if (
                isinstance(rejected, pd.DataFrame)
                and not rejected.empty
            ):
                raise TradingPipelineError(
                    f"{timeframe} contains rejected candles."
                )

            accepted = manager.get_candles()

            if accepted.empty:
                raise TradingPipelineError(
                    f"{timeframe} has no accepted candles."
                )

            accepted = self._select_single_source(
                accepted,
                timeframe,
            )

            self._validate_freshness(
                timeframe,
                accepted,
                analysis_time,
            )

            prepared[timeframe] = accepted

        return prepared

    @staticmethod
    def _requires_rejection_persistence(
        signal: dict[str, Any],
    ) -> bool:
        """Return whether a signal belongs in rejection history."""

        return (
            signal.get("status") == "REJECTED"
            and signal.get("signal") == "NO_TRADE"
        )

    def _ensure_rejection_persisted(
        self,
        signal: dict[str, Any],
    ) -> bool:
        """Ensure a rejected signal exists in rejection storage."""

        if not self._requires_rejection_persistence(signal):
            return False

        signal_id = signal["signal_id"]

        try:
            existing_rejection = self.rejection_store.get(
                signal_id
            )

            if existing_rejection is not None:
                return True

            self.rejection_store.append(signal)

            if not self.rejection_store.contains(signal_id):
                raise TradingPipelinePersistenceError(
                    "Rejection record was not found after append."
                )

            return True

        except TradingPipelinePersistenceError:
            raise

        except (
            SignalRejectionError,
            OSError,
        ) as exc:
            raise TradingPipelinePersistenceError(
                "Unable to persist rejection record "
                f"for {signal_id}: {exc}"
            ) from exc

    def _persist_signal(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist signal history and ensure rejection consistency."""

        signal_id = signal.get("signal_id")

        if not isinstance(signal_id, str) or not signal_id:
            raise TradingPipelineError(
                "Signal engine returned an invalid signal identifier."
            )

        try:
            existing = self.history_store.get(signal_id)

            if existing is not None:

                rejection_saved = self._ensure_rejection_persisted(
                    existing
                )

                return {
                    "status": "DUPLICATE",
                    "history_saved": True,
                    "rejection_saved": rejection_saved,
                    "signal": existing,
                }

            history_record = self.history_store.append(
                signal
            )

        except TradingPipelinePersistenceError:
            raise

        except (
            SignalHistoryError,
            OSError,
        ) as exc:
            raise TradingPipelinePersistenceError(
                f"Unable to persist signal history: {exc}"
            ) from exc

        rejection_saved = self._ensure_rejection_persisted(
            history_record
        )

        return {
            "status": (
                "DUPLICATE"
                if history_record.get("duplicate") is True
                else "SAVED"
            ),
            "history_saved": True,
            "rejection_saved": rejection_saved,
            "signal": history_record,
        }

    @staticmethod
    def _validate_signal_safety(
        signal: dict[str, Any],
    ) -> None:
        """Verify signal metadata before any persistence operation."""

        data_quality = signal.get("data_quality")

        if not isinstance(data_quality, dict):
            raise TradingPipelineError(
                "Signal engine returned invalid data-quality metadata."
            )

        if data_quality.get("future_data_used") is not False:
            raise TradingPipelineError(
                "Signal engine did not confirm future-data protection."
            )

        if data_quality.get("closed_candles_only") is not True:
            raise TradingPipelineError(
                "Signal engine did not confirm closed-candle usage."
            )

        execution = signal.get("execution")

        if not isinstance(execution, dict):
            raise TradingPipelineError(
                "Signal execution metadata is missing."
            )

        if (
            execution.get("manual_review_only") is not True
            or execution.get("auto_trading") is not False
            or execution.get("exchange_connection") is not False
            or execution.get("order_execution") is not False
        ):
            raise TradingPipelineError(
                "Signal execution safety verification failed."
            )

    def run(
        self,
        data: dict[str, pd.DataFrame],
        *,
        analysis_time: datetime | None = None,
        account_equity: float | None = None,
        leverage: float = 1.0,
    ) -> dict[str, Any]:
        """Run the complete validated trading signal pipeline."""

        timestamp = self._normalize_analysis_time(
            analysis_time
        )

        self._validate_input(data)

        prepared = self._prepare_candles(
            data,
            timestamp,
        )

        try:
            signal = self.engine.generate(
                prepared,
                analysis_time=timestamp,
                account_equity=account_equity,
                leverage=leverage,
            )

        except TradingSignalEngineError as exc:
            raise TradingPipelineError(
                f"Signal generation failed: {exc}"
            ) from exc

        if not isinstance(signal, dict):
            raise TradingPipelineError(
                "Signal engine returned an invalid result."
            )

        self._validate_signal_safety(signal)

        persistence = self._persist_signal(
            signal
        )

        return {
            "pipeline_status": "SUCCESS",
            "analysis_timestamp": timestamp.isoformat(),
            "validated_timeframes": list(self.timeframes),
            "candle_counts": {
                timeframe: len(prepared[timeframe])
                for timeframe in self.timeframes
            },
            "signal": signal,
            "persistence": {
                "status": persistence["status"],
                "history_saved": persistence["history_saved"],
                "rejection_saved": persistence["rejection_saved"],
            },
            "execution": {
                "manual_review_only": True,
                "auto_trading": False,
                "exchange_connection": False,
                "order_execution": False,
            },
        }


def run_trading_pipeline(
    data: dict[str, pd.DataFrame],
    *,
    analysis_time: datetime | None = None,
    account_equity: float | None = None,
    leverage: float = 1.0,
    history_path: str | Path | None = None,
    rejection_path: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience function for one complete pipeline run."""

    pipeline = TradingPipeline(
        history_path=history_path,
        rejection_path=rejection_path,
        config_dir=config_dir,
    )

    return pipeline.run(
        data,
        analysis_time=analysis_time,
        account_equity=account_equity,
        leverage=leverage,
    )


if __name__ == "__main__":
    print(
        "Bitcoin trading pipeline loaded. "
        "No exchange connection or order execution is enabled."
    )