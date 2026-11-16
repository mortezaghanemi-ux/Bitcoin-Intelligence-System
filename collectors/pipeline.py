"""Kraken OHLCV collection and validation pipeline."""

from __future__ import annotations

import math

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from collectors.kraken_ohlcv import (
    TIMEFRAME_MINUTES,
    KrakenCollectorError,
    collect_kraken_ohlcv,
)

from config.loader import (
    ConfigurationError,
    load_config,
)

from validation.pipeline import (
    ValidationPipelineError,
    run_candle_validation,
)


class KrakenPipelineError(RuntimeError):
    """Raised when the Kraken data pipeline fails."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_reports.csv"
)


def _validate_utc_datetime(value: datetime) -> datetime:
    """Validate and normalize a UTC datetime."""

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError(
            "Timestamp must be a timezone-aware UTC datetime."
        )

    return value.astimezone(timezone.utc)


def _validate_max_age(value: Any) -> float:
    """Validate a positive finite maximum age."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "max_age_minutes must be a positive finite number."
        )

    return float(value)


def _resolve_max_age(
    max_age_minutes: float | None,
    config: dict[str, Any],
) -> float | None:
    """
    Resolve candle freshness from project configuration.

    An explicit max_age_minutes value takes precedence
    over the configured freshness limit.
    """

    if max_age_minutes is not None:
        return _validate_max_age(max_age_minutes)

    try:

        quality = config["data_quality"]

        freshness = quality["freshness"]

    except (KeyError, TypeError) as exc:

        raise KrakenPipelineError(
            "Missing or invalid data quality freshness configuration."
        ) from exc

    if not isinstance(freshness, dict):

        raise KrakenPipelineError(
            "freshness configuration must be a mapping."
        )

    if "enabled" not in freshness:

        raise KrakenPipelineError(
            "Missing freshness.enabled configuration."
        )

    if "reject_stale_data" not in freshness:

        raise KrakenPipelineError(
            "Missing freshness.reject_stale_data configuration."
        )

    enabled = freshness["enabled"]

    reject_stale = freshness["reject_stale_data"]

    if not isinstance(enabled, bool):

        raise KrakenPipelineError(
            "freshness.enabled must be a boolean."
        )

    if not isinstance(reject_stale, bool):

        raise KrakenPipelineError(
            "freshness.reject_stale_data must be a boolean."
        )

    if not enabled or not reject_stale:

        return None

    if "max_age_minutes" not in freshness:

        raise KrakenPipelineError(
            "Missing max_age_minutes freshness configuration."
        )

    limits = freshness["max_age_minutes"]

    if not isinstance(limits, dict):

        raise KrakenPipelineError(
            "freshness.max_age_minutes must be a mapping."
        )

    if "candles" not in limits:

        raise KrakenPipelineError(
            "Missing candles freshness configuration."
        )

    try:

        return _validate_max_age(
            limits["candles"]
        )

    except ValueError as exc:

        raise KrakenPipelineError(
            f"Invalid candles freshness configuration: {exc}"
        ) from exc


def run_kraken_pipeline(
    pair: str = "XBTUSD",
    timeframe: str = "5m",
    *,
    output_dir: str | Path | None = None,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    collection_time: datetime | None = None,
    max_age_minutes: float | None = None,
    timeout: float = 15,
) -> dict[str, Any]:
    """
    Collect Kraken OHLCV data and validate new closed candles.

    Raw provider records are preserved by the collector.

    Only validated records are returned as accepted data.
    """

    # Step 1: Validate timeframe.

    if timeframe not in TIMEFRAME_MINUTES:

        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    # Step 2: Resolve collection timestamp.

    checked_at = (
        datetime.now(timezone.utc)
        if collection_time is None
        else _validate_utc_datetime(collection_time)
    )

    # Step 3: Load project configuration.

    try:

        config = load_config()

    except ConfigurationError as exc:

        raise KrakenPipelineError(
            f"Cannot load project configuration: {exc}"
        ) from exc

    # Step 4: Resolve freshness policy.

    age_limit = _resolve_max_age(
        max_age_minutes,
        config,
    )

    # Step 5: Collect and persist original Kraken data.

    try:

        records = collect_kraken_ohlcv(
            pair=pair,
            timeframe=timeframe,
            output_dir=output_dir,
            collection_time=checked_at,
            timeout=timeout,
        )

    except (KrakenCollectorError, ValueError) as exc:

        raise KrakenPipelineError(
            f"Kraken data collection failed: {exc}"
        ) from exc

    # Step 6: Handle absence of new closed candles.

    if not records:

        return {
            "status": "NO_NEW_DATA",
            "pair": pair.upper(),
            "timeframe": timeframe,
            "collected_records": 0,
            "accepted": pd.DataFrame(),
            "rejected": pd.DataFrame(),
            "validation": None,
            "reports": [],
            "max_age_minutes": age_limit,
            "message": (
                "No new closed candles were available."
            ),
        }

    # Step 7: Prepare collected records for validation.

    try:

        data = pd.DataFrame(records)

        required_columns = [
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            raise KrakenPipelineError(
                "Collected data is missing required columns: "
                + ", ".join(missing_columns)
            )

        validation_data = data.loc[
            :,
            required_columns,
        ].copy(deep=True)

    except KrakenPipelineError:

        raise

    except Exception as exc:

        raise KrakenPipelineError(
            f"Cannot prepare collected data: {exc}"
        ) from exc

    # Step 8: Validate collected candles.

    try:

        validation_result = run_candle_validation(
            validation_data,
            report_path=report_path,
            validation_time=checked_at,
            max_age_minutes=age_limit,
        )

    except (
        ValidationPipelineError,
        ValueError,
        TypeError,
    ) as exc:

        raise KrakenPipelineError(
            f"Kraken candle validation failed: {exc}"
        ) from exc

    # Step 9: Extract validation results.

    try:

        accepted = validation_result["accepted"]

        rejected = validation_result["rejected"]

        validation_report = validation_result["validation"]

        reports = validation_result["reports"]

    except (KeyError, TypeError) as exc:

        raise KrakenPipelineError(
            f"Invalid validation pipeline response: {exc}"
        ) from exc

    if (
        not isinstance(accepted, pd.DataFrame)
        or not isinstance(rejected, pd.DataFrame)
    ):

        raise KrakenPipelineError(
            "Validation pipeline returned invalid data structures."
        )

    # Step 10: Return pipeline results.

    status = (
        "PASS"
        if rejected.empty
        else "FAIL"
    )

    return {
        "status": status,
        "pair": pair.upper(),
        "timeframe": timeframe,
        "collected_records": len(records),
        "accepted": accepted,
        "rejected": rejected,
        "validation": validation_report,
        "reports": reports,
        "max_age_minutes": age_limit,
        "message": (
            "All collected candles passed validation."
            if rejected.empty
            else "Some collected candles were rejected."
        ),
    }