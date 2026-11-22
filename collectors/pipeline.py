
"""Kraken OHLCV collection and validation pipeline."""

from __future__ import annotations

import math

from collections.abc import Mapping
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

REQUIRED_COLUMNS = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)


def _validate_utc_datetime(value: datetime) -> datetime:
    """Validate and normalize a UTC datetime."""

    if not isinstance(value, datetime):
        raise ValueError(
            "Timestamp must be a timezone-aware UTC datetime."
        )

    if value.tzinfo is None:
        raise ValueError(
            "Timestamp must be a timezone-aware UTC datetime."
        )

    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid UTC timestamp."
        ) from exc

    if offset != timedelta(0):
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

    if not isinstance(config, Mapping):
        raise KrakenPipelineError(
            "Project configuration must be a mapping."
        )

    try:
        quality = config["data_quality"]
        freshness = quality["freshness"]

    except (KeyError, TypeError) as exc:
        raise KrakenPipelineError(
            "Missing or invalid data quality freshness configuration."
        ) from exc

    if not isinstance(freshness, Mapping):
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
            "Missing candles freshness configuration: "
            "max_age_minutes is not configured."
        )

    limits = freshness["max_age_minutes"]

    if not isinstance(limits, Mapping):
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


def _empty_result(
    pair: str,
    timeframe: str,
    max_age_minutes: float | None,
) -> dict[str, Any]:
    """Build a consistent response when no new candles exist."""

    return {
        "status": "NO_NEW_DATA",
        "pair": pair.upper(),
        "timeframe": timeframe,
        "collected_records": 0,
        "accepted": pd.DataFrame(),
        "rejected": pd.DataFrame(),
        "validation": None,
        "reports": [],
        "max_age_minutes": max_age_minutes,
        "message": (
            "No new closed candles were available."
        ),
    }


def _prepare_validation_data(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Prepare collected records for candle validation."""

    if not isinstance(records, list):
        raise KrakenPipelineError(
            "Kraken collector must return a list of records."
        )

    if not records:
        return pd.DataFrame()

    for index, record in enumerate(records):

        if not isinstance(record, Mapping):
            raise KrakenPipelineError(
                f"Invalid collected record at index {index}."
            )

    try:
        data = pd.DataFrame(records)

    except (TypeError, ValueError) as exc:
        raise KrakenPipelineError(
            f"Cannot prepare collected data: {exc}"
        ) from exc

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise KrakenPipelineError(
            "Collected data is missing required columns: "
            + ", ".join(missing_columns)
        )

    return data.loc[
        :,
        list(REQUIRED_COLUMNS),
    ].copy(deep=True)


def _extract_validation_result(
    validation_result: Any,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Any,
    list,
]:
    """Validate and extract the validation pipeline response."""

    if not isinstance(validation_result, Mapping):
        raise KrakenPipelineError(
            "Validation pipeline returned an invalid response."
        )

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

    if not isinstance(reports, list):
        raise KrakenPipelineError(
            "Validation pipeline returned invalid reports."
        )

    return (
        accepted.copy(deep=True),
        rejected.copy(deep=True),
        validation_report,
        reports,
    )


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
    Invalid records are returned separately as rejected data.

    This pipeline never generates trading signals or executes orders.
    """

    # Step 1: Validate timeframe.

    if (
        not isinstance(timeframe, str)
        or timeframe not in TIMEFRAME_MINUTES
    ):
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    # Step 2: Validate pair.

    if (
        not isinstance(pair, str)
        or not pair
        or not pair.isalnum()
    ):
        raise ValueError(
            "Invalid trading pair."
        )

    normalized_pair = pair.upper()

    # Step 3: Validate timeout.

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout must be a positive finite number."
        )

    # Step 4: Resolve collection timestamp.

    checked_at = (
        datetime.now(timezone.utc)
        if collection_time is None
        else _validate_utc_datetime(collection_time)
    )

    # Step 5: Load project configuration.

    try:
        config = load_config()

    except ConfigurationError as exc:
        raise KrakenPipelineError(
            f"Cannot load project configuration: {exc}"
        ) from exc

    # Step 6: Resolve freshness policy.

    age_limit = _resolve_max_age(
        max_age_minutes,
        config,
    )

    # Step 7: Collect Kraken data.

    try:
        records = collect_kraken_ohlcv(
            pair=normalized_pair,
            timeframe=timeframe,
            output_dir=output_dir,
            collection_time=checked_at,
            timeout=timeout,
        )

    except (
        KrakenCollectorError,
        ValueError,
    ) as exc:
        raise KrakenPipelineError(
            f"Kraken data collection failed: {exc}"
        ) from exc

    # Step 8: Handle absence of new closed candles.

    if not records:
        return _empty_result(
            normalized_pair,
            timeframe,
            age_limit,
        )

    # Step 9: Prepare collected records.

    validation_data = _prepare_validation_data(
        records
    )

    if validation_data.empty:
        return _empty_result(
            normalized_pair,
            timeframe,
            age_limit,
        )

    # Step 10: Validate collected candles.

    try:
        validation_result = run_candle_validation(
            validation_data,
            report_path=report_path,
            validation_time=checked_at,
            report_timestamp=datetime.now(timezone.utc),
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

    # Step 11: Extract and verify validation results.

    (
        accepted,
        rejected,
        validation_report,
        reports,
    ) = _extract_validation_result(
        validation_result
    )

    # Step 12: Determine pipeline status.

    status = (
        "PASS"
        if rejected.empty
        else "FAIL"
    )

    # Step 13: Return structured results.

    return {
        "status": status,
        "pair": normalized_pair,
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


if __name__ == "__main__":
    print(
        "Kraken collection and validation pipeline loaded."
    )
