"""Kraken OHLCV collection and validation pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from collectors.kraken_ohlcv import (
    TIMEFRAME_MINUTES,
    KrakenCollectorError,
    collect_kraken_ohlcv,
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


def _resolve_max_age(
    timeframe: str,
    max_age_minutes: float | None,
) -> float:
    """Resolve the maximum allowed candle age."""

    if max_age_minutes is not None:

        if (
            isinstance(max_age_minutes, bool)
            or not isinstance(max_age_minutes, (int, float))
            or max_age_minutes <= 0
        ):
            raise ValueError(
                "max_age_minutes must be a positive number."
            )

        return float(max_age_minutes)

    interval = TIMEFRAME_MINUTES[timeframe]

    # Allow up to three candle intervals for collection delays.
    return float(interval * 3)


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
    Collect Kraken OHLCV data and validate newly collected candles.

    Raw provider records are preserved by the collector.

    Only validated records are returned as accepted data.
    """

    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    checked_at = (
        datetime.now(timezone.utc)
        if collection_time is None
        else _validate_utc_datetime(collection_time)
    )

    age_limit = _resolve_max_age(
        timeframe,
        max_age_minutes,
    )

    # Step 1: Collect and persist original Kraken data.

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

    # Step 2: Handle the absence of new closed candles.

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
            "message": (
                "No new closed candles were available."
            ),
        }

    # Step 3: Convert collected records into a DataFrame.

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

    # Step 4: Validate collected candles.

    try:

        validation_result = run_candle_validation(
            validation_data,
            report_path=report_path,
            validation_time=checked_at,
            max_age_minutes=age_limit,
        )

    except (ValidationPipelineError, ValueError, TypeError) as exc:

        raise KrakenPipelineError(
            f"Kraken candle validation failed: {exc}"
        ) from exc

    # Step 5: Return validated pipeline results.

    accepted = validation_result["accepted"]
    rejected = validation_result["rejected"]

    return {
        "status": (
            "PASS"
            if rejected.empty
            else "FAIL"
        ),
        "pair": pair.upper(),
        "timeframe": timeframe,
        "collected_records": len(records),
        "accepted": accepted,
        "rejected": rejected,
        "validation": validation_result["validation"],
        "reports": validation_result["reports"],
        "message": (
            "All collected candles passed validation."
            if rejected.empty
            else "Some collected candles were rejected."
        ),
    }