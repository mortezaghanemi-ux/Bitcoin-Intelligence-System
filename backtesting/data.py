"""Historical OHLCV data loader for the backtesting module.

Reads append-only JSONL candle files from the raw Kraken data directory,
validates historical records, and prepares DataFrames for BacktestingEngine.

This module never modifies raw source files.
"""

from __future__ import annotations

import json
import math

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from validation.candles import validate_candles


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "market"
    / "ohlcv"
)

SUPPORTED_TIMEFRAMES = (
    "5m",
    "15m",
    "1h",
    "4h",
)

TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

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

OUTPUT_COLUMNS = list(REQUIRED_COLUMNS)


class BacktestingDataError(RuntimeError):
    """Raised when historical backtesting data cannot be loaded safely."""


def _validate_pair(pair: str) -> str:
    """Validate and normalize a trading pair."""

    if (
        not isinstance(pair, str)
        or not pair.strip()
        or not pair.strip().isalnum()
    ):
        raise ValueError("pair must contain only letters and numbers.")

    return pair.strip().upper()


def _parse_utc_datetime(
    value: datetime | str | None,
    name: str,
) -> datetime | None:
    """Parse a timezone-aware UTC datetime."""

    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a valid UTC datetime."
        ) from exc

    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(
            f"{name} must be timezone-aware UTC."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(
            f"{name} must use UTC."
        )

    return timestamp.tz_convert("UTC").to_pydatetime()


def _resolve_data_file(
    timeframe: str,
    pair: str,
    data_dir: Path,
) -> Path:
    """Resolve the historical JSONL file for one timeframe."""

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    filename = f"kraken_{pair}_{timeframe}.jsonl"

    return data_dir / filename


def _read_jsonl(
    path: Path,
    timeframe: str,
) -> list[dict[str, Any]]:
    """Read historical JSONL records without modifying the source."""

    if not path.exists():
        raise BacktestingDataError(
            f"Historical data file does not exist: {path}"
        )

    if not path.is_file():
        raise BacktestingDataError(
            f"Historical data path is not a file: {path}"
        )

    records = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise BacktestingDataError(
                        f"Invalid JSON in {path.name} "
                        f"at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise BacktestingDataError(
                        f"Invalid record structure in {path.name} "
                        f"at line {line_number}."
                    )

                records.append(record)

    except OSError as exc:
        raise BacktestingDataError(
            f"Cannot read {path.name}: {exc}"
        ) from exc

    if not records:
        raise BacktestingDataError(
            f"No historical records found in {path.name}."
        )

    return records


def _validate_record_metadata(
    records: list[dict[str, Any]],
    timeframe: str,
    pair: str,
) -> None:
    """Ensure every record belongs to the requested dataset."""

    for index, record in enumerate(records, start=1):

        record_pair = record.get("pair")
        record_timeframe = record.get("timeframe")
        record_source = record.get("source")

        if record_pair != pair:
            raise BacktestingDataError(
                f"{timeframe} record {index} has an unexpected pair."
            )

        if record_timeframe != timeframe:
            raise BacktestingDataError(
                f"{timeframe} record {index} has an unexpected timeframe."
            )

        if (
            not isinstance(record_source, str)
            or not record_source.strip()
        ):
            raise BacktestingDataError(
                f"{timeframe} record {index} has an invalid source."
            )


def _build_dataframe(
    records: list[dict[str, Any]],
    timeframe: str,
) -> pd.DataFrame:
    """Build a standardized OHLCV DataFrame."""

    rows = []

    for index, record in enumerate(records, start=1):

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in record
        ]

        if missing:
            raise BacktestingDataError(
                f"{timeframe} record {index} is missing fields: "
                + ", ".join(missing)
            )

        rows.append({
            column: record[column]
            for column in REQUIRED_COLUMNS
        })

    dataframe = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if dataframe.empty:
        raise BacktestingDataError(
            f"{timeframe} dataset is empty."
        )

    return dataframe


def _validate_historical_candles(
    dataframe: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """Validate historical candles without applying live-data freshness rules."""

    parsed_close_times = []

    for index, value in enumerate(
        dataframe["close_time"],
        start=1,
    ):
        try:
            timestamp = pd.Timestamp(value)

        except (TypeError, ValueError, OverflowError) as exc:
            raise BacktestingDataError(
                f"{timeframe} contains an invalid close_time "
                f"at row {index}."
            ) from exc

        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise BacktestingDataError(
                f"{timeframe} close_time must be timezone-aware UTC."
            )

        if timestamp.utcoffset() != timedelta(0):
            raise BacktestingDataError(
                f"{timeframe} close_time must use UTC."
            )

        parsed_close_times.append(
            timestamp.tz_convert("UTC")
        )

    validation_time = max(parsed_close_times).to_pydatetime()

    try:
        accepted, rejected, report = validate_candles(
            dataframe,
            validation_time=validation_time,
            max_age_minutes=None,
            require_closed=True,
        )

    except Exception as exc:
        raise BacktestingDataError(
            f"{timeframe} candle validation failed: {exc}"
        ) from exc

    if not rejected.empty:
        error_messages = (
            rejected["validation_errors"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise BacktestingDataError(
            f"{timeframe} contains rejected historical candles. "
            f"Rejected: {len(rejected)}. "
            f"Errors: {error_messages[:10]}"
        )

    if accepted.empty:
        raise BacktestingDataError(
            f"{timeframe} has no valid historical candles."
        )

    if report["status"] != "PASS":
        raise BacktestingDataError(
            f"{timeframe} validation did not pass."
        )

    accepted = accepted.loc[
        :,
        OUTPUT_COLUMNS,
    ].copy()

    accepted = accepted.sort_values(
        "open_time",
        kind="stable",
    ).reset_index(drop=True)

    interval = pd.Timedelta(
        minutes=TIMEFRAME_MINUTES[timeframe]
    )

    if len(accepted) > 1:

        differences = (
            accepted["open_time"]
            .diff()
            .iloc[1:]
        )

        if not differences.eq(interval).all():
            raise BacktestingDataError(
                f"{timeframe} contains missing or overlapping "
                "historical candle intervals."
            )

    return accepted


def load_timeframe_data(
    timeframe: str,
    *,
    pair: str = "XBTUSD",
    data_dir: str | Path | None = None,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
) -> pd.DataFrame:
    """Load and validate historical candles for one timeframe.

    Args:
        timeframe: One of 5m, 15m, 1h or 4h.
        pair: Kraken trading pair.
        data_dir: Directory containing raw JSONL candle files.
        start_time: Optional inclusive UTC candle open-time boundary.
        end_time: Optional inclusive UTC candle close-time boundary.

    Returns:
        A validated DataFrame containing standardized OHLCV columns.
    """

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    normalized_pair = _validate_pair(pair)

    start = _parse_utc_datetime(
        start_time,
        "start_time",
    )

    end = _parse_utc_datetime(
        end_time,
        "end_time",
    )

    if start is not None and end is not None and start >= end:
        raise ValueError(
            "start_time must be earlier than end_time."
        )

    directory = Path(
        data_dir
        if data_dir is not None
        else DEFAULT_DATA_DIR
    )

    path = _resolve_data_file(
        timeframe,
        normalized_pair,
        directory,
    )

    records = _read_jsonl(
        path,
        timeframe,
    )

    _validate_record_metadata(
        records,
        timeframe,
        normalized_pair,
    )

    dataframe = _build_dataframe(
        records,
        timeframe,
    )

    dataframe = _validate_historical_candles(
        dataframe,
        timeframe,
    )

    if start is not None:
        dataframe = dataframe.loc[
            dataframe["open_time"] >= pd.Timestamp(start)
        ]

    if end is not None:
        dataframe = dataframe.loc[
            dataframe["close_time"] <= pd.Timestamp(end)
        ]

    dataframe = dataframe.reset_index(drop=True)

    if dataframe.empty:
        raise BacktestingDataError(
            f"No {timeframe} candles remain after applying "
            "the requested time range."
        )

    return dataframe


def load_backtest_data(
    *,
    pair: str = "XBTUSD",
    data_dir: str | Path | None = None,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all four required trading timeframes for backtesting.

    Returns:
        {
            "5m": DataFrame,
            "15m": DataFrame,
            "1h": DataFrame,
            "4h": DataFrame,
        }

    The returned dictionary is compatible with BacktestingEngine.run().
    """

    normalized_pair = _validate_pair(pair)

    start = _parse_utc_datetime(
        start_time,
        "start_time",
    )

    end = _parse_utc_datetime(
        end_time,
        "end_time",
    )

    if start is not None and end is not None and start >= end:
        raise ValueError(
            "start_time must be earlier than end_time."
        )

    datasets = {}

    for timeframe in SUPPORTED_TIMEFRAMES:

        datasets[timeframe] = load_timeframe_data(
            timeframe,
            pair=normalized_pair,
            data_dir=data_dir,
            start_time=start,
            end_time=end,
        )

    return datasets


def get_backtest_data_summary(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Return a compact summary of loaded backtesting datasets."""

    if not isinstance(datasets, dict):
        raise TypeError(
            "datasets must be a dictionary."
        )

    if set(datasets) != set(SUPPORTED_TIMEFRAMES):
        raise BacktestingDataError(
            "Exactly 5m, 15m, 1h and 4h datasets are required."
        )

    summary = {}

    for timeframe in SUPPORTED_TIMEFRAMES:

        dataframe = datasets[timeframe]

        if not isinstance(dataframe, pd.DataFrame):
            raise BacktestingDataError(
                f"{timeframe} must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise BacktestingDataError(
                f"{timeframe} dataset is empty."
            )

        summary[timeframe] = {
            "records": len(dataframe),
            "start_time": (
                pd.Timestamp(
                    dataframe["open_time"].iloc[0]
                ).isoformat()
            ),
            "end_time": (
                pd.Timestamp(
                    dataframe["close_time"].iloc[-1]
                ).isoformat()
            ),
            "source": sorted(
                dataframe["source"]
                .astype(str)
                .unique()
                .tolist()
            ),
        }

    return {
        "status": "READY",
        "timeframes": summary,
    }