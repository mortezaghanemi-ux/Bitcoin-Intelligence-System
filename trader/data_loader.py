"""Load and validate persisted Kraken candle history for trading."""

from __future__ import annotations

import json
import math

from pathlib import Path
from typing import Any

import pandas as pd

from collectors.kraken_ohlcv import TIMEFRAME_MINUTES


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "market"
    / "ohlcv"
)

TRADING_TIMEFRAMES = (
    "5m",
    "15m",
    "1h",
    "4h",
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

REQUIRED_METADATA = (
    "schema_version",
    "source",
    "pair",
    "timeframe",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class TradingDataLoaderError(RuntimeError):
    """Raised when persisted trading candle data is invalid."""


def _validate_pair(pair: str) -> str:
    """Validate and normalize a Kraken trading pair."""

    if (
        not isinstance(pair, str)
        or not pair
        or not pair.isalnum()
    ):
        raise ValueError("Invalid trading pair.")

    return pair.upper()


def _validate_timeframe(timeframe: str) -> str:
    """Validate a supported trading timeframe."""

    if (
        not isinstance(timeframe, str)
        or timeframe not in TRADING_TIMEFRAMES
        or timeframe not in TIMEFRAME_MINUTES
    ):
        raise ValueError(
            f"Unsupported trading timeframe: {timeframe}"
        )

    return timeframe


def _validate_data_directory(
    data_dir: str | Path | None,
) -> Path:
    """Resolve and validate the raw candle directory."""

    if data_dir is None:
        directory = DEFAULT_DATA_DIR
    else:
        try:
            directory = Path(data_dir).expanduser()
        except (TypeError, ValueError) as exc:
            raise TradingDataLoaderError(
                "Invalid candle data directory."
            ) from exc

    try:
        directory = directory.resolve()
    except (OSError, RuntimeError) as exc:
        raise TradingDataLoaderError(
            f"Cannot resolve candle data directory: {exc}"
        ) from exc

    if directory.exists() and not directory.is_dir():
        raise TradingDataLoaderError(
            "Candle data path is not a directory."
        )

    return directory


def _parse_utc_timestamp(
    value: Any,
    *,
    field: str,
    line_number: int,
) -> pd.Timestamp:
    """Parse a timestamp and require an explicit UTC timezone."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TradingDataLoaderError(
            f"Invalid {field} at JSONL line {line_number}."
        ) from exc

    if pd.isna(timestamp):
        raise TradingDataLoaderError(
            f"Missing {field} at JSONL line {line_number}."
        )

    if timestamp.tzinfo is None:
        raise TradingDataLoaderError(
            f"{field} must include a UTC timezone "
            f"at JSONL line {line_number}."
        )

    try:
        offset = timestamp.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise TradingDataLoaderError(
            f"Invalid timezone in {field} "
            f"at JSONL line {line_number}."
        ) from exc

    if offset is None or offset.total_seconds() != 0:
        raise TradingDataLoaderError(
            f"{field} must use UTC "
            f"at JSONL line {line_number}."
        )

    return timestamp.tz_convert("UTC")


def _validate_record(
    record: Any,
    *,
    pair: str,
    timeframe: str,
    line_number: int,
) -> dict[str, Any]:
    """Validate one persisted Kraken candle record."""

    if not isinstance(record, dict):
        raise TradingDataLoaderError(
            f"Invalid JSONL record at line {line_number}: "
            "record must be an object."
        )

    missing = [
        field
        for field in REQUIRED_METADATA
        if field not in record
    ]

    if missing:
        raise TradingDataLoaderError(
            f"Missing fields at JSONL line {line_number}: "
            + ", ".join(missing)
        )

    if (
        isinstance(record["schema_version"], bool)
        or record["schema_version"] != 1
    ):
        raise TradingDataLoaderError(
            f"Unsupported schema_version at line {line_number}."
        )

    if record["pair"] != pair:
        raise TradingDataLoaderError(
            f"Unexpected trading pair at line {line_number}."
        )

    if record["timeframe"] != timeframe:
        raise TradingDataLoaderError(
            f"Unexpected timeframe at line {line_number}."
        )

    if record["source"] != "kraken":
        raise TradingDataLoaderError(
            f"Unexpected data source at line {line_number}."
        )

    opened = _parse_utc_timestamp(
        record["open_time"],
        field="open_time",
        line_number=line_number,
    )

    closed = _parse_utc_timestamp(
        record["close_time"],
        field="close_time",
        line_number=line_number,
    )

    if closed <= opened:
        raise TradingDataLoaderError(
            f"Invalid candle time range at line {line_number}."
        )

    expected_seconds = TIMEFRAME_MINUTES[timeframe] * 60

    actual_seconds = (
        closed - opened
    ).total_seconds()

    if actual_seconds != expected_seconds:
        raise TradingDataLoaderError(
            f"Invalid candle duration at line {line_number}."
        )

    if opened.value % (
        expected_seconds * 1_000_000_000
    ) != 0:
        raise TradingDataLoaderError(
            f"Misaligned candle timestamp at line {line_number}."
        )

    return {
        column: record[column]
        for column in REQUIRED_COLUMNS
    }


def _read_jsonl_file(
    file_path: Path,
    *,
    pair: str,
    timeframe: str,
) -> pd.DataFrame:
    """Read and validate the complete JSONL candle history."""

    if not file_path.exists():
        raise TradingDataLoaderError(
            f"Historical candle file does not exist: {file_path.name}"
        )

    if not file_path.is_file():
        raise TradingDataLoaderError(
            f"Candle history path is not a file: {file_path.name}"
        )

    records = []

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                if not line.strip():
                    continue

                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TradingDataLoaderError(
                        f"Corrupted JSONL data in "
                        f"{file_path.name}, line {line_number}."
                    ) from exc

                validated = _validate_record(
                    raw_record,
                    pair=pair,
                    timeframe=timeframe,
                    line_number=line_number,
                )

                records.append(validated)

    except TradingDataLoaderError:
        raise
    except (OSError, UnicodeError) as exc:
        raise TradingDataLoaderError(
            f"Cannot read {file_path.name}: {exc}"
        ) from exc

    if not records:
        raise TradingDataLoaderError(
            f"No candle records found in {file_path.name}."
        )

    try:
        return pd.DataFrame(
            records,
            columns=REQUIRED_COLUMNS,
        )
    except (TypeError, ValueError) as exc:
        raise TradingDataLoaderError(
            f"Cannot build candle DataFrame for "
            f"{timeframe}: {exc}"
        ) from exc


def _validate_numeric_values(
    data: pd.DataFrame,
    *,
    timeframe: str,
) -> None:
    """Validate and convert OHLCV values to finite floats."""

    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    for column in numeric_columns:

        converted = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if converted.isna().any():
            bad_index = int(
                converted[converted.isna()].index[0]
            )

            raise TradingDataLoaderError(
                f"{timeframe} contains a missing or "
                f"non-numeric {column} at row {bad_index}."
            )

        try:
            values = converted.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TradingDataLoaderError(
                f"{timeframe} contains invalid {column} values."
            ) from exc

        finite_mask = values.map(math.isfinite)

        if not finite_mask.all():
            bad_index = int(
                finite_mask[~finite_mask].index[0]
            )

            raise TradingDataLoaderError(
                f"{timeframe} contains a non-finite "
                f"{column} at row {bad_index}."
            )

        if column != "volume":
            invalid = values <= 0
        else:
            invalid = values < 0

        if invalid.any():
            bad_index = int(
                invalid[invalid].index[0]
            )

            raise TradingDataLoaderError(
                f"{timeframe} contains an invalid "
                f"{column} at row {bad_index}."
            )

        # Important: downstream calculations must receive numbers,
        # not the original Kraken numeric strings.
        data[column] = values


def _prepare_timeframe_data(
    data: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    """Normalize and verify one timeframe history."""

    if not isinstance(data, pd.DataFrame):
        raise TradingDataLoaderError(
            f"{timeframe} history must be a DataFrame."
        )

    if data.empty:
        raise TradingDataLoaderError(
            f"{timeframe} history is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise TradingDataLoaderError(
            f"{timeframe} history is missing columns: "
            + ", ".join(missing)
        )

    result = data.loc[
        :,
        list(REQUIRED_COLUMNS),
    ].copy(deep=True)

    result["open_time"] = pd.to_datetime(
        result["open_time"],
        utc=True,
        errors="coerce",
    )

    result["close_time"] = pd.to_datetime(
        result["close_time"],
        utc=True,
        errors="coerce",
    )

    if (
        result["open_time"].isna().any()
        or result["close_time"].isna().any()
    ):
        raise TradingDataLoaderError(
            f"{timeframe} contains invalid timestamps."
        )

    if result["open_time"].duplicated().any():
        raise TradingDataLoaderError(
            f"{timeframe} contains duplicate candle timestamps."
        )

    if result["close_time"].duplicated().any():
        raise TradingDataLoaderError(
            f"{timeframe} contains duplicate candle close times."
        )

    result = result.sort_values(
        "open_time",
        kind="stable",
    ).reset_index(drop=True)

    _validate_numeric_values(
        result,
        timeframe=timeframe,
    )

    # OHLC validation is now numeric, not lexicographic.
    invalid_high = (
        result["high"]
        < result[["open", "low", "close"]].max(axis=1)
    )

    if invalid_high.any():
        bad_index = int(
            invalid_high[invalid_high].index[0]
        )

        raise TradingDataLoaderError(
            f"{timeframe} contains invalid high price "
            f"at row {bad_index}."
        )

    invalid_low = (
        result["low"]
        > result[["open", "high", "close"]].min(axis=1)
    )

    if invalid_low.any():
        bad_index = int(
            invalid_low[invalid_low].index[0]
        )

        raise TradingDataLoaderError(
            f"{timeframe} contains invalid low price "
            f"at row {bad_index}."
        )

    if not (
        result["source"].astype(str) == "kraken"
    ).all():
        raise TradingDataLoaderError(
            f"{timeframe} contains an unexpected data source."
        )

    return result


def load_timeframe_history(
    timeframe: str,
    *,
    pair: str = "XBTUSD",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load complete persisted history for one timeframe."""

    normalized_pair = _validate_pair(pair)

    normalized_timeframe = _validate_timeframe(
        timeframe
    )

    directory = _validate_data_directory(
        data_dir
    )

    file_path = directory / (
        f"kraken_{normalized_pair}_{normalized_timeframe}.jsonl"
    )

    data = _read_jsonl_file(
        file_path,
        pair=normalized_pair,
        timeframe=normalized_timeframe,
    )

    return _prepare_timeframe_data(
        data,
        timeframe=normalized_timeframe,
    )


def load_trading_candles(
    *,
    pair: str = "XBTUSD",
    data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load complete Kraken history for all trading timeframes."""

    normalized_pair = _validate_pair(pair)

    directory = _validate_data_directory(
        data_dir
    )

    if not directory.exists():
        raise TradingDataLoaderError(
            f"Raw candle data directory does not exist: {directory}"
        )

    result = {}

    for timeframe in TRADING_TIMEFRAMES:

        result[timeframe] = load_timeframe_history(
            timeframe,
            pair=normalized_pair,
            data_dir=directory,
        )

    return result


if __name__ == "__main__":

    try:
        histories = load_trading_candles()

        print(
            "Kraken trading history loaded successfully."
        )

        for timeframe in TRADING_TIMEFRAMES:
            print(
                f"{timeframe}: "
                f"{len(histories[timeframe])} candles"
            )

    except TradingDataLoaderError as exc:
        print(
            f"Trading data loading failed: {exc}"
        )