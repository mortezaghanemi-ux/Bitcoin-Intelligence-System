"""Trader candle management and historical data protection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import math
import pandas as pd

from validation.candles import validate_candles


SUPPORTED_TIMEFRAMES = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
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


class CandleManagerError(RuntimeError):
    """Raised when candle management fails."""


def _validate_utc_datetime(value: datetime) -> None:
    """Ensure that a datetime is timezone-aware UTC."""

    if not isinstance(value, datetime):
        raise ValueError(
            "Timestamp must be a datetime instance."
        )

    if value.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware UTC."
        )

    try:
        if value.utcoffset() != timedelta(0):
            raise ValueError(
                "Timestamp must use UTC timezone."
            )
    except (TypeError, OverflowError) as exc:
        raise ValueError(
            "Invalid UTC timestamp."
        ) from exc


def timeframe_to_seconds(timeframe: str) -> int:
    """Return the duration of a supported timeframe."""

    if not isinstance(timeframe, str):
        raise ValueError(
            "timeframe must be a string."
        )

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    return SUPPORTED_TIMEFRAMES[timeframe]


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp only when it is explicitly UTC."""

    try:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return None

        if timestamp.tzinfo is None:
            return None

        if timestamp.utcoffset() != timedelta(0):
            return None

        return timestamp.tz_convert("UTC")

    except (TypeError, ValueError, OverflowError):
        return None


def _empty_candles() -> pd.DataFrame:
    """Return an empty candle DataFrame with standard columns."""

    return pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )


def _values_equal(first: Any, second: Any) -> bool:
    """Compare candle values safely."""

    first_timestamp = _parse_timestamp(first)
    second_timestamp = _parse_timestamp(second)

    if first_timestamp is not None or second_timestamp is not None:
        return (
            first_timestamp is not None
            and second_timestamp is not None
            and first_timestamp == second_timestamp
        )

    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second

    try:
        first_number = float(first)
        second_number = float(second)

        if (
            math.isfinite(first_number)
            and math.isfinite(second_number)
        ):
            return first_number == second_number

    except (TypeError, ValueError, OverflowError):
        pass

    try:
        if pd.isna(first) and pd.isna(second):
            return True
    except (TypeError, ValueError):
        pass

    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _record_key(record: dict) -> tuple | None:
    """Build a stable candle identity from UTC open time and source."""

    opened = _parse_timestamp(
        record.get("open_time")
    )

    source = record.get("source")

    if opened is None or pd.isna(source):
        return None

    source = str(source).strip()

    if not source:
        return None

    return opened, source


class CandleManager:
    """
    Manage validated trading candles.

    Features:
    - Supports 5m, 15m, 1h and 4h candles.
    - Accepts closed candles only.
    - Rejects invalid and duplicate records.
    - Detects missing candle intervals.
    - Preserves previously stored historical records.
    - Never overwrites an existing candle.
    """

    def __init__(self, timeframe: str):
        self.timeframe = timeframe

        self.interval_seconds = timeframe_to_seconds(
            timeframe
        )

        self.interval = pd.Timedelta(
            seconds=self.interval_seconds
        )

        self._candles = _empty_candles()

    @property
    def candles(self) -> pd.DataFrame:
        """Return a copy of stored candle records."""

        return self._candles.copy(deep=True)

    def clear(self) -> None:
        """Clear all in-memory candle records."""

        self._candles = _empty_candles()

    def _validate_timeframe(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Reject candles whose duration differs from the timeframe."""

        accepted_rows = []
        rejected_rows = []

        for _, row in data.iterrows():

            opened = row["open_time"]
            closed = row["close_time"]

            if (closed - opened) != self.interval:

                rejected = row.to_dict()

                rejected["validation_errors"] = (
                    "invalid_timeframe_duration"
                )

                rejected_rows.append(rejected)

            else:
                accepted_rows.append(
                    row.to_dict()
                )

        accepted = pd.DataFrame(
            accepted_rows,
            columns=REQUIRED_COLUMNS,
        )

        rejected = pd.DataFrame(
            rejected_rows,
        )

        if not rejected.empty:
            rejected = rejected[
                list(REQUIRED_COLUMNS)
                + ["validation_errors"]
            ]

        return accepted, rejected

    def detect_missing_candles(
        self,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Detect missing candle intervals separately for each source.

        Only gaps between existing records are reported.
        """

        source_data = (
            self._candles
            if data is None
            else data
        )

        if not isinstance(source_data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if source_data.empty:
            return pd.DataFrame(
                columns=[
                    "source",
                    "missing_open_time",
                ]
            )

        missing_rows = []

        for source, group in source_data.groupby(
            "source",
            sort=True,
        ):

            ordered = group.sort_values(
                "open_time"
            )

            timestamps = ordered[
                "open_time"
            ].tolist()

            for previous, current in zip(
                timestamps,
                timestamps[1:],
            ):

                expected = previous + self.interval

                while expected < current:

                    missing_rows.append({
                        "source": source,
                        "missing_open_time": expected,
                    })

                    expected += self.interval

        return pd.DataFrame(
            missing_rows,
            columns=[
                "source",
                "missing_open_time",
            ],
        )

    def _check_historical_conflicts(
        self,
        incoming: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate historical duplicates and conflicts before validation.

        Existing historical records always take priority.
        """

        if self._candles.empty:
            return (
                incoming.copy(deep=True),
                pd.DataFrame(
                    columns=list(REQUIRED_COLUMNS)
                    + ["validation_errors"]
                ),
            )

        existing_index = {}

        for _, row in self._candles.iterrows():

            record = row.to_dict()
            key = _record_key(record)

            if key is not None:
                existing_index[key] = record

        remaining_rows = []
        rejected_rows = []

        for _, row in incoming.iterrows():

            record = row.to_dict()
            key = _record_key(record)

            if key is None or key not in existing_index:
                remaining_rows.append(record)
                continue

            original = existing_index[key]

            identical = all(
                _values_equal(
                    original[column],
                    record[column],
                )
                for column in REQUIRED_COLUMNS
            )

            if identical:
                # Identical historical candle:
                # ignore it without creating a duplicate.
                continue

            rejected = record.copy()

            rejected["validation_errors"] = (
                "historical_candle_conflict"
            )

            rejected_rows.append(rejected)

        remaining = pd.DataFrame(
            remaining_rows,
            columns=REQUIRED_COLUMNS,
        )

        conflicts = pd.DataFrame(
            rejected_rows,
            columns=list(REQUIRED_COLUMNS)
            + ["validation_errors"],
        )

        return remaining, conflicts

    def ingest(
        self,
        data: pd.DataFrame,
        *,
        validation_time: datetime | None = None,
        max_age_minutes: float | None = None,
    ) -> dict:
        """
        Validate and ingest new candle records.

        Existing historical candles always take priority.
        Conflicting records cannot overwrite stored candles.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise CandleManagerError(
                "Cannot ingest an empty candle dataset."
            )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            raise CandleManagerError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        checked_at = (
            validation_time
            if validation_time is not None
            else datetime.now(timezone.utc)
        )

        _validate_utc_datetime(checked_at)

        checked_at = checked_at.astimezone(
            timezone.utc
        )

        incoming = data.loc[
            :,
            REQUIRED_COLUMNS,
        ].copy(deep=True)

        # Step 1: Protect historical records before OHLC validation.

        new_incoming, historical_conflicts = (
            self._check_historical_conflicts(
                incoming
            )
        )

        # Step 2: Validate only genuinely new records.

        if new_incoming.empty:

            accepted = pd.DataFrame(
                columns=REQUIRED_COLUMNS
            )

            rejected = pd.DataFrame(
                columns=list(REQUIRED_COLUMNS)
                + ["validation_errors"]
            )

            validation_report = {
                "dataset": "candles",
                "validation_timestamp": pd.Timestamp(
                    checked_at
                ).isoformat(),
                "records_checked": 0,
                "valid_records": 0,
                "rejected_records": 0,
                "status": "PASS",
                "errors": 0,
            }

        else:

            accepted, rejected, validation_report = (
                validate_candles(
                    new_incoming,
                    validation_time=checked_at,
                    max_age_minutes=max_age_minutes,
                    require_closed=True,
                )
            )

        # Step 3: Validate timeframe duration.

        timeframe_accepted, timeframe_rejected = (
            self._validate_timeframe(
                accepted
            )
        )

        if not timeframe_rejected.empty:

            rejected = pd.concat(
                [
                    rejected,
                    timeframe_rejected,
                ],
                ignore_index=True,
            )

        # Step 4: Preserve existing records.

        accepted_rows = []

        for _, row in timeframe_accepted.iterrows():

            accepted_rows.append(
                row.to_dict()
            )

        if accepted_rows:

            new_records = pd.DataFrame(
                accepted_rows,
                columns=REQUIRED_COLUMNS,
            )

            if self._candles.empty:

                updated = new_records.copy(
                    deep=True
                )

            else:

                updated = pd.concat(
                    [
                        self._candles,
                        new_records,
                    ],
                    ignore_index=True,
                )

            updated = updated.sort_values(
                ["source", "open_time"],
                kind="stable",
            ).reset_index(drop=True)

            self._candles = updated

        # Step 5: Combine all rejection categories.

        rejection_frames = []

        if not rejected.empty:
            rejection_frames.append(
                rejected
            )

        if not historical_conflicts.empty:
            rejection_frames.append(
                historical_conflicts
            )

        if rejection_frames:

            rejected = pd.concat(
                rejection_frames,
                ignore_index=True,
            )

        else:

            rejected = pd.DataFrame(
                columns=list(REQUIRED_COLUMNS)
                + ["validation_errors"]
            )

        # Step 6: Detect missing intervals.

        missing = self.detect_missing_candles()

        conflict_count = len(
            historical_conflicts
        )

        final_report = {
            **validation_report,
            "timeframe": self.timeframe,
            "accepted_records": len(accepted_rows),
            "rejected_records": len(rejected),
            "duplicate_conflicts": conflict_count,
            "missing_intervals": len(missing),
            "stored_records": len(self._candles),
            "status": (
                "PASS"
                if rejected.empty
                else "FAIL"
            ),
        }

        return {
            "accepted": pd.DataFrame(
                accepted_rows,
                columns=REQUIRED_COLUMNS,
            ),
            "rejected": rejected.reset_index(
                drop=True
            ),
            "missing": missing,
            "candles": self.candles,
            "report": final_report,
            "status": final_report["status"],
        }

    def get_candles(
        self,
        *,
        source: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return stored candles with optional source and row limits."""

        result = self._candles.copy(
            deep=True
        )

        if source is not None:

            result = result.loc[
                result["source"] == source
            ]

        result = result.sort_values(
            "open_time",
            kind="stable",
        )

        if limit is not None:

            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit <= 0
            ):
                raise ValueError(
                    "limit must be a positive integer."
                )

            result = result.tail(limit)

        return result.reset_index(
            drop=True
        )

    def get_latest_closed_candle(
        self,
        *,
        source: str | None = None,
    ) -> pd.Series | None:
        """Return the latest stored closed candle."""

        candles = self.get_candles(
            source=source,
        )

        if candles.empty:
            return None

        return candles.iloc[-1].copy()

    def get_missing_candles(self) -> pd.DataFrame:
        """Return currently detected missing intervals."""

        return self.detect_missing_candles()