"""Trader candle management and historical data protection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

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

            errors = []

            if (
                closed - opened
            ) != self.interval:
                errors.append(
                    "invalid_timeframe_duration"
                )

            if errors:
                rejected = row.to_dict()

                rejected["validation_errors"] = ";".join(
                    errors
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

        # Validate candle records.

        accepted, rejected, validation_report = (
            validate_candles(
                incoming,
                validation_time=checked_at,
                max_age_minutes=max_age_minutes,
                require_closed=True,
            )
        )

        # Validate timeframe duration.

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

        # Preserve existing records.

        existing = self._candles.copy(
            deep=True
        )

        accepted_rows = []
        duplicate_rows = []

        existing_index = {}

        for _, row in existing.iterrows():

            key = (
                row["open_time"],
                row["source"],
            )

            existing_index[key] = row.to_dict()

        for _, row in timeframe_accepted.iterrows():

            record = row.to_dict()

            key = (
                record["open_time"],
                record["source"],
            )

            if key in existing_index:

                original = existing_index[key]

                same_record = all(
                    original[column] == record[column]
                    for column in REQUIRED_COLUMNS
                )

                if not same_record:

                    duplicate = record.copy()

                    duplicate["validation_errors"] = (
                        "historical_candle_conflict"
                    )

                    duplicate_rows.append(
                        duplicate
                    )

                continue

            accepted_rows.append(record)

            existing_index[key] = record

        # Add newly accepted records without modifying history.

        if accepted_rows:

            new_records = pd.DataFrame(
                accepted_rows,
                columns=REQUIRED_COLUMNS,
            )

            updated = pd.concat(
                [
                    existing,
                    new_records,
                ],
                ignore_index=True,
            )

            updated = updated.sort_values(
                ["source", "open_time"],
                kind="stable",
            ).reset_index(drop=True)

            self._candles = updated

        # Add historical conflicts to rejected records.

        if duplicate_rows:

            duplicate_frame = pd.DataFrame(
                duplicate_rows
            )

            rejected = pd.concat(
                [
                    rejected,
                    duplicate_frame,
                ],
                ignore_index=True,
            )

        # Detect missing intervals after ingestion.

        missing = self.detect_missing_candles()

        final_report = {
            **validation_report,
            "timeframe": self.timeframe,
            "accepted_records": len(accepted_rows),
            "rejected_records": len(rejected),
            "duplicate_conflicts": len(duplicate_rows),
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