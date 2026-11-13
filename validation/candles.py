"""Bitcoin OHLCV candle validation."""

from __future__ import annotations

import math

from datetime import datetime, timedelta, timezone

import pandas as pd


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

NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
)


def _is_utc_datetime(value):

    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def _parse_utc_timestamp(value):

    try:

        timestamp = pd.Timestamp(value)

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() != timedelta(0)
        ):
            return None

        return timestamp.tz_convert("UTC")

    except (TypeError, ValueError, OverflowError):

        return None


def validate_candles(
    data: pd.DataFrame,
    validation_time: datetime | None = None,
    max_age_minutes: float | None = None,
    require_closed: bool = True,
):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    checked_at = (
        validation_time
        or datetime.now(timezone.utc)
    )

    if not _is_utc_datetime(checked_at):
        raise ValueError(
            "validation_time must be a timezone-aware UTC datetime"
        )

    if max_age_minutes is not None:

        if (
            isinstance(max_age_minutes, bool)
            or not isinstance(max_age_minutes, (int, float))
            or not math.isfinite(max_age_minutes)
            or max_age_minutes < 0
        ):
            raise ValueError(
                "max_age_minutes must be a finite non-negative number"
            )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    checked_at_ts = pd.Timestamp(
        checked_at
    ).tz_convert("UTC")

    valid = data.copy(
        deep=True
    ).reset_index(drop=True)

    errors = [
        []
        for _ in range(len(valid))
    ]

    # Timestamp validation

    for column in ("open_time", "close_time"):

        parsed = [
            _parse_utc_timestamp(value)
            for value in valid[column]
        ]

        valid[column] = parsed

        for index, timestamp in enumerate(parsed):

            if timestamp is None:
                errors[index].append(
                    f"{column}_must_be_valid_utc"
                )

    # Numeric validation

    for column in NUMERIC_COLUMNS:

        original_values = valid[column].tolist()

        converted = pd.to_numeric(
            valid[column],
            errors="coerce",
        )

        normalized = []

        for index, (original, value) in enumerate(
            zip(original_values, converted)
        ):

            if isinstance(original, bool) or pd.isna(value):

                errors[index].append(
                    f"{column}_must_be_finite_numeric"
                )

                normalized.append(float("nan"))
                continue

            try:
                number = float(value)

            except (TypeError, ValueError, OverflowError):

                errors[index].append(
                    f"{column}_must_be_finite_numeric"
                )

                normalized.append(float("nan"))
                continue

            if not math.isfinite(number):

                errors[index].append(
                    f"{column}_must_be_finite_numeric"
                )

                normalized.append(float("nan"))

            else:
                normalized.append(number)

        valid[column] = normalized

    # OHLC validation

    for index, row in valid.iterrows():

        source = row["source"]

        if (
            pd.isna(source)
            or not str(source).strip()
        ):
            errors[index].append("source_required")

        prices = [
            row[column]
            for column in PRICE_COLUMNS
        ]

        if all(pd.notna(value) for value in prices):

            if any(value <= 0 for value in prices):
                errors[index].append(
                    "prices_must_be_positive"
                )

            if row["high"] < max(
                row["open"],
                row["close"],
                row["low"],
            ):
                errors[index].append(
                    "invalid_high_price"
                )

            if row["low"] > min(
                row["open"],
                row["close"],
                row["high"],
            ):
                errors[index].append(
                    "invalid_low_price"
                )

        if (
            pd.notna(row["volume"])
            and row["volume"] < 0
        ):
            errors[index].append(
                "volume_must_be_non_negative"
            )

        opened = row["open_time"]
        closed = row["close_time"]

        if opened is not None and closed is not None:

            if closed <= opened:
                errors[index].append(
                    "close_time_must_follow_open_time"
                )

            if require_closed and closed > checked_at_ts:
                errors[index].append(
                    "candle_not_closed"
                )

            if max_age_minutes is not None:

                age_minutes = (
                    checked_at_ts - closed
                ).total_seconds() / 60

                if age_minutes > max_age_minutes:
                    errors[index].append(
                        "stale_candle"
                    )

    # Duplicate detection

    duplicate_mask = valid.duplicated(
        subset=["open_time", "source"],
        keep="first",
    )

    for index, is_duplicate in enumerate(duplicate_mask):

        if is_duplicate:
            errors[index].append(
                "duplicate_candle"
            )

    # Separate accepted and rejected records

    error_text = [
        ";".join(dict.fromkeys(row_errors))
        for row_errors in errors
    ]

    rejected_mask = [
        bool(message)
        for message in error_text
    ]

    valid["validation_errors"] = error_text

    accepted = (
        valid.loc[
            [not rejected for rejected in rejected_mask]
        ]
        .drop(columns=["validation_errors"])
        .reset_index(drop=True)
    )

    rejected = (
        valid.loc[rejected_mask]
        .reset_index(drop=True)
    )

    # Validation report

    report = {
        "dataset": "candles",
        "validation_timestamp": checked_at_ts.isoformat(),
        "records_checked": len(valid),
        "valid_records": len(accepted),
        "rejected_records": len(rejected),
        "status": (
            "PASS"
            if rejected.empty
            else "FAIL"
        ),
        "errors": sum(
            len(row_errors)
            for row_errors in errors
        ),
    }

    return accepted, rejected, report