"""Bitcoin OHLCV candle validation."""

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


def validate_candles(
    data,
    validation_time=None,
    max_age_minutes=None,
    require_closed=True,
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    checked_at = validation_time or datetime.now(timezone.utc)

    if (
        checked_at.tzinfo is None
        or checked_at.utcoffset() != timedelta(0)
    ):
        raise ValueError("validation_time must be UTC")

    if max_age_minutes is not None and max_age_minutes < 0:
        raise ValueError("max_age_minutes cannot be negative")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    valid = data.copy(deep=True).reset_index(drop=True)

    errors = [[] for _ in range(len(valid))]

    # Timestamp validation

    for column in ("open_time", "close_time"):

        parsed = []

        for value in valid[column]:

            try:
                ts = pd.Timestamp(value)

                if (
                    ts.tzinfo is None
                    or ts.utcoffset() != timedelta(0)
                ):
                    parsed.append(None)

                else:
                    parsed.append(ts.tz_convert("UTC"))

            except (TypeError, ValueError, OverflowError):

                parsed.append(None)

        valid[column] = parsed

        for i, timestamp in enumerate(parsed):

            if timestamp is None:
                errors[i].append(
                    f"{column}_must_be_valid_utc"
                )

    # Numeric validation

    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    for column in numeric_columns:

        valid[column] = pd.to_numeric(
            valid[column],
            errors="coerce",
        )

        for i, value in enumerate(valid[column]):

            if pd.isna(value):
                errors[i].append(
                    f"{column}_must_be_numeric"
                )

    # OHLC validation

    for i, row in valid.iterrows():

        if pd.isna(row["source"]) or not str(
            row["source"]
        ).strip():

            errors[i].append("source_required")

        prices = [
            row["open"],
            row["high"],
            row["low"],
            row["close"],
        ]

        if all(pd.notna(x) for x in prices):

            if any(x <= 0 for x in prices):
                errors[i].append(
                    "prices_must_be_positive"
                )

            if row["high"] < max(
                row["open"],
                row["close"],
                row["low"],
            ):
                errors[i].append("invalid_high_price")

            if row["low"] > min(
                row["open"],
                row["close"],
                row["high"],
            ):
                errors[i].append("invalid_low_price")

        if (
            pd.notna(row["volume"])
            and row["volume"] < 0
        ):
            errors[i].append(
                "volume_must_be_non_negative"
            )

        opened = row["open_time"]
        closed = row["close_time"]

        if (
            opened is not None
            and closed is not None
            and not pd.isna(opened)
            and not pd.isna(closed)
        ):

            if closed <= opened:
                errors[i].append(
                    "close_time_must_follow_open_time"
                )

            if (
                require_closed
                and closed.to_pydatetime() > checked_at
            ):
                errors[i].append("candle_not_closed")

            if max_age_minutes is not None:

                age = (
                    checked_at - closed.to_pydatetime()
                ).total_seconds() / 60

                if age > max_age_minutes:
                    errors[i].append("stale_candle")

    # Duplicate detection

    duplicate_mask = valid.duplicated(
        subset=["open_time", "source"],
        keep="first",
    )

    for i, is_duplicate in enumerate(duplicate_mask):

        if is_duplicate:
            errors[i].append("duplicate_candle")

    # Separate accepted and rejected records

    error_text = [
        ";".join(dict.fromkeys(row_errors))
        for row_errors in errors
    ]

    rejected_mask = [
        bool(error)
        for error in error_text
    ]

    valid["validation_errors"] = error_text

    accepted = (
        valid.loc[
            [not x for x in rejected_mask]
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
        "validation_timestamp": checked_at.isoformat(),
        "records_checked": len(valid),
        "valid_records": len(accepted),
        "rejected_records": len(rejected),
        "status": (
            "PASS"
            if len(rejected) == 0
            else "FAIL"
        ),
        "errors": sum(len(x) for x in errors),
    }

    return accepted, rejected, report