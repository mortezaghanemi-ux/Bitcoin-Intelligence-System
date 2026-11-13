"""Automated tests for Bitcoin data validation."""

import csv
from datetime import datetime, timezone

import pandas as pd
import pytest

from validation.candles import validate_candles
from validation.report import (
    build_report,
    save_report,
    ValidationReportError,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026, 11, 13, 12, 0, tzinfo=UTC
)


def valid_candle():
    return {
        "open_time": "2026-11-13T11:00:00+00:00",
        "close_time": "2026-11-13T11:05:00+00:00",
        "open": 95000,
        "high": 96000,
        "low": 94000,
        "close": 95500,
        "volume": 120.5,
        "source": "kraken",
    }


# --------------------------------------------------
# Candle validation tests
# --------------------------------------------------

def test_valid_candle():

    accepted, rejected, report = validate_candles(
        pd.DataFrame([valid_candle()]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 1
    assert len(rejected) == 0
    assert report["status"] == "PASS"


def test_invalid_high_price():

    candle = valid_candle()
    candle["high"] = 93000

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 0
    assert len(rejected) == 1

    assert "invalid_high_price" in rejected.iloc[0][
        "validation_errors"
    ]


def test_invalid_low_price():

    candle = valid_candle()
    candle["low"] = 96000

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 0
    assert len(rejected) == 1


def test_negative_volume():

    candle = valid_candle()
    candle["volume"] = -10

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 0
    assert len(rejected) == 1


def test_unclosed_candle():

    candle = valid_candle()
    candle["close_time"] = "2026-11-13T12:05:00+00:00"

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 0

    assert "candle_not_closed" in rejected.iloc[0][
        "validation_errors"
    ]


def test_non_utc_timestamp():

    candle = valid_candle()
    candle["open_time"] = "2026-11-13T12:00:00+03:30"

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 0
    assert len(rejected) == 1


def test_duplicate_candles():

    candle = valid_candle()

    data = pd.DataFrame([
        candle,
        candle,
    ])

    accepted, rejected, report = validate_candles(
        data,
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 1
    assert len(rejected) == 1

    assert "duplicate_candle" in rejected.iloc[0][
        "validation_errors"
    ]


def test_stale_candle():

    candle = valid_candle()

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
        max_age_minutes=30,
    )

    assert len(accepted) == 0

    assert "stale_candle" in rejected.iloc[0][
        "validation_errors"
    ]


def test_missing_columns():

    with pytest.raises(ValueError):

        validate_candles(
            pd.DataFrame([{"open": 95000}]),
            validation_time=CHECK_TIME,
        )


def test_non_utc_validation_time():

    with pytest.raises(ValueError):

        validate_candles(
            pd.DataFrame([valid_candle()]),
            validation_time=datetime(
                2026, 11, 13, 12, 0
            ),
        )


# --------------------------------------------------
# Numeric validation tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_invalid_numeric_values(column, bad_value):

    candle = valid_candle()
    candle[column] = bad_value

    accepted, rejected, report = validate_candles(
        pd.DataFrame([candle]),
        validation_time=CHECK_TIME,
    )

    assert len(accepted) == 0
    assert len(rejected) == 1

    assert (
        f"{column}_must_be_finite_numeric"
        in rejected.iloc[0]["validation_errors"]
    )


@pytest.mark.parametrize(
    "bad_age",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_invalid_max_age_minutes(bad_age):

    with pytest.raises(ValueError):

        validate_candles(
            pd.DataFrame([valid_candle()]),
            validation_time=CHECK_TIME,
            max_age_minutes=bad_age,
        )


# --------------------------------------------------
# Validation report tests
# --------------------------------------------------

def test_report_creation():

    report = build_report(
        dataset="candles",
        source="kraken",
        records_checked=100,
        valid_records=95,
        rejected_records=5,
        errors=["invalid_price"],
        validation_timestamp=CHECK_TIME,
    )

    assert report["status"] == "FAIL"
    assert report["records_checked"] == "100"
    assert report["valid_records"] == "95"


def test_warning_report():

    report = build_report(
        dataset="candles",
        source="kraken",
        records_checked=100,
        valid_records=100,
        rejected_records=0,
        warnings=["low_volume"],
        validation_timestamp=CHECK_TIME,
    )

    assert report["status"] == "WARNING"


def test_report_count_consistency():

    with pytest.raises(ValidationReportError):

        build_report(
            dataset="candles",
            source="kraken",
            records_checked=100,
            valid_records=90,
            rejected_records=5,
            validation_timestamp=CHECK_TIME,
        )


def test_report_history(tmp_path):

    path = tmp_path / "validation_report.csv"

    report = build_report(
        dataset="candles",
        source="kraken",
        records_checked=100,
        valid_records=100,
        rejected_records=0,
        validation_timestamp=CHECK_TIME,
    )

    save_report(report, path)

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["status"] == "PASS"


def test_duplicate_report_rejected(tmp_path):

    path = tmp_path / "validation_report.csv"

    report = build_report(
        dataset="candles",
        source="kraken",
        records_checked=10,
        valid_records=10,
        rejected_records=0,
        validation_timestamp=CHECK_TIME,
    )

    save_report(report, path)

    with pytest.raises(ValidationReportError):
        save_report(report, path)


def test_report_history_preserved(tmp_path):

    path = tmp_path / "validation_report.csv"

    for day in (13, 14):

        report = build_report(
            dataset="candles",
            source="kraken",
            records_checked=10,
            valid_records=10,
            rejected_records=0,
            validation_timestamp=datetime(
                2026, 11, day, 12, 0, tzinfo=UTC
            ),
        )

        save_report(report, path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(csv.DictReader(file))

    assert len(rows) == 2


def test_negative_age_limit():

    with pytest.raises(ValueError):

        validate_candles(
            pd.DataFrame([valid_candle()]),
            validation_time=CHECK_TIME,
            max_age_minutes=-1,
        )