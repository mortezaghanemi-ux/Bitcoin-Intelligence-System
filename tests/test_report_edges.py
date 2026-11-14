"""Edge case tests for validation reports."""

import csv

import pytest

from datetime import datetime, timezone

from validation.report import (
    REPORT_COLUMNS,
    ValidationReportError,
    build_report,
    save_report,
    record_report,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026, 11, 13, 12, 0,
    tzinfo=UTC,
)


def make_report(**overrides):
    data = {
        "dataset": "candles",
        "source": "kraken",
        "records_checked": 10,
        "valid_records": 10,
        "rejected_records": 0,
        "validation_timestamp": CHECK_TIME,
    }

    data.update(overrides)

    return build_report(**data)


# -----------------------------------------
# Dataset and source validation
# -----------------------------------------

@pytest.mark.parametrize(
    "field",
    ["dataset", "source"],
)
@pytest.mark.parametrize(
    "value",
    ["", "   ", None, 123],
)
def test_rejects_invalid_dataset_and_source(
    field,
    value,
):
    with pytest.raises(ValidationReportError):
        make_report(**{field: value})


# -----------------------------------------
# Record count validation
# -----------------------------------------

@pytest.mark.parametrize(
    "field",
    [
        "records_checked",
        "valid_records",
        "rejected_records",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-1, True, 1.5, "10"],
)
def test_rejects_invalid_record_counts(
    field,
    value,
):
    with pytest.raises(ValidationReportError):
        make_report(**{field: value})


# -----------------------------------------
# Timestamp validation
# -----------------------------------------

@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 11, 13, 12, 0),
        datetime(
            2026, 11, 13, 12, 0,
            tzinfo=timezone.utc,
        ).replace(
            tzinfo=timezone(
                __import__("datetime").timedelta(hours=3)
            )
        ),
    ],
)
def test_rejects_invalid_timestamps(timestamp):
    with pytest.raises(ValidationReportError):
        make_report(
            validation_timestamp=timestamp
        )


# -----------------------------------------
# Warning and error validation
# -----------------------------------------

@pytest.mark.parametrize(
    "field",
    ["warnings", "errors"],
)
@pytest.mark.parametrize(
    "value",
    [
        "not a list",
        b"invalid",
        ["", "valid"],
        ["valid", None],
        [123],
    ],
)
def test_rejects_invalid_messages(
    field,
    value,
):
    with pytest.raises(ValidationReportError):
        make_report(**{field: value})


def test_none_messages_are_allowed():
    report = make_report(
        warnings=None,
        errors=None,
    )

    assert report["warnings"] == "[]"
    assert report["errors"] == "[]"


# -----------------------------------------
# Report schema validation
# -----------------------------------------

def test_rejects_non_dictionary_report(tmp_path):
    with pytest.raises(ValidationReportError):
        save_report(
            [],
            tmp_path / "report.csv",
        )


def test_rejects_incomplete_report_schema(tmp_path):
    report = make_report()
    report.pop("source")

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


def test_rejects_invalid_report_status(tmp_path):
    report = make_report()
    report["status"] = "UNKNOWN"

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


# -----------------------------------------
# Existing report file validation
# -----------------------------------------

def test_rejects_incompatible_existing_schema(tmp_path):
    path = tmp_path / "report.csv"

    path.write_text(
        "wrong,header\nvalue,value\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationReportError,
        match="incompatible schema",
    ):
        save_report(
            make_report(),
            path,
        )


def test_creates_missing_parent_directories(tmp_path):
    path = (
        tmp_path
        / "nested"
        / "reports"
        / "validation.csv"
    )

    save_report(
        make_report(),
        path,
    )

    assert path.is_file()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1


# -----------------------------------------
# Report recording
# -----------------------------------------

def test_record_report_creates_report(tmp_path):
    path = tmp_path / "reports" / "validation.csv"

    result = record_report(
        output_path=path,
        dataset="candles",
        source="kraken",
        records_checked=10,
        valid_records=9,
        rejected_records=1,
        errors=["invalid_price"],
        validation_timestamp=CHECK_TIME,
    )

    assert result["status"] == "FAIL"
    assert path.is_file()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        assert tuple(reader.fieldnames) == REPORT_COLUMNS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["status"] == "FAIL"