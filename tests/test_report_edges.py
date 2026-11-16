"""Edge case tests for validation reports."""

import csv
import json

from datetime import datetime, timezone, timedelta

import pytest

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


def write_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REPORT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


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


def test_dataset_and_source_are_normalized():
    report = make_report(
        dataset="  candles  ",
        source="  kraken  ",
    )

    assert report["dataset"] == "candles"
    assert report["source"] == "kraken"


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


@pytest.mark.parametrize(
    "counts",
    [
        (10, 9, 0),
        (10, 10, 1),
        (0, 1, 0),
        (0, 0, 1),
    ],
)
def test_rejects_inconsistent_record_counts(counts):
    checked, valid, rejected = counts

    with pytest.raises(ValidationReportError):
        make_report(
            records_checked=checked,
            valid_records=valid,
            rejected_records=rejected,
        )


def test_accepts_zero_record_counts():
    report = make_report(
        records_checked=0,
        valid_records=0,
        rejected_records=0,
    )

    assert report["records_checked"] == "0"
    assert report["valid_records"] == "0"
    assert report["rejected_records"] == "0"


# -----------------------------------------
# Timestamp validation
# -----------------------------------------

@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 11, 13, 12, 0),
        datetime(
            2026, 11, 13, 12, 0,
            tzinfo=timezone(
                timedelta(hours=3)
            ),
        ),
        "2026-11-13T12:00:00+00:00",
        123,
    ],
)
def test_rejects_invalid_timestamps(timestamp):
    with pytest.raises(ValidationReportError):
        make_report(
            validation_timestamp=timestamp
        )


def test_accepts_utc_timestamp():
    report = make_report(
        validation_timestamp=CHECK_TIME,
    )

    assert report["validation_timestamp"] == (
        "2026-11-13T12:00:00+00:00"
    )


def test_default_timestamp_is_utc():
    report = make_report(
        validation_timestamp=None,
    )

    timestamp = datetime.fromisoformat(
        report["validation_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


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
        {"message": "invalid"},
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


def test_warning_messages_create_warning_status():
    report = make_report(
        warnings=["low_volume"],
    )

    assert report["status"] == "WARNING"

    assert json.loads(
        report["warnings"]
    ) == ["low_volume"]


def test_error_messages_create_fail_status():
    report = make_report(
        errors=["invalid_price"],
    )

    assert report["status"] == "FAIL"


def test_rejected_records_create_fail_status():
    report = make_report(
        records_checked=10,
        valid_records=9,
        rejected_records=1,
    )

    assert report["status"] == "FAIL"


def test_generator_messages_are_supported():
    report = make_report(
        warnings=(
            message
            for message in ["warning_one", "warning_two"]
        ),
    )

    assert json.loads(
        report["warnings"]
    ) == ["warning_one", "warning_two"]


def test_empty_message_list_creates_pass_status():
    report = make_report(
        warnings=[],
        errors=[],
    )

    assert report["status"] == "PASS"


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


def test_rejects_extra_report_field(tmp_path):
    report = make_report()
    report["unexpected"] = "value"

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


@pytest.mark.parametrize(
    "status",
    ["UNKNOWN", "", "pass", "FAILURE", None],
)
def test_rejects_invalid_report_status(
    tmp_path,
    status,
):
    report = make_report()
    report["status"] = status

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


@pytest.mark.parametrize(
    "field",
    [
        "report_id",
        "dataset",
        "source",
        "validation_timestamp",
        "warnings",
        "errors",
    ],
)
@pytest.mark.parametrize(
    "value",
    ["", "   ", None, 123],
)
def test_rejects_invalid_required_string_fields(
    tmp_path,
    field,
    value,
):
    report = make_report()
    report[field] = value

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


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
    ["-1", "1.5", "abc", "", " 1", "1 "],
)
def test_rejects_invalid_serialized_counts(
    tmp_path,
    field,
    value,
):
    report = make_report()
    report[field] = value

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


def test_rejects_inconsistent_serialized_counts(tmp_path):
    report = make_report()
    report["valid_records"] = "9"

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


def test_rejects_report_id_mismatch(tmp_path):
    report = make_report()
    report["report_id"] = "wrong-id"

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


def test_rejects_non_normalized_timestamp(tmp_path):
    report = make_report()

    report["validation_timestamp"] = (
        "2026-11-13T12:00:00Z"
    )

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


def test_rejects_non_utc_report_timestamp(tmp_path):
    report = make_report()

    report["validation_timestamp"] = (
        "2026-11-13T15:00:00+03:00"
    )

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


def test_rejects_invalid_json_messages(tmp_path):
    report = make_report()
    report["warnings"] = "invalid-json"

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


@pytest.mark.parametrize(
    "field",
    ["warnings", "errors"],
)
@pytest.mark.parametrize(
    "value",
    [
        '"string"',
        '{"message": "value"}',
        "123",
        "null",
    ],
)
def test_rejects_non_list_json_messages(
    tmp_path,
    field,
    value,
):
    report = make_report()
    report[field] = value

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


@pytest.mark.parametrize(
    "field",
    ["warnings", "errors"],
)
@pytest.mark.parametrize(
    "value",
    [
        '[""]',
        '["valid", null]',
        '[123]',
    ],
)
def test_rejects_invalid_json_message_items(
    tmp_path,
    field,
    value,
):
    report = make_report()
    report[field] = value

    with pytest.raises(ValidationReportError):
        save_report(
            report,
            tmp_path / "report.csv",
        )


@pytest.mark.parametrize(
    "contents",
    [
        {
            "warnings": '["warning"]',
            "errors": "[]",
            "status": "PASS",
        },
        {
            "warnings": "[]",
            "errors": '["error"]',
            "status": "WARNING",
        },
        {
            "warnings": "[]",
            "errors": "[]",
            "status": "FAIL",
        },
    ],
)
def test_rejects_inconsistent_report_status(
    tmp_path,
    contents,
):
    report = make_report()

    report.update(contents)

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

    rows = read_csv(path)

    assert len(rows) == 1


def test_rejects_duplicate_report_id(tmp_path):
    path = tmp_path / "report.csv"

    report = make_report()

    save_report(report, path)

    with pytest.raises(
        ValidationReportError,
        match="duplicate validation report",
    ):
        save_report(report, path)

    assert len(read_csv(path)) == 1


def test_rejects_duplicate_ids_in_existing_file(tmp_path):
    path = tmp_path / "report.csv"

    report = make_report()

    write_csv(
        path,
        [report, report],
    )

    with pytest.raises(
        ValidationReportError,
        match="duplicate report IDs",
    ):
        save_report(
            make_report(
                source="coingecko",
            ),
            path,
        )


def test_rejects_invalid_existing_report_row(tmp_path):
    path = tmp_path / "report.csv"

    report = make_report()
    report["status"] = "UNKNOWN"

    write_csv(
        path,
        [report],
    )

    with pytest.raises(ValidationReportError):
        save_report(
            make_report(
                source="coingecko",
            ),
            path,
        )


def test_rejects_existing_row_with_extra_column(tmp_path):
    path = tmp_path / "report.csv"

    path.write_text(
        ",".join(REPORT_COLUMNS)
        + ",unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationReportError):
        save_report(
            make_report(),
            path,
        )


def test_appends_without_overwriting_history(tmp_path):
    path = tmp_path / "report.csv"

    first = make_report()

    second = make_report(
        source="coingecko",
    )

    save_report(first, path)
    save_report(second, path)

    rows = read_csv(path)

    assert len(rows) == 2
    assert rows[0]["source"] == "kraken"
    assert rows[1]["source"] == "coingecko"


# -----------------------------------------
# Atomic file replacement
# -----------------------------------------

def test_atomic_replace_failure_preserves_original_file(
    tmp_path,
    monkeypatch,
):
    import validation.report as report_module

    path = tmp_path / "report.csv"

    original = make_report()

    save_report(original, path)

    original_content = path.read_text(
        encoding="utf-8"
    )

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        report_module.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="simulated replace failure",
    ):
        save_report(
            make_report(
                source="coingecko",
            ),
            path,
        )

    assert path.read_text(
        encoding="utf-8"
    ) == original_content

    temporary_files = list(
        tmp_path.glob("*.tmp")
    )

    assert temporary_files == []


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


def test_record_report_returns_normalized_report(tmp_path):
    result = record_report(
        output_path=tmp_path / "report.csv",
        dataset=" candles ",
        source=" kraken ",
        records_checked=5,
        valid_records=5,
        rejected_records=0,
        warnings=["low_volume"],
        validation_timestamp=CHECK_TIME,
    )

    assert result["dataset"] == "candles"
    assert result["source"] == "kraken"
    assert result["status"] == "WARNING"


def test_record_report_rejects_invalid_input(tmp_path):
    with pytest.raises(ValidationReportError):
        record_report(
            output_path=tmp_path / "report.csv",
            dataset="candles",
            source="kraken",
            records_checked=5,
            valid_records=6,
            rejected_records=0,
            validation_timestamp=CHECK_TIME,
        )

    assert not (
        tmp_path / "report.csv"
    ).exists()