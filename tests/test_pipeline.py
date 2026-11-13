"""Integration tests for the candle validation pipeline."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from validation.pipeline import (
    ValidationPipelineError,
    run_candle_validation,
)


CHECK_TIME = datetime(
    2026, 11, 13, 12, 0, tzinfo=timezone.utc
)


def candle(
    source="kraken",
    high=96000,
    open_time="2026-11-13T11:50:00+00:00",
):
    return {
        "open_time": open_time,
        "close_time": "2026-11-13T11:55:00+00:00",
        "open": 95000,
        "high": high,
        "low": 94000,
        "close": 95500,
        "volume": 120.5,
        "source": source,
    }


def test_pipeline_accepts_valid_candles_and_writes_report(tmp_path):

    data = pd.DataFrame([candle()])
    report_path = tmp_path / "reports" / "validation.csv"

    result = run_candle_validation(
        data,
        report_path=report_path,
        validation_time=CHECK_TIME,
    )

    assert result["status"] == "PASS"
    assert len(result["accepted"]) == 1
    assert result["rejected"].empty
    assert len(result["reports"]) == 1
    assert report_path.is_file()

    saved = pd.read_csv(report_path)

    assert len(saved) == 1
    assert saved.loc[0, "source"] == "kraken"
    assert saved.loc[0, "status"] == "PASS"


def test_pipeline_reports_each_source_independently(tmp_path):

    second = candle(
        source="coinbase",
        high=93000,
        open_time="2026-11-13T11:55:00+00:00",
    )

    second["close_time"] = "2026-11-13T12:00:00+00:00"

    data = pd.DataFrame([
        candle(source="kraken"),
        second,
    ])

    result = run_candle_validation(
        data,
        report_path=tmp_path / "reports.csv",
        validation_time=CHECK_TIME,
    )

    assert result["status"] == "FAIL"
    assert len(result["accepted"]) == 1
    assert len(result["rejected"]) == 1
    assert len(result["reports"]) == 2

    reports = pd.read_csv(
        tmp_path / "reports.csv"
    ).set_index("source")

    assert reports.loc["kraken", "valid_records"] == 1
    assert reports.loc["kraken", "rejected_records"] == 0

    assert reports.loc["coinbase", "valid_records"] == 0
    assert reports.loc["coinbase", "rejected_records"] == 1

    assert reports.loc["coinbase", "status"] == "FAIL"


def test_pipeline_does_not_modify_input_dataframe(tmp_path):

    data = pd.DataFrame([candle()])
    original = data.copy(deep=True)

    run_candle_validation(
        data,
        report_path=tmp_path / "reports.csv",
        validation_time=CHECK_TIME,
    )

    pd.testing.assert_frame_equal(data, original)


def test_pipeline_rejects_empty_dataset():

    data = pd.DataFrame(columns=[
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    ])

    with pytest.raises(
        ValidationPipelineError,
        match="empty",
    ):

        run_candle_validation(
            data,
            validation_time=CHECK_TIME,
        )


def test_pipeline_rejects_missing_source_column(tmp_path):

    data = pd.DataFrame(
        [candle()]
    ).drop(columns=["source"])

    with pytest.raises(
        ValidationPipelineError,
        match="Missing required columns",
    ):

        run_candle_validation(
            data,
            report_path=tmp_path / "reports.csv",
            validation_time=CHECK_TIME,
        )


def test_pipeline_rejects_non_utc_validation_time(tmp_path):

    data = pd.DataFrame([candle()])

    local_time = datetime(
        2026, 11, 13, 12, 0
    )

    with pytest.raises(
        ValueError,
        match="UTC",
    ):

        run_candle_validation(
            data,
            report_path=tmp_path / "reports.csv",
            validation_time=local_time,
        )


def test_pipeline_prevents_duplicate_report_for_same_run(tmp_path):

    data = pd.DataFrame([candle()])

    report_path = tmp_path / "reports.csv"

    run_candle_validation(
        data,
        report_path=report_path,
        validation_time=CHECK_TIME,
    )

    with pytest.raises(
        ValidationPipelineError,
        match="duplicate validation report",
    ):

        run_candle_validation(
            data,
            report_path=report_path,
            validation_time=CHECK_TIME,
        )