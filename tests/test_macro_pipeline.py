"""Tests for the FRED macroeconomic collection pipeline."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

import collectors.macro_pipeline as macro_pipeline

from collectors.fred import FREDCollectorError
from collectors.macro_pipeline import (
    MacroPipelineError,
    run_fred_pipeline,
)
from config.macro_series_loader import MacroConfigurationError


API_KEY = "A" * 32

COLLECTION_TIME = datetime(
    2026,
    11,
    19,
    10,
    30,
    tzinfo=timezone.utc,
)


def valid_config():
    """Return a valid macro configuration fixture."""

    return {
        "enabled": True,
        "provider": "fred",
        "timezone": "UTC",
        "series": [
            {
                "id": "DGS10",
                "name": "treasury_10_year_yield",
                "description": "US 10-Year Treasury Yield",
                "enabled": True,
            },
            {
                "id": "FEDFUNDS",
                "name": "federal_funds_rate",
                "description": "Federal Funds Effective Rate",
                "enabled": True,
            },
        ],
        "collection": {
            "observation_start": "2020-01-01",
            "observation_end": "2026-11-19",
        },
        "validation": {},
    }


def valid_record(
    series_id="DGS10",
    value=4.25,
    value_status="VALID",
):
    """Return a valid FRED observation."""

    return {
        "schema_version": 1,
        "source": "fred",
        "series_id": series_id,
        "observation_date": "2026-11-18",
        "realtime_start": "2026-11-18",
        "realtime_end": "2026-11-19",
        "value": value,
        "raw_value": str(value),
        "value_status": value_status,
        "collected_at": "2026-11-19T10:30:00Z",
        "provider_timestamp": None,
    }


@pytest.fixture
def mocked_config(monkeypatch):
    """Mock the macro configuration loader."""

    config = valid_config()

    def fake_loader(config_path=None):
        return config

    monkeypatch.setattr(
        macro_pipeline,
        "load_macro_series_config",
        fake_loader,
    )

    return config


def test_successful_fred_pipeline(
    monkeypatch,
    mocked_config,
    tmp_path,
):
    """Pipeline should collect and summarize valid records."""

    records = [
        valid_record("DGS10", 4.25),
        valid_record("FEDFUNDS", 3.75),
    ]

    def fake_collector(**kwargs):
        return records

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        fake_collector,
    )

    result = run_fred_pipeline(
        output_dir=tmp_path,
    )

    assert result["status"] == "SUCCESS"
    assert result["provider"] == "fred"
    assert result["timezone"] == "UTC"

    assert result["requested_series"] == [
        "DGS10",
        "FEDFUNDS",
    ]

    assert result["collected_records"] == 2

    assert result["counts_by_series"] == {
        "DGS10": 1,
        "FEDFUNDS": 1,
    }

    assert result["records"] == records

    assert result["output_file"] == str(
        tmp_path / "fred_observations.jsonl"
    )


def test_all_enabled_series_are_forwarded(
    monkeypatch,
    mocked_config,
):
    """All enabled series must reach the collector."""

    captured = {}

    def fake_collector(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        fake_collector,
    )

    run_fred_pipeline()

    assert captured["series_ids"] == [
        "DGS10",
        "FEDFUNDS",
    ]


def test_pipeline_forwards_all_arguments(
    monkeypatch,
    mocked_config,
    tmp_path,
):
    """Pipeline must forward collection parameters unchanged."""

    captured = {}

    def fake_collector(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        fake_collector,
    )

    config_path = tmp_path / "macro.yaml"
    output_dir = tmp_path / "output"

    result = run_fred_pipeline(
        config_path=config_path,
        output_dir=output_dir,
        collection_time=COLLECTION_TIME,
        timeout=15.0,
        page_size=500,
        api_key=API_KEY,
    )

    assert captured["series_ids"] == [
        "DGS10",
        "FEDFUNDS",
    ]

    assert captured["observation_start"] == "2020-01-01"
    assert captured["observation_end"] == "2026-11-19"

    assert captured["output_dir"] == output_dir
    assert captured["collection_time"] == COLLECTION_TIME
    assert captured["timeout"] == 15.0
    assert captured["page_size"] == 500
    assert captured["api_key"] == API_KEY

    assert result["status"] == "NO_NEW_DATA"


def test_empty_collection_returns_no_new_data(
    monkeypatch,
    mocked_config,
    tmp_path,
):
    """Empty collector result must not be treated as failure."""

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [],
    )

    result = run_fred_pipeline(
        output_dir=tmp_path,
    )

    assert result["status"] == "NO_NEW_DATA"
    assert result["collected_records"] == 0
    assert result["records"] == []

    assert result["counts_by_series"] == {
        "DGS10": 0,
        "FEDFUNDS": 0,
    }

    assert result["message"] == (
        "No new FRED observations were available."
    )


def test_invalid_configuration_is_translated(
    monkeypatch,
):
    """Configuration errors must become pipeline errors."""

    def fake_loader(config_path=None):
        raise MacroConfigurationError(
            "Invalid test configuration."
        )

    monkeypatch.setattr(
        macro_pipeline,
        "load_macro_series_config",
        fake_loader,
    )

    with pytest.raises(
        MacroPipelineError,
        match="Invalid macroeconomic configuration",
    ):
        run_fred_pipeline()


@pytest.mark.parametrize(
    "series",
    [
        None,
        [],
        "DGS10",
        [None],
        [{}],
        [{"id": None, "enabled": True}],
        [{"id": "", "enabled": True}],
        [{"id": "DGS10", "enabled": False}],
        [{"id": "DGS10", "enabled": 1}],
    ],
)
def test_invalid_series_configuration_is_rejected(
    monkeypatch,
    series,
):
    """Invalid or disabled series must not reach the collector."""

    config = valid_config()
    config["series"] = series

    monkeypatch.setattr(
        macro_pipeline,
        "load_macro_series_config",
        lambda config_path=None: config,
    )

    collector_called = False

    def fake_collector(**kwargs):
        nonlocal collector_called
        collector_called = True
        return []

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        fake_collector,
    )

    with pytest.raises(MacroPipelineError):
        run_fred_pipeline()

    assert collector_called is False


@pytest.mark.parametrize(
    "collection",
    [
        None,
        [],
        "invalid",
    ],
)
def test_invalid_collection_configuration_is_rejected(
    monkeypatch,
    collection,
):
    """Collection configuration must be a mapping."""

    config = valid_config()
    config["collection"] = collection

    monkeypatch.setattr(
        macro_pipeline,
        "load_macro_series_config",
        lambda config_path=None: config,
    )

    with pytest.raises(
        MacroPipelineError,
        match="Invalid macro collection configuration",
    ):
        run_fred_pipeline()


@pytest.mark.parametrize(
    "exception",
    [
        FREDCollectorError("FRED failed"),
        ValueError("Invalid collector input"),
        OSError("Storage failure"),
    ],
)
def test_collector_exceptions_are_translated(
    monkeypatch,
    mocked_config,
    exception,
):
    """Expected collector failures must become pipeline errors."""

    def fake_collector(**kwargs):
        raise exception

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        fake_collector,
    )

    with pytest.raises(
        MacroPipelineError,
        match="FRED macroeconomic collection failed",
    ):
        run_fred_pipeline()


def test_collector_result_must_be_a_list(
    monkeypatch,
    mocked_config,
):
    """Collector must return a list."""

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: {"records": []},
    )

    with pytest.raises(
        MacroPipelineError,
        match="invalid result",
    ):
        run_fred_pipeline()


def test_collector_records_must_be_dictionaries(
    monkeypatch,
    mocked_config,
):
    """Every returned record must be a dictionary."""

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [None],
    )

    with pytest.raises(
        MacroPipelineError,
        match="Invalid FRED record at index 0",
    ):
        run_fred_pipeline()


def test_unexpected_data_source_is_rejected(
    monkeypatch,
    mocked_config,
):
    """Records from another provider must be rejected."""

    record = valid_record()
    record["source"] = "another_provider"

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [record],
    )

    with pytest.raises(
        MacroPipelineError,
        match="Unexpected data source",
    ):
        run_fred_pipeline()


def test_unexpected_series_id_is_rejected(
    monkeypatch,
    mocked_config,
):
    """Unrequested series must not enter the result."""

    record = valid_record("UNKNOWN_SERIES")

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [record],
    )

    with pytest.raises(
        MacroPipelineError,
        match="Unexpected FRED series",
    ):
        run_fred_pipeline()


@pytest.mark.parametrize(
    "value_status",
    [
        None,
        "",
        "INVALID",
        "valid",
        1,
    ],
)
def test_invalid_value_status_is_rejected(
    monkeypatch,
    mocked_config,
    value_status,
):
    """Only VALID and MISSING statuses are accepted."""

    record = valid_record()
    record["value_status"] = value_status

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [record],
    )

    with pytest.raises(
        MacroPipelineError,
        match="Invalid FRED value status",
    ):
        run_fred_pipeline()


def test_missing_value_status_is_accepted(
    monkeypatch,
    mocked_config,
):
    """Missing economic observations must be preserved."""

    record = valid_record(
        series_id="DGS10",
        value=None,
        value_status="MISSING",
    )

    record["raw_value"] = "."

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [record],
    )

    result = run_fred_pipeline()

    assert result["status"] == "SUCCESS"
    assert result["records"][0]["value"] is None
    assert result["records"][0]["value_status"] == "MISSING"


def test_multiple_records_are_counted_correctly(
    monkeypatch,
    mocked_config,
):
    """Per-series record counts must be accurate."""

    records = [
        valid_record("DGS10", 4.25),
        valid_record("DGS10", 4.30),
        valid_record("FEDFUNDS", 3.75),
    ]

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: records,
    )

    result = run_fred_pipeline()

    assert result["collected_records"] == 3

    assert result["counts_by_series"] == {
        "DGS10": 2,
        "FEDFUNDS": 1,
    }


def test_default_output_path_is_used(
    monkeypatch,
    mocked_config,
):
    """Default output directory must match the collector."""

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [],
    )

    result = run_fred_pipeline()

    expected = (
        macro_pipeline.DEFAULT_OUTPUT_DIR
        / "fred_observations.jsonl"
    )

    assert Path(result["output_file"]) == expected


def test_api_key_is_not_exposed_in_result(
    monkeypatch,
    mocked_config,
    tmp_path,
):
    """Private API credentials must not appear in the result."""

    monkeypatch.setattr(
        macro_pipeline,
        "collect_fred_observations",
        lambda **kwargs: [],
    )

    result = run_fred_pipeline(
        output_dir=tmp_path,
        api_key=API_KEY,
    )

    assert API_KEY not in repr(result)
    assert API_KEY not in result["output_file"]
    assert "api_key" not in result