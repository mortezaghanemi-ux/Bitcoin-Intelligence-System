
"""Tests for the Kraken collection and validation pipeline."""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

import collectors.pipeline as pipeline


UTC = timezone.utc

CHECK_TIME = datetime(
    2026,
    11,
    16,
    4,
    0,
    tzinfo=UTC,
)


@pytest.fixture
def valid_config():
    """Return a valid minimal configuration for pipeline tests."""

    return {
        "data_quality": {
            "freshness": {
                "enabled": True,
                "reject_stale_data": True,
                "max_age_minutes": {
                    "candles": 15,
                },
                "candles_by_timeframe": {
                    "5m": 15,
                    "15m": 45,
                    "1h": 180,
                    "4h": 720,
                },
            },
        },
    }


@pytest.fixture
def valid_record():
    """Return a valid collected Kraken candle."""

    return {
        "open_time": "2026-11-16T03:55:00Z",
        "close_time": "2026-11-16T04:00:00Z",
        "open": "95000",
        "high": "96000",
        "low": "94000",
        "close": "95500",
        "volume": "12.5",
        "source": "kraken",
        "pair": "XBTUSD",
        "timeframe": "5m",
    }


@pytest.fixture
def successful_validation():
    """Return a successful validation response."""

    accepted = pd.DataFrame(
        [
            {
                "open_time": "2026-11-16T03:55:00Z",
                "close_time": "2026-11-16T04:00:00Z",
                "open": 95000,
                "high": 96000,
                "low": 94000,
                "close": 95500,
                "volume": 12.5,
                "source": "kraken",
            }
        ]
    )

    return {
        "accepted": accepted,
        "rejected": pd.DataFrame(),
        "validation": {
            "status": "PASS",
            "records_checked": 1,
        },
        "reports": ["validation-report"],
    }


@pytest.fixture
def mock_pipeline_dependencies(
    monkeypatch,
    valid_config,
    valid_record,
    successful_validation,
):
    """Mock external dependencies used by the pipeline."""

    calls = {}

    def fake_load_config():
        calls["config"] = True
        return valid_config

    def fake_collect(**kwargs):
        calls["collection"] = kwargs
        return [valid_record.copy()]

    def fake_validate(data, **kwargs):
        calls["validation_data"] = data.copy(deep=True)
        calls["validation"] = kwargs
        return successful_validation

    monkeypatch.setattr(
        pipeline,
        "load_config",
        fake_load_config,
    )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        fake_collect,
    )

    monkeypatch.setattr(
        pipeline,
        "run_candle_validation",
        fake_validate,
    )

    return calls


def test_successful_pipeline(mock_pipeline_dependencies):
    """The pipeline should return PASS for valid candles."""

    result = pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    assert result["status"] == "PASS"
    assert result["pair"] == "XBTUSD"
    assert result["timeframe"] == "5m"
    assert result["collected_records"] == 1
    assert len(result["accepted"]) == 1
    assert result["rejected"].empty
    assert result["max_age_minutes"] == 15
    assert result["validation"]["status"] == "PASS"


def test_loads_configuration_before_collection(
    mock_pipeline_dependencies,
):
    """Configuration must be loaded before collection."""

    pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    assert mock_pipeline_dependencies["config"] is True
    assert "collection" in mock_pipeline_dependencies


def test_uses_configured_candle_freshness(
    mock_pipeline_dependencies,
):
    """Configured candle freshness must reach validation."""

    pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    assert (
        mock_pipeline_dependencies["validation"][
            "max_age_minutes"
        ]
        == 15
    )


@pytest.mark.parametrize(
    "timeframe,expected_age",
    [
        ("5m", 15),
        ("15m", 45),
        ("1h", 180),
        ("4h", 720),
    ],
)
def test_uses_timeframe_specific_candle_freshness(
    mock_pipeline_dependencies,
    timeframe,
    expected_age,
):
    """Each timeframe must use its configured freshness limit."""

    result = pipeline.run_kraken_pipeline(
        timeframe=timeframe,
        collection_time=CHECK_TIME,
    )

    assert result["max_age_minutes"] == expected_age

    assert (
        mock_pipeline_dependencies["validation"][
            "max_age_minutes"
        ]
        == expected_age
    )


def test_explicit_freshness_override_takes_precedence(
    mock_pipeline_dependencies,
):
    """Explicit freshness must override project configuration."""

    result = pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
        max_age_minutes=30,
    )

    assert result["max_age_minutes"] == 30

    assert (
        mock_pipeline_dependencies["validation"][
            "max_age_minutes"
        ]
        == 30
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
    ],
)
def test_rejects_invalid_freshness_override(
    mock_pipeline_dependencies,
    invalid_value,
):
    """Invalid freshness overrides must be rejected."""

    with pytest.raises(ValueError):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
            max_age_minutes=invalid_value,
        )


def test_rejects_unsupported_timeframe():
    """Unsupported timeframes must fail before configuration loading."""

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        pipeline.run_kraken_pipeline(
            timeframe="2m",
        )


def test_rejects_naive_collection_time():
    """Naive collection timestamps must be rejected."""

    naive_time = datetime(
        2026,
        11,
        16,
        4,
        0,
    )

    with pytest.raises(ValueError, match="UTC"):
        pipeline.run_kraken_pipeline(
            collection_time=naive_time,
        )


def test_rejects_non_utc_collection_time():
    """Non-UTC timestamps must be rejected."""

    non_utc_time = datetime(
        2026,
        11,
        16,
        7,
        30,
        tzinfo=timezone(
            timedelta(hours=3, minutes=30)
        ),
    )

    with pytest.raises(ValueError, match="UTC"):
        pipeline.run_kraken_pipeline(
            collection_time=non_utc_time,
        )


def test_handles_configuration_error(monkeypatch):
    """Configuration errors must become pipeline errors."""

    def fail_config():
        raise pipeline.ConfigurationError(
            "Invalid configuration"
        )

    monkeypatch.setattr(
        pipeline,
        "load_config",
        fail_config,
    )

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="Cannot load project configuration",
    ):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
        )


def test_handles_missing_freshness_configuration(
    monkeypatch,
    valid_config,
):
    """Missing candle freshness configuration must fail safely."""

    freshness = valid_config["data_quality"]["freshness"]

    del freshness["max_age_minutes"]
    del freshness["candles_by_timeframe"]

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="Missing candles freshness configuration",
    ):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
        )


@pytest.mark.parametrize(
    "enabled,reject_stale",
    [
        (False, True),
        (False, False),
    ],
)
def test_disabled_freshness_returns_none(
    valid_config,
    enabled,
    reject_stale,
):
    """Disabled freshness policy must return None."""

    valid_config["data_quality"]["freshness"][
        "enabled"
    ] = enabled

    valid_config["data_quality"]["freshness"][
        "reject_stale_data"
    ] = reject_stale

    result = pipeline._resolve_max_age(
        None,
        valid_config,
    )

    assert result is None


def test_rejects_invalid_freshness_enabled_type(
    valid_config,
):
    """Freshness enabled must be a boolean."""

    valid_config["data_quality"]["freshness"][
        "enabled"
    ] = "true"

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="freshness.enabled must be a boolean",
    ):
        pipeline._resolve_max_age(
            None,
            valid_config,
        )


def test_rejects_invalid_reject_stale_type(
    valid_config,
):
    """Reject stale policy must be a boolean."""

    valid_config["data_quality"]["freshness"][
        "reject_stale_data"
    ] = "true"

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="freshness.reject_stale_data must be a boolean",
    ):
        pipeline._resolve_max_age(
            None,
            valid_config,
        )


def test_rejects_non_mapping_freshness_limits(
    valid_config,
):
    """Freshness limits must be a mapping."""

    valid_config["data_quality"]["freshness"][
        "max_age_minutes"
    ] = 15

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="must be a mapping",
    ):
        pipeline._resolve_max_age(
            None,
            valid_config,
        )


def test_returns_no_new_data_without_validation(
    monkeypatch,
    valid_config,
):
    """No new candles must skip validation."""

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        lambda **kwargs: [],
    )

    def unexpected_validation(*args, **kwargs):
        pytest.fail(
            "Validation must not run when no new data exists."
        )

    monkeypatch.setattr(
        pipeline,
        "run_candle_validation",
        unexpected_validation,
    )

    result = pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    assert result["status"] == "NO_NEW_DATA"
    assert result["collected_records"] == 0
    assert result["accepted"].empty
    assert result["rejected"].empty
    assert result["validation"] is None
    assert result["reports"] == []


def test_handles_collection_error(
    monkeypatch,
    valid_config,
):
    """Collector errors must become pipeline errors."""

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    def fail_collection(**kwargs):
        raise pipeline.KrakenCollectorError(
            "API unavailable"
        )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        fail_collection,
    )

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="Kraken data collection failed",
    ):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
        )


def test_handles_missing_required_collected_columns(
    monkeypatch,
    valid_config,
):
    """Missing columns must stop validation."""

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        lambda **kwargs: [
            {
                "open_time": "2026-11-16T03:55:00Z",
                "close_time": "2026-11-16T04:00:00Z",
            }
        ],
    )

    def unexpected_validation(*args, **kwargs):
        pytest.fail(
            "Validation must not run with missing columns."
        )

    monkeypatch.setattr(
        pipeline,
        "run_candle_validation",
        unexpected_validation,
    )

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="missing required columns",
    ):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
        )


def test_handles_validation_pipeline_error(
    monkeypatch,
    valid_config,
    valid_record,
):
    """Validation errors must become pipeline errors."""

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        lambda **kwargs: [valid_record],
    )

    def fail_validation(*args, **kwargs):
        raise pipeline.ValidationPipelineError(
            "Validation failed"
        )

    monkeypatch.setattr(
        pipeline,
        "run_candle_validation",
        fail_validation,
    )

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="Kraken candle validation failed",
    ):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
        )


def test_returns_fail_when_records_are_rejected(
    monkeypatch,
    valid_config,
    valid_record,
):
    """Rejected candles must produce FAIL status."""

    rejected = pd.DataFrame(
        [
            {
                **valid_record,
                "validation_errors": "stale_candle",
            }
        ]
    )

    validation_result = {
        "accepted": pd.DataFrame(),
        "rejected": rejected,
        "validation": {
            "status": "FAIL",
        },
        "reports": ["validation-report"],
    }

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        lambda **kwargs: [valid_record],
    )

    monkeypatch.setattr(
        pipeline,
        "run_candle_validation",
        lambda *args, **kwargs: validation_result,
    )

    result = pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    assert result["status"] == "FAIL"
    assert result["accepted"].empty
    assert len(result["rejected"]) == 1
    assert "rejected" in result["message"].lower()


def test_rejects_invalid_validation_response(
    monkeypatch,
    valid_config,
    valid_record,
):
    """Invalid validation data structures must fail safely."""

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda: valid_config,
    )

    monkeypatch.setattr(
        pipeline,
        "collect_kraken_ohlcv",
        lambda **kwargs: [valid_record],
    )

    monkeypatch.setattr(
        pipeline,
        "run_candle_validation",
        lambda *args, **kwargs: {
            "accepted": [],
            "rejected": [],
            "validation": {},
            "reports": [],
        },
    )

    with pytest.raises(
        pipeline.KrakenPipelineError,
        match="invalid data structures",
    ):
        pipeline.run_kraken_pipeline(
            collection_time=CHECK_TIME,
        )


def test_passes_only_required_columns_to_validation(
    mock_pipeline_dependencies,
):
    """Collector metadata must not leak into candle validation."""

    pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    columns = list(
        mock_pipeline_dependencies[
            "validation_data"
        ].columns
    )

    assert columns == [
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    ]


def test_passes_collection_time_to_collector(
    mock_pipeline_dependencies,
):
    """Collector must receive the exact UTC collection time."""

    pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
    )

    assert (
        mock_pipeline_dependencies["collection"][
            "collection_time"
        ]
        == CHECK_TIME
    )


def test_passes_timeout_to_collector(
    mock_pipeline_dependencies,
):
    """Custom timeout must reach the collector."""

    pipeline.run_kraken_pipeline(
        collection_time=CHECK_TIME,
        timeout=25,
    )

    assert (
        mock_pipeline_dependencies["collection"][
            "timeout"
        ]
        == 25
    )
