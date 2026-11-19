"""Tests for the end-to-end Kraken trading runtime."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import trader.runtime as runtime_module

from trader.runtime import (
    TradingRuntimeError,
    run_trading_runtime,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026,
    11,
    19,
    12,
    0,
    tzinfo=UTC,
)

TIMEFRAMES = (
    "5m",
    "15m",
    "1h",
    "4h",
)


# ---------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------


def make_config():
    """Build a valid runtime configuration."""

    return {
        "trader": {
            "trader": {
                "enabled": True,
                "execution": {
                    "manual_review_only": True,
                    "auto_trading": False,
                    "exchange_connection": False,
                    "exchange_api_keys_required": False,
                },
                "timeframes": list(TIMEFRAMES),
            }
        },
        "data_quality": {
            "freshness": {
                "enabled": True,
                "reject_stale_data": True,
                "max_age_minutes": {
                    "candles": 15,
                },
            }
        },
    }


def make_candles():
    """Create non-empty candle datasets for all timeframes."""

    result = {}

    for timeframe in TIMEFRAMES:

        result[timeframe] = pd.DataFrame({
            "open_time": [
                CHECK_TIME - timedelta(minutes=10)
            ],
            "close_time": [
                CHECK_TIME - timedelta(minutes=5)
            ],
            "open": [95000.0],
            "high": [95100.0],
            "low": [94900.0],
            "close": [95050.0],
            "volume": [10.0],
            "source": ["kraken"],
        })

    return result


def make_signal():
    """Build a valid manual-review signal."""

    return {
        "signal_id": "a" * 64,
        "signal": "LONG",
        "status": "REQUIRES_MANUAL_REVIEW",
        "analysis_timestamp": CHECK_TIME.isoformat(),
    }


def make_trading_result():
    """Build a successful trading pipeline response."""

    return {
        "pipeline_status": "SUCCESS",
        "signal": make_signal(),
        "persistence": {
            "status": "SAVED",
            "history_saved": True,
            "rejection_saved": False,
        },
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
    }


@pytest.fixture
def runtime_environment(monkeypatch, tmp_path):
    """Prepare isolated runtime dependencies."""

    collection_calls = []
    trading_calls = []

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: make_config(),
    )

    def fake_collection(**kwargs):

        collection_calls.append(kwargs)

        return {
            "status": "PASS",
            "pair": kwargs["pair"],
            "timeframe": kwargs["timeframe"],
            "collected_records": 1,
            "accepted": pd.DataFrame({
                "close": [95000.0],
            }),
            "rejected": pd.DataFrame(),
            "message": "Validation passed.",
        }

    def fake_history_loader(*, pair, data_dir):

        assert pair == "XBTUSD"

        return make_candles()

    def fake_trading_pipeline(
        data,
        **kwargs,
    ):

        trading_calls.append({
            "data": data,
            "kwargs": kwargs,
        })

        return make_trading_result()

    monkeypatch.setattr(
        runtime_module,
        "run_kraken_pipeline",
        fake_collection,
    )

    monkeypatch.setattr(
        runtime_module,
        "load_trading_candles",
        fake_history_loader,
    )

    monkeypatch.setattr(
        runtime_module,
        "run_trading_pipeline",
        fake_trading_pipeline,
    )

    return {
        "collection_calls": collection_calls,
        "trading_calls": trading_calls,
        "data_dir": tmp_path / "raw",
        "report_path": tmp_path / "reports" / "validation.csv",
        "history_path": tmp_path / "signals" / "history.jsonl",
        "rejection_path": tmp_path / "signals" / "rejections.jsonl",
    }


# ---------------------------------------------------------
# Successful runtime tests
# ---------------------------------------------------------


def test_runtime_completes_successfully(
    runtime_environment,
):

    result = run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    assert result["runtime_status"] == "SUCCESS"

    assert result["pair"] == "XBTUSD"

    assert result["signal"]["signal"] == "LONG"

    assert result["execution"]["manual_review_only"] is True

    assert result["execution"]["auto_trading"] is False

    assert result["execution"]["exchange_connection"] is False

    assert result["execution"]["order_execution"] is False


def test_runtime_collects_all_four_timeframes(
    runtime_environment,
):

    run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    calls = runtime_environment["collection_calls"]

    assert len(calls) == 4

    collected_timeframes = {
        call["timeframe"]
        for call in calls
    }

    assert collected_timeframes == set(TIMEFRAMES)


def test_runtime_uses_correct_freshness_limits(
    runtime_environment,
):

    run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    expected_limits = {
        "5m": 20,
        "15m": 30,
        "1h": 75,
        "4h": 255,
    }

    calls = runtime_environment["collection_calls"]

    actual_limits = {
        call["timeframe"]: call["max_age_minutes"]
        for call in calls
    }

    assert actual_limits == expected_limits


def test_runtime_uses_utc_analysis_timestamp(
    runtime_environment,
):

    result = run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    assert result["analysis_timestamp"] == CHECK_TIME.isoformat()

    for call in runtime_environment["collection_calls"]:

        assert call["collection_time"] == CHECK_TIME


def test_runtime_passes_leverage_to_trading_pipeline(
    runtime_environment,
):

    run_trading_runtime(
        analysis_time=CHECK_TIME,
        leverage=2,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    calls = runtime_environment["trading_calls"]

    assert len(calls) == 1

    assert calls[0]["kwargs"]["leverage"] == 2


def test_runtime_passes_account_equity_to_trading_pipeline(
    runtime_environment,
):

    run_trading_runtime(
        analysis_time=CHECK_TIME,
        account_equity=10000,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    calls = runtime_environment["trading_calls"]

    assert calls[0]["kwargs"]["account_equity"] == 10000


def test_runtime_passes_persistence_paths(
    runtime_environment,
):

    run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    calls = runtime_environment["trading_calls"]

    kwargs = calls[0]["kwargs"]

    assert kwargs["history_path"] == (
        runtime_environment["history_path"]
    )

    assert kwargs["rejection_path"] == (
        runtime_environment["rejection_path"]
    )


def test_runtime_creates_storage_directories(
    runtime_environment,
):

    run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    assert runtime_environment["data_dir"].is_dir()

    assert runtime_environment["report_path"].parent.is_dir()

    assert runtime_environment["history_path"].parent.is_dir()

    assert runtime_environment["rejection_path"].parent.is_dir()


# ---------------------------------------------------------
# Pair and timestamp validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "pair",
    [
        "",
        "BTC/USD",
        "BTC-USD",
        "BTC USD",
        "../XBTUSD",
        None,
        123,
    ],
)
def test_runtime_rejects_invalid_pair(pair):

    with pytest.raises(ValueError):

        run_trading_runtime(
            pair=pair,
        )


def test_runtime_normalizes_pair():

    assert runtime_module._normalize_pair(
        "xbtusd"
    ) == "XBTUSD"


def test_runtime_accepts_utc_analysis_time():

    result = runtime_module._normalize_analysis_time(
        CHECK_TIME
    )

    assert result == CHECK_TIME


def test_runtime_rejects_naive_analysis_time():

    with pytest.raises(ValueError):

        runtime_module._normalize_analysis_time(
            datetime(2026, 11, 19, 12, 0)
        )


def test_runtime_rejects_non_utc_analysis_time():

    non_utc = datetime(
        2026,
        11,
        19,
        15,
        0,
        tzinfo=timezone(
            timedelta(hours=3)
        ),
    )

    with pytest.raises(ValueError):

        runtime_module._normalize_analysis_time(
            non_utc
        )


def test_runtime_rejects_invalid_analysis_time_type():

    with pytest.raises(ValueError):

        runtime_module._normalize_analysis_time(
            "2026-11-19T12:00:00Z"
        )


# ---------------------------------------------------------
# Numeric parameter validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        "2",
        None,
    ],
)
def test_runtime_rejects_invalid_leverage(value):

    with pytest.raises(ValueError):

        run_trading_runtime(
            leverage=value,
        )


def test_runtime_rejects_leverage_above_two():

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):

        run_trading_runtime(
            leverage=3,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -100,
        True,
        float("nan"),
        float("inf"),
        "1000",
    ],
)
def test_runtime_rejects_invalid_account_equity(value):

    with pytest.raises(ValueError):

        run_trading_runtime(
            account_equity=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        float("nan"),
        float("inf"),
        "15",
    ],
)
def test_runtime_rejects_invalid_timeout(value):

    with pytest.raises(ValueError):

        run_trading_runtime(
            timeout=value,
        )


# ---------------------------------------------------------
# Configuration safety tests
# ---------------------------------------------------------


def test_runtime_rejects_disabled_trader(
    monkeypatch,
):

    config = make_config()

    config["trader"]["trader"]["enabled"] = False

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: config,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="disabled",
    ):

        runtime_module._load_runtime_configuration()


@pytest.mark.parametrize(
    "key,value",
    [
        ("manual_review_only", False),
        ("auto_trading", True),
        ("exchange_connection", True),
        ("exchange_api_keys_required", True),
    ],
)
def test_runtime_rejects_unsafe_execution_configuration(
    monkeypatch,
    key,
    value,
):

    config = make_config()

    config["trader"]["trader"]["execution"][key] = value

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: config,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="Unsafe execution configuration",
    ):

        runtime_module._load_runtime_configuration()


def test_runtime_rejects_invalid_timeframe_configuration(
    monkeypatch,
):

    config = make_config()

    config["trader"]["trader"]["timeframes"] = [
        "5m",
        "15m",
        "1h",
    ]

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: config,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="timeframes",
    ):

        runtime_module._load_runtime_configuration()


def test_runtime_rejects_disabled_freshness_validation(
    monkeypatch,
):

    config = make_config()

    config["data_quality"]["freshness"]["enabled"] = False

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: config,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="freshness",
    ):

        runtime_module._load_runtime_configuration()


def test_runtime_rejects_disabled_stale_data_protection(
    monkeypatch,
):

    config = make_config()

    config["data_quality"]["freshness"][
        "reject_stale_data"
    ] = False

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: config,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="freshness",
    ):

        runtime_module._load_runtime_configuration()


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        float("nan"),
        float("inf"),
        "15",
        None,
    ],
)
def test_runtime_rejects_invalid_candle_freshness_limit(
    monkeypatch,
    value,
):

    config = make_config()

    config["data_quality"]["freshness"][
        "max_age_minutes"
    ]["candles"] = value

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: config,
    )

    with pytest.raises(TradingRuntimeError):

        runtime_module._load_runtime_configuration()


def test_runtime_wraps_configuration_error(
    monkeypatch,
):

    def failing_loader():
        from config.loader import ConfigurationError

        raise ConfigurationError(
            "Invalid project configuration."
        )

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        failing_loader,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="Unable to load project configuration",
    ):

        runtime_module._load_runtime_configuration()


# ---------------------------------------------------------
# Kraken collection failure tests
# ---------------------------------------------------------


def test_runtime_rejects_failed_kraken_validation(
    runtime_environment,
    monkeypatch,
):

    def failed_collection(**kwargs):

        return {
            "status": "FAIL",
            "pair": kwargs["pair"],
            "timeframe": kwargs["timeframe"],
            "collected_records": 1,
            "accepted": pd.DataFrame(),
            "rejected": pd.DataFrame({
                "reason": ["invalid_candle"],
            }),
        }

    monkeypatch.setattr(
        runtime_module,
        "run_kraken_pipeline",
        failed_collection,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="validation did not pass",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


def test_runtime_accepts_no_new_data_status(
    runtime_environment,
    monkeypatch,
):

    def no_new_data(**kwargs):

        return {
            "status": "NO_NEW_DATA",
            "pair": kwargs["pair"],
            "timeframe": kwargs["timeframe"],
            "collected_records": 0,
            "accepted": pd.DataFrame(),
            "rejected": pd.DataFrame(),
            "message": "No new candles.",
        }

    monkeypatch.setattr(
        runtime_module,
        "run_kraken_pipeline",
        no_new_data,
    )

    result = run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    assert result["runtime_status"] == "SUCCESS"

    for timeframe in TIMEFRAMES:

        assert (
            result["collection"][timeframe]["status"]
            == "NO_NEW_DATA"
        )


def test_runtime_rejects_pass_without_accepted_candles(
    runtime_environment,
    monkeypatch,
):

    def invalid_collection(**kwargs):

        return {
            "status": "PASS",
            "pair": kwargs["pair"],
            "timeframe": kwargs["timeframe"],
            "collected_records": 1,
            "accepted": pd.DataFrame(),
            "rejected": pd.DataFrame(),
        }

    monkeypatch.setattr(
        runtime_module,
        "run_kraken_pipeline",
        invalid_collection,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="without accepted candles",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


def test_runtime_wraps_kraken_collection_error(
    runtime_environment,
    monkeypatch,
):

    def failing_collection(**kwargs):

        from collectors.pipeline import KrakenPipelineError

        raise KrakenPipelineError(
            "Kraken unavailable."
        )

    monkeypatch.setattr(
        runtime_module,
        "run_kraken_pipeline",
        failing_collection,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="Kraken collection failed",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


# ---------------------------------------------------------
# Historical candle loading tests
# ---------------------------------------------------------


def test_runtime_rejects_empty_timeframe_history(
    runtime_environment,
    monkeypatch,
):

    def empty_history(*, pair, data_dir):

        data = make_candles()

        data["4h"] = pd.DataFrame()

        return data

    monkeypatch.setattr(
        runtime_module,
        "load_trading_candles",
        empty_history,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="No historical candles",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


def test_runtime_wraps_history_loading_error(
    runtime_environment,
    monkeypatch,
):

    def failing_loader(*, pair, data_dir):

        from trader.data_loader import TradingDataLoaderError

        raise TradingDataLoaderError(
            "Invalid historical data."
        )

    monkeypatch.setattr(
        runtime_module,
        "load_trading_candles",
        failing_loader,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="Unable to load complete Kraken candle history",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


# ---------------------------------------------------------
# Trading pipeline failure tests
# ---------------------------------------------------------


def test_runtime_rejects_unsuccessful_trading_pipeline(
    runtime_environment,
    monkeypatch,
):

    def failed_pipeline(data, **kwargs):

        return {
            "pipeline_status": "FAILED",
        }

    monkeypatch.setattr(
        runtime_module,
        "run_trading_pipeline",
        failed_pipeline,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="did not report SUCCESS",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


def test_runtime_rejects_missing_signal(
    runtime_environment,
    monkeypatch,
):

    def invalid_pipeline(data, **kwargs):

        result = make_trading_result()

        result["signal"] = None

        return result

    monkeypatch.setattr(
        runtime_module,
        "run_trading_pipeline",
        invalid_pipeline,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="invalid signal",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("manual_review_only", False),
        ("auto_trading", True),
        ("exchange_connection", True),
        ("order_execution", True),
    ],
)
def test_runtime_rejects_unsafe_pipeline_execution_result(
    runtime_environment,
    monkeypatch,
    key,
    value,
):

    def unsafe_pipeline(data, **kwargs):

        result = make_trading_result()

        result["execution"][key] = value

        return result

    monkeypatch.setattr(
        runtime_module,
        "run_trading_pipeline",
        unsafe_pipeline,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="execution safety verification failed",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


def test_runtime_wraps_trading_pipeline_error(
    runtime_environment,
    monkeypatch,
):

    def failing_pipeline(data, **kwargs):

        from trader.pipeline import TradingPipelineError

        raise TradingPipelineError(
            "Signal generation failed."
        )

    monkeypatch.setattr(
        runtime_module,
        "run_trading_pipeline",
        failing_pipeline,
    )

    with pytest.raises(
        TradingRuntimeError,
        match="Trading signal pipeline failed",
    ):

        run_trading_runtime(
            analysis_time=CHECK_TIME,
            data_dir=runtime_environment["data_dir"],
            report_path=runtime_environment["report_path"],
            history_path=runtime_environment["history_path"],
            rejection_path=runtime_environment["rejection_path"],
        )


# ---------------------------------------------------------
# Execution safety tests
# ---------------------------------------------------------


def test_runtime_never_enables_automatic_trading(
    runtime_environment,
):

    result = run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    assert result["execution"] == {
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
        "order_execution": False,
    }


def test_runtime_does_not_require_exchange_api_keys(
    runtime_environment,
):

    result = run_trading_runtime(
        analysis_time=CHECK_TIME,
        data_dir=runtime_environment["data_dir"],
        report_path=runtime_environment["report_path"],
        history_path=runtime_environment["history_path"],
        rejection_path=runtime_environment["rejection_path"],
    )

    assert result["runtime_status"] == "SUCCESS"

    assert result["execution"]["exchange_connection"] is False

    assert result["execution"]["order_execution"] is False