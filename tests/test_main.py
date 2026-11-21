
"""Tests for the Bitcoin Intelligence System main application."""

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import main as app


# ---------------------------------------------------------
# Shared values
# ---------------------------------------------------------

UTC_TIME = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------
# Pair validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "pair, expected",
    [
        ("XBTUSD", "XBTUSD"),
        ("xbtusd", "XBTUSD"),
        (" BTCUSD ", "BTCUSD"),
        ("ETHUSD", "ETHUSD"),
    ],
)
def test_validate_pair_normalizes_valid_pairs(pair, expected):
    assert app._validate_pair(pair) == expected


@pytest.mark.parametrize(
    "pair",
    [
        "",
        " ",
        "BTC/USD",
        "BTC-USD",
        "BTC USD",
        "../XBTUSD",
        "BTC.USD",
        None,
        123,
        "A" * 21,
    ],
)
def test_validate_pair_rejects_invalid_values(pair):
    with pytest.raises(app.ApplicationError):
        app._validate_pair(pair)


# ---------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1.0),
        (2, 2.0),
        (0.5, 0.5),
        (10000, 10000.0),
    ],
)
def test_validate_positive_number_accepts_valid_values(value, expected):
    assert app._validate_positive_number(value, "value") == expected


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
def test_validate_positive_number_rejects_invalid_values(value):
    with pytest.raises(app.ApplicationError):
        app._validate_positive_number(value, "value")


# ---------------------------------------------------------
# JSON output
# ---------------------------------------------------------

def test_print_json_outputs_valid_json(capsys):
    app._print_json(
        {
            "status": "SUCCESS",
            "value": 10,
        }
    )

    output = capsys.readouterr().out
    result = json.loads(output)

    assert result["status"] == "SUCCESS"
    assert result["value"] == 10


def test_print_json_supports_datetime_values(capsys):
    app._print_json(
        {
            "timestamp": UTC_TIME,
        }
    )

    result = json.loads(capsys.readouterr().out)

    assert result["timestamp"] == str(UTC_TIME)


def test_print_json_rejects_nan_values():
    with pytest.raises(app.ApplicationError):
        app._print_json(
            {
                "value": float("nan"),
            }
        )


# ---------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------

def test_load_project_config_returns_dictionary(monkeypatch):
    expected = {
        "settings": {},
        "trader": {},
    }

    monkeypatch.setattr(
        "config.loader.load_config",
        lambda: expected,
    )

    assert app._load_project_config() is expected


def test_load_project_config_rejects_invalid_result(monkeypatch):
    monkeypatch.setattr(
        "config.loader.load_config",
        lambda: [],
    )

    with pytest.raises(
        app.ApplicationError,
        match="invalid result",
    ):
        app._load_project_config()


def test_load_project_config_wraps_configuration_error(monkeypatch):
    from config.loader import ConfigurationError

    def failing_loader():
        raise ConfigurationError("Invalid configuration")

    monkeypatch.setattr(
        "config.loader.load_config",
        failing_loader,
    )

    with pytest.raises(
        app.ApplicationError,
        match="configuration is invalid",
    ):
        app._load_project_config()


# ---------------------------------------------------------
# Configuration check
# ---------------------------------------------------------

def valid_configuration():
    return {
        "settings": {
            "system": {
                "name": "Bitcoin-Intelligence-System",
                "version": "1.0",
                "timezone": "UTC",
            }
        },
        "trader": {
            "trader": {
                "execution": {
                    "auto_trading": False,
                    "exchange_connection": False,
                    "manual_review_only": True,
                }
            }
        },
    }


def test_run_config_check_returns_success(monkeypatch):
    monkeypatch.setattr(
        app,
        "_load_project_config",
        valid_configuration,
    )

    result = app.run_config_check()

    assert result["status"] == "SUCCESS"
    assert result["timezone"] == "UTC"
    assert result["safety"]["manual_review_only"] is True
    assert result["safety"]["automatic_trading"] is False
    assert result["safety"]["exchange_connection"] is False
    assert result["safety"]["order_execution"] is False


@pytest.mark.parametrize(
    "key, value",
    [
        ("auto_trading", True),
        ("exchange_connection", True),
        ("manual_review_only", False),
    ],
)
def test_run_config_check_rejects_unsafe_trading_configuration(
    monkeypatch,
    key,
    value,
):
    config = valid_configuration()
    config["trader"]["trader"]["execution"][key] = value

    monkeypatch.setattr(
        app,
        "_load_project_config",
        lambda: config,
    )

    with pytest.raises(
        app.ApplicationError,
        match="safety configuration",
    ):
        app.run_config_check()


# ---------------------------------------------------------
# Collection workflow
# ---------------------------------------------------------

def collection_configuration():
    return {
        "schedules": {
            "schedules": {
                "candles": {
                    "enabled": True,
                    "timeframes": ["5m", "15m", "1h", "4h"],
                }
            }
        }
    }


def test_run_collection_returns_success(monkeypatch):
    calls = []

    def fake_pipeline(*, pair, timeframe, **_ignored):
        calls.append((pair, timeframe))

        return {
            "status": "PASS",
            "collected_records": 10,
            "accepted": [1, 2, 3],
            "rejected": [],
            "message": "Validated",
        }

    monkeypatch.setattr(
        app,
        "_load_project_config",
        collection_configuration,
    )

    monkeypatch.setattr(
        app,
        "_utc_now",
        lambda: UTC_TIME,
    )

    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        fake_pipeline,
    )

    result = app.run_collection(
        pair="xbtusd",
        timeframes=["5m", "1h"],
    )

    assert result["status"] == "SUCCESS"
    assert result["pair"] == "XBTUSD"
    assert len(result["timeframes"]) == 2
    assert result["failed_timeframes"] == []
    assert calls == [
        ("XBTUSD", "5m"),
        ("XBTUSD", "1h"),
    ]


def test_run_collection_rejects_disabled_collection(monkeypatch):
    config = collection_configuration()
    config["schedules"]["schedules"]["candles"]["enabled"] = False

    monkeypatch.setattr(
        app,
        "_load_project_config",
        lambda: config,
    )

    with pytest.raises(
        app.ApplicationError,
        match="disabled",
    ):
        app.run_collection()


def test_run_collection_rejects_unknown_timeframe(monkeypatch):
    monkeypatch.setattr(
        app,
        "_load_project_config",
        collection_configuration,
    )

    with pytest.raises(
        app.ApplicationError,
        match="timeframes",
    ):
        app.run_collection(timeframes=["1m"])


def test_run_collection_records_pipeline_failure(monkeypatch):
    monkeypatch.setattr(
        app,
        "_load_project_config",
        collection_configuration,
    )

    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        lambda **kwargs: {
            "status": "FAIL",
            "collected_records": 0,
            "accepted": [],
            "rejected": [],
        },
    )

    result = app.run_collection(timeframes=["5m"])

    assert result["status"] == "FAILED"
    assert result["failed_timeframes"] == ["5m"]


def test_run_collection_records_pipeline_exception(monkeypatch):
    monkeypatch.setattr(
        app,
        "_load_project_config",
        collection_configuration,
    )

    def failing_pipeline(**kwargs):
        raise OSError("Provider unavailable")

    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        failing_pipeline,
    )

    result = app.run_collection(timeframes=["5m"])

    assert result["status"] == "FAILED"
    assert result["timeframes"][0]["status"] == "ERROR"


# ---------------------------------------------------------
# Market data validation
# ---------------------------------------------------------

def test_collect_validated_analysis_data_returns_copy(monkeypatch):
    import pandas as pd

    original = pd.DataFrame(
        {
            "open": [100],
            "high": [110],
            "low": [90],
            "close": [105],
            "volume": [10],
        }
    )

    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        lambda **kwargs: {
            "status": "PASS",
            "accepted": original,
        },
    )

    result = app._collect_validated_analysis_data(
        "XBTUSD",
        "1h",
    )

    assert result.equals(original)
    assert result is not original


def test_collect_validated_analysis_data_rejects_failed_validation(
    monkeypatch,
):
    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        lambda **kwargs: {
            "status": "FAIL",
            "accepted": [],
        },
    )

    with pytest.raises(
        app.ApplicationError,
        match="validation did not pass",
    ):
        app._collect_validated_analysis_data(
            "XBTUSD",
            "1h",
        )


def test_collect_validated_analysis_data_rejects_empty_data(
    monkeypatch,
):
    import pandas as pd

    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        lambda **kwargs: {
            "status": "PASS",
            "accepted": pd.DataFrame(),
        },
    )

    with pytest.raises(
        app.ApplicationError,
        match="No accepted closed candles",
    ):
        app._collect_validated_analysis_data(
            "XBTUSD",
            "1h",
        )


def test_collect_validated_analysis_data_wraps_pipeline_error(
    monkeypatch,
):
    def failing_pipeline(**kwargs):
        raise OSError("Network unavailable")

    monkeypatch.setattr(
        "collectors.pipeline.run_kraken_pipeline",
        failing_pipeline,
    )

    with pytest.raises(
        app.ApplicationError,
        match="Market data collection failed",
    ):
        app._collect_validated_analysis_data(
            "XBTUSD",
            "1h",
        )


# ---------------------------------------------------------
# Trader workflow
# ---------------------------------------------------------

def safe_trading_result():
    return {
        "runtime_status": "SUCCESS",
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
    }


def test_run_trader_passes_scheduler_arguments(monkeypatch):
    calls = []

    def fake_runtime(**kwargs):
        calls.append(kwargs)
        return safe_trading_result()

    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        fake_runtime,
    )

    candles = {"1h": object()}

    result = app.run_trader(
        pair="xbtusd",
        leverage=2,
        account_equity=10000,
        analysis_time=UTC_TIME,
        candle_data=candles,
    )

    assert result["runtime_status"] == "SUCCESS"
    assert len(calls) == 1
    assert calls[0]["pair"] == "XBTUSD"
    assert calls[0]["leverage"] == 2.0
    assert calls[0]["account_equity"] == 10000.0
    assert calls[0]["analysis_time"] == UTC_TIME
    assert calls[0]["candle_data"] is candles


def test_run_trader_preserves_standalone_arguments(monkeypatch):
    calls = []

    def fake_runtime(**kwargs):
        calls.append(kwargs)
        return safe_trading_result()

    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        fake_runtime,
    )

    app.run_trader()

    assert calls == [
        {
            "pair": "XBTUSD",
            "leverage": 1.0,
            "account_equity": None,
        }
    ]


@pytest.mark.parametrize(
    "leverage",
    [
        0,
        -1,
        True,
        float("nan"),
        float("inf"),
        "2",
    ],
)
def test_run_trader_rejects_invalid_leverage(leverage):
    with pytest.raises(app.ApplicationError):
        app.run_trader(leverage=leverage)


def test_run_trader_rejects_leverage_above_two():
    with pytest.raises(
        app.ApplicationError,
        match="maximum of 2",
    ):
        app.run_trader(leverage=3)


def test_run_trader_rejects_invalid_account_equity():
    with pytest.raises(app.ApplicationError):
        app.run_trader(account_equity=0)


def test_run_trader_rejects_failed_runtime(monkeypatch):
    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        lambda **kwargs: {
            "runtime_status": "FAILED",
            "execution": {},
        },
    )

    with pytest.raises(
        app.ApplicationError,
        match="did not report SUCCESS",
    ):
        app.run_trader()


@pytest.mark.parametrize(
    "key, value",
    [
        ("manual_review_only", False),
        ("auto_trading", True),
        ("exchange_connection", True),
        ("order_execution", True),
    ],
)
def test_run_trader_rejects_unsafe_execution_result(
    monkeypatch,
    key,
    value,
):
    result = safe_trading_result()
    result["execution"][key] = value

    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        lambda **kwargs: result,
    )

    with pytest.raises(
        app.ApplicationError,
        match="safety verification failed",
    ):
        app.run_trader()


def test_run_trader_wraps_runtime_error(monkeypatch):
    from trader.runtime import TradingRuntimeError

    def failing_runtime(**kwargs):
        raise TradingRuntimeError("Runtime failure")

    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        failing_runtime,
    )

    with pytest.raises(
        app.ApplicationError,
        match="Trading analysis runtime failed",
    ):
        app.run_trader()


# ---------------------------------------------------------
# Command line interface
# ---------------------------------------------------------

def test_argument_parser_requires_command():
    parser = app.build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_argument_parser_accepts_check_config():
    parser = app.build_argument_parser()

    arguments = parser.parse_args(["check-config"])

    assert arguments.command == "check-config"


def test_argument_parser_accepts_collection_arguments():
    parser = app.build_argument_parser()

    arguments = parser.parse_args(
        [
            "collect",
            "--pair",
            "XBTUSD",
            "--timeframes",
            "5m",
            "1h",
        ]
    )

    assert arguments.command == "collect"
    assert arguments.pair == "XBTUSD"
    assert arguments.timeframes == ["5m", "1h"]


def test_argument_parser_accepts_trader_arguments():
    parser = app.build_argument_parser()

    arguments = parser.parse_args(
        [
            "trader",
            "--pair",
            "XBTUSD",
            "--leverage",
            "2",
            "--account-equity",
            "10000",
        ]
    )

    assert arguments.command == "trader"
    assert arguments.leverage == 2
    assert arguments.account_equity == 10000


def test_main_runs_config_check(monkeypatch, capsys):
    monkeypatch.setattr(
        app,
        "run_config_check",
        lambda: {"status": "SUCCESS"},
    )

    result = app.main(["check-config"])

    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "SUCCESS"


def test_main_returns_failure_for_failed_workflow(monkeypatch, capsys):
    monkeypatch.setattr(
        app,
        "run_config_check",
        lambda: {"status": "FAILED"},
    )

    result = app.main(["check-config"])

    assert result == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"


def test_main_handles_application_error(monkeypatch):
    def failing_check():
        raise app.ApplicationError("Configuration failure")

    monkeypatch.setattr(
        app,
        "run_config_check",
        failing_check,
    )

    assert app.main(["check-config"]) == 1


def test_main_handles_keyboard_interrupt(monkeypatch):
    def interrupted_check():
        raise KeyboardInterrupt

    monkeypatch.setattr(
        app,
        "run_config_check",
        interrupted_check,
    )

    assert app.main(["check-config"]) == 130


def test_main_handles_unexpected_exception(monkeypatch):
    def failing_check():
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr(
        app,
        "run_config_check",
        failing_check,
    )

    assert app.main(["check-config"]) == 1


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

def test_run_dashboard_returns_zero(monkeypatch):
    monkeypatch.setattr(
        "dashboard.app.run_dashboard_server",
        lambda **kwargs: None,
    )

    assert app.run_dashboard() == 0


def test_run_dashboard_wraps_server_error(monkeypatch):
    from dashboard.app import DashboardServerError

    def failing_server(**kwargs):
        raise DashboardServerError("Server unavailable")

    monkeypatch.setattr(
        "dashboard.app.run_dashboard_server",
        failing_server,
    )

    with pytest.raises(
        app.ApplicationError,
        match="Dashboard startup failed",
    ):
        app.run_dashboard()
