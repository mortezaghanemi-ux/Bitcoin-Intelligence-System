"""Tests for the backtest / import-history commands and signal context."""

import pytest

import main as app
from backtesting.performance import (
    evaluate_performance,
    save_performance_report,
)
from tests.test_backtest_performance import make_result


def test_import_history_missing_file_raises_application_error(tmp_path):
    with pytest.raises(app.ApplicationError):
        app.run_import_history(str(tmp_path / "missing.csv"))


def test_import_history_rejects_bad_pair(tmp_path):
    with pytest.raises(app.ApplicationError):
        app.run_import_history(str(tmp_path / "x.csv"), pair="BAD PAIR")


@pytest.mark.parametrize("target", [0, 4, -1])
def test_backtest_rejects_invalid_target(target):
    with pytest.raises(app.ApplicationError):
        app.run_backtest_evaluation(target=target)


def test_backtest_without_data_raises_application_error(tmp_path):
    with pytest.raises(app.ApplicationError):
        app.run_backtest_evaluation(data_dir=str(tmp_path))


def test_parser_accepts_new_commands():
    parser = app.build_argument_parser()

    imported = parser.parse_args(["import-history", "file.csv"])
    assert imported.command == "import-history"
    assert imported.pair == "XBTUSD"

    backtest = parser.parse_args(
        ["backtest", "--target", "2", "--step", "6", "--last-days", "30"]
    )
    assert backtest.target == 2
    assert backtest.step == 6
    assert backtest.last_days == 30.0

    with pytest.raises(SystemExit):
        parser.parse_args(["backtest", "--target", "9"])


def trading_result(signal_type, confidence):
    return {
        "runtime_status": "SUCCESS",
        "signal": {"signal": signal_type, "confidence": confidence},
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
    }


def test_run_trader_attaches_measured_performance(monkeypatch, tmp_path):
    report = evaluate_performance(
        make_result(
            [10, -5] * 10,
            confidences=[85] * 20,
            directions=["LONG"] * 20,
        ),
        minimum_sample=10,
    )
    save_performance_report(report, tmp_path)

    monkeypatch.setattr(
        "backtesting.performance.DEFAULT_PERFORMANCE_DIR", tmp_path
    )
    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        lambda **kwargs: trading_result("LONG", 85.0),
    )

    result = app.run_trader()

    context = result["historical_performance"]
    assert context["status"] == "MEASURED"
    assert context["win_rate_pct"] == 50.0
    assert result["signal"]["signal"] == "LONG"  # signal untouched


def test_run_trader_does_not_attach_for_no_trade(monkeypatch):
    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        lambda **kwargs: trading_result("NO_TRADE", 40.0),
    )

    assert "historical_performance" not in app.run_trader()


def test_run_trader_flags_missing_backtest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backtesting.performance.DEFAULT_PERFORMANCE_DIR", tmp_path
    )
    monkeypatch.setattr(
        "trader.runtime.run_trading_runtime",
        lambda **kwargs: trading_result("SHORT", 70.0),
    )

    context = app.run_trader()["historical_performance"]

    assert context["status"] == "NO_BACKTEST"
