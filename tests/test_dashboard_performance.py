"""Dashboard tests for the measured-performance section."""

import json
from datetime import datetime, timezone

from backtesting.performance import (
    evaluate_performance,
    save_performance_report,
)
from dashboard.data import (
    _attach_historical_performance,
    load_dashboard_data,
    load_performance,
)

from tests.test_backtest_performance import make_result

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def make_signal(timestamp):
    return {
        "signal_id": "sig-1",
        "analysis_timestamp": timestamp.isoformat(),
        "status": "REQUIRES_MANUAL_REVIEW",
        "signal": "LONG",
        "direction": "BULLISH",
        "timeframe": "5m",
        "current_price": 90000.0,
        "confidence": 85.0,
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
        "data_quality": {
            "status": "VALIDATED",
            "closed_candles_only": True,
            "future_data_used": False,
        },
    }


def saved_report(directory):
    report = evaluate_performance(
        make_result(
            [10, -5] * 10,
            confidences=[85] * 20,
            directions=["LONG"] * 20,
        ),
        minimum_sample=10,
        generated_at=NOW,
    )
    save_performance_report(report, directory)

    return report


def test_load_performance_unavailable_without_report(tmp_path):
    result = load_performance(performance_directory=tmp_path)

    assert result["status"] == "UNAVAILABLE"
    assert result["report"] is None


def test_load_performance_available_with_report(tmp_path):
    saved_report(tmp_path)

    result = load_performance(performance_directory=tmp_path)

    assert result["status"] == "AVAILABLE"
    assert result["report"]["overall"]["trades"] == 20
    assert "not a prediction" in result["message"]


def test_attach_does_not_mutate_and_handles_missing_report():
    signals = [make_signal(NOW)]

    attached = _attach_historical_performance(signals, None)

    assert "historical_performance" not in signals[0]
    assert (
        attached[0]["historical_performance"]["status"] == "NO_BACKTEST"
    )


def test_snapshot_includes_performance_and_signal_context(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps(make_signal(NOW)) + "\n")

    performance_dir = tmp_path / "perf"
    saved_report(performance_dir)

    snapshot = load_dashboard_data(
        history_path=history,
        performance_directory=performance_dir,
        candle_directory=tmp_path / "candles",
        report_directory=tmp_path / "reports",
        now=NOW,
    )

    assert snapshot["performance"]["status"] == "AVAILABLE"

    context = snapshot["trading"]["signals"][0]["historical_performance"]

    assert context["status"] == "MEASURED"
    assert context["win_rate_pct"] == 50.0
    assert context["trades"] == 20
    assert any(
        "not predictions" in item for item in snapshot["limitations"]
    )


def test_snapshot_without_backtest_is_explicit(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps(make_signal(NOW)) + "\n")

    snapshot = load_dashboard_data(
        history_path=history,
        performance_directory=tmp_path / "none",
        candle_directory=tmp_path / "candles",
        report_directory=tmp_path / "reports",
        now=NOW,
    )

    assert snapshot["performance"]["status"] == "UNAVAILABLE"
    assert (
        snapshot["trading"]["signals"][0]["historical_performance"][
            "status"
        ]
        == "NO_BACKTEST"
    )
