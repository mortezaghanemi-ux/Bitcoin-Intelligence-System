
"""Tests for simulated trading performance reports."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from reports.engine import (
    ReportEngine,
    ReportEngineError,
)
from reports.trading_performance_report import (
    PERIODS_PER_YEAR,
    TradingPerformanceReportError,
    build_trading_performance_report,
    create_trading_performance_report,
)


UTC = timezone.utc

GENERATED_AT = datetime(
    2026, 11, 20, 7, 0,
    tzinfo=UTC,
)


def make_backtest_result(**overrides):
    """Build a valid minimal BacktestingEngine result."""

    result = {
        "status": "COMPLETED",
        "execution_timeframe": "5m",
        "execution": {
            "simulation_only": True,
            "exchange_connection": False,
            "order_execution": False,
            "leverage": 1.0,
            "fee_rate": 0.0006,
            "slippage_bps": 2.0,
        },
        "summary": {
            "initial_equity": 10000.0,
            "final_equity": 10100.0,
            "net_profit": 100.0,
            "total_trades": 1,
        },
        "trades": [
            {
                "signal_id": "signal-001",
                "net_pnl": 100.0,
            },
        ],
        "equity_curve": [
            {
                "timestamp": "2026-11-20T07:00:00+00:00",
                "equity": 10050.0,
            },
            {
                "timestamp": "2026-11-20T07:05:00+00:00",
                "equity": 10100.0,
            },
        ],
        "limitations": [
            "Historical candle-based simulation.",
            "Funding payments are not modeled.",
        ],
    }

    result.update(overrides)

    return result


# -----------------------------------------
# Valid report construction
# -----------------------------------------

def test_builds_valid_trading_performance_report():
    result = make_backtest_result()

    report = build_trading_performance_report(
        result,
        generated_at=GENERATED_AT,
    )

    assert report["report_type"] == "trading_performance"
    assert report["generated_at"] == GENERATED_AT
    assert report["source"] == "Bitcoin-Intelligence-System"

    assert report["metadata"]["performance_type"] == (
        "BACKTEST_ONLY"
    )

    assert report["metadata"]["execution_timeframe"] == "5m"
    assert report["metadata"]["periods_per_year"] == 105120
    assert report["metadata"]["trade_count"] == 1
    assert report["metadata"]["simulation_only"] is True
    assert report["metadata"]["live_performance_verified"] is False


def test_report_contains_calculated_metrics():
    report = build_trading_performance_report(
        make_backtest_result(),
        generated_at=GENERATED_AT,
    )

    metrics = report["payload"]["metrics"]

    assert metrics["status"] == "SUCCESS"
    assert metrics["summary"]["initial_equity"] == 10000.0
    assert metrics["summary"]["final_equity"] == 10100.0
    assert metrics["summary"]["net_profit"] == 100.0
    assert metrics["summary"]["total_trades"] == 1
    assert metrics["summary"]["winning_trades"] == 1
    assert metrics["summary"]["losing_trades"] == 0


def test_report_contains_backtest_metadata():
    report = build_trading_performance_report(
        make_backtest_result(),
        generated_at=GENERATED_AT,
    )

    payload = report["payload"]

    assert payload["performance_type"] == "BACKTEST_ONLY"
    assert payload["execution_timeframe"] == "5m"
    assert payload["backtest"]["status"] == "COMPLETED"
    assert payload["backtest"]["trade_count"] == 1
    assert payload["backtest"]["equity_curve_points"] == 2


def test_report_preserves_simulation_safety_settings():
    report = build_trading_performance_report(
        make_backtest_result(),
        generated_at=GENERATED_AT,
    )

    execution = report["payload"]["execution"]

    assert execution["simulation_only"] is True
    assert execution["exchange_connection"] is False
    assert execution["order_execution"] is False


def test_report_preserves_backtest_limitations():
    report = build_trading_performance_report(
        make_backtest_result(),
        generated_at=GENERATED_AT,
    )

    assert report["payload"]["limitations"] == [
        "Historical candle-based simulation.",
        "Funding payments are not modeled.",
    ]


def test_report_explicitly_disclaims_live_performance():
    report = build_trading_performance_report(
        make_backtest_result(),
        generated_at=GENERATED_AT,
    )

    assert report["metadata"]["live_performance_verified"] is False

    assert "does not represent verified live trading performance" in (
        report["payload"]["interpretation"]
    )


# -----------------------------------------
# Timeframe annualization
# -----------------------------------------

@pytest.mark.parametrize(
    "timeframe,expected_periods",
    [
        ("5m", 105120),
        ("15m", 35040),
        ("1h", 8760),
        ("4h", 2190),
    ],
)
def test_uses_timeframe_specific_annualization(
    timeframe,
    expected_periods,
):
    result = make_backtest_result(
        execution_timeframe=timeframe,
    )

    report = build_trading_performance_report(
        result,
        generated_at=GENERATED_AT,
    )

    assert report["metadata"]["periods_per_year"] == (
        expected_periods
    )

    assert report["payload"]["metrics"]["metadata"][
        "periods_per_year"
    ] == expected_periods


def test_custom_annualization_periods_are_used():
    report = build_trading_performance_report(
        make_backtest_result(),
        generated_at=GENERATED_AT,
        periods_per_year=50000,
    )

    assert report["metadata"]["periods_per_year"] == 50000

    assert report["payload"]["metrics"]["metadata"][
        "periods_per_year"
    ] == 50000


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
        "105120",
        None,
    ],
)
def test_rejects_invalid_custom_annualization_periods(value):
    with pytest.raises(TradingPerformanceReportError):
        build_trading_performance_report(
            make_backtest_result(),
            generated_at=GENERATED_AT,
            periods_per_year=value,
        )


# -----------------------------------------
# Backtest result validation
# -----------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_rejects_non_dictionary_backtest_result(value):
    with pytest.raises(TradingPerformanceReportError):
        build_trading_performance_report(value)


@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "RUNNING",
        "FAILED",
        "SUCCESS",
        "completed",
    ],
)
def test_rejects_non_completed_backtest(status):
    result = make_backtest_result(status=status)

    with pytest.raises(
        TradingPerformanceReportError,
        match="completed backtests",
    ):
        build_trading_performance_report(result)


@pytest.mark.parametrize(
    "timeframe",
    [
        None,
        "",
        "1m",
        "1d",
        "30m",
        "5M",
    ],
)
def test_rejects_unsupported_execution_timeframe(timeframe):
    result = make_backtest_result(
        execution_timeframe=timeframe,
    )

    with pytest.raises(
        TradingPerformanceReportError,
        match="timeframe",
    ):
        build_trading_performance_report(result)


@pytest.mark.parametrize(
    "execution",
    [
        None,
        [],
        "invalid",
    ],
)
def test_rejects_invalid_execution_metadata(execution):
    result = make_backtest_result(
        execution=execution,
    )

    with pytest.raises(
        TradingPerformanceReportError,
        match="execution metadata",
    ):
        build_trading_performance_report(result)


@pytest.mark.parametrize(
    "field,invalid_value",
    [
        ("simulation_only", False),
        ("simulation_only", 1),
        ("exchange_connection", True),
        ("exchange_connection", None),
        ("order_execution", True),
        ("order_execution", "false"),
    ],
)
def test_rejects_unsafe_execution_settings(
    field,
    invalid_value,
):
    result = make_backtest_result()
    result["execution"][field] = invalid_value

    with pytest.raises(
        TradingPerformanceReportError,
        match="safety setting",
    ):
        build_trading_performance_report(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", None),
        ("summary", []),
        ("trades", None),
        ("trades", {}),
        ("equity_curve", None),
        ("equity_curve", {}),
    ],
)
def test_rejects_invalid_backtest_sections(
    field,
    value,
):
    result = make_backtest_result(**{field: value})

    with pytest.raises(TradingPerformanceReportError):
        build_trading_performance_report(result)


@pytest.mark.parametrize(
    "limitations",
    [
        "not a list",
        {},
        [""],
        ["valid", None],
        [123],
    ],
)
def test_rejects_invalid_backtest_limitations(limitations):
    result = make_backtest_result(
        limitations=limitations,
    )

    with pytest.raises(
        TradingPerformanceReportError,
        match="limitations",
    ):
        build_trading_performance_report(result)


# -----------------------------------------
# Metrics validation and error handling
# -----------------------------------------

def test_rejects_invalid_metrics_input():
    result = make_backtest_result()

    result["trades"] = [
        {
            "signal_id": "signal-001",
            "net_pnl": float("nan"),
        }
    ]

    with pytest.raises(
        TradingPerformanceReportError,
        match="calculate backtest performance",
    ):
        build_trading_performance_report(result)


def test_rejects_duplicate_trade_signal_ids():
    result = make_backtest_result()

    result["trades"] = [
        {
            "signal_id": "signal-001",
            "net_pnl": 50.0,
        },
        {
            "signal_id": "signal-001",
            "net_pnl": 50.0,
        },
    ]

    with pytest.raises(
        TradingPerformanceReportError,
        match="calculate backtest performance",
    ):
        build_trading_performance_report(result)


def test_rejects_non_increasing_equity_timestamps():
    result = make_backtest_result()

    result["equity_curve"] = [
        {
            "timestamp": "2026-11-20T07:05:00+00:00",
            "equity": 10050.0,
        },
        {
            "timestamp": "2026-11-20T07:05:00+00:00",
            "equity": 10100.0,
        },
    ]

    with pytest.raises(
        TradingPerformanceReportError,
        match="calculate backtest performance",
    ):
        build_trading_performance_report(result)


def test_rejects_equity_curve_final_value_mismatch():
    result = make_backtest_result()

    result["equity_curve"][-1]["equity"] = 9999.0

    with pytest.raises(
        TradingPerformanceReportError,
        match="calculate backtest performance",
    ):
        build_trading_performance_report(result)


def test_accepts_empty_trade_history_when_equity_is_valid():
    result = make_backtest_result(
        summary={
            "initial_equity": 10000.0,
            "final_equity": 10000.0,
        },
        trades=[],
        equity_curve=[
            {
                "timestamp": "2026-11-20T07:00:00+00:00",
                "equity": 10000.0,
            },
            {
                "timestamp": "2026-11-20T07:05:00+00:00",
                "equity": 10000.0,
            },
        ],
    )

    report = build_trading_performance_report(
        result,
        generated_at=GENERATED_AT,
    )

    assert report["metadata"]["trade_count"] == 0
    assert report["payload"]["metrics"]["summary"]["total_trades"] == 0


# -----------------------------------------
# Input immutability
# -----------------------------------------

def test_build_report_does_not_modify_backtest_result():
    result = make_backtest_result()
    original = deepcopy(result)

    build_trading_performance_report(
        result,
        generated_at=GENERATED_AT,
    )

    assert result == original


# -----------------------------------------
# Report persistence
# -----------------------------------------

def test_create_report_saves_report(tmp_path):
    result = create_trading_performance_report(
        make_backtest_result(),
        output_directory=tmp_path,
        generated_at=GENERATED_AT,
    )

    assert result["status"] == "SUCCESS"
    assert result["report_type"] == "trading_performance"
    assert result["storage"]["saved"] is True

    saved_path = result["storage"]["path"]

    assert saved_path.endswith(".json")


def test_create_report_can_use_report_engine(tmp_path):
    engine = ReportEngine(tmp_path)

    result = create_trading_performance_report(
        make_backtest_result(),
        report_engine=engine,
        generated_at=GENERATED_AT,
    )

    assert result["status"] == "SUCCESS"
    assert result["report_type"] == "trading_performance"

    history = engine.get_history(
        report_type="trading_performance",
    )

    assert len(history) == 1
    assert history[0]["report_id"] == result["report_id"]


def test_create_report_rejects_engine_and_directory_together(
    tmp_path,
):
    engine = ReportEngine(tmp_path)

    with pytest.raises(
        TradingPerformanceReportError,
        match="not both",
    ):
        create_trading_performance_report(
            make_backtest_result(),
            report_engine=engine,
            output_directory=str(tmp_path),
        )


def test_create_report_rejects_invalid_backtest_before_saving(
    tmp_path,
):
    result = make_backtest_result(
        status="FAILED",
    )

    with pytest.raises(TradingPerformanceReportError):
        create_trading_performance_report(
            result,
            output_directory=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_create_report_wraps_report_engine_errors(tmp_path):
    class FailingReportEngine:
        def generate(self, **kwargs):
            raise ReportEngineError("simulated storage failure")

    with pytest.raises(
        TradingPerformanceReportError,
        match="Unable to save",
    ):
        create_trading_performance_report(
            make_backtest_result(),
            report_engine=FailingReportEngine(),
            generated_at=GENERATED_AT,
        )


# -----------------------------------------
# Default timestamp
# -----------------------------------------

def test_default_report_timestamp_is_utc():
    report = build_trading_performance_report(
        make_backtest_result(),
    )

    assert isinstance(report["generated_at"], datetime)
    assert report["generated_at"].tzinfo is not None
    assert report["generated_at"].utcoffset() == timedelta(0)


def test_timeframe_period_mapping_is_complete():
    assert set(PERIODS_PER_YEAR) == {
        "5m",
        "15m",
        "1h",
        "4h",
    }
