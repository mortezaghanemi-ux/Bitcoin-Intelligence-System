
"""Trading performance reporting based on completed backtests.

This module evaluates simulated trading performance only.
It does not claim to measure live account performance and never
connects to an exchange or executes orders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backtesting.metrics import (
    BacktestingMetricsError,
    calculate_backtest_metrics,
)
from reports.engine import (
    DEFAULT_REPORT_DIRECTORY,
    ReportEngine,
    ReportEngineError,
)


class TradingPerformanceReportError(RuntimeError):
    """Raised when a trading performance report cannot be created."""


PERIODS_PER_YEAR = {
    "5m": 105120,
    "15m": 35040,
    "1h": 8760,
    "4h": 2190,
}

REQUIRED_EXECUTION_SAFETY = {
    "simulation_only": True,
    "exchange_connection": False,
    "order_execution": False,
}


def _validate_backtest_result(
    result: Any,
) -> dict[str, Any]:
    """Validate the result returned by BacktestingEngine.run()."""

    if not isinstance(result, dict):
        raise TradingPerformanceReportError(
            "Backtest result must be a dictionary."
        )

    if result.get("status") != "COMPLETED":
        raise TradingPerformanceReportError(
            "Only completed backtests can be reported."
        )

    timeframe = result.get("execution_timeframe")

    if timeframe not in PERIODS_PER_YEAR:
        raise TradingPerformanceReportError(
            "Unsupported or missing backtest execution timeframe."
        )

    execution = result.get("execution")

    if not isinstance(execution, dict):
        raise TradingPerformanceReportError(
            "Backtest execution metadata is missing."
        )

    for field, expected in REQUIRED_EXECUTION_SAFETY.items():
        if execution.get(field) is not expected:
            raise TradingPerformanceReportError(
                f"Invalid backtest execution safety setting: {field}."
            )

    summary = result.get("summary")

    if not isinstance(summary, dict):
        raise TradingPerformanceReportError(
            "Backtest summary is missing or invalid."
        )

    trades = result.get("trades")

    if not isinstance(trades, list):
        raise TradingPerformanceReportError(
            "Backtest trades must be a list."
        )

    equity_curve = result.get("equity_curve")

    if not isinstance(equity_curve, list):
        raise TradingPerformanceReportError(
            "Backtest equity curve must be a list."
        )

    limitations = result.get("limitations", [])

    if (
        not isinstance(limitations, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in limitations
        )
    ):
        raise TradingPerformanceReportError(
            "Backtest limitations must be a list of non-empty strings."
        )

    return result


def build_trading_performance_report(
    backtest_result: dict[str, Any],
    *,
    generated_at: datetime | str | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Build a validated trading performance report without saving it.

    Performance figures are calculated from the supplied completed
    backtest. They must not be interpreted as live trading results.
    """

    result = _validate_backtest_result(
        backtest_result
    )

    timeframe = result["execution_timeframe"]

    annualization_periods = (
        PERIODS_PER_YEAR[timeframe]
        if periods_per_year is None
        else periods_per_year
    )

    try:
        metrics = calculate_backtest_metrics(
            result,
            periods_per_year=annualization_periods,
        )

    except BacktestingMetricsError as exc:
        raise TradingPerformanceReportError(
            f"Unable to calculate backtest performance: {exc}"
        ) from exc

    if not isinstance(metrics, dict):
        raise TradingPerformanceReportError(
            "Backtesting metrics returned an invalid result."
        )

    if metrics.get("status") != "SUCCESS":
        raise TradingPerformanceReportError(
            "Backtesting metrics did not report SUCCESS."
        )

    report_time = (
        generated_at
        if generated_at is not None
        else datetime.now(timezone.utc)
    )

    execution = result["execution"]

    payload = {
        "performance_type": "BACKTEST_ONLY",
        "execution_timeframe": timeframe,
        "metrics": metrics,
        "backtest": {
            "status": result["status"],
            "summary": dict(result["summary"]),
            "trade_count": len(result["trades"]),
            "equity_curve_points": len(result["equity_curve"]),
        },
        "execution": {
            "simulation_only": execution["simulation_only"],
            "exchange_connection": execution["exchange_connection"],
            "order_execution": execution["order_execution"],
            "leverage": execution.get("leverage"),
            "fee_rate": execution.get("fee_rate"),
            "slippage_bps": execution.get("slippage_bps"),
        },
        "limitations": list(result.get("limitations", [])),
        "interpretation": (
            "This report measures simulated backtest performance only. "
            "It does not represent verified live trading performance."
        ),
    }

    return {
        "report_type": "trading_performance",
        "generated_at": report_time,
        "source": "Bitcoin-Intelligence-System",
        "metadata": {
            "performance_type": "BACKTEST_ONLY",
            "execution_timeframe": timeframe,
            "periods_per_year": annualization_periods,
            "trade_count": len(result["trades"]),
            "simulation_only": True,
            "live_performance_verified": False,
        },
        "payload": payload,
    }


def create_trading_performance_report(
    backtest_result: dict[str, Any],
    *,
    report_engine: ReportEngine | None = None,
    output_directory: str | None = None,
    generated_at: datetime | str | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Create and persist a trading performance report."""

    if report_engine is not None and output_directory is not None:
        raise TradingPerformanceReportError(
            "Provide report_engine or output_directory, not both."
        )

    report_data = build_trading_performance_report(
        backtest_result,
        generated_at=generated_at,
        periods_per_year=periods_per_year,
    )

    if report_engine is None:
        report_engine = (
            ReportEngine(output_directory)
            if output_directory is not None
            else ReportEngine(DEFAULT_REPORT_DIRECTORY)
        )

    try:
        return report_engine.generate(
            report_type=report_data["report_type"],
            payload=report_data["payload"],
            generated_at=report_data["generated_at"],
            source=report_data["source"],
            metadata=report_data["metadata"],
        )

    except ReportEngineError as exc:
        raise TradingPerformanceReportError(
            f"Unable to save trading performance report: {exc}"
        ) from exc


if __name__ == "__main__":
    print("Bitcoin trading performance report module loaded.")
    print("Backtest performance only; no live execution is performed.")
