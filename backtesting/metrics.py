
"""Performance metrics for Bitcoin Intelligence System backtests.

This module analyzes the output of BacktestingEngine.
It does not execute trades or modify the supplied result.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Any


class BacktestingMetricsError(ValueError):
    """Raised when backtest performance metrics cannot be calculated."""


DEFAULT_PERIODS_PER_YEAR = 105120  # 5-minute periods in a 365-day year


def _finite_number(value: Any, field_name: str) -> float:
    """Validate and return a finite numeric value."""

    if isinstance(value, bool):
        raise BacktestingMetricsError(
            f"{field_name} must be a finite number."
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BacktestingMetricsError(
            f"{field_name} must be a finite number."
        ) from exc

    if not math.isfinite(number):
        raise BacktestingMetricsError(
            f"{field_name} must be finite."
        )

    return number


def _parse_utc_timestamp(
    value: Any,
    field_name: str,
) -> datetime:
    """Parse a timestamp and require an explicit UTC timezone."""

    if isinstance(value, datetime):
        timestamp = value

    elif isinstance(value, str) and value.strip():
        try:
            timestamp = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise BacktestingMetricsError(
                f"{field_name} is not a valid timestamp."
            ) from exc

    else:
        raise BacktestingMetricsError(
            f"{field_name} must be a timezone-aware UTC timestamp."
        )

    if timestamp.tzinfo is None:
        raise BacktestingMetricsError(
            f"{field_name} must include timezone information."
        )

    try:
        offset = timestamp.utcoffset()
    except (ValueError, OverflowError) as exc:
        raise BacktestingMetricsError(
            f"{field_name} has an invalid timezone."
        ) from exc

    if offset != timedelta(0):
        raise BacktestingMetricsError(
            f"{field_name} must use UTC."
        )

    return timestamp


def _validate_result(
    backtest_result: dict[str, Any],
) -> tuple[
    float,
    float,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate the structure and values returned by BacktestingEngine."""

    if not isinstance(backtest_result, dict):
        raise BacktestingMetricsError(
            "backtest_result must be a dictionary."
        )

    if backtest_result.get("status") != "COMPLETED":
        raise BacktestingMetricsError(
            "Only completed backtests can be analyzed."
        )

    summary = backtest_result.get("summary")

    if not isinstance(summary, dict):
        raise BacktestingMetricsError(
            "Backtest summary must be a dictionary."
        )

    initial_equity = _finite_number(
        summary.get("initial_equity"),
        "summary.initial_equity",
    )

    final_equity = _finite_number(
        summary.get("final_equity"),
        "summary.final_equity",
    )

    if initial_equity <= 0:
        raise BacktestingMetricsError(
            "Initial equity must be greater than zero."
        )

    trades = backtest_result.get("trades")

    if not isinstance(trades, list):
        raise BacktestingMetricsError(
            "Backtest trades must be a list."
        )

    validated_trades = []
    seen_trade_ids = set()

    for index, trade in enumerate(trades):
        if not isinstance(trade, dict):
            raise BacktestingMetricsError(
                f"Trade at index {index} must be a dictionary."
            )

        net_pnl = _finite_number(
            trade.get("net_pnl"),
            f"trades[{index}].net_pnl",
        )

        if "signal_id" in trade:
            signal_id = trade["signal_id"]

            if signal_id is not None:
                if (
                    not isinstance(signal_id, str)
                    or not signal_id.strip()
                ):
                    raise BacktestingMetricsError(
                        f"Trade at index {index} has an invalid signal_id."
                    )

                if signal_id in seen_trade_ids:
                    raise BacktestingMetricsError(
                        "Duplicate trade signal IDs are not allowed."
                    )

                seen_trade_ids.add(signal_id)

        validated_trades.append({
            "net_pnl": net_pnl,
        })

    equity_curve = backtest_result.get("equity_curve")

    if not isinstance(equity_curve, list):
        raise BacktestingMetricsError(
            "Backtest equity_curve must be a list."
        )

    validated_curve = []
    previous_timestamp = None

    for index, point in enumerate(equity_curve):
        if not isinstance(point, dict):
            raise BacktestingMetricsError(
                f"Equity point at index {index} must be a dictionary."
            )

        timestamp = _parse_utc_timestamp(
            point.get("timestamp"),
            f"equity_curve[{index}].timestamp",
        )

        equity = _finite_number(
            point.get("equity"),
            f"equity_curve[{index}].equity",
        )

        if equity < 0:
            raise BacktestingMetricsError(
                f"Equity at index {index} cannot be negative."
            )

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise BacktestingMetricsError(
                "Equity curve timestamps must be strictly increasing."
            )

        previous_timestamp = timestamp

        validated_curve.append({
            "timestamp": timestamp,
            "equity": equity,
        })

    if validated_curve:
        curve_final_equity = validated_curve[-1]["equity"]

        if not math.isclose(
            curve_final_equity,
            final_equity,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise BacktestingMetricsError(
                "Final equity does not match the last equity curve point."
            )

    return (
        initial_equity,
        final_equity,
        validated_trades,
        validated_curve,
    )


def _calculate_maximum_drawdown(
    initial_equity: float,
    equity_curve: list[dict[str, Any]],
) -> tuple[float, float]:
    """Calculate maximum drawdown as an amount and percentage."""

    peak = initial_equity
    maximum_drawdown_amount = 0.0
    maximum_drawdown_pct = 0.0

    for point in equity_curve:
        equity = point["equity"]

        if equity > peak:
            peak = equity

        if peak <= 0:
            continue

        drawdown_amount = peak - equity
        drawdown_pct = (
            drawdown_amount / peak
        ) * 100.0

        maximum_drawdown_amount = max(
            maximum_drawdown_amount,
            drawdown_amount,
        )

        maximum_drawdown_pct = max(
            maximum_drawdown_pct,
            drawdown_pct,
        )

    return (
        maximum_drawdown_amount,
        maximum_drawdown_pct,
    )


def _calculate_sharpe_ratio(
    initial_equity: float,
    equity_curve: list[dict[str, Any]],
    periods_per_year: int,
) -> float | None:
    """Calculate annualized Sharpe ratio using realized equity returns.

    The risk-free rate is assumed to be zero.
    The first return is measured against initial equity.
    """

    if not equity_curve:
        return None

    returns = []
    previous_equity = initial_equity

    for point in equity_curve:
        current_equity = point["equity"]

        if previous_equity <= 0:
            return None

        period_return = (
            current_equity - previous_equity
        ) / previous_equity

        if not math.isfinite(period_return):
            raise BacktestingMetricsError(
                "Equity returns contain a non-finite value."
            )

        returns.append(period_return)
        previous_equity = current_equity

    if len(returns) < 2:
        return None

    mean_return = math.fsum(returns) / len(returns)

    variance = math.fsum(
        (value - mean_return) ** 2
        for value in returns
    ) / (len(returns) - 1)

    if variance <= 0:
        return None

    standard_deviation = math.sqrt(variance)

    sharpe = (
        mean_return / standard_deviation
    ) * math.sqrt(periods_per_year)

    if not math.isfinite(sharpe):
        return None

    return sharpe


def _calculate_profit_factor(
    gross_profit: float,
    gross_loss: float,
) -> float | None:
    """Calculate profit factor without returning infinity."""

    if gross_loss == 0:
        return None

    return gross_profit / abs(gross_loss)


def calculate_backtest_metrics(
    backtest_result: dict[str, Any],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """
    Calculate detailed performance metrics for a completed backtest.

    Args:
        backtest_result:
            The complete dictionary returned by BacktestingEngine.run().

        periods_per_year:
            Number of return observations per year.
            The default assumes a five-minute execution timeframe.

    Returns:
        A dictionary containing:
        - Total return and net profit.
        - Win rate and trade statistics.
        - Profit factor and expectancy.
        - Maximum drawdown.
        - Annualized Sharpe ratio.
        - Calculation metadata.

    The input result and its nested objects are never modified.
    """

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise BacktestingMetricsError(
            "periods_per_year must be a positive integer."
        )

    (
        initial_equity,
        final_equity,
        trades,
        equity_curve,
    ) = _validate_result(backtest_result)

    total_trades = len(trades)

    winning_trades = [
        trade
        for trade in trades
        if trade["net_pnl"] > 0
    ]

    losing_trades = [
        trade
        for trade in trades
        if trade["net_pnl"] < 0
    ]

    breakeven_trades = (
        total_trades
        - len(winning_trades)
        - len(losing_trades)
    )

    gross_profit = math.fsum(
        trade["net_pnl"]
        for trade in winning_trades
    )

    gross_loss = math.fsum(
        trade["net_pnl"]
        for trade in losing_trades
    )

    net_profit = final_equity - initial_equity

    total_return_pct = (
        net_profit / initial_equity
    ) * 100.0

    win_rate_pct = (
        len(winning_trades) / total_trades
    ) * 100.0 if total_trades else 0.0

    average_win = (
        gross_profit / len(winning_trades)
        if winning_trades
        else None
    )

    average_loss = (
        gross_loss / len(losing_trades)
        if losing_trades
        else None
    )

    expectancy = (
        math.fsum(
            trade["net_pnl"]
            for trade in trades
        ) / total_trades
        if total_trades
        else None
    )

    profit_factor = _calculate_profit_factor(
        gross_profit,
        gross_loss,
    )

    (
        maximum_drawdown_amount,
        maximum_drawdown_pct,
    ) = _calculate_maximum_drawdown(
        initial_equity,
        equity_curve,
    )

    sharpe_ratio = _calculate_sharpe_ratio(
        initial_equity,
        equity_curve,
        periods_per_year,
    )

    return {
        "status": "SUCCESS",
        "analysis_type": "BACKTEST_PERFORMANCE",
        "summary": {
            "initial_equity": round(initial_equity, 8),
            "final_equity": round(final_equity, 8),
            "net_profit": round(net_profit, 8),
            "total_return_pct": round(
                total_return_pct,
                6,
            ),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "breakeven_trades": breakeven_trades,
            "win_rate_pct": round(
                win_rate_pct,
                4,
            ),
        },
        "profitability": {
            "gross_profit": round(gross_profit, 8),
            "gross_loss": round(gross_loss, 8),
            "profit_factor": (
                round(profit_factor, 8)
                if profit_factor is not None
                else None
            ),
            "average_win": (
                round(average_win, 8)
                if average_win is not None
                else None
            ),
            "average_loss": (
                round(average_loss, 8)
                if average_loss is not None
                else None
            ),
            "expectancy_per_trade": (
                round(expectancy, 8)
                if expectancy is not None
                else None
            ),
        },
        "risk": {
            "maximum_drawdown_amount": round(
                maximum_drawdown_amount,
                8,
            ),
            "maximum_drawdown_pct": round(
                maximum_drawdown_pct,
                6,
            ),
            "annualized_sharpe_ratio": (
                round(sharpe_ratio, 8)
                if sharpe_ratio is not None
                else None
            ),
        },
        "metadata": {
            "trade_results_based_on": "net_pnl_after_fees",
            "equity_curve_points": len(equity_curve),
            "periods_per_year": periods_per_year,
            "risk_free_rate": 0.0,
            "sharpe_ratio_annualized": True,
            "input_data_modified": False,
            "profit_factor_unavailable_when_no_losses": True,
            "sharpe_unavailable_when_insufficient_or_constant_returns": True,
        },
    }


if __name__ == "__main__":
    print("Bitcoin Backtesting Metrics module loaded.")
