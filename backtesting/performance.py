"""Historical performance evaluation of the trading signal engine.

Turns a completed backtest into a measured win-rate report, saves it,
and looks up the measured performance that matches a live signal.

Important: every number here is a measurement of what the rules did on
past data. It is not a prediction and not a promise about future trades.
"""

from __future__ import annotations

import json
import math

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.engine import run_backtest
from backtesting.metrics import calculate_backtest_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PERFORMANCE_DIR = PROJECT_ROOT / "data" / "backtests"

LATEST_FILE = "performance_latest.json"

MINIMUM_SAMPLE_SIZE = 30

CONFIDENCE_BUCKETS = (
    ("60-69", 60.0, 70.0),
    ("70-79", 70.0, 80.0),
    ("80-100", 80.0, 100.01),
)

DISCLAIMER = (
    "Measured on past data by simulation. It is not a prediction, "
    "and future results can differ."
)


class PerformanceError(RuntimeError):
    """Raised when performance evaluation fails."""


def wilson_interval(
    wins: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float] | None:
    """95% Wilson score interval for a win rate, in percent."""

    if total <= 0:
        return None

    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total + z * z / (4 * total * total)
        )
        / denominator
    )

    return (
        round(max(0.0, centre - margin) * 100.0, 2),
        round(min(1.0, centre + margin) * 100.0, 2),
    )


def summarize_trades(
    trades: list[dict[str, Any]],
    *,
    minimum_sample: int = MINIMUM_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Summarize a list of closed simulated trades."""

    total = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]

    gross_profit = math.fsum(t["net_pnl"] for t in wins)
    gross_loss = math.fsum(t["net_pnl"] for t in losses)

    interval = wilson_interval(len(wins), total)

    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": total - len(wins) - len(losses),
        "win_rate_pct": (
            round(len(wins) / total * 100.0, 2) if total else None
        ),
        "win_rate_ci95_pct": (
            list(interval) if interval is not None else None
        ),
        "profit_factor": (
            round(gross_profit / abs(gross_loss), 4)
            if gross_loss < 0
            else None
        ),
        "expectancy_per_trade": (
            round(
                math.fsum(t["net_pnl"] for t in trades) / total,
                8,
            )
            if total
            else None
        ),
        "sample_status": (
            "SUFFICIENT"
            if total >= minimum_sample
            else "INSUFFICIENT_SAMPLE"
        ),
    }


def _bucket_for(confidence: Any) -> str | None:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
    ):
        return None

    for name, low, high in CONFIDENCE_BUCKETS:
        if low <= confidence < high:
            return name

    return None


def evaluate_performance(
    backtest_result: dict[str, Any],
    *,
    pair: str = "XBTUSD",
    train_fraction: float = 0.7,
    minimum_sample: int = MINIMUM_SAMPLE_SIZE,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a performance report from a completed backtest."""

    if (
        not isinstance(backtest_result, dict)
        or backtest_result.get("status") != "COMPLETED"
    ):
        raise PerformanceError(
            "Only completed backtests can be evaluated."
        )

    if not 0.1 <= train_fraction <= 0.9:
        raise PerformanceError(
            "train_fraction must be between 0.1 and 0.9."
        )

    trades = list(backtest_result["trades"])
    curve = backtest_result["equity_curve"]

    if not curve:
        raise PerformanceError("Backtest has no equity curve.")

    period_start = pd.Timestamp(curve[0]["timestamp"])
    period_end = pd.Timestamp(curve[-1]["timestamp"])
    split_time = period_start + (period_end - period_start) * train_fraction

    in_sample = [
        t
        for t in trades
        if pd.Timestamp(t["decision_time"]) <= split_time
    ]
    out_of_sample = [
        t
        for t in trades
        if pd.Timestamp(t["decision_time"]) > split_time
    ]

    by_direction = {
        direction: summarize_trades(
            [t for t in trades if t["direction"] == direction],
            minimum_sample=minimum_sample,
        )
        for direction in ("LONG", "SHORT")
    }

    by_confidence = {
        name: summarize_trades(
            [t for t in trades if _bucket_for(t.get("confidence")) == name],
            minimum_sample=minimum_sample,
        )
        for name, _, _ in CONFIDENCE_BUCKETS
    }

    by_exit_reason: dict[str, int] = {}

    for trade in trades:
        reason = trade.get("exit_reason", "UNKNOWN")
        by_exit_reason[reason] = by_exit_reason.get(reason, 0) + 1

    metrics = (
        calculate_backtest_metrics(
            backtest_result,
            periods_per_year=105120,
        )
        if curve
        else {}
    )

    moment = generated_at or datetime.now(timezone.utc)

    return {
        "status": "COMPLETED",
        "generated_at": moment.astimezone(timezone.utc).isoformat(),
        "pair": pair,
        "execution_timeframe": backtest_result["execution_timeframe"],
        "target": backtest_result["execution"].get("target", "TP1"),
        "period": {
            "start": period_start.isoformat(),
            "end": period_end.isoformat(),
            "days": round(
                (period_end - period_start).total_seconds() / 86400.0,
                2,
            ),
            "in_sample_until": split_time.isoformat(),
        },
        "minimum_sample_size": minimum_sample,
        "overall": summarize_trades(
            trades, minimum_sample=minimum_sample
        ),
        "in_sample": summarize_trades(
            in_sample, minimum_sample=minimum_sample
        ),
        "out_of_sample": summarize_trades(
            out_of_sample, minimum_sample=minimum_sample
        ),
        "by_direction": by_direction,
        "by_confidence": by_confidence,
        "by_exit_reason": by_exit_reason,
        "risk": {
            "total_return_pct": metrics.get("summary", {}).get(
                "total_return_pct"
            ),
            "maximum_drawdown_pct": metrics.get("risk", {}).get(
                "maximum_drawdown_pct"
            ),
            "profit_factor": metrics.get("profitability", {}).get(
                "profit_factor"
            ),
        },
        "assumptions": {
            "fee_rate": backtest_result["execution"].get("fee_rate"),
            "slippage_bps": backtest_result["execution"].get("slippage_bps"),
            "leverage": backtest_result["execution"].get("leverage"),
            "decision_step_candles": backtest_result["execution"].get(
                "decision_step", 1
            ),
        },
        "limitations": list(backtest_result.get("limitations", []))
        + [DISCLAIMER],
    }


def save_performance_report(
    report: dict[str, Any],
    directory: str | Path | None = None,
) -> Path:
    """Save the report as latest and keep a timestamped history copy."""

    target = Path(directory or DEFAULT_PERFORMANCE_DIR)
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)

    stamp = (
        report["generated_at"]
        .replace(":", "")
        .replace("-", "")
        .replace("+0000", "Z")
        .split(".")[0]
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)

    (history / f"performance_{report['target']}_{stamp}.json").write_text(
        text, encoding="utf-8"
    )

    latest = target / LATEST_FILE
    latest.write_text(text, encoding="utf-8")

    return latest


def load_latest_performance(
    directory: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load the latest saved report, or None when none exists."""

    path = Path(directory or DEFAULT_PERFORMANCE_DIR) / LATEST_FILE

    if not path.is_file():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None


def lookup_signal_performance(
    performance: dict[str, Any] | None,
    signal: dict[str, Any],
) -> dict[str, Any]:
    """Find the measured history that matches a live signal.

    Uses the most specific group with enough trades:
    direction + confidence bucket, then direction, then overall.
    """

    if not performance:
        return {
            "status": "NO_BACKTEST",
            "message": "No backtest has been run yet.",
        }

    minimum = performance.get("minimum_sample_size", MINIMUM_SAMPLE_SIZE)

    candidates = []

    bucket = _bucket_for(signal.get("confidence"))
    direction = signal.get("signal")

    if bucket and bucket in performance.get("by_confidence", {}):
        candidates.append(
            (
                f"confidence {bucket}",
                performance["by_confidence"][bucket],
            )
        )

    if direction in performance.get("by_direction", {}):
        candidates.append(
            (f"{direction} signals", performance["by_direction"][direction])
        )

    candidates.append(("all signals", performance["overall"]))

    for scope, group in candidates:
        if group["trades"] >= minimum:
            return _lookup_result(performance, scope, group, "MEASURED")

    scope, group = candidates[-1]

    return _lookup_result(
        performance, scope, group, "INSUFFICIENT_SAMPLE"
    )


def _lookup_result(
    performance: dict[str, Any],
    scope: str,
    group: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "scope": scope,
        "target": performance.get("target"),
        "win_rate_pct": group["win_rate_pct"],
        "win_rate_ci95_pct": group["win_rate_ci95_pct"],
        "trades": group["trades"],
        "period_days": performance.get("period", {}).get("days"),
        "generated_at": performance.get("generated_at"),
        "disclaimer": DISCLAIMER,
    }


def run_performance_evaluation(
    data: dict[str, pd.DataFrame],
    *,
    pair: str = "XBTUSD",
    target_index: int = 0,
    initial_equity: float = 10000.0,
    leverage: float = 1.0,
    fee_rate: float = 0.0006,
    slippage_bps: float = 2.0,
    lookback_candles: int | None = 150,
    decision_step: int = 12,
    train_fraction: float = 0.7,
    minimum_sample: int = MINIMUM_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Run a backtest on stored candles and return the report."""

    result = run_backtest(
        data,
        initial_equity=initial_equity,
        leverage=leverage,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        lookback_candles=lookback_candles,
        target_index=target_index,
        decision_step=decision_step,
    )

    return evaluate_performance(
        result,
        pair=pair,
        train_fraction=train_fraction,
        minimum_sample=minimum_sample,
    )
