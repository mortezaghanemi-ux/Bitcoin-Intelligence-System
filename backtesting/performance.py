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


# Activation gate: the rules count as having a measured edge only when
# every check passes. Anything else is shown as "no edge proven".
GATE_MINIMUM_PROFIT_FACTOR = 1.2
GATE_MINIMUM_TRADES = 30
GATE_MINIMUM_OUT_OF_SAMPLE_TRADES = 15
WALK_FORWARD_FOLDS = 4
GATE_MINIMUM_PROFITABLE_FOLDS = 3


class PerformanceError(RuntimeError):
    """Raised when performance evaluation fails."""


def walk_forward_folds(
    trades: list[dict[str, Any]],
    folds: int = WALK_FORWARD_FOLDS,
) -> dict[str, Any]:
    """Split trades in time order into equal groups and score each.

    A rule set that only works in one stretch of history shows up as
    one good fold and several bad ones.
    """

    ordered = sorted(trades, key=lambda t: t["decision_time"])
    size = len(ordered) // folds if folds else 0
    results = []

    if size == 0:
        return {"folds": [], "profitable_folds": 0, "fold_count": folds}

    for index in range(folds):
        start = index * size
        end = (index + 1) * size if index < folds - 1 else len(ordered)
        summary = summarize_trades(ordered[start:end], minimum_sample=1)
        results.append(
            {
                "from": ordered[start]["decision_time"],
                "to": ordered[end - 1]["decision_time"],
                "trades": summary["trades"],
                "win_rate_pct": summary["win_rate_pct"],
                "profit_factor": summary["profit_factor"],
                "net_pnl": round(
                    math.fsum(
                        t["net_pnl"] for t in ordered[start:end]
                    ),
                    8,
                ),
            }
        )

    profitable = sum(1 for fold in results if fold["net_pnl"] > 0)

    return {
        "folds": results,
        "profitable_folds": profitable,
        "fold_count": folds,
    }


def confidence_informativeness(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check whether a higher confidence score really wins more often."""

    pairs = [
        (float(t["confidence"]), 1.0 if t["net_pnl"] > 0 else 0.0)
        for t in trades
        if isinstance(t.get("confidence"), (int, float))
    ]

    if len(pairs) < 100:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "correlation": None,
            "informative": False,
        }

    scores = pd.Series([p[0] for p in pairs])
    wins = pd.Series([p[1] for p in pairs])

    if scores.nunique() < 2 or wins.nunique() < 2:
        return {
            "status": "NO_VARIATION",
            "correlation": None,
            "informative": False,
        }

    correlation = float(scores.corr(wins))
    # About two standard errors for a correlation at this sample size.
    threshold = 2.0 / math.sqrt(len(pairs))

    return {
        "status": "MEASURED",
        "correlation": round(correlation, 4),
        "informative": correlation > threshold,
        "threshold": round(threshold, 4),
    }


def activation_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the rules that decide whether signals may be trusted."""

    overall = report["overall"]
    out = report["out_of_sample"]
    folds = report.get("walk_forward", {})

    def pf(group):
        return group.get("profit_factor")

    checks = {
        "overall_trades": overall["trades"] >= GATE_MINIMUM_TRADES,
        "overall_profit_factor": (
            pf(overall) is not None
            and pf(overall) >= GATE_MINIMUM_PROFIT_FACTOR
        ),
        "out_of_sample_trades": (
            out["trades"] >= GATE_MINIMUM_OUT_OF_SAMPLE_TRADES
        ),
        "out_of_sample_profit_factor": (
            pf(out) is not None
            and pf(out) >= GATE_MINIMUM_PROFIT_FACTOR
        ),
        "walk_forward_consistency": (
            folds.get("profitable_folds", 0)
            >= GATE_MINIMUM_PROFITABLE_FOLDS
        ),
    }

    holdout = report.get("holdout")

    if holdout is not None:
        checks["holdout_trades"] = (
            holdout["trades"] >= GATE_MINIMUM_TRADES
        )
        checks["holdout_profit_factor"] = (
            pf(holdout) is not None
            and pf(holdout) >= GATE_MINIMUM_PROFIT_FACTOR
        )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rules": {
            "minimum_profit_factor": GATE_MINIMUM_PROFIT_FACTOR,
            "minimum_trades": GATE_MINIMUM_TRADES,
            "minimum_out_of_sample_trades": (
                GATE_MINIMUM_OUT_OF_SAMPLE_TRADES
            ),
            "minimum_profitable_folds": (
                f"{GATE_MINIMUM_PROFITABLE_FOLDS} of {WALK_FORWARD_FOLDS}"
            ),
        },
    }


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
    profile: dict[str, Any] | None = None,
    holdout_before: str | None = None,
) -> dict[str, Any]:
    """Build a performance report from a completed backtest.

    holdout_before names a cut-off date. Trades before it were never
    used to choose the profile, so they form an honest confirmation
    test; they are scored separately and must also pass the gate.
    """

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

    report = {
        "status": "COMPLETED",
        "generated_at": moment.astimezone(timezone.utc).isoformat(),
        "pair": pair,
        "profile": profile or {"name": "baseline"},
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

    if holdout_before is not None:
        cutoff = pd.Timestamp(holdout_before, tz="UTC")
        held = [
            t for t in trades
            if pd.Timestamp(t["decision_time"]) < cutoff
        ]
        report["holdout"] = {
            "before": cutoff.isoformat(),
            **summarize_trades(held, minimum_sample=minimum_sample),
            "walk_forward": walk_forward_folds(held),
        }

    report["walk_forward"] = walk_forward_folds(trades)
    report["confidence_check"] = confidence_informativeness(trades)
    report["activation_gate"] = activation_gate(report)

    return report


def save_performance_report(
    report: dict[str, Any],
    directory: str | Path | None = None,
    *,
    set_latest: bool = True,
) -> Path:
    """Save a timestamped history copy, a per-profile copy and,
    optionally, the report the dashboard shows (performance_latest)."""

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

    profile_name = report.get("profile", {}).get("name", "baseline")

    (history / f"performance_{report['target']}_{stamp}.json").write_text(
        text, encoding="utf-8"
    )

    per_profile = target / f"performance_{profile_name}_{report['target']}.json"
    per_profile.write_text(text, encoding="utf-8")

    latest = target / LATEST_FILE

    if set_latest:
        latest.write_text(text, encoding="utf-8")

    return latest if set_latest else per_profile


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


def edge_verdict(performance: dict[str, Any]) -> str:
    """Say whether the measured history shows the rules making money.

    NO_EDGE_MEASURED: profit factor below 1 overall or out of sample.
    EDGE_MEASURED: the full activation gate passed.
    UNCLEAR: anything else (too few trades, mixed folds, ...).
    """

    overall = performance.get("overall", {})
    out = performance.get("out_of_sample", {})

    pf_all = overall.get("profit_factor")
    pf_out = out.get("profit_factor")

    if pf_all is not None and pf_all < 1:
        return "NO_EDGE_MEASURED"

    if pf_out is not None and out.get("trades", 0) and pf_out < 1:
        return "NO_EDGE_MEASURED"

    gate = performance.get("activation_gate")

    if gate is None and "walk_forward" not in performance:
        # Older reports without a gate: fall back to the simple rule.
        if (
            pf_all is not None
            and pf_all > 1
            and pf_out is not None
            and pf_out > 1
            and overall.get("sample_status") == "SUFFICIENT"
            and out.get("sample_status") == "SUFFICIENT"
        ):
            return "EDGE_MEASURED"

        return "UNCLEAR"

    if gate and gate.get("passed"):
        return "EDGE_MEASURED"

    return "UNCLEAR"


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
        "profit_factor": group.get("profit_factor"),
        "overall_profit_factor": performance["overall"].get(
            "profit_factor"
        ),
        "out_of_sample_profit_factor": performance.get(
            "out_of_sample", {}
        ).get("profit_factor"),
        "total_return_pct": performance.get("risk", {}).get(
            "total_return_pct"
        ),
        "maximum_drawdown_pct": performance.get("risk", {}).get(
            "maximum_drawdown_pct"
        ),
        "edge_verdict": edge_verdict(performance),
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
    profile: Any = None,
) -> dict[str, Any]:
    """Run a backtest on stored candles and return the report.

    A profile selects the signal rules (see trader.strategy_profile).
    Fills are simulated on the profile's fill timeframe.
    """

    signal_engine = None
    fill_timeframe = "5m"

    if profile is not None:
        from trader.signal_engine.engine import TradingSignalEngine

        signal_engine = TradingSignalEngine(profile=profile)
        fill_timeframe = profile.fill_timeframe

    result = run_backtest(
        data,
        signal_engine=signal_engine,
        execution_timeframe=fill_timeframe,
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
        profile=profile.to_dict() if profile is not None else None,
    )
