"""Track what happened to real signals after they were issued.

The backtest says how the rules did on the past. This module checks how
the signals the system actually issued turned out, using the candles
collected afterwards, and compares that with the backtest. If live
results are clearly worse, the signals are flagged as paused.

Rules mirror the backtest: one position at a time, entry at the
signal's reference price, stop checked before the target inside the same
candle. Fees and slippage are not included, so live results here are
slightly optimistic.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MINIMUM_RESOLVED = 20
MAXIMUM_OPEN_HOURS = 48


def _parse(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None

    if stamp.tzinfo is None:
        return None

    return stamp.tz_convert("UTC")


def read_candles(path: str | Path) -> list[dict[str, Any]]:
    """Read a candle JSONL file, ordered and without duplicates."""

    file = Path(path)

    if not file.is_file():
        return []

    candles: dict[pd.Timestamp, dict[str, Any]] = {}

    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            opened = _parse(record["open_time"])
            candle = {
                "open_time": opened,
                "high": float(record["high"]),
                "low": float(record["low"]),
            }
        except (ValueError, KeyError, TypeError):
            continue

        if (
            opened is None
            or not math.isfinite(candle["high"])
            or not math.isfinite(candle["low"])
        ):
            continue

        candles[opened] = candle

    return [candles[key] for key in sorted(candles)]


def read_signals(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)

    if not file.is_file():
        return []

    signals = []

    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue

        if (
            isinstance(record, dict)
            and record.get("signal") in ("LONG", "SHORT")
            and _parse(record.get("analysis_timestamp")) is not None
        ):
            signals.append(record)

    signals.sort(key=lambda s: _parse(s["analysis_timestamp"]))

    return signals


def resolve_signal(
    signal: dict[str, Any],
    candles: list[dict[str, Any]],
    *,
    target_index: int = 0,
    maximum_hours: float = MAXIMUM_OPEN_HOURS,
) -> dict[str, Any]:
    """Follow one signal through later candles until stop or target."""

    start = _parse(signal["analysis_timestamp"])
    is_long = signal["signal"] == "LONG"

    try:
        entry = float(signal["current_price"])
        stop = float(signal["stop_loss"])
        target = float(signal["take_profit"][target_index]["price"])
    except (KeyError, IndexError, TypeError, ValueError):
        return {"outcome": "INVALID", "resolved_at": None, "r": None}

    risk = (entry - stop) if is_long else (stop - entry)
    reward = (target - entry) if is_long else (entry - target)

    if risk <= 0 or reward <= 0:
        return {"outcome": "INVALID", "resolved_at": None, "r": None}

    deadline = start + timedelta(hours=maximum_hours)

    for candle in candles:
        if candle["open_time"] < start:
            continue

        stopped = candle["low"] <= stop if is_long else candle["high"] >= stop
        reached = (
            candle["high"] >= target if is_long else candle["low"] <= target
        )

        if stopped:
            return {
                "outcome": "STOP_LOSS",
                "resolved_at": candle["open_time"].isoformat(),
                "r": -1.0,
            }

        if reached:
            return {
                "outcome": "TARGET",
                "resolved_at": candle["open_time"].isoformat(),
                "r": round(reward / risk, 4),
            }

        if candle["open_time"] > deadline:
            return {
                "outcome": "EXPIRED",
                "resolved_at": candle["open_time"].isoformat(),
                "r": None,
            }

    return {"outcome": "OPEN", "resolved_at": None, "r": None}


def track_signals(
    signals: list[dict[str, Any]],
    candles: list[dict[str, Any]],
    *,
    target_index: int = 0,
) -> list[dict[str, Any]]:
    """Resolve signals one position at a time, in time order."""

    results = []
    busy_until: pd.Timestamp | None = None

    for signal in signals:
        issued = _parse(signal["analysis_timestamp"])

        if busy_until is not None and issued < busy_until:
            continue

        result = resolve_signal(
            signal, candles, target_index=target_index
        )

        if result["outcome"] == "INVALID":
            continue

        results.append(
            {
                "signal_id": signal.get("signal_id"),
                "issued_at": issued.isoformat(),
                "direction": signal["signal"],
                **result,
            }
        )

        if result["outcome"] == "OPEN":
            # Later signals cannot be judged until this one finishes.
            break

        busy_until = _parse(result["resolved_at"])

    return results


def summarize_live(results: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [r for r in results if r["outcome"] in ("TARGET", "STOP_LOSS")]
    wins = [r for r in resolved if r["outcome"] == "TARGET"]
    losses = [r for r in resolved if r["outcome"] == "STOP_LOSS"]

    gross_win = math.fsum(r["r"] for r in wins)
    gross_loss = abs(math.fsum(r["r"] for r in losses))

    return {
        "tracked": len(results),
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "open": sum(1 for r in results if r["outcome"] == "OPEN"),
        "expired": sum(1 for r in results if r["outcome"] == "EXPIRED"),
        "win_rate_pct": (
            round(len(wins) / len(resolved) * 100.0, 2)
            if resolved
            else None
        ),
        "profit_factor": (
            round(gross_win / gross_loss, 4) if gross_loss > 0 else None
        ),
        "expectancy_r": (
            round(
                math.fsum(r["r"] for r in resolved) / len(resolved), 4
            )
            if resolved
            else None
        ),
    }


def compare_with_backtest(
    live: dict[str, Any],
    backtest: dict[str, Any] | None,
    *,
    minimum_resolved: int = MINIMUM_RESOLVED,
) -> dict[str, Any]:
    """Decide whether live results are consistent with the backtest."""

    if live["resolved"] < minimum_resolved:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "signals_paused": False,
            "message": (
                f"Only {live['resolved']} signals have finished "
                f"(need {minimum_resolved}); live results are not yet "
                "reliable."
            ),
        }

    if live["profit_factor"] is not None and live["profit_factor"] < 1:
        return {
            "status": "UNDERPERFORMING",
            "signals_paused": True,
            "message": (
                "Live signals lost more than they won "
                f"(profit factor {live['profit_factor']})."
            ),
        }

    interval = (
        (backtest or {}).get("overall", {}).get("win_rate_ci95_pct")
    )

    if (
        interval
        and live["win_rate_pct"] is not None
        and live["win_rate_pct"] < interval[0]
    ):
        return {
            "status": "WORSE_THAN_BACKTEST",
            "signals_paused": True,
            "message": (
                f"Live win rate {live['win_rate_pct']}% is below the "
                f"backtest's 95% range ({interval[0]}-{interval[1]}%)."
            ),
        }

    return {
        "status": "CONSISTENT",
        "signals_paused": False,
        "message": "Live results are consistent with the backtest.",
    }


def build_live_performance(
    *,
    history_path: str | Path,
    candle_path: str | Path,
    backtest: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    target_name = (backtest or {}).get("target", "TP1")
    target_index = (
        int(target_name[2:]) - 1
        if isinstance(target_name, str) and target_name[2:].isdigit()
        else 0
    )

    signals = read_signals(history_path)
    candles = read_candles(candle_path)
    results = track_signals(signals, candles, target_index=target_index)
    summary = summarize_live(results)

    return {
        "status": "AVAILABLE" if results else "UNAVAILABLE",
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "target": f"TP{target_index + 1}",
        "summary": summary,
        "comparison": compare_with_backtest(summary, backtest),
        "recent": results[-10:],
        "limitations": [
            "Live tracking ignores fees and slippage.",
            "One position at a time; overlapping signals are skipped.",
            "Results are past outcomes, not a prediction.",
        ],
    }
