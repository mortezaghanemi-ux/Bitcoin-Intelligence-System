"""Backtest across the separate continuous stretches of history.

A shard is one date range (for example one calendar year). For every
stretch that overlaps it, the signal engine is replayed on that part
only, so no signal or trade ever straddles a hole in the data. The
results are merged into one result that the normal performance
evaluation can score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MINIMUM_SHARD_DAYS = 20.0


class SegmentError(RuntimeError):
    """Raised for segment backtest problems."""


def load_index(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "segments.json"

    if not path.is_file():
        raise SegmentError(f"No segments.json in {root}.")

    return json.loads(path.read_text(encoding="utf-8"))


def _overlap(
    segment: dict[str, Any],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    begin = pd.Timestamp(segment["start"])
    finish = pd.Timestamp(segment["end"])

    if start is not None:
        begin = max(begin, start)

    if end is not None:
        finish = min(finish, end)

    if finish <= begin:
        return None

    return begin, finish


def run_shard(
    root: str | Path,
    *,
    profile: Any = None,
    target_index: int = 0,
    decision_step: int = 12,
    start: str | None = None,
    end: str | None = None,
    pair: str = "XBTUSD",
    minimum_days: float = MINIMUM_SHARD_DAYS,
) -> dict[str, Any]:
    """Replay one profile over every stretch inside [start, end)."""

    from backtesting.data import load_backtest_data
    from backtesting.engine import run_backtest
    from trader.signal_engine.engine import TradingSignalEngine

    index = load_index(root)
    lower = pd.Timestamp(start, tz="UTC") if start else None
    upper = pd.Timestamp(end, tz="UTC") if end else None

    results = []
    used = []
    skipped = []

    for segment in index["segments"]:
        window = _overlap(segment, lower, upper)

        if window is None:
            continue

        days = (window[1] - window[0]).total_seconds() / 86400

        if days < minimum_days:
            skipped.append(
                {"segment": segment["name"], "reason": f"{days:.1f} days"}
            )
            continue

        try:
            data = load_backtest_data(
                pair=pair,
                data_dir=Path(root) / segment["name"],
                start_time=window[0].to_pydatetime(),
                end_time=window[1].to_pydatetime(),
            )
            engine = (
                TradingSignalEngine(profile=profile)
                if profile is not None
                else None
            )
            result = run_backtest(
                data,
                signal_engine=engine,
                execution_timeframe=(
                    profile.fill_timeframe if profile is not None else "5m"
                ),
                target_index=target_index,
                decision_step=decision_step,
            )
        except Exception as error:
            skipped.append(
                {
                    "segment": segment["name"],
                    "reason": f"{type(error).__name__}: {error}"[:160],
                }
            )
            continue

        results.append(result)
        used.append(
            {
                "segment": segment["name"],
                "from": window[0].isoformat(),
                "to": window[1].isoformat(),
                "days": round(days, 2),
                "trades": len(result["trades"]),
            }
        )

    return {
        "status": "SHARD_COMPLETED",
        "start": start,
        "end": end,
        "used": used,
        "skipped": skipped,
        "results": results,
    }


def combine_results(
    results: list[dict[str, Any]],
    *,
    initial_equity: float = 10000.0,
) -> dict[str, Any]:
    """Merge per-stretch backtests into one chronological result.

    Each stretch starts again from the same initial equity, so profits
    are added rather than compounded across holes. The equity curve is
    shifted by the profit made before each stretch.
    """

    if not results:
        raise SegmentError("No backtest results to combine.")

    ordered = sorted(
        results,
        key=lambda r: r["equity_curve"][0]["timestamp"],
    )

    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    offset = 0.0

    for result in ordered:
        start_equity = float(result["equity_curve"][0]["equity"])

        for point in result["equity_curve"]:
            curve.append(
                {
                    "timestamp": point["timestamp"],
                    "equity": initial_equity
                    + offset
                    + (float(point["equity"]) - start_equity),
                }
            )

        trades.extend(result["trades"])
        offset += float(result["equity_curve"][-1]["equity"]) - start_equity

    trades.sort(key=lambda t: t["decision_time"])
    first = ordered[0]

    return {
        "status": "COMPLETED",
        "execution_timeframe": first.get("execution_timeframe"),
        "execution": first.get("execution", {}),
        "summary": {
            "initial_equity": initial_equity,
            "final_equity": initial_equity + offset,
            "stretches": len(ordered),
        },
        "trades": trades,
        "equity_curve": curve,
        "limitations": list(first.get("limitations", []))
        + [
            "Results are merged from separate continuous stretches; "
            "each starts from the same equity and profits are added."
        ],
    }
