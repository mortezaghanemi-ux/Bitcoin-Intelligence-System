"""Summarize many backtest reports into one comparison table."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def load_reports(directory: str | Path) -> list[dict[str, Any]]:
    reports = []

    for path in sorted(Path(directory).rglob("performance_*_TP*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if isinstance(value, dict) and "overall" in value:
            reports.append(value)

    return reports


def summarize(reports: list[dict[str, Any]]) -> str:
    header = (
        "| Profile | Target | Trades | Win % | PF | OOS trades | OOS PF "
        "| Folds + | Return % | Max DD % | Gate |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []

    for report in sorted(
        reports,
        key=lambda r: (
            not r["activation_gate"]["passed"],
            -(r["overall"].get("profit_factor") or 0),
        ),
    ):
        overall = report["overall"]
        out = report["out_of_sample"]
        walk = report.get("walk_forward", {})
        risk = report.get("risk", {})

        rows.append(
            "| {profile} | {target} | {trades} | {win} | {pf} | {ot} "
            "| {opf} | {folds}/{count} | {ret} | {dd} | {gate} |".format(
                profile=report.get("profile", {}).get("name", "baseline"),
                target=report.get("target"),
                trades=overall["trades"],
                win=_fmt(overall["win_rate_pct"], 1),
                pf=_fmt(overall["profit_factor"]),
                ot=out["trades"],
                opf=_fmt(out["profit_factor"]),
                folds=walk.get("profitable_folds", 0),
                count=walk.get("fold_count", 0),
                ret=_fmt(risk.get("total_return_pct"), 1),
                dd=_fmt(risk.get("maximum_drawdown_pct"), 1),
                gate="PASS" if report["activation_gate"]["passed"] else "no",
            )
        )

    passed = [
        r for r in reports if r["activation_gate"]["passed"]
    ]
    verdict = (
        f"{len(passed)} of {len(reports)} runs passed the activation gate."
        if reports
        else "No reports found."
    )

    return (
        "## Strategy experiments\n\n"
        + header
        + "\n".join(rows)
        + "\n\n"
        + verdict
        + "\n\nGate: profit factor >= 1.2 overall and out of sample, "
        "enough trades, and at least 3 of 4 time folds profitable. "
        "Past results are not a prediction.\n"
    )


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    print(summarize(load_reports(directory)))
