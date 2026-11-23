from datetime import datetime, timezone

from backtesting.experiment_summary import load_reports, summarize
from backtesting.performance import evaluate_performance, save_performance_report
from tests.test_backtest_performance import make_result

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def test_summary_table_orders_passing_runs_first(tmp_path):
    good = evaluate_performance(
        make_result([12, 12, -10] * 20),
        minimum_sample=5, generated_at=NOW, profile={"name": "good"},
    )
    bad = evaluate_performance(
        make_result([10, -12] * 20),
        minimum_sample=5, generated_at=NOW, profile={"name": "bad"},
    )

    save_performance_report(bad, tmp_path, set_latest=False)
    save_performance_report(good, tmp_path, set_latest=False)

    text = summarize(load_reports(tmp_path))
    lines = [x for x in text.splitlines() if x.startswith("| good") or x.startswith("| bad")]

    assert lines[0].startswith("| good") and lines[0].endswith("PASS |")
    assert lines[1].startswith("| bad") and lines[1].endswith("no |")
    assert "1 of 2 runs passed" in text


def test_summary_without_reports(tmp_path):
    assert "No reports found" in summarize(load_reports(tmp_path))
