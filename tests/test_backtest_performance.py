"""Tests for performance evaluation, history import and engine fixes."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestingEngine, BacktestingEngineError
from backtesting.importer import (
    HistoryImportError,
    import_kraken_history,
)
from backtesting.performance import (
    PerformanceError,
    evaluate_performance,
    load_latest_performance,
    lookup_signal_performance,
    save_performance_report,
    summarize_trades,
    wilson_interval,
)

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def make_result(pnls, confidences=None, directions=None):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    trades = []

    for i, pnl in enumerate(pnls):
        trades.append(
            {
                "signal_id": f"s{i}",
                "direction": (directions or ["LONG"] * len(pnls))[i],
                "confidence": (confidences or [75] * len(pnls))[i],
                "decision_time": (start + pd.Timedelta(hours=i)).isoformat(),
                "net_pnl": float(pnl),
                "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
            }
        )

    equity = 10000.0
    curve = [{"timestamp": start.isoformat(), "equity": equity}]

    for i, pnl in enumerate(pnls):
        equity += pnl
        curve.append(
            {
                "timestamp": (start + pd.Timedelta(hours=i + 1)).isoformat(),
                "equity": equity,
            }
        )

    return {
        "status": "COMPLETED",
        "execution_timeframe": "5m",
        "execution": {
            "simulation_only": True,
            "exchange_connection": False,
            "order_execution": False,
            "target": "TP1",
            "fee_rate": 0.0006,
            "slippage_bps": 2.0,
            "leverage": 1.0,
        },
        "summary": {
            "initial_equity": 10000.0,
            "final_equity": equity,
        },
        "trades": trades,
        "equity_curve": curve,
        "limitations": ["Simulation only."],
    }


def test_wilson_interval_bounds():
    low, high = wilson_interval(8, 10)

    assert 0 < low < 80 < high < 100
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 100.0


def test_summarize_trades_counts_and_sample_flag():
    trades = [{"net_pnl": 10}, {"net_pnl": -5}, {"net_pnl": 0}]

    summary = summarize_trades(trades, minimum_sample=30)

    assert summary["trades"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["breakeven"] == 1
    assert summary["win_rate_pct"] == pytest.approx(33.33, abs=0.01)
    assert summary["profit_factor"] == 2.0
    assert summary["sample_status"] == "INSUFFICIENT_SAMPLE"

    empty = summarize_trades([])
    assert empty["win_rate_pct"] is None


def test_evaluate_performance_groups_and_split():
    pnls = [10, -5, 10, 10, -5, 10, 10, -5, 10, 10]
    confidences = [65, 75, 85, 85, 75, 65, 85, 75, 85, 85]
    directions = ["LONG", "SHORT"] * 5

    report = evaluate_performance(
        make_result(pnls, confidences, directions),
        train_fraction=0.7,
        minimum_sample=5,
        generated_at=NOW,
    )

    assert report["overall"]["trades"] == 10
    assert report["overall"]["wins"] == 7
    assert (
        report["in_sample"]["trades"] + report["out_of_sample"]["trades"]
        == 10
    )
    assert report["by_direction"]["LONG"]["trades"] == 5
    assert report["by_confidence"]["80-100"]["trades"] == 5
    assert report["by_confidence"]["60-69"]["trades"] == 2
    assert report["by_exit_reason"]["STOP_LOSS"] == 3
    assert report["risk"]["total_return_pct"] is not None
    assert report["risk"]["maximum_drawdown_pct"] is not None
    assert "not a prediction" in report["limitations"][-1]


def test_evaluate_rejects_bad_input():
    with pytest.raises(PerformanceError):
        evaluate_performance({"status": "FAILED"})

    with pytest.raises(PerformanceError):
        evaluate_performance(make_result([1, -1]), train_fraction=0.99)


def test_save_and_load_round_trip(tmp_path):
    report = evaluate_performance(
        make_result([10, -5, 10]), generated_at=NOW
    )

    path = save_performance_report(report, tmp_path)

    assert path.name == "performance_latest.json"
    assert list((tmp_path / "history").glob("performance_TP1_*.json"))
    assert load_latest_performance(tmp_path)["overall"]["trades"] == 3
    assert load_latest_performance(tmp_path / "missing") is None


def test_lookup_prefers_specific_group_with_enough_trades():
    report = evaluate_performance(
        make_result(
            [10, -5] * 10,
            confidences=[85] * 20,
            directions=["LONG"] * 20,
        ),
        minimum_sample=10,
        generated_at=NOW,
    )

    found = lookup_signal_performance(
        report, {"signal": "LONG", "confidence": 85}
    )

    assert found["status"] == "MEASURED"
    assert found["scope"] == "confidence 80-100"
    assert found["win_rate_pct"] == 50.0
    assert found["trades"] == 20
    assert "not a prediction" in found["disclaimer"]

    # SHORT has no trades, so it falls back to all signals.
    fallback = lookup_signal_performance(
        report, {"signal": "SHORT", "confidence": 62}
    )
    assert fallback["scope"] == "all signals"


def test_lookup_flags_small_sample_and_missing_backtest():
    report = evaluate_performance(
        make_result([10, -5, 10]), generated_at=NOW
    )

    small = lookup_signal_performance(
        report, {"signal": "LONG", "confidence": 75}
    )
    assert small["status"] == "INSUFFICIENT_SAMPLE"

    assert (
        lookup_signal_performance(None, {"signal": "LONG"})["status"]
        == "NO_BACKTEST"
    )


def test_engine_parameter_validation():
    with pytest.raises(BacktestingEngineError):
        BacktestingEngine(target_index=3)

    with pytest.raises(BacktestingEngineError):
        BacktestingEngine(decision_step=0)

    with pytest.raises(BacktestingEngineError):
        BacktestingEngine(lookback_candles=5)


def write_csv(path, candles, gap_at=None):
    rng = np.random.default_rng(3)
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    close = 90000 * np.exp(np.cumsum(rng.normal(0, 0.001, candles)))
    rows = []

    for i in range(candles):
        if gap_at is not None and i in gap_at:
            continue

        price = close[i]
        rows.append(
            f"{start + i * 300},{price},{price * 1.001},"
            f"{price * 0.999},{price},1.5,3"
        )

    path.write_text("\n".join(rows))


def test_import_history_builds_all_timeframes(tmp_path):
    csv = tmp_path / "XBTUSD_5.csv"
    write_csv(csv, 288 * 3, gap_at={50, 51})

    result = import_kraken_history(csv, output_dir=tmp_path / "out")

    assert result["status"] == "IMPORTED"
    assert result["gaps"]["filled_candles"] == 2
    assert result["candles"]["5m"] == 288 * 3
    assert result["candles"]["4h"] == 18
    assert result["candles"]["1h"] == 72

    first = json.loads(
        (tmp_path / "out" / "kraken_XBTUSD_4h.jsonl")
        .read_text()
        .splitlines()[0]
    )

    assert first["source"] == "kraken_history"
    assert first["timeframe"] == "4h"
    assert first["open_time"].startswith("2026-01-01T00:00")


def test_import_history_rejects_bad_files(tmp_path):
    with pytest.raises(HistoryImportError):
        import_kraken_history(tmp_path / "missing.csv")

    short = tmp_path / "short.csv"
    write_csv(short, 20)

    with pytest.raises(HistoryImportError):
        import_kraken_history(short, output_dir=tmp_path / "o")

    misaligned = tmp_path / "bad.csv"
    misaligned.write_text(
        "\n".join(f"{1767225600 + i * 301},1,1,1,1,1,1" for i in range(200))
    )

    with pytest.raises(HistoryImportError):
        import_kraken_history(misaligned, output_dir=tmp_path / "o2")


def test_import_history_drops_data_before_long_gap(tmp_path):
    csv = tmp_path / "gap.csv"
    # a 100-candle hole, far above the fill limit
    write_csv(csv, 288 * 3, gap_at=set(range(100, 200)))

    result = import_kraken_history(csv, output_dir=tmp_path / "out")

    assert result["gaps"]["long_gap_count"] == 1
    assert result["gaps"]["long_gaps_dropped_rows"] == 100
    assert result["candles"]["5m"] < 288 * 3 - 100


def test_equity_curve_replaces_duplicate_timestamp():
    engine = BacktestingEngine()
    stamp = pd.Timestamp("2026-01-01T00:05:00", tz="UTC")

    engine._record_equity(stamp)
    engine.equity = 9900.0
    engine._record_equity(stamp)

    assert len(engine.equity_curve) == 1
    assert engine.equity_curve[0]["equity"] == 9900.0


def test_signal_engine_analysis_window_is_validated():
    from trader.signal_engine.engine import (
        TradingSignalEngine,
        TradingSignalEngineError,
    )

    assert TradingSignalEngine().analysis_window == 120
    assert TradingSignalEngine(analysis_window=None).analysis_window is None

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine(analysis_window=10)


def test_edge_verdict_flags_losing_rules():
    from backtesting.performance import edge_verdict

    losing = evaluate_performance(
        make_result([10, -12] * 10), minimum_sample=5, generated_at=NOW
    )

    assert edge_verdict(losing) == "NO_EDGE_MEASURED"

    found = lookup_signal_performance(
        losing, {"signal": "LONG", "confidence": 75}
    )

    assert found["edge_verdict"] == "NO_EDGE_MEASURED"
    assert found["overall_profit_factor"] < 1
    assert found["total_return_pct"] < 0


def test_edge_verdict_requires_both_parts_profitable():
    from backtesting.performance import edge_verdict

    winning = evaluate_performance(
        make_result([12, 12, -10] * 20), minimum_sample=5, generated_at=NOW
    )

    assert edge_verdict(winning) == "EDGE_MEASURED"
    assert edge_verdict(
        {"overall": {"profit_factor": 1.5, "sample_status": "SUFFICIENT"},
         "out_of_sample": {"profit_factor": 0.7, "trades": 30,
                           "sample_status": "SUFFICIENT"}}
    ) == "NO_EDGE_MEASURED"


def test_walk_forward_and_gate():
    from backtesting.performance import (
        activation_gate,
        walk_forward_folds,
    )

    steady = evaluate_performance(
        make_result([12, 12, -10] * 20), minimum_sample=5, generated_at=NOW
    )

    assert steady["walk_forward"]["profitable_folds"] == 4
    assert steady["activation_gate"]["passed"] is True

    # Profit comes only from the first quarter of the trades.
    uneven = [30] * 3 + [-10] * 3 + [-1] * 12 + [12, -10] * 6 + [-5] * 12
    report = evaluate_performance(
        make_result(uneven * 2), minimum_sample=5, generated_at=NOW
    )

    assert report["activation_gate"]["checks"]["walk_forward_consistency"] is False
    assert report["activation_gate"]["passed"] is False
    assert walk_forward_folds([])["folds"] == []
    assert activation_gate(steady)["rules"]["minimum_profit_factor"] == 1.2


def test_confidence_check_needs_enough_trades():
    from backtesting.performance import confidence_informativeness

    assert confidence_informativeness([])["status"] == "INSUFFICIENT_SAMPLE"

    trades = [
        {"confidence": 60 + (i % 40), "net_pnl": 1 if (i % 40) > 20 else -1}
        for i in range(200)
    ]

    result = confidence_informativeness(trades)

    assert result["status"] == "MEASURED"
    assert result["informative"] is True


def test_profile_is_recorded_and_saved_per_profile(tmp_path):
    report = evaluate_performance(
        make_result([10, -5, 10]),
        generated_at=NOW,
        profile={"name": "h1_test"},
    )

    path = save_performance_report(report, tmp_path, set_latest=False)

    assert path.name == "performance_h1_test_TP1.json"
    assert not (tmp_path / "performance_latest.json").exists()
