import json

import numpy as np
import pandas as pd
import pytest

import main
from backtesting.importer import (
    HistoryImportError,
    import_kraken_history_segments,
)
from backtesting.performance import evaluate_performance
from backtesting.segments import SegmentError, combine_results, load_index
from tests.test_backtest_performance import make_result

START = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())


def write_stretches(path, stretches):
    """stretches: list of (days, gap_candles_before)."""

    rng = np.random.default_rng(1)
    rows = []
    cursor = START
    price = 90000.0

    for days, gap in stretches:
        cursor += gap * 300

        for _ in range(int(days * 288)):
            price *= float(np.exp(rng.normal(0, 0.001)))
            rows.append(
                f"{cursor},{price},{price*1.001},{price*0.999},{price},2,3"
            )
            cursor += 300

    path.write_text("\n".join(rows))


def test_segmented_import_keeps_every_long_stretch(tmp_path):
    csv = tmp_path / "XBTUSD_5.csv"
    # 20 days, a 3-day hole, 16 days, a 2-day hole, 5 days (too short)
    write_stretches(csv, [(20, 0), (16, 864), (5, 576)])

    result = import_kraken_history_segments(
        csv, output_dir=tmp_path / "seg", minimum_days=14
    )

    index = load_index(tmp_path / "seg")

    assert result["segment_count"] == 2
    assert result["skipped_short_stretches"] == 1
    assert [s["name"][:8] for s in index["segments"]] == [
        "segment_", "segment_"
    ]
    assert all(s["days"] >= 14 for s in index["segments"])

    for segment in index["segments"]:
        for timeframe in ("5m", "15m", "1h", "4h"):
            assert (
                tmp_path / "seg" / segment["name"]
                / f"kraken_XBTUSD_{timeframe}.jsonl"
            ).is_file()


def test_segmented_import_rejects_all_short(tmp_path):
    csv = tmp_path / "x.csv"
    write_stretches(csv, [(3, 0), (3, 864)])

    with pytest.raises(HistoryImportError):
        import_kraken_history_segments(
            csv, output_dir=tmp_path / "s", minimum_days=14
        )


def shifted(result, days, pnl_scale=1.0):
    offset = pd.Timedelta(days=days)

    for trade in result["trades"]:
        trade["decision_time"] = (
            pd.Timestamp(trade["decision_time"]) + offset
        ).isoformat()
        trade["net_pnl"] *= pnl_scale

    for point in result["equity_curve"]:
        point["timestamp"] = (
            pd.Timestamp(point["timestamp"]) + offset
        ).isoformat()

    return result


def test_combine_adds_profit_across_stretches():
    first = make_result([10, -5, 10])
    second = shifted(make_result([20, -5]), days=30)

    merged = combine_results([second, first])

    assert [t["net_pnl"] for t in merged["trades"]] == [10, -5, 10, 20, -5]
    assert merged["summary"]["final_equity"] == pytest.approx(10030.0)
    assert merged["summary"]["stretches"] == 2

    equities = [p["equity"] for p in merged["equity_curve"]]

    # Second stretch continues from the first one's final equity.
    assert equities[3] == pytest.approx(10015.0)
    assert equities[-1] == pytest.approx(10030.0)

    with pytest.raises(SegmentError):
        combine_results([])


def test_holdout_is_scored_separately_and_gates():
    pnls = [12, 12, -10] * 20
    result = make_result(pnls)
    # decision_time starts 2026-01-01 and moves hourly; cut after 45 trades
    report = evaluate_performance(
        result,
        minimum_sample=5,
        holdout_before="2026-01-02T21:00:00",
    )

    assert report["holdout"]["trades"] == 45
    assert report["holdout"]["profit_factor"] == pytest.approx(2.4)
    assert "holdout_profit_factor" in report["activation_gate"]["checks"]

    small = evaluate_performance(
        make_result(pnls), minimum_sample=5,
        holdout_before="2026-01-01T05:00:00",
    )

    assert small["activation_gate"]["checks"]["holdout_trades"] is False
    assert small["activation_gate"]["passed"] is False


def test_evaluate_shards_command_end_to_end(tmp_path):
    shard = {
        "profile": "h1_regime",
        "target": 2,
        "used": [{"segment": "s1", "trades": 60}],
        "results": [make_result([12, 12, -10] * 20)],
    }
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "shard_h1_regime_TP2_2026.json").write_text(
        json.dumps(shard)
    )
    # A shard for another profile must be ignored.
    other = dict(shard, profile="baseline")
    (tmp_path / "in" / "shard_baseline_TP2_2026.json").write_text(
        json.dumps(other)
    )

    result = main.run_evaluate_shards(
        inputs=str(tmp_path / "in"),
        profile="h1_regime",
        target=2,
        holdout_before="2026-01-03T00:00:00",
        output_dir=str(tmp_path / "out"),
    )

    assert result["stretches"] == 1
    assert result["overall"]["trades"] == 60
    assert (tmp_path / "out" / "performance_h1_regime_TP2.json").is_file()
    assert not (tmp_path / "out" / "performance_latest.json").exists()

    with pytest.raises(main.ApplicationError):
        main.run_evaluate_shards(
            inputs=str(tmp_path / "in"), profile="nope", target=2,
            holdout_before=None, output_dir=str(tmp_path / "out"),
        )
