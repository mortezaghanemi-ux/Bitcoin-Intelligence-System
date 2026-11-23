import json
from datetime import datetime, timedelta, timezone

from trader.live_tracking import (
    build_live_performance,
    compare_with_backtest,
    read_candles,
    resolve_signal,
    summarize_live,
    track_signals,
)

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def candle(minutes, high, low):
    return {
        "open_time": json_time(T0 + timedelta(minutes=minutes)),
        "high": high,
        "low": low,
    }


def json_time(value):
    import pandas as pd

    return pd.Timestamp(value)


def signal(direction="LONG", minutes=0, entry=100.0, stop=98.0, tp=104.0):
    return {
        "signal_id": f"s{minutes}",
        "signal": direction,
        "analysis_timestamp": (T0 + timedelta(minutes=minutes)).isoformat(),
        "current_price": entry,
        "stop_loss": stop,
        "take_profit": [{"level": 1, "price": tp}],
    }


def test_long_target_and_stop():
    win = resolve_signal(signal(), [candle(5, 101, 99), candle(10, 104.5, 100)])
    loss = resolve_signal(signal(), [candle(5, 101, 97.5)])

    assert win["outcome"] == "TARGET" and win["r"] == 2.0
    assert loss["outcome"] == "STOP_LOSS" and loss["r"] == -1.0


def test_stop_wins_when_both_hit_in_one_candle():
    result = resolve_signal(signal(), [candle(5, 110, 90)])

    assert result["outcome"] == "STOP_LOSS"


def test_short_signal_directions():
    short = signal("SHORT", entry=100, stop=102, tp=96)

    assert resolve_signal(short, [candle(5, 100.5, 95)])["outcome"] == "TARGET"
    assert resolve_signal(short, [candle(5, 103, 99)])["outcome"] == "STOP_LOSS"


def test_open_expired_and_invalid():
    assert resolve_signal(signal(), [candle(5, 101, 99)])["outcome"] == "OPEN"
    assert resolve_signal(signal(), [candle(60 * 50, 101, 99)])["outcome"] == "EXPIRED"
    assert resolve_signal(signal(stop=101), [])["outcome"] == "INVALID"


def test_one_position_at_a_time():
    signals = [signal(minutes=0), signal(minutes=10), signal(minutes=40)]
    candles = [candle(5, 101, 99), candle(30, 105, 100), candle(45, 101, 97)]

    results = track_signals(signals, candles)

    # Second signal arrives while the first is open and is skipped.
    assert [r["signal_id"] for r in results] == ["s0", "s40"]
    assert [r["outcome"] for r in results] == ["TARGET", "STOP_LOSS"]


def test_summary_and_comparison():
    results = [
        {"outcome": "TARGET", "r": 2.0},
        {"outcome": "STOP_LOSS", "r": -1.0},
        {"outcome": "OPEN", "r": None},
    ]

    summary = summarize_live(results)

    assert summary["resolved"] == 2 and summary["open"] == 1
    assert summary["win_rate_pct"] == 50.0
    assert summary["profit_factor"] == 2.0

    few = compare_with_backtest(summary, None)

    assert few["status"] == "INSUFFICIENT_SAMPLE"
    assert few["signals_paused"] is False


def test_comparison_flags_underperformance():
    live = {"resolved": 30, "profit_factor": 0.7, "win_rate_pct": 40.0}

    assert compare_with_backtest(live, None)["signals_paused"] is True

    worse = {"resolved": 30, "profit_factor": 1.1, "win_rate_pct": 40.0}
    backtest = {"overall": {"win_rate_ci95_pct": [50.0, 60.0]}}

    assert compare_with_backtest(worse, backtest)["status"] == "WORSE_THAN_BACKTEST"

    fine = {"resolved": 30, "profit_factor": 1.4, "win_rate_pct": 55.0}

    assert compare_with_backtest(fine, backtest)["status"] == "CONSISTENT"


def test_build_from_files_ignores_bad_lines(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps(signal()) + "\nnot json\n"
        + json.dumps({"signal": "NO_TRADE"}) + "\n"
    )
    candles = tmp_path / "kraken_XBTUSD_5m.jsonl"
    candles.write_text(
        json.dumps(
            {
                "open_time": (T0 + timedelta(minutes=5)).isoformat(),
                "high": 105, "low": 99,
            }
        )
        + "\n{broken\n"
    )

    assert len(read_candles(candles)) == 1

    result = build_live_performance(
        history_path=history, candle_path=candles, now=T0
    )

    assert result["status"] == "AVAILABLE"
    assert result["summary"]["wins"] == 1
    assert build_live_performance(
        history_path=tmp_path / "none", candle_path=candles
    )["status"] == "UNAVAILABLE"
