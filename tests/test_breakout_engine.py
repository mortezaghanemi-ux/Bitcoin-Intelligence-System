from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.breakout_engine import DonchianBreakoutEngine
from trader.strategy_profile import build_engine, get_profile


def _candles(closes):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i, c in enumerate(closes):
        open_time = start + timedelta(hours=4 * i)
        rows.append({
            "open_time": open_time,
            "close_time": open_time + timedelta(hours=4),
            "open": c, "high": c + 1, "low": c - 1, "close": c,
            "volume": 1.0,
        })
    return pd.DataFrame(rows)


def _run(closes, minutes_after=5):
    candles = _candles(closes)
    when = candles.iloc[-1]["close_time"].to_pydatetime() + timedelta(
        minutes=minutes_after)
    engine = DonchianBreakoutEngine(channel=20)
    return engine.generate({"4h": candles}, analysis_time=when)


def test_fresh_breakout_long():
    signal = _run([100.0] * 40 + [130.0])
    assert signal["signal"] == "LONG"
    assert signal["stop_loss"] < 130.0
    prices = [t["price"] for t in signal["take_profit"]]
    assert prices == sorted(prices) and prices[0] > 130.0


def test_breakdown_short():
    signal = _run([100.0] * 40 + [70.0])
    assert signal["signal"] == "SHORT"
    assert signal["stop_loss"] > 70.0


def test_second_candle_is_not_fresh_breakout():
    assert _run([100.0] * 40 + [130.0, 131.0])["signal"] == "NO_TRADE"


def test_stale_candle_is_ignored():
    assert _run([100.0] * 40 + [130.0], minutes_after=120)["signal"] == "NO_TRADE"


def test_profiles_build_the_right_engine():
    assert isinstance(build_engine(get_profile("donchian55")), DonchianBreakoutEngine)
    assert build_engine(get_profile("h4_base")).__class__.__name__ == "TradingSignalEngine"
    assert get_profile("donchian55").breakout_channel == 55
