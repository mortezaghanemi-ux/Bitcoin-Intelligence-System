from datetime import datetime, timezone

import pandas as pd
import pytest

from tests.test_trader_signal_engine import (
    CHECK_TIME,
    make_candles,
    make_market_data,
)
from trader.signal_engine.engine import TradingSignalEngine
from trader.strategy_profile import (
    StrategyProfile,
    StrategyProfileError,
    evaluate_filters,
    get_profile,
    load_profiles,
)


def call(profile, direction="BULLISH", directions=None, candles=None):
    execution = candles if candles is not None else make_candles("1h", 100)

    return evaluate_filters(
        profile,
        analysis_time=CHECK_TIME,
        candles_by_timeframe={"4h": make_candles("4h", 100)},
        overall_direction=direction,
        timeframe_directions=directions or {},
        execution_candles=execution,
        atr=1.0,
    )


def test_shipped_profiles_load_and_are_valid():
    profiles, active = load_profiles()

    assert active == "baseline"
    assert {"baseline", "h1_base", "h1_regime"} <= set(profiles)
    assert profiles["h1_base"].execution_timeframe == "1h"


def test_invalid_profiles_are_rejected():
    with pytest.raises(StrategyProfileError):
        StrategyProfile(name="x", execution_timeframe="2h")

    with pytest.raises(StrategyProfileError):
        StrategyProfile(name="x", allowed_regimes=("SIDEWAYS",))

    with pytest.raises(StrategyProfileError):
        StrategyProfile(name="bad name")

    with pytest.raises(StrategyProfileError):
        get_profile("missing")


def test_empty_profile_never_rejects():
    reasons, _ = call(StrategyProfile(name="empty"))

    assert reasons == []


def test_higher_timeframe_bias_mismatch_is_rejected():
    profile = StrategyProfile(name="p", higher_timeframe_bias="4h")

    mismatch, _ = call(profile, directions={"4h": "BEARISH"})
    match, _ = call(profile, directions={"4h": "BULLISH"})

    assert any("4h direction" in item for item in mismatch)
    assert match == []


def test_blackout_window_blocks_signals():
    start = (CHECK_TIME - pd.Timedelta(hours=1)).isoformat()
    end = (CHECK_TIME + pd.Timedelta(hours=1)).isoformat()
    profile = StrategyProfile(name="p", blackout_windows=((start, end),))

    reasons, _ = call(profile)

    assert any("blackout" in item for item in reasons)


def test_pullback_filter_rejects_extended_price():
    candles = make_candles("1h", 100)
    # make_candles trends steadily upward, far above its EMA20.
    profile = StrategyProfile(name="p", maximum_pullback_atr=0.5)

    reasons, details = call(profile, candles=candles)

    assert "ema20_distance_atr" in details
    assert any("pullback" in item for item in reasons)


def test_regime_filter_fails_closed_without_usable_data():
    profile = StrategyProfile(name="p", allowed_regimes=("TRENDING",))

    reasons, _ = evaluate_filters(
        profile,
        analysis_time=CHECK_TIME,
        candles_by_timeframe={"4h": make_candles("4h", 5)},
        overall_direction="BULLISH",
        timeframe_directions={},
        execution_candles=make_candles("1h", 100),
        atr=1.0,
    )

    assert any("Regime analysis unavailable" in item for item in reasons)


def test_neutral_direction_skips_filters():
    profile = StrategyProfile(name="p", higher_timeframe_bias="4h")

    reasons, _ = call(profile, direction="NEUTRAL")

    assert reasons == []


def test_engine_applies_profile_and_changes_signal_id():
    data = make_market_data()
    plain = TradingSignalEngine().generate(data, analysis_time=CHECK_TIME)
    profile = StrategyProfile(
        name="h1", execution_timeframe="1h", higher_timeframe_bias="4h"
    )
    engine = TradingSignalEngine(profile=profile)
    result = engine.generate(data, analysis_time=CHECK_TIME)

    assert engine.execution_timeframe == "1h"
    assert result["parameters"]["profile"] == "h1"
    assert result["timeframe"] == "1h"
    assert result["signal_id"] != plain["signal_id"]
