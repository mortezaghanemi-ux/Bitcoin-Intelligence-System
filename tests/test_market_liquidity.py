"""Tests for Bitcoin market liquidity analysis."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_analysis.liquidity import (
    ACTIVE,
    BUY_SIDE,
    SELL_SIDE,
    SWEPT,
    LiquidityAnalysisError,
    analyze_liquidity,
)


UTC = timezone.utc
START = datetime(2026, 11, 14, 0, 0, tzinfo=UTC)
ANALYSIS_TIME = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------


def make_candles(candle_count=40):
    """Create deterministic OHLCV candles."""

    rows = []

    for index in range(candle_count):
        price = 100.0

        rows.append(
            {
                "open_time": START + timedelta(minutes=5 * index),
                "close_time": START + timedelta(minutes=5 * (index + 1)),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 100.0,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def analyze(data, **kwargs):
    """Run liquidity analysis with common test parameters."""

    parameters = {
        "swing_window": 2,
        "minimum_touches": 2,
        "level_tolerance": 0.001,
        "minimum_candles": 30,
        "analysis_time": ANALYSIS_TIME,
    }

    parameters.update(kwargs)

    return analyze_liquidity(
        data=data,
        **parameters,
    )


def add_equal_highs(data):
    """Create two confirmed equal swing highs."""

    data.loc[8, ["open", "high", "low", "close"]] = [
        105.0, 110.0, 104.0, 105.0
    ]

    data.loc[16, ["open", "high", "low", "close"]] = [
        105.0, 110.0, 104.0, 105.0
    ]

    return data


def add_equal_lows(data):
    """Create two confirmed equal swing lows."""

    data.loc[8, ["open", "high", "low", "close"]] = [
        95.0, 96.0, 90.0, 95.0
    ]

    data.loc[16, ["open", "high", "low", "close"]] = [
        95.0, 96.0, 90.0, 95.0
    ]

    return data


def get_pool(result, side):
    """Return the first liquidity pool of the requested side."""

    pools = [
        pool
        for pool in result["pools"]
        if pool["type"] == side
    ]

    assert pools, f"No {side} liquidity pool was detected."

    return pools[0]


# ---------------------------------------------------------------------
# Basic analysis
# ---------------------------------------------------------------------


def test_analysis_returns_expected_structure():
    result = analyze(make_candles())

    expected_keys = {
        "analysis_timestamp",
        "status",
        "source",
        "candles_analyzed",
        "last_candle_close_time",
        "swing_highs",
        "swing_lows",
        "swings",
        "buy_side_liquidity",
        "sell_side_liquidity",
        "pools",
        "sweeps",
        "summary",
        "parameters",
    }

    assert expected_keys.issubset(result.keys())


def test_analysis_returns_success_status():
    result = analyze(make_candles())

    assert result["status"] == "SUCCESS"


def test_analysis_reports_candle_count():
    data = make_candles()

    result = analyze(data)

    assert result["candles_analyzed"] == len(data)


def test_analysis_reports_sources():
    result = analyze(make_candles())

    assert result["source"] == ["kraken"]


def test_analysis_uses_utc_timestamp():
    result = analyze(make_candles())

    timestamp = pd.Timestamp(
        result["analysis_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_analysis_reports_last_candle_close_time():
    data = make_candles()

    result = analyze(data)

    expected = data["close_time"].iloc[-1].isoformat()

    assert result["last_candle_close_time"] == expected


def test_analysis_returns_summary_dictionary():
    result = analyze(make_candles())

    assert isinstance(result["summary"], dict)


def test_analysis_returns_parameters_dictionary():
    result = analyze(make_candles())

    assert isinstance(result["parameters"], dict)


# ---------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------


def test_detects_confirmed_swing_high():
    data = make_candles()

    data.loc[10, ["open", "high", "low", "close"]] = [
        105.0, 110.0, 104.0, 105.0
    ]

    result = analyze(data)

    swing = next(
        item
        for item in result["swing_highs"]
        if item["index"] == 10
    )

    assert swing["price"] == 110.0
    assert swing["confirmation_index"] == 12


def test_detects_confirmed_swing_low():
    data = make_candles()

    data.loc[10, ["open", "high", "low", "close"]] = [
        95.0, 96.0, 90.0, 95.0
    ]

    result = analyze(data)

    swing = next(
        item
        for item in result["swing_lows"]
        if item["index"] == 10
    )

    assert swing["price"] == 90.0
    assert swing["confirmation_index"] == 12


def test_flat_market_has_no_confirmed_swings():
    result = analyze(make_candles())

    assert result["swing_highs"] == []
    assert result["swing_lows"] == []


def test_swing_confirmation_index_is_after_swing_index():
    data = make_candles()

    data.loc[10, "high"] = 110.0

    result = analyze(data)

    swing = next(
        item
        for item in result["swing_highs"]
        if item["index"] == 10
    )

    assert swing["confirmation_index"] > swing["index"]


def test_swing_confirmation_time_is_after_swing_time():
    data = make_candles()

    data.loc[10, "high"] = 110.0

    result = analyze(data)

    swing = next(
        item
        for item in result["swing_highs"]
        if item["index"] == 10
    )

    assert pd.Timestamp(
        swing["confirmation_time"]
    ) > pd.Timestamp(swing["time"])


# ---------------------------------------------------------------------
# Liquidity pool detection
# ---------------------------------------------------------------------


def test_detects_buy_side_liquidity_pool():
    data = add_equal_highs(make_candles())

    result = analyze(data)

    pool = get_pool(result, BUY_SIDE)

    assert pool["touch_count"] >= 2
    assert pool["level"] == pytest.approx(110.0)


def test_detects_sell_side_liquidity_pool():
    data = add_equal_lows(make_candles())

    result = analyze(data)

    pool = get_pool(result, SELL_SIDE)

    assert pool["touch_count"] >= 2
    assert pool["level"] == pytest.approx(90.0)


def test_equal_highs_are_grouped_into_one_pool():
    data = add_equal_highs(make_candles())

    result = analyze(data)

    pools = [
        pool
        for pool in result["pools"]
        if pool["type"] == BUY_SIDE
        and pool["level"] == pytest.approx(110.0)
    ]

    assert len(pools) == 1


def test_equal_lows_are_grouped_into_one_pool():
    data = add_equal_lows(make_candles())

    result = analyze(data)

    pools = [
        pool
        for pool in result["pools"]
        if pool["type"] == SELL_SIDE
        and pool["level"] == pytest.approx(90.0)
    ]

    assert len(pools) == 1


def test_flat_market_has_no_liquidity_pools():
    result = analyze(make_candles())

    assert result["pools"] == []
    assert result["buy_side_liquidity"] == []
    assert result["sell_side_liquidity"] == []


def test_pool_has_confirmation_index():
    data = add_equal_highs(make_candles())

    result = analyze(data)

    pool = get_pool(result, BUY_SIDE)

    assert isinstance(pool["confirmation_index"], int)
    assert pool["confirmation_index"] >= 0


def test_pool_is_active_without_sweep():
    data = add_equal_highs(make_candles())

    result = analyze(data)

    pool = get_pool(result, BUY_SIDE)

    assert pool["status"] == ACTIVE


# ---------------------------------------------------------------------
# Liquidity sweep detection
# ---------------------------------------------------------------------


def test_detects_buy_side_liquidity_sweep():
    data = add_equal_highs(make_candles())

    data.loc[20, ["open", "high", "low", "close"]] = [
        106.0, 112.0, 104.0, 105.0
    ]

    result = analyze(data)

    pool = get_pool(result, BUY_SIDE)

    assert pool["status"] == SWEPT
    assert result["sweeps"]


def test_detects_sell_side_liquidity_sweep():
    data = add_equal_lows(make_candles())

    data.loc[20, ["open", "high", "low", "close"]] = [
        94.0, 96.0, 88.0, 95.0
    ]

    result = analyze(data)

    pool = get_pool(result, SELL_SIDE)

    assert pool["status"] == SWEPT
    assert result["sweeps"]


def test_buy_side_sweep_requires_close_below_level():
    data = add_equal_highs(make_candles())

    data.loc[20, ["open", "high", "low", "close"]] = [
        109.0, 112.0, 108.0, 111.0
    ]

    result = analyze(data)

    pool = get_pool(result, BUY_SIDE)

    assert pool["status"] == ACTIVE


def test_sell_side_sweep_requires_close_above_level():
    data = add_equal_lows(make_candles())

    data.loc[20, ["open", "high", "low", "close"]] = [
        91.0, 92.0, 88.0, 89.0
    ]

    result = analyze(data)

    pool = get_pool(result, SELL_SIDE)

    assert pool["status"] == ACTIVE


def test_sweep_is_not_detected_before_pool_confirmation():
    data = add_equal_highs(make_candles())

    # First swing is confirmed at index 10.
    # The second swing is confirmed at index 18.
    # Index 11 is after the first confirmation
    # but before the second confirmation.
    data.loc[11, ["open", "high", "low", "close"]] = [
        106.0, 112.0, 104.0, 105.0
    ]

    result = analyze(data)

    pool = get_pool(result, BUY_SIDE)

    assert pool["confirmation_index"] == 18
    assert pool["status"] == ACTIVE
    assert pool["sweeps"] == []


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("swing_window", 0),
        ("swing_window", -1),
        ("swing_window", True),
        ("minimum_touches", 0),
        ("minimum_touches", 1),
        ("minimum_touches", True),
        ("minimum_candles", 0),
        ("minimum_candles", -1),
        ("minimum_candles", True),
    ],
)
def test_rejects_invalid_integer_parameters(parameter, value):
    with pytest.raises(LiquidityAnalysisError):
        analyze(
            make_candles(),
            **{parameter: value},
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -0.1,
        0.05,
        1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_invalid_level_tolerance(value):
    with pytest.raises(LiquidityAnalysisError):
        analyze(
            make_candles(),
            level_tolerance=value,
        )


def test_rejects_minimum_candles_below_swing_requirement():
    with pytest.raises(LiquidityAnalysisError):
        analyze(
            make_candles(),
            swing_window=3,
            minimum_candles=6,
        )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_rejects_non_dataframe_input():
    with pytest.raises(TypeError):
        analyze([])


def test_rejects_empty_dataframe():
    with pytest.raises(LiquidityAnalysisError):
        analyze(pd.DataFrame())


def test_rejects_missing_required_column():
    data = make_candles().drop(columns=["volume"])

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


def test_rejects_insufficient_candles():
    data = make_candles(candle_count=10)

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


def test_rejects_naive_analysis_timestamp():
    with pytest.raises(LiquidityAnalysisError):
        analyze(
            make_candles(),
            analysis_time=datetime(2026, 11, 14, 12, 0),
        )


def test_rejects_non_utc_analysis_timestamp():
    non_utc = timezone(timedelta(hours=3))

    with pytest.raises(LiquidityAnalysisError):
        analyze(
            make_candles(),
            analysis_time=datetime(
                2026, 11, 14, 12, 0,
                tzinfo=non_utc,
            ),
        )


def test_rejects_unsorted_candles():
    data = make_candles()

    data = data.iloc[::-1].reset_index(drop=True)

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


def test_rejects_duplicate_candle_timestamps():
    data = make_candles()

    data.loc[1, "open_time"] = data.loc[0, "open_time"]

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


def test_rejects_invalid_ohlc():
    data = make_candles()

    data.loc[10, "high"] = 90.0

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


def test_rejects_open_candles():
    data = make_candles()

    data.loc[len(data) - 1, "close_time"] = (
        ANALYSIS_TIME + timedelta(minutes=5)
    )

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


def test_rejects_non_utc_candle_timestamps():
    data = make_candles()

    non_utc = timezone(timedelta(hours=3))

    data["open_time"] = data["open_time"].map(
        lambda value: value.astimezone(non_utc)
    )

    data["close_time"] = data["close_time"].map(
        lambda value: value.astimezone(non_utc)
    )

    with pytest.raises(LiquidityAnalysisError):
        analyze(data)


# ---------------------------------------------------------------------
# Data integrity and determinism
# ---------------------------------------------------------------------


def test_analysis_does_not_modify_input_dataframe():
    data = add_equal_highs(make_candles())

    original = data.copy(deep=True)

    analyze(data)

    pd.testing.assert_frame_equal(data, original)


def test_analysis_is_deterministic():
    data = add_equal_highs(make_candles())

    first = analyze(data)
    second = analyze(data)

    assert first == second


def test_analysis_does_not_generate_trading_signals():
    result = analyze(make_candles())

    forbidden_keys = {
        "signal",
        "entry",
        "stop_loss",
        "take_profit",
        "order",
        "trade",
    }

    assert forbidden_keys.isdisjoint(result.keys())


def test_all_detected_pools_have_valid_status():
    data = add_equal_highs(make_candles())

    result = analyze(data)

    for pool in result["pools"]:
        assert pool["status"] in {ACTIVE, SWEPT}


def test_sweeps_are_returned_as_a_list():
    data = add_equal_highs(make_candles())

    data.loc[20, ["open", "high", "low", "close"]] = [
        106.0, 112.0, 104.0, 105.0
    ]

    result = analyze(data)

    assert isinstance(result["sweeps"], list)