"""Edge-case tests for Bitcoin market liquidity analysis."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from market_analysis.liquidity import (
    ACTIVE,
    BUY_SIDE,
    SELL_SIDE,
    SWEPT,
    LiquidityAnalysisError,
    _detect_liquidity_sweeps,
    _find_confirmed_swings,
    _group_liquidity_levels,
    analyze_liquidity,
)


UTC = timezone.utc

ANALYSIS_TIME = datetime(
    2026, 11, 17, 12, 0, tzinfo=UTC
)


def make_candles(
    highs,
    lows,
    closes=None,
    start_time=None,
    interval_minutes=5,
):
    """Build deterministic, closed UTC OHLCV candles."""

    if start_time is None:
        start_time = datetime(
            2026, 11, 17, 0, 0, tzinfo=UTC
        )

    if closes is None:
        closes = [
            (high + low) / 2
            for high, low in zip(highs, lows)
        ]

    rows = []

    for index, (high, low, close) in enumerate(
        zip(highs, lows, closes)
    ):
        open_time = start_time + timedelta(
            minutes=interval_minutes * index
        )

        close_time = open_time + timedelta(
            minutes=interval_minutes
        )

        open_price = float(close)

        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 100.0 + index,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def make_valid_liquidity_data(count=40):
    """Create valid candles with confirmed repeated swing highs."""

    highs = []
    lows = []
    closes = []

    for index in range(count):
        base = 100.0 + (index % 4)

        highs.append(base + 2.0)
        lows.append(base - 2.0)
        closes.append(base)

    # Create separated, confirmed swing highs.
    for index in (6, 14, 22, 30):
        highs[index] = 120.0
        lows[index] = 98.0
        closes[index] = 110.0

    # Create separated, confirmed swing lows.
    for index in (10, 18, 26, 34):
        highs[index] = 102.0
        lows[index] = 80.0
        closes[index] = 90.0

    return make_candles(
        highs=highs,
        lows=lows,
        closes=closes,
    )


def test_analyze_liquidity_returns_expected_structure():
    data = make_valid_liquidity_data()

    result = analyze_liquidity(
        data,
        swing_window=2,
        minimum_touches=2,
        minimum_candles=30,
        analysis_time=ANALYSIS_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert result["candles_analyzed"] == len(data)
    assert result["source"] == ["kraken"]
    assert result["summary"]["total_pools"] >= 0
    assert isinstance(result["pools"], list)
    assert isinstance(result["sweeps"], list)


def test_analysis_does_not_modify_input_dataframe():
    data = make_valid_liquidity_data()
    original = data.copy(deep=True)

    analyze_liquidity(
        data,
        swing_window=2,
        minimum_touches=2,
        minimum_candles=30,
        analysis_time=ANALYSIS_TIME,
    )

    pd.testing.assert_frame_equal(data, original)


def test_analysis_rejects_non_dataframe_input():
    with pytest.raises(TypeError):
        analyze_liquidity(
            [],
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_empty_dataframe():
    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            pd.DataFrame(),
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_missing_required_columns():
    data = make_valid_liquidity_data().drop(
        columns=["volume"]
    )

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            analysis_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "analysis_time",
    [
        datetime(2026, 11, 17, 12, 0),
        datetime(
            2026, 11, 17, 12, 0,
            tzinfo=timezone(timedelta(hours=3)),
        ),
        "invalid",
        pd.NaT,
    ],
)
def test_analysis_rejects_invalid_analysis_time(
    analysis_time,
):
    data = make_valid_liquidity_data()

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=analysis_time,
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("swing_window", 0),
        ("swing_window", -1),
        ("swing_window", True),
        ("swing_window", 2.5),
        ("minimum_touches", 0),
        ("minimum_touches", 1),
        ("minimum_touches", True),
        ("minimum_candles", 0),
        ("minimum_candles", -1),
        ("minimum_candles", True),
    ],
)
def test_analysis_rejects_invalid_integer_parameters(
    parameter,
    value,
):
    data = make_valid_liquidity_data()

    parameters = {
        "swing_window": 2,
        "minimum_touches": 2,
        "minimum_candles": 30,
        "analysis_time": ANALYSIS_TIME,
    }

    parameters[parameter] = value

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(data, **parameters)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -0.1,
        0.05,
        1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        "0.001",
    ],
)
def test_analysis_rejects_invalid_level_tolerance(value):
    data = make_valid_liquidity_data()

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            level_tolerance=value,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_insufficient_minimum_candles():
    data = make_valid_liquidity_data()

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=3,
            minimum_candles=6,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_insufficient_input_rows():
    data = make_valid_liquidity_data(count=20)

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_unclosed_candles():
    data = make_valid_liquidity_data()

    data.loc[
        data.index[-1],
        "close_time",
    ] = ANALYSIS_TIME + timedelta(minutes=5)

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_duplicate_open_times():
    data = make_valid_liquidity_data()

    data.loc[1, "open_time"] = data.loc[0, "open_time"]
    data.loc[1, "close_time"] = (
        data.loc[0, "close_time"] + timedelta(minutes=5)
    )

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_duplicate_close_times():
    data = make_valid_liquidity_data()

    data.loc[1, "close_time"] = data.loc[0, "close_time"]

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_unsorted_candles():
    data = make_valid_liquidity_data()

    data = data.iloc[::-1].reset_index(drop=True)

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_invalid_source():
    data = make_valid_liquidity_data()

    data.loc[5, "source"] = ""

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_non_finite_prices():
    data = make_valid_liquidity_data()

    data.loc[5, "close"] = np.inf

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_negative_volume():
    data = make_valid_liquidity_data()

    data.loc[5, "volume"] = -1

    with pytest.raises(LiquidityAnalysisError):
        analyze_liquidity(
            data,
            swing_window=2,
            minimum_touches=2,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_find_confirmed_swings_returns_empty_for_short_data():
    data = make_valid_liquidity_data(count=4)

    result = _find_confirmed_swings(
        data,
        swing_window=2,
    )

    assert result == []


def test_find_confirmed_swings_excludes_unconfirmed_edges():
    data = make_valid_liquidity_data(count=9)

    result = _find_confirmed_swings(
        data,
        swing_window=2,
    )

    for swing in result:
        assert swing["index"] >= 2
        assert swing["index"] < len(data) - 2
        assert swing["confirmation_index"] < len(data)


def test_find_confirmed_swings_requires_strict_extreme():
    highs = [10, 11, 12, 12, 11, 10, 9]
    lows = [5, 6, 7, 7, 6, 5, 4]

    data = make_candles(
        highs=highs,
        lows=lows,
    )

    result = _find_confirmed_swings(
        data,
        swing_window=1,
    )

    assert not any(
        swing["index"] in (2, 3)
        for swing in result
    )


def test_group_liquidity_levels_ignores_single_touch_groups():
    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 100.0,
            "time": "2026-11-17T00:15:00+00:00",
            "confirmation_index": 3,
            "confirmation_time": "2026-11-17T00:20:00+00:00",
        },
        {
            "type": "LOW",
            "index": 5,
            "price": 80.0,
            "time": "2026-11-17T00:30:00+00:00",
            "confirmation_index": 6,
            "confirmation_time": "2026-11-17T00:35:00+00:00",
        },
    ]

    result = _group_liquidity_levels(
        swings,
        minimum_touches=2,
        level_tolerance=0.001,
    )

    assert result == []


def test_group_liquidity_levels_separates_highs_and_lows():
    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 100.0,
            "time": "2026-11-17T00:15:00+00:00",
            "confirmation_index": 3,
            "confirmation_time": "2026-11-17T00:20:00+00:00",
        },
        {
            "type": "LOW",
            "index": 5,
            "price": 100.0,
            "time": "2026-11-17T00:30:00+00:00",
            "confirmation_index": 6,
            "confirmation_time": "2026-11-17T00:35:00+00:00",
        },
        {
            "type": "HIGH",
            "index": 8,
            "price": 100.02,
            "time": "2026-11-17T00:45:00+00:00",
            "confirmation_index": 9,
            "confirmation_time": "2026-11-17T00:50:00+00:00",
        },
        {
            "type": "LOW",
            "index": 11,
            "price": 100.02,
            "time": "2026-11-17T01:00:00+00:00",
            "confirmation_index": 12,
            "confirmation_time": "2026-11-17T01:05:00+00:00",
        },
    ]

    result = _group_liquidity_levels(
        swings,
        minimum_touches=2,
        level_tolerance=0.001,
    )

    assert len(result) == 2

    assert {
        pool["type"]
        for pool in result
    } == {BUY_SIDE, SELL_SIDE}


def test_group_liquidity_levels_calculates_mean_price():
    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 100.0,
            "time": "2026-11-17T00:15:00+00:00",
            "confirmation_index": 3,
            "confirmation_time": "2026-11-17T00:20:00+00:00",
        },
        {
            "type": "HIGH",
            "index": 8,
            "price": 100.05,
            "time": "2026-11-17T00:45:00+00:00",
            "confirmation_index": 9,
            "confirmation_time": "2026-11-17T00:50:00+00:00",
        },
    ]

    result = _group_liquidity_levels(
        swings,
        minimum_touches=2,
        level_tolerance=0.001,
    )

    assert len(result) == 1
    assert result[0]["type"] == BUY_SIDE
    assert result[0]["touch_count"] == 2
    assert result[0]["level"] == pytest.approx(100.025)


def test_group_liquidity_levels_rejects_no_data_gracefully():
    result = _group_liquidity_levels(
        [],
        minimum_touches=2,
        level_tolerance=0.001,
    )

    assert result == []


def test_detect_liquidity_sweeps_ignores_pre_confirmation_candles():
    data = make_candles(
        highs=[10, 12, 11, 13, 11, 10],
        lows=[8, 9, 8, 9, 8, 7],
        closes=[9, 10, 9, 10, 9, 8],
    )

    pools = [
        {
            "type": BUY_SIDE,
            "level": 11.0,
            "confirmation_index": 3,
        }
    ]

    evaluated, sweeps = _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert sweeps == []
    assert evaluated[0]["status"] == ACTIVE


def test_detect_liquidity_sweeps_detects_buy_side_sweep():
    data = make_candles(
        highs=[10, 11, 12, 13, 14, 16],
        lows=[8, 9, 10, 11, 12, 12],
        closes=[9, 10, 11, 12, 13, 14],
    )

    pools = [
        {
            "type": BUY_SIDE,
            "level": 15.0,
            "confirmation_index": 3,
        }
    ]

    evaluated, sweeps = _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert len(sweeps) == 1
    assert sweeps[0]["candle_index"] == 5
    assert sweeps[0]["type"] == BUY_SIDE
    assert evaluated[0]["status"] == SWEPT
    assert evaluated[0]["sweep_count"] == 1


def test_detect_liquidity_sweeps_detects_sell_side_sweep():
    data = make_candles(
        highs=[20, 19, 18, 17, 16, 18],
        lows=[18, 17, 16, 15, 14, 12],
        closes=[19, 18, 17, 16, 15, 17],
    )

    pools = [
        {
            "type": SELL_SIDE,
            "level": 13.0,
            "confirmation_index": 3,
        }
    ]

    evaluated, sweeps = _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert len(sweeps) == 1
    assert sweeps[0]["candle_index"] == 5
    assert sweeps[0]["type"] == SELL_SIDE
    assert evaluated[0]["status"] == SWEPT


def test_detect_liquidity_sweeps_does_not_mutate_original_pools():
    data = make_candles(
        highs=[10, 11, 12, 13, 14, 16],
        lows=[8, 9, 10, 11, 12, 12],
        closes=[9, 10, 11, 12, 13, 14],
    )

    pools = [
        {
            "type": BUY_SIDE,
            "level": 15.0,
            "confirmation_index": 3,
        }
    ]

    original = [
        dict(pool)
        for pool in pools
    ]

    _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert pools == original


def test_detect_liquidity_sweeps_returns_active_pool_without_sweep():
    data = make_candles(
        highs=[10, 11, 12, 13, 14, 14],
        lows=[8, 9, 10, 11, 12, 12],
        closes=[9, 10, 11, 12, 13, 13],
    )

    pools = [
        {
            "type": BUY_SIDE,
            "level": 20.0,
            "confirmation_index": 3,
        }
    ]

    evaluated, sweeps = _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert sweeps == []
    assert evaluated[0]["status"] == ACTIVE
    assert evaluated[0]["sweep_count"] == 0
    assert evaluated[0]["first_sweep_time"] is None
    assert evaluated[0]["last_sweep_time"] is None


def test_detect_liquidity_sweeps_handles_multiple_pools():
    data = make_candles(
        highs=[10, 11, 12, 13, 14, 16],
        lows=[8, 9, 10, 11, 12, 7],
        closes=[9, 10, 11, 12, 13, 10],
    )

    pools = [
        {
            "type": BUY_SIDE,
            "level": 15.0,
            "confirmation_index": 3,
        },
        {
            "type": SELL_SIDE,
            "level": 8.0,
            "confirmation_index": 3,
        },
    ]

    evaluated, sweeps = _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert len(evaluated) == 2
    assert len(sweeps) == 2
    assert {
        sweep["type"]
        for sweep in sweeps
    } == {BUY_SIDE, SELL_SIDE}


def test_detect_liquidity_sweeps_ignores_equal_level_touch():
    data = make_candles(
        highs=[10, 11, 12, 13, 14, 15],
        lows=[8, 9, 10, 11, 12, 12],
        closes=[9, 10, 11, 12, 13, 15],
    )

    pools = [
        {
            "type": BUY_SIDE,
            "level": 15.0,
            "confirmation_index": 3,
        }
    ]

    evaluated, sweeps = _detect_liquidity_sweeps(
        data,
        pools,
    )

    assert sweeps == []
    assert evaluated[0]["status"] == ACTIVE


def test_analysis_summary_counts_match_detected_items():
    data = make_valid_liquidity_data()

    result = analyze_liquidity(
        data,
        swing_window=2,
        minimum_touches=2,
        minimum_candles=30,
        analysis_time=ANALYSIS_TIME,
    )

    summary = result["summary"]

    assert summary["total_pools"] == len(result["pools"])

    assert summary["buy_side_pool_count"] == sum(
        pool["type"] == BUY_SIDE
        for pool in result["pools"]
    )

    assert summary["sell_side_pool_count"] == sum(
        pool["type"] == SELL_SIDE
        for pool in result["pools"]
    )

    assert summary["total_sweeps"] == len(result["sweeps"])

    assert summary["active_pool_count"] + summary[
        "swept_pool_count"
    ] == summary["total_pools"]