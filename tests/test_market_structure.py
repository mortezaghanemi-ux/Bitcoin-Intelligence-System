"""Tests for Bitcoin market price structure analysis."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_analysis.structure import (
    BOS,
    CHOCH,
    BEARISH,
    BULLISH,
    NEUTRAL,
    HH,
    HL,
    LH,
    LL,
    PriceStructureError,
    _classify_swing_points,
    _detect_developing_swings,
    _detect_structure_breaks,
    _find_swings,
    _validate_parameters,
    analyze_price_structure,
)


UTC = timezone.utc

ANALYSIS_TIME = datetime(
    2026, 11, 14, 12, 0, tzinfo=UTC
)


def make_candles(
    count=40,
    base_price=100.0,
):
    """Create deterministic valid OHLCV candles."""

    rows = []

    start = datetime(
        2026, 11, 14, 0, 0, tzinfo=UTC
    )

    for index in range(count):
        open_time = start + timedelta(
            minutes=5 * index
        )

        close_time = open_time + timedelta(
            minutes=5
        )

        price = base_price

        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 10.0,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def test_find_confirmed_swing_high():
    """A swing high requires right-side confirmation candles."""

    data = make_candles(15)

    data.loc[6, "high"] = 110.0

    swings = _find_swings(
        data,
        swing_window=2,
    )

    high_swings = [
        item
        for item in swings
        if item["type"] == "HIGH"
    ]

    assert len(high_swings) == 1

    swing = high_swings[0]

    assert swing["index"] == 6
    assert swing["price"] == 110.0
    assert swing["confirmation_index"] == 8
    assert swing["time"] == data.loc[
        6, "close_time"
    ].isoformat()
    assert swing["confirmation_time"] == data.loc[
        8, "close_time"
    ].isoformat()


def test_find_confirmed_swing_low():
    """A swing low must be lower than surrounding candles."""

    data = make_candles(15)

    data.loc[6, "low"] = 90.0

    swings = _find_swings(
        data,
        swing_window=2,
    )

    low_swings = [
        item
        for item in swings
        if item["type"] == "LOW"
    ]

    assert len(low_swings) == 1

    assert low_swings[0]["index"] == 6
    assert low_swings[0]["price"] == 90.0
    assert low_swings[0]["confirmation_index"] == 8


def test_last_unconfirmed_candles_are_not_swings():
    """A candidate without enough right candles is not confirmed."""

    data = make_candles(10)

    data.loc[8, "high"] = 120.0
    data.loc[9, "high"] = 125.0

    swings = _find_swings(
        data,
        swing_window=2,
    )

    assert all(
        item["index"] < 8
        for item in swings
    )


def test_find_swings_returns_empty_for_flat_market():
    """Equal highs and lows must not create swing points."""

    data = make_candles(20)

    swings = _find_swings(
        data,
        swing_window=3,
    )

    assert swings == []


def test_classify_higher_high_and_lower_high():
    """Swing highs must be classified correctly."""

    swings = [
        {
            "type": "HIGH",
            "price": 100.0,
        },
        {
            "type": "HIGH",
            "price": 110.0,
        },
        {
            "type": "HIGH",
            "price": 105.0,
        },
    ]

    result = _classify_swing_points(swings)

    assert result[0]["classification"] is None
    assert result[1]["classification"] == HH
    assert result[2]["classification"] == LH


def test_classify_higher_low_and_lower_low():
    """Swing lows must be classified correctly."""

    swings = [
        {
            "type": "LOW",
            "price": 90.0,
        },
        {
            "type": "LOW",
            "price": 95.0,
        },
        {
            "type": "LOW",
            "price": 85.0,
        },
    ]

    result = _classify_swing_points(swings)

    assert result[0]["classification"] is None
    assert result[1]["classification"] == HL
    assert result[2]["classification"] == LL


def test_classify_equal_high_and_equal_low():
    """Equal swing levels must be identified."""

    swings = [
        {"type": "HIGH", "price": 100.0},
        {"type": "HIGH", "price": 100.0},
        {"type": "LOW", "price": 90.0},
        {"type": "LOW", "price": 90.0},
    ]

    result = _classify_swing_points(swings)

    assert result[1]["classification"] == "EQH"
    assert result[3]["classification"] == "EQL"


def test_detect_bullish_bos():
    """A close above a confirmed swing high creates bullish BOS."""

    data = make_candles(12)

    data.loc[6, "close"] = 112.0

    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 110.0,
            "time": data.loc[2, "close_time"].isoformat(),
            "confirmation_index": 4,
            "confirmation_time": data.loc[
                4, "close_time"
            ].isoformat(),
        }
    ]

    events, trend = _detect_structure_breaks(
        data,
        swings,
    )

    assert len(events) == 1
    assert events[0]["type"] == BOS
    assert events[0]["direction"] == BULLISH
    assert events[0]["level"] == 110.0
    assert trend == BULLISH


def test_detect_bearish_bos():
    """A close below a confirmed swing low creates bearish BOS."""

    data = make_candles(12)

    data.loc[6, "close"] = 88.0

    swings = [
        {
            "type": "LOW",
            "index": 2,
            "price": 90.0,
            "time": data.loc[2, "close_time"].isoformat(),
            "confirmation_index": 4,
            "confirmation_time": data.loc[
                4, "close_time"
            ].isoformat(),
        }
    ]

    events, trend = _detect_structure_breaks(
        data,
        swings,
    )

    assert len(events) == 1
    assert events[0]["type"] == BOS
    assert events[0]["direction"] == BEARISH
    assert events[0]["level"] == 90.0
    assert trend == BEARISH


def test_no_break_before_swing_confirmation():
    """A level cannot be broken before its confirmation."""

    data = make_candles(12)

    data.loc[3, "close"] = 115.0

    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 110.0,
            "time": data.loc[2, "close_time"].isoformat(),
            "confirmation_index": 4,
            "confirmation_time": data.loc[
                4, "close_time"
            ].isoformat(),
        }
    ]

    events, _ = _detect_structure_breaks(
        data,
        swings,
    )

    assert events == []


def test_no_break_when_close_equals_level():
    """Touching a level is not a confirmed close beyond it."""

    data = make_candles(12)

    data.loc[6, "close"] = 110.0

    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 110.0,
            "time": data.loc[2, "close_time"].isoformat(),
            "confirmation_index": 4,
            "confirmation_time": data.loc[
                4, "close_time"
            ].isoformat(),
        }
    ]

    events, _ = _detect_structure_breaks(
        data,
        swings,
    )

    assert events == []


def test_broken_level_is_not_repeated():
    """The same swing level must not generate duplicate breaks."""

    data = make_candles(12)

    data.loc[6:, "close"] = 115.0

    swings = [
        {
            "type": "HIGH",
            "index": 2,
            "price": 110.0,
            "time": data.loc[2, "close_time"].isoformat(),
            "confirmation_index": 4,
            "confirmation_time": data.loc[
                4, "close_time"
            ].isoformat(),
        }
    ]

    events, _ = _detect_structure_breaks(
        data,
        swings,
    )

    assert len(events) == 1


def test_detect_bullish_choch_after_bearish_break():
    """A bullish break after bearish structure is CHoCH."""

    data = make_candles(15)

    data.loc[6, "close"] = 88.0
    data.loc[10, "close"] = 112.0

    swings = [
        {
            "type": "LOW",
            "index": 2,
            "price": 90.0,
            "time": data.loc[2, "close_time"].isoformat(),
            "confirmation_index": 4,
            "confirmation_time": data.loc[
                4, "close_time"
            ].isoformat(),
        },
        {
            "type": "HIGH",
            "index": 7,
            "price": 110.0,
            "time": data.loc[7, "close_time"].isoformat(),
            "confirmation_index": 9,
            "confirmation_time": data.loc[
                9, "close_time"
            ].isoformat(),
        },
    ]

    events, trend = _detect_structure_breaks(
        data,
        swings,
    )

    assert len(events) == 2

    assert events[0]["type"] == BOS
    assert events[0]["direction"] == BEARISH

    assert events[1]["type"] == CHOCH
    assert events[1]["direction"] == BULLISH

    assert trend == BULLISH


def test_developing_swing_is_not_confirmed():
    """Developing candidates must remain separate from confirmed swings."""

    data = make_candles(10)

    data.loc[9, "high"] = 110.0
    data.loc[9, "low"] = 90.0

    result = _detect_developing_swings(
        data,
        swing_window=3,
    )

    assert len(result) == 2

    assert result[0]["type"] == "POTENTIAL_HIGH"
    assert result[1]["type"] == "POTENTIAL_LOW"

    assert all(
        item["status"] == "DEVELOPING"
        for item in result
    )


def test_invalid_swing_window():
    """Invalid swing window must be rejected."""

    with pytest.raises(PriceStructureError):
        _validate_parameters(
            swing_window=0,
            minimum_candles=30,
        )


def test_boolean_swing_window_is_rejected():
    """Boolean values must not be accepted as integers."""

    with pytest.raises(PriceStructureError):
        _validate_parameters(
            swing_window=True,
            minimum_candles=30,
        )


def test_minimum_candles_must_be_sufficient():
    """Minimum candles must support the swing window."""

    with pytest.raises(PriceStructureError):
        _validate_parameters(
            swing_window=5,
            minimum_candles=10,
        )


def test_analysis_requires_dataframe():
    """Non-DataFrame input must be rejected."""

    with pytest.raises(TypeError):
        analyze_price_structure(
            data=[],
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_empty_dataframe():
    """Empty input must be rejected."""

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=pd.DataFrame(),
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_missing_columns():
    """Missing OHLCV columns must be rejected."""

    data = pd.DataFrame(
        {
            "open": [100.0],
            "close": [100.0],
        }
    )

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=data,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_naive_analysis_time():
    """Analysis timestamp must be timezone-aware."""

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=make_candles(),
            analysis_time=datetime(2026, 11, 14),
        )


def test_analysis_rejects_non_utc_analysis_time():
    """Non-UTC analysis timestamp must be rejected."""

    non_utc = datetime(
        2026,
        11,
        14,
        12,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=make_candles(),
            analysis_time=non_utc,
        )


def test_analysis_rejects_future_candles():
    """Candles closing after analysis time must be rejected."""

    data = make_candles(30)

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=data,
            analysis_time=datetime(
                2026, 11, 14, 0, 30, tzinfo=UTC
            ),
        )


def test_analysis_rejects_unsorted_candles():
    """Input candles must be chronological."""

    data = make_candles(30)

    data = pd.concat(
        [
            data.iloc[:10],
            data.iloc[11:],
            data.iloc[10:11],
        ],
        ignore_index=True,
    )

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=data,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_duplicate_candles():
    """Duplicate candle timestamps must be rejected."""

    data = make_candles(30)

    data.loc[10] = data.loc[9]

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=data,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_invalid_ohlc():
    """Invalid OHLC relationships must be rejected."""

    data = make_candles(30)

    data.loc[10, "high"] = 50.0

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=data,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_rejects_insufficient_candles():
    """The configured minimum candle count must be enforced."""

    data = make_candles(20)

    with pytest.raises(PriceStructureError):
        analyze_price_structure(
            data=data,
            minimum_candles=30,
            analysis_time=ANALYSIS_TIME,
        )


def test_analysis_does_not_modify_input():
    """Analysis must not mutate the original DataFrame."""

    data = make_candles(40)
    original = data.copy(deep=True)

    analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_analysis_output_schema():
    """Successful analysis must return the expected output schema."""

    data = make_candles(40)

    result = analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    expected_keys = {
        "analysis_timestamp",
        "status",
        "source",
        "trend",
        "swing_window",
        "candles_analyzed",
        "last_candle_close_time",
        "swing_highs",
        "swing_lows",
        "swings",
        "structure_breaks",
        "developing_structures",
        "levels",
        "summary",
        "parameters",
    }

    assert set(result.keys()) == expected_keys
    assert result["status"] == "SUCCESS"
    assert result["trend"] in (
        BULLISH,
        BEARISH,
        NEUTRAL,
    )
    assert result["source"] == ["kraken"]
    assert result["candles_analyzed"] == 40


def test_analysis_is_deterministic():
    """Identical inputs must produce identical results."""

    data = make_candles(40)

    first = analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    second = analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    assert first == second


def test_analysis_uses_utc_timestamps():
    """Output timestamps must use UTC."""

    data = make_candles(40)

    result = analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    assert result["analysis_timestamp"].endswith("+00:00")
    assert result["last_candle_close_time"].endswith("+00:00")


def test_flat_market_has_no_confirmed_swings():
    """A flat market must not create false swing points."""

    data = make_candles(40)

    result = analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    assert result["swing_highs"] == []
    assert result["swing_lows"] == []
    assert result["structure_breaks"] == []
    assert result["trend"] == NEUTRAL


def test_summary_counts_match_events():
    """Summary counts must match the generated events."""

    data = make_candles(40)

    result = analyze_price_structure(
        data=data,
        analysis_time=ANALYSIS_TIME,
    )

    assert result["summary"]["swing_high_count"] == len(
        result["swing_highs"]
    )

    assert result["summary"]["swing_low_count"] == len(
        result["swing_lows"]
    )

    assert result["summary"]["bos_count"] == sum(
        event["type"] == BOS
        for event in result["structure_breaks"]
    )

    assert result["summary"]["choch_count"] == sum(
        event["type"] == CHOCH
        for event in result["structure_breaks"]
    )