"""Tests for trading signal confidence and pattern recency."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.signal_engine.engine import (
    _calculate_confidence,
    _select_pattern_direction,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026,
    11,
    16,
    12,
    0,
    tzinfo=UTC,
)


# ---------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------


def test_neutral_market_must_not_receive_high_confidence():
    confidence, reasons = _calculate_confidence(
        overall_direction="NEUTRAL",
        alignment_ratio=1.0,
        structure_direction="NEUTRAL",
        order_block_direction="NEUTRAL",
        pattern_direction="NEUTRAL",
        volume_direction="NEUTRAL",
        conflict_count=0,
    )

    assert confidence == 0.0
    assert confidence < 60.0
    assert isinstance(reasons, list)


def test_neutral_market_with_bullish_confirmations_is_not_tradeable():
    confidence, _ = _calculate_confidence(
        overall_direction="NEUTRAL",
        alignment_ratio=1.0,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=0,
    )

    assert confidence == 0.0


def test_bullish_confirmations_increase_confidence():
    baseline, _ = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=0.5,
        structure_direction="NEUTRAL",
        order_block_direction="NEUTRAL",
        pattern_direction="NEUTRAL",
        volume_direction="NEUTRAL",
        conflict_count=0,
    )

    confirmed, _ = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=0.5,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=0,
    )

    assert confirmed > baseline


def test_conflicting_timeframes_reduce_confidence():
    aligned, _ = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=0.8,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=0,
    )

    conflicting, _ = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=0.8,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=2,
    )

    assert conflicting < aligned


@pytest.mark.parametrize(
    "alignment_ratio",
    [0.0, 0.25, 0.5, 0.75, 1.0],
)
def test_confidence_remains_within_valid_range(alignment_ratio):
    confidence, _ = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=alignment_ratio,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=0,
    )

    assert 0.0 <= confidence <= 100.0


# ---------------------------------------------------------
# Pattern fixtures
# ---------------------------------------------------------


def make_pattern(
    direction,
    close_time,
    timeframe="5m",
):
    return {
        "direction": direction,
        "close_time": close_time,
        "timeframe": timeframe,
        "confidence": 80.0,
    }


# ---------------------------------------------------------
# Pattern recency
# ---------------------------------------------------------


def test_old_patterns_must_not_outvote_latest_pattern():
    old_time = CHECK_TIME - timedelta(hours=5)
    recent_time = CHECK_TIME - timedelta(minutes=5)

    patterns = pd.DataFrame(
        [
            make_pattern("BULLISH", old_time),
            make_pattern(
                "BULLISH",
                old_time + timedelta(minutes=5),
            ),
            make_pattern(
                "BULLISH",
                old_time + timedelta(minutes=10),
            ),
            make_pattern(
                "BULLISH",
                old_time + timedelta(minutes=15),
            ),
            make_pattern(
                "BULLISH",
                old_time + timedelta(minutes=20),
            ),
            make_pattern("BEARISH", recent_time),
        ]
    )

    result = _select_pattern_direction({
        "5m": patterns,
    })

    assert result == "BEARISH"


def test_latest_pattern_is_selected_independently_per_timeframe():
    patterns_5m = pd.DataFrame(
        [
            make_pattern(
                "BULLISH",
                CHECK_TIME - timedelta(minutes=30),
                "5m",
            ),
            make_pattern(
                "BEARISH",
                CHECK_TIME - timedelta(minutes=5),
                "5m",
            ),
        ]
    )

    patterns_15m = pd.DataFrame(
        [
            make_pattern(
                "BULLISH",
                CHECK_TIME - timedelta(minutes=60),
                "15m",
            ),
            make_pattern(
                "BULLISH",
                CHECK_TIME - timedelta(minutes=15),
                "15m",
            ),
        ]
    )

    result = _select_pattern_direction({
        "5m": patterns_5m,
        "15m": patterns_15m,
    })

    # Latest 5m pattern is bearish.
    # Latest 15m pattern is bullish.
    # Conflicting timeframe directions must return NEUTRAL.
    assert result == "NEUTRAL"


def test_equal_latest_bullish_and_bearish_patterns_return_neutral():
    patterns_5m = pd.DataFrame(
        [
            make_pattern(
                "BULLISH",
                CHECK_TIME - timedelta(minutes=5),
                "5m",
            ),
        ]
    )

    patterns_15m = pd.DataFrame(
        [
            make_pattern(
                "BEARISH",
                CHECK_TIME - timedelta(minutes=15),
                "15m",
            ),
        ]
    )

    result = _select_pattern_direction({
        "5m": patterns_5m,
        "15m": patterns_15m,
    })

    assert result == "NEUTRAL"


def test_empty_pattern_results_return_neutral():
    result = _select_pattern_direction({
        "5m": pd.DataFrame(),
        "15m": pd.DataFrame(),
        "1h": pd.DataFrame(),
        "4h": pd.DataFrame(),
    })

    assert result == "NEUTRAL"


def test_neutral_patterns_do_not_create_directional_bias():
    patterns = pd.DataFrame(
        [
            make_pattern(
                "NEUTRAL",
                CHECK_TIME - timedelta(minutes=5),
            ),
            make_pattern(
                "NEUTRAL",
                CHECK_TIME - timedelta(minutes=10),
            ),
        ]
    )

    result = _select_pattern_direction({
        "5m": patterns,
    })

    assert result == "NEUTRAL"