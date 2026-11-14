"""Tests for Bitcoin market confluence analysis."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from market_analysis.confluence import (
    MarketConfluenceError,
    analyze_market_confluence,
)


ANALYSIS_TIME = datetime(
    2026, 11, 14, 12, 0,
    tzinfo=timezone.utc,
)

ANALYSIS_TIMESTAMP = "2026-11-14T11:05:00+00:00"
CLOSE_TIME = "2026-11-14T11:00:00+00:00"


def build_analyses():
    """Build consistent sample outputs from upstream modules."""

    common = {
        "analysis_timestamp": ANALYSIS_TIMESTAMP,
        "status": "SUCCESS",
        "source": ["kraken"],
        "candles_analyzed": 100,
        "last_candle_close_time": CLOSE_TIME,
    }

    return {
        "trend": {
            **common,
            "trend": "BULLISH",
            "trend_score": 75.0,
            "confidence": 80.0,
            "market_price": 95000.0,
        },
        "structure": {
            **common,
            "trend": "BULLISH",
            "structure_breaks": [
                {
                    "type": "BOS",
                    "direction": "BULLISH",
                    "status": "CONFIRMED",
                    "break_price": 95000.0,
                    "break_time": CLOSE_TIME,
                }
            ],
        },
        "zones": {
            **common,
            "zones": [
                {
                    "type": "DEMAND",
                    "status": "ACTIVE",
                    "lower": 94000.0,
                    "upper": 94500.0,
                },
                {
                    "type": "SUPPLY",
                    "status": "ACTIVE",
                    "lower": 96000.0,
                    "upper": 96500.0,
                },
            ],
        },
        "liquidity": {
            **common,
            "sweeps": [
                {
                    "side": "SELL_SIDE",
                    "status": "CONFIRMED",
                    "price": 94000.0,
                    "time": CLOSE_TIME,
                }
            ],
            "buy_side_liquidity": [],
            "sell_side_liquidity": [],
        },
        "regime": {
            **common,
            "direction": "BULLISH",
            "regime": "TRENDING",
            "trend_strength": "STRONG",
            "market_price": 95000.0,
        },
    }


def run_confluence(analyses=None, **kwargs):
    """Run confluence with consistent sample results."""

    data = analyses if analyses is not None else build_analyses()

    parameters = {
        "trend_analysis": data["trend"],
        "structure_analysis": data["structure"],
        "zones_analysis": data["zones"],
        "liquidity_analysis": data["liquidity"],
        "regime_analysis": data["regime"],
        "analysis_time": ANALYSIS_TIME,
    }

    parameters.update(kwargs)

    return analyze_market_confluence(**parameters)


def test_confluence_returns_success():
    result = run_confluence()

    assert result["status"] == "SUCCESS"


def test_confluence_contains_direction():
    result = run_confluence()

    assert result["direction"] in {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }


def test_confluence_contains_final_score():
    result = run_confluence()

    assert isinstance(
        result["confluence_score"],
        (int, float),
    )

    assert -100 <= result["confluence_score"] <= 100


def test_confluence_contains_component_scores():
    result = run_confluence()

    expected = {
        "trend",
        "structure",
        "zones",
        "liquidity",
        "regime",
    }

    assert expected.issubset(
        set(result["component_scores"])
    )


def test_confluence_contains_weighted_scores():
    result = run_confluence()

    assert isinstance(result["weighted_scores"], dict)


def test_confluence_contains_weights():
    result = run_confluence()

    assert isinstance(result["weights"], dict)


def test_confluence_contains_evidence():
    result = run_confluence()

    assert "bullish" in result["evidence"]
    assert "bearish" in result["evidence"]
    assert "neutral" in result["evidence"]


def test_confluence_preserves_analysis_timestamp():
    result = run_confluence()

    assert result["analysis_timestamp"] == ANALYSIS_TIME.isoformat()


def test_confluence_preserves_last_candle_close_time():
    result = run_confluence()

    assert result["last_candle_close_time"] == CLOSE_TIME


def test_confluence_preserves_candle_count():
    result = run_confluence()

    assert result["candles_analyzed"] == 100


def test_confluence_preserves_sources():
    result = run_confluence()

    assert result["source"] == ["kraken"]


def test_confluence_does_not_modify_input():
    analyses = build_analyses()
    original = deepcopy(analyses)

    run_confluence(analyses)

    assert analyses == original


def test_bullish_inputs_produce_bullish_direction():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = 100.0
    analyses["trend"]["trend"] = "BULLISH"

    analyses["structure"]["trend"] = "BULLISH"

    analyses["regime"]["direction"] = "BULLISH"
    analyses["regime"]["regime"] = "TRENDING"

    result = run_confluence(analyses)

    assert result["direction"] == "BULLISH"


def test_bearish_inputs_produce_bearish_direction():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = -100.0
    analyses["trend"]["trend"] = "BEARISH"

    analyses["structure"]["trend"] = "BEARISH"
    analyses["structure"]["structure_breaks"] = [
        {
            "type": "BOS",
            "direction": "BEARISH",
            "status": "CONFIRMED",
        }
    ]

    analyses["liquidity"]["sweeps"] = [
        {
            "side": "BUY_SIDE",
            "status": "CONFIRMED",
        }
    ]

    analyses["regime"]["direction"] = "BEARISH"

    result = run_confluence(analyses)

    assert result["direction"] == "BEARISH"


def test_neutral_inputs_produce_neutral_direction():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = 0.0
    analyses["trend"]["trend"] = "NEUTRAL"

    analyses["structure"]["trend"] = "NEUTRAL"
    analyses["structure"]["structure_breaks"] = []

    analyses["zones"]["zones"] = []
    analyses["liquidity"]["sweeps"] = []

    analyses["regime"]["direction"] = "NEUTRAL"
    analyses["regime"]["regime"] = "RANGING"

    result = run_confluence(analyses)

    assert result["direction"] == "NEUTRAL"


def test_custom_weights_are_accepted():
    weights = {
        "trend": 0.30,
        "structure": 0.25,
        "zones": 0.15,
        "liquidity": 0.15,
        "regime": 0.15,
    }

    result = run_confluence(weights=weights)

    assert result["status"] == "SUCCESS"


def test_custom_weights_are_normalized():
    weights = {
        "trend": 3,
        "structure": 2,
        "zones": 1,
        "liquidity": 2,
        "regime": 2,
    }

    result = run_confluence(weights=weights)

    assert sum(result["weights"].values()) == pytest.approx(1.0)


def test_missing_analysis_status_is_rejected():
    analyses = build_analyses()
    analyses["trend"]["status"] = "FAILED"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_missing_analysis_timestamp_is_rejected():
    analyses = build_analyses()
    del analyses["trend"]["analysis_timestamp"]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_analysis_timestamp_is_rejected():
    analyses = build_analyses()
    analyses["trend"]["analysis_timestamp"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_naive_analysis_timestamp_is_rejected():
    analyses = build_analyses()
    analyses["trend"]["analysis_timestamp"] = (
        "2026-11-14T11:05:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_non_utc_analysis_timestamp_is_rejected():
    analyses = build_analyses()
    analyses["trend"]["analysis_timestamp"] = (
        "2026-11-14T12:05:00+01:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_future_analysis_timestamp_is_rejected():
    analyses = build_analyses()
    analyses["trend"]["analysis_timestamp"] = (
        "2026-11-14T12:30:00+00:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_different_close_times_are_rejected():
    analyses = build_analyses()

    analyses["regime"]["last_candle_close_time"] = (
        "2026-11-14T10:55:00+00:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_different_candle_counts_are_rejected():
    analyses = build_analyses()
    analyses["liquidity"]["candles_analyzed"] = 99

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_missing_source_is_rejected():
    analyses = build_analyses()
    analyses["zones"]["source"] = []

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_weights_are_rejected():
    weights = {
        "trend": -1,
        "structure": 1,
        "zones": 1,
        "liquidity": 1,
        "regime": 1,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(weights=weights)


def test_non_numeric_weights_are_rejected():
    weights = {
        "trend": "high",
        "structure": 1,
        "zones": 1,
        "liquidity": 1,
        "regime": 1,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(weights=weights)


def test_zero_total_weights_are_rejected():
    weights = {
        "trend": 0,
        "structure": 0,
        "zones": 0,
        "liquidity": 0,
        "regime": 0,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(weights=weights)


def test_missing_weight_key_is_rejected():
    weights = {
        "trend": 1,
        "structure": 1,
        "zones": 1,
        "liquidity": 1,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(weights=weights)


def test_invalid_bullish_threshold_is_rejected():
    with pytest.raises(MarketConfluenceError):
        run_confluence(bullish_threshold=-1)


def test_invalid_bearish_threshold_is_rejected():
    with pytest.raises(MarketConfluenceError):
        run_confluence(bearish_threshold=-1)


def test_threshold_above_maximum_is_rejected():
    with pytest.raises(MarketConfluenceError):
        run_confluence(bullish_threshold=101)


def test_analysis_time_must_be_utc():
    with pytest.raises(MarketConfluenceError):
        run_confluence(
            analysis_time=datetime(
                2026, 11, 14, 12, 0
            )
        )


def test_missing_trend_score_is_rejected():
    analyses = build_analyses()
    del analyses["trend"]["trend_score"]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_trend_score_is_rejected():
    analyses = build_analyses()
    analyses["trend"]["trend_score"] = float("nan")

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_missing_structure_events_are_handled():
    analyses = build_analyses()
    del analyses["structure"]["structure_breaks"]

    result = run_confluence(analyses)

    assert result["status"] == "SUCCESS"


def test_empty_zones_are_handled():
    analyses = build_analyses()
    analyses["zones"]["zones"] = []

    result = run_confluence(analyses)

    assert result["status"] == "SUCCESS"


def test_empty_liquidity_sweeps_are_handled():
    analyses = build_analyses()
    analyses["liquidity"]["sweeps"] = []

    result = run_confluence(analyses)

    assert result["status"] == "SUCCESS"


def test_result_score_is_finite():
    result = run_confluence()

    assert result["confluence_score"] == pytest.approx(
        result["confluence_score"]
    )

    assert abs(result["confluence_score"]) != float("inf")


def test_result_contains_analysis_details():
    result = run_confluence()

    assert isinstance(result["details"], dict)
    assert "structure_events" in result["details"]
    assert "zones" in result["details"]
    assert "liquidity" in result["details"]
    assert "regime" in result["details"]


def test_result_contains_market_price():
    result = run_confluence()

    assert result["market_price"] == 95000.0


def test_result_contains_limitations():
    result = run_confluence()

    assert isinstance(result["limitations"], list)
    assert len(result["limitations"]) > 0


def test_duplicate_sources_are_merged():
    analyses = build_analyses()

    analyses["regime"]["source"] = [
        "kraken",
        "coingecko",
    ]

    result = run_confluence(analyses)

    assert result["source"] == [
        "coingecko",
        "kraken",
    ]


def test_invalid_structure_direction_is_rejected():
    analyses = build_analyses()
    analyses["structure"]["trend"] = "UNKNOWN"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_regime_direction_is_rejected():
    analyses = build_analyses()
    analyses["regime"]["direction"] = "UNKNOWN"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_regime_type_is_rejected():
    analyses = build_analyses()
    analyses["regime"]["regime"] = "UNKNOWN"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_zones_container_is_rejected():
    analyses = build_analyses()
    analyses["zones"]["zones"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_sweeps_container_is_rejected():
    analyses = build_analyses()
    analyses["liquidity"]["sweeps"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_invalid_structure_events_container_is_rejected():
    analyses = build_analyses()
    analyses["structure"]["structure_breaks"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)