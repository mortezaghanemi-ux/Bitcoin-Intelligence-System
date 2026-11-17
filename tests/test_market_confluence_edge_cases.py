"""Edge-case tests for Bitcoin market confluence analysis."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from market_analysis.confluence import (
    MarketConfluenceError,
    analyze_market_confluence,
)


ANALYSIS_TIME = datetime(
    2026,
    11,
    14,
    12,
    0,
    tzinfo=timezone.utc,
)

ANALYSIS_TIMESTAMP = "2026-11-14T11:05:00+00:00"
CLOSE_TIME = "2026-11-14T11:00:00+00:00"


def build_analyses():
    """Build valid and synchronized upstream analysis results."""

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
            "structure_breaks": [],
        },
        "zones": {
            **common,
            "zones": [],
        },
        "liquidity": {
            **common,
            "sweeps": [],
        },
        "regime": {
            **common,
            "direction": "BULLISH",
            "regime": "TRENDING",
            "trend_strength": "STRONG",
            "market_price": 95000.0,
        },
    }


def run_confluence(
    analyses=None,
    **kwargs,
):
    """Run confluence using valid default analysis results."""

    data = (
        deepcopy(analyses)
        if analyses is not None
        else build_analyses()
    )

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


# ---------------------------------------------------------------------
# Analysis timestamp edge cases
# ---------------------------------------------------------------------


def test_missing_analysis_time_uses_utc_clock():
    result = run_confluence(
        analysis_time=None,
    )

    timestamp = datetime.fromisoformat(
        result["analysis_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_rejects_naive_analysis_time():
    invalid_time = datetime(
        2026,
        11,
        14,
        12,
        0,
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            analysis_time=invalid_time,
        )


def test_rejects_non_utc_analysis_time():
    invalid_time = datetime(
        2026,
        11,
        14,
        13,
        0,
        tzinfo=timezone(timedelta(hours=1)),
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            analysis_time=invalid_time,
        )


@pytest.mark.parametrize(
    "analysis_time",
    [
        "2026-11-14T12:00:00+00:00",
        123,
    ],
)
def test_rejects_invalid_analysis_time_types(analysis_time):
    with pytest.raises(MarketConfluenceError):
        run_confluence(
            analysis_time=analysis_time,
        )


def test_accepts_utc_analysis_time():
    result = run_confluence(
        analysis_time=ANALYSIS_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert result["analysis_timestamp"] == ANALYSIS_TIME.isoformat()


# ---------------------------------------------------------------------
# Weight validation edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "weights",
    [
        [],
        "invalid",
        123,
    ],
)
def test_rejects_invalid_weight_containers(weights):
    with pytest.raises(MarketConfluenceError):
        run_confluence(
            weights=weights,
        )


def test_none_weights_use_default_weights():
    result = run_confluence(
        weights=None,
    )

    assert result["status"] == "SUCCESS"

    assert result["weights"] == {
        "trend": 0.30,
        "structure": 0.25,
        "zones": 0.15,
        "liquidity": 0.15,
        "regime": 0.15,
    }


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "0.5",
        None,
    ],
)
def test_rejects_invalid_individual_weights(invalid_value):
    weights = {
        "trend": 1,
        "structure": 1,
        "zones": 1,
        "liquidity": 1,
        "regime": 1,
    }

    weights["trend"] = invalid_value

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            weights=weights,
        )


def test_rejects_weights_with_missing_keys():
    weights = {
        "trend": 1,
        "structure": 1,
        "zones": 1,
        "liquidity": 1,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            weights=weights,
        )


def test_rejects_weights_with_extra_keys():
    weights = {
        "trend": 1,
        "structure": 1,
        "zones": 1,
        "liquidity": 1,
        "regime": 1,
        "extra": 1,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            weights=weights,
        )


def test_rejects_zero_total_weights():
    weights = {
        "trend": 0,
        "structure": 0,
        "zones": 0,
        "liquidity": 0,
        "regime": 0,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            weights=weights,
        )


def test_rejects_overflowing_weight_total():
    weights = {
        "trend": 1e308,
        "structure": 1e308,
        "zones": 1e308,
        "liquidity": 1e308,
        "regime": 1e308,
    }

    with pytest.raises(MarketConfluenceError):
        run_confluence(
            weights=weights,
        )


def test_custom_weights_are_normalized():
    weights = {
        "trend": 3,
        "structure": 2,
        "zones": 1,
        "liquidity": 2,
        "regime": 2,
    }

    result = run_confluence(
        weights=weights,
    )

    assert sum(
        result["weights"].values()
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Upstream analysis validation edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_rejects_non_dictionary_analysis_results(invalid_result):
    analyses = build_analyses()
    analyses["trend"] = invalid_result

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_missing_analysis_status():
    analyses = build_analyses()
    del analyses["trend"]["status"]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_unsuccessful_analysis_status():
    analyses = build_analyses()
    analyses["trend"]["status"] = "FAILED"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_missing_analysis_timestamp():
    analyses = build_analyses()
    del analyses["trend"]["analysis_timestamp"]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        "",
        "invalid",
        None,
        123,
    ],
)
def test_rejects_invalid_upstream_timestamps(invalid_timestamp):
    analyses = build_analyses()

    analyses["trend"]["analysis_timestamp"] = (
        invalid_timestamp
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_naive_upstream_timestamp():
    analyses = build_analyses()

    analyses["trend"]["analysis_timestamp"] = (
        "2026-11-14T11:05:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_non_utc_upstream_timestamp():
    analyses = build_analyses()

    analyses["trend"]["analysis_timestamp"] = (
        "2026-11-14T12:05:00+01:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_future_analysis_timestamp():
    analyses = build_analyses()

    analyses["trend"]["analysis_timestamp"] = (
        "2026-11-14T12:30:00+00:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_missing_candle_close_time():
    analyses = build_analyses()

    del analyses["trend"]["last_candle_close_time"]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_future_candle_close_time():
    analyses = build_analyses()

    analyses["trend"]["last_candle_close_time"] = (
        "2026-11-14T12:30:00+00:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_different_candle_close_times():
    analyses = build_analyses()

    analyses["regime"]["last_candle_close_time"] = (
        "2026-11-14T10:55:00+00:00"
    )

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
        1.5,
        True,
        "100",
        None,
    ],
)
def test_rejects_invalid_candle_counts(invalid_count):
    analyses = build_analyses()

    analyses["trend"]["candles_analyzed"] = invalid_count

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_different_candle_counts():
    analyses = build_analyses()

    analyses["regime"]["candles_analyzed"] = 99

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


@pytest.mark.parametrize(
    "invalid_source",
    [
        None,
        [],
        "",
        "kraken",
        [""],
        [None],
        [123],
    ],
)
def test_rejects_invalid_source_metadata(invalid_source):
    analyses = build_analyses()

    analyses["trend"]["source"] = invalid_source

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


# ---------------------------------------------------------------------
# Trend scoring edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_score",
    [
        None,
        "75",
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_invalid_trend_scores(invalid_score):
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = invalid_score

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_trend_score_is_clamped_to_maximum():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = 500

    result = run_confluence(analyses)

    assert result["component_scores"]["trend"] == 100.0


def test_trend_score_is_clamped_to_minimum():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = -500

    result = run_confluence(analyses)

    assert result["component_scores"]["trend"] == -100.0


# ---------------------------------------------------------------------
# Structure scoring edge cases
# ---------------------------------------------------------------------


def test_rejects_invalid_structure_direction():
    analyses = build_analyses()

    analyses["structure"]["trend"] = "UNKNOWN"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_invalid_structure_events_container():
    analyses = build_analyses()

    analyses["structure"]["structure_breaks"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_non_dictionary_structure_event():
    analyses = build_analyses()

    analyses["structure"]["structure_breaks"] = [
        "invalid"
    ]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_structure_score_is_clamped():
    analyses = build_analyses()

    analyses["structure"]["trend"] = "BULLISH"

    analyses["structure"]["structure_breaks"] = [
        {
            "type": "BOS",
            "direction": "BULLISH",
        }
        for _ in range(20)
    ]

    result = run_confluence(analyses)

    assert result["component_scores"]["structure"] == 100.0


def test_missing_structure_events_use_empty_list():
    analyses = build_analyses()

    del analyses["structure"]["structure_breaks"]

    result = run_confluence(analyses)

    assert result["status"] == "SUCCESS"

    assert result["details"]["structure_events"] == {
        "BOS": 0,
        "CHoCH": 0,
    }


# ---------------------------------------------------------------------
# Supply and demand zone scoring edge cases
# ---------------------------------------------------------------------


def test_rejects_invalid_zones_container():
    analyses = build_analyses()

    analyses["zones"]["zones"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_non_dictionary_zone():
    analyses = build_analyses()

    analyses["zones"]["zones"] = [
        "invalid"
    ]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_zone_score_is_clamped():
    analyses = build_analyses()

    analyses["zones"]["zones"] = [
        {
            "type": "DEMAND",
            "status": "ACTIVE",
        }
        for _ in range(20)
    ]

    result = run_confluence(analyses)

    assert result["component_scores"]["zones"] == 60.0


def test_mitigated_zones_are_not_active():
    analyses = build_analyses()

    analyses["zones"]["zones"] = [
        {
            "type": "DEMAND",
            "status": "MITIGATED",
        }
    ]

    result = run_confluence(analyses)

    assert result["details"]["zones"]["mitigated"] == 1
    assert result["details"]["zones"]["active_demand"] == 0


def test_broken_zones_are_not_active():
    analyses = build_analyses()

    analyses["zones"]["zones"] = [
        {
            "type": "SUPPLY",
            "status": "BROKEN",
        }
    ]

    result = run_confluence(analyses)

    assert result["details"]["zones"]["broken"] == 1
    assert result["details"]["zones"]["active_supply"] == 0


# ---------------------------------------------------------------------
# Liquidity scoring edge cases
# ---------------------------------------------------------------------


def test_rejects_invalid_sweeps_container():
    analyses = build_analyses()

    analyses["liquidity"]["sweeps"] = "invalid"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_non_dictionary_liquidity_sweep():
    analyses = build_analyses()

    analyses["liquidity"]["sweeps"] = [
        "invalid"
    ]

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_liquidity_score_is_clamped():
    analyses = build_analyses()

    analyses["liquidity"]["sweeps"] = [
        {
            "side": "SELL_SIDE",
        }
        for _ in range(20)
    ]

    result = run_confluence(analyses)

    assert result["component_scores"]["liquidity"] == 100.0


def test_unknown_liquidity_sweeps_are_counted():
    analyses = build_analyses()

    analyses["liquidity"]["sweeps"] = [
        {
            "side": "UNKNOWN",
        }
    ]

    result = run_confluence(analyses)

    assert result["details"]["liquidity"]["unknown_sweeps"] == 1


# ---------------------------------------------------------------------
# Market regime scoring edge cases
# ---------------------------------------------------------------------


def test_rejects_invalid_regime_direction():
    analyses = build_analyses()

    analyses["regime"]["direction"] = "UNKNOWN"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_rejects_invalid_regime_type():
    analyses = build_analyses()

    analyses["regime"]["regime"] = "UNKNOWN"

    with pytest.raises(MarketConfluenceError):
        run_confluence(analyses)


def test_ranging_regime_reduces_directional_score():
    analyses = build_analyses()

    analyses["regime"]["direction"] = "BULLISH"
    analyses["regime"]["regime"] = "RANGING"

    result = run_confluence(analyses)

    assert result["component_scores"]["regime"] == 17.5


def test_transitional_regime_reduces_directional_score():
    analyses = build_analyses()

    analyses["regime"]["direction"] = "BULLISH"
    analyses["regime"]["regime"] = "TRANSITIONAL"

    result = run_confluence(analyses)

    assert result["component_scores"]["regime"] == 25.0


# ---------------------------------------------------------------------
# Threshold validation edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -1,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "20",
        None,
    ],
)
def test_rejects_invalid_bullish_thresholds(threshold):
    with pytest.raises(MarketConfluenceError):
        run_confluence(
            bullish_threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -1,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "20",
        None,
    ],
)
def test_rejects_invalid_bearish_thresholds(threshold):
    with pytest.raises(MarketConfluenceError):
        run_confluence(
            bearish_threshold=threshold,
        )


def test_score_equal_to_bullish_threshold_is_bullish():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = 20.0
    analyses["structure"]["trend"] = "NEUTRAL"
    analyses["regime"]["direction"] = "NEUTRAL"

    weights = {
        "trend": 1,
        "structure": 0,
        "zones": 0,
        "liquidity": 0,
        "regime": 0,
    }

    result = run_confluence(
        analyses,
        weights=weights,
        bullish_threshold=20,
    )

    assert result["confluence_score"] == 20.0
    assert result["direction"] == "BULLISH"


def test_score_equal_to_bearish_threshold_is_bearish():
    analyses = build_analyses()

    analyses["trend"]["trend_score"] = -20.0
    analyses["structure"]["trend"] = "NEUTRAL"
    analyses["regime"]["direction"] = "NEUTRAL"

    weights = {
        "trend": 1,
        "structure": 0,
        "zones": 0,
        "liquidity": 0,
        "regime": 0,
    }

    result = run_confluence(
        analyses,
        weights=weights,
        bearish_threshold=20,
    )

    assert result["confluence_score"] == -20.0
    assert result["direction"] == "BEARISH"


# ---------------------------------------------------------------------
# Result integrity and consistency
# ---------------------------------------------------------------------


def test_result_score_is_finite():
    result = run_confluence()

    assert np.isfinite(
        result["confluence_score"]
    )


def test_result_score_stays_within_valid_range():
    result = run_confluence()

    assert -100 <= result["confluence_score"] <= 100


def test_result_weights_sum_to_one():
    result = run_confluence()

    assert sum(
        result["weights"].values()
    ) == pytest.approx(1.0)


def test_result_contains_all_five_components():
    result = run_confluence()

    expected = {
        "trend",
        "structure",
        "zones",
        "liquidity",
        "regime",
    }

    assert set(result["component_scores"]) == expected


def test_result_contains_heuristic_limitation():
    result = run_confluence()

    assert any(
        "not probabilities" in limitation
        for limitation in result["limitations"]
    )


def test_result_does_not_modify_input_analyses():
    analyses = build_analyses()
    original = deepcopy(analyses)

    run_confluence(analyses)

    assert analyses == original


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


def test_result_contains_no_trading_execution():
    result = run_confluence()

    assert any(
        "does not execute trades" in limitation
        for limitation in result["limitations"]
    )