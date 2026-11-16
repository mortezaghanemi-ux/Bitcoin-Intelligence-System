"""Validation edge-case tests for the Bitcoin forecast engine."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from forecast.engine import (
    BULLISH,
    ForecastEngineError,
    generate_forecast,
)


UTC = timezone.utc

ANALYSIS_TIME = datetime(
    2026, 11, 16, 12, 0, tzinfo=UTC
)

CANDLE_CLOSE_TIME = datetime(
    2026, 11, 16, 11, 55, tzinfo=UTC
)


def make_market_analysis():
    return {
        "status": "SUCCESS",
        "analysis_timestamp": ANALYSIS_TIME,
        "market_price": 95000,
        "source": ["kraken"],
        "last_candle_close_time": CANDLE_CLOSE_TIME,
        "candles_analyzed": 200,
        "confluence": {
            "status": "SUCCESS",
            "analysis_timestamp": ANALYSIS_TIME,
            "last_candle_close_time": CANDLE_CLOSE_TIME,
            "direction": BULLISH,
            "confluence_score": 70,
            "agreement_ratio": 0.8,
            "component_scores": {
                "trend": 70,
                "structure": 60,
            },
        },
    }


def generate_valid_forecast(analysis=None):
    return generate_forecast(
        market_analysis=(
            analysis if analysis is not None
            else make_market_analysis()
        ),
        forecast_time=ANALYSIS_TIME,
        horizon_days=7,
    )


# =========================================================
# Market analysis status validation
# =========================================================

@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "FAILED",
        "PENDING",
        "SUCCESSFUL",
        200,
        True,
    ],
)
def test_invalid_market_analysis_status_is_rejected(status):
    analysis = make_market_analysis()
    analysis["status"] = status

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


def test_missing_market_analysis_status_is_rejected():
    analysis = make_market_analysis()
    del analysis["status"]

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


# =========================================================
# Timestamp validation
# =========================================================

@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        "invalid-timestamp",
        "2026-99-99T99:99:99Z",
        123,
        True,
        [],
        {},
    ],
)
def test_invalid_analysis_timestamp_is_rejected(timestamp):
    analysis = make_market_analysis()
    analysis["analysis_timestamp"] = timestamp

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        "invalid-timestamp",
        "2026-99-99T99:99:99Z",
        123,
        True,
        [],
        {},
    ],
)
def test_invalid_candle_close_timestamp_is_rejected(timestamp):
    analysis = make_market_analysis()
    analysis["last_candle_close_time"] = timestamp

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


def test_candle_closing_after_analysis_is_rejected():
    analysis = make_market_analysis()

    future_close = ANALYSIS_TIME + timedelta(seconds=1)

    analysis["last_candle_close_time"] = future_close
    analysis["confluence"]["last_candle_close_time"] = future_close

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


def test_confluence_analysis_timestamp_is_required():
    analysis = make_market_analysis()
    del analysis["confluence"]["analysis_timestamp"]

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


def test_confluence_candle_timestamp_is_required():
    analysis = make_market_analysis()
    del analysis["confluence"]["last_candle_close_time"]

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


# =========================================================
# Source normalization
# =========================================================

def test_sources_are_sorted_and_deduplicated():
    analysis = make_market_analysis()
    analysis["source"] = [
        "kraken",
        "coingecko",
        "kraken",
        "mempool",
    ]

    result = generate_valid_forecast(analysis)

    assert result["input_metadata"]["sources"] == [
        "coingecko",
        "kraken",
        "mempool",
    ]


def test_source_whitespace_is_preserved():
    analysis = make_market_analysis()
    analysis["source"] = [
        "kraken",
        " coingecko ",
    ]

    result = generate_valid_forecast(analysis)

    assert result["input_metadata"]["sources"] == [
        " coingecko ",
        "kraken",
    ]


def test_source_list_with_only_whitespace_is_rejected():
    analysis = make_market_analysis()
    analysis["source"] = [" ", "\t", "\n"]

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


# =========================================================
# Confluence validation
# =========================================================

@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "FAILED",
        "PENDING",
        200,
        True,
    ],
)
def test_invalid_confluence_status_is_rejected(status):
    analysis = make_market_analysis()
    analysis["confluence"]["status"] = status

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


def test_missing_confluence_status_is_rejected():
    analysis = make_market_analysis()
    del analysis["confluence"]["status"]

    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(analysis)


def test_empty_component_scores_are_accepted():
    analysis = make_market_analysis()
    analysis["confluence"]["component_scores"] = {}

    result = generate_valid_forecast(analysis)

    assert result["status"] == "SUCCESS"
    assert result["influencing_factors"] == []


def test_component_scores_are_not_modified():
    analysis = make_market_analysis()
    original = deepcopy(analysis["confluence"]["component_scores"])

    generate_valid_forecast(analysis)

    assert analysis["confluence"]["component_scores"] == original


# =========================================================
# Numeric normalization
# =========================================================

def test_integer_market_price_is_normalized_to_float():
    analysis = make_market_analysis()
    analysis["market_price"] = 95000

    result = generate_valid_forecast(analysis)

    assert result["market_price"] == 95000.0
    assert isinstance(result["market_price"], float)


def test_integer_confluence_score_is_normalized_to_float():
    analysis = make_market_analysis()
    analysis["confluence"]["confluence_score"] = 70

    result = generate_valid_forecast(analysis)

    assert result["input_metadata"]["confluence_score"] == 70.0
    assert isinstance(
        result["input_metadata"]["confluence_score"],
        float,
    )


def test_integer_agreement_ratio_is_normalized_to_float():
    analysis = make_market_analysis()
    analysis["confluence"]["agreement_ratio"] = 1

    result = generate_valid_forecast(analysis)

    assert result["input_metadata"]["agreement_ratio"] == 1.0
    assert isinstance(
        result["input_metadata"]["agreement_ratio"],
        float,
    )


# =========================================================
# Input integrity
# =========================================================

def test_validation_does_not_modify_market_analysis():
    analysis = make_market_analysis()
    original = deepcopy(analysis)

    generate_valid_forecast(analysis)

    assert analysis == original


def test_forecast_generation_preserves_original_source_order():
    analysis = make_market_analysis()
    analysis["source"] = [
        "kraken",
        "coingecko",
        "mempool",
    ]

    original_sources = list(analysis["source"])

    generate_valid_forecast(analysis)

    assert analysis["source"] == original_sources