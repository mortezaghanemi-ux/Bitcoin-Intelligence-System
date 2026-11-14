from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from forecast.engine import (
    BULLISH,
    BEARISH,
    NEUTRAL,
    ForecastEngineError,
    generate_forecast,
)


UTC = timezone.utc

ANALYSIS_TIME = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
CANDLE_CLOSE_TIME = datetime(2026, 11, 14, 11, 0, tzinfo=UTC)


def make_market_analysis(
    direction=BULLISH,
    score=70.0,
    agreement_ratio=0.8,
):
    return {
        "status": "SUCCESS",
        "analysis_timestamp": ANALYSIS_TIME,
        "market_price": 95000.0,
        "source": ["kraken"],
        "last_candle_close_time": CANDLE_CLOSE_TIME,
        "candles_analyzed": 200,
        "confluence": {
            "status": "SUCCESS",
            "analysis_timestamp": ANALYSIS_TIME,
            "last_candle_close_time": CANDLE_CLOSE_TIME,
            "direction": direction,
            "confluence_score": score,
            "agreement_ratio": agreement_ratio,
            "component_scores": {
                "trend": score,
                "structure": score,
                "liquidity": score,
                "regime": score,
            },
        },
    }


def generate_valid_forecast(**kwargs):
    parameters = {
        "market_analysis": make_market_analysis(),
        "forecast_time": ANALYSIS_TIME,
        "horizon_days": 7,
    }
    parameters.update(kwargs)
    return generate_forecast(**parameters)


# =========================================================
# Successful forecast generation
# =========================================================

def test_generate_forecast_successfully():
    result = generate_valid_forecast()

    assert isinstance(result, dict)
    assert result["status"] == "SUCCESS"
    assert result["asset"] == "BTC"


def test_forecast_contains_required_fields():
    result = generate_valid_forecast()

    required_fields = {
        "forecast_id",
        "status",
        "asset",
        "forecast_timestamp",
        "analysis_timestamp",
        "last_candle_close_time",
        "forecast_horizon",
        "market_price",
        "probabilities",
        "confidence_score",
        "market_condition",
        "influencing_factors",
        "risk_factors",
        "model",
        "input_metadata",
        "limitations",
    }

    assert required_fields.issubset(result.keys())


def test_forecast_id_is_not_empty():
    result = generate_valid_forecast()

    assert isinstance(result["forecast_id"], str)
    assert result["forecast_id"].startswith("BTC-")


def test_market_price_matches_analysis():
    result = generate_valid_forecast()

    assert result["market_price"] == 95000.0


def test_forecast_horizon_is_seven_days():
    result = generate_valid_forecast()

    assert result["forecast_horizon"]["days"] == 7


def test_forecast_start_matches_forecast_timestamp():
    result = generate_valid_forecast()

    assert (
        result["forecast_horizon"]["start"]
        == result["forecast_timestamp"]
    )


def test_forecast_end_matches_seven_day_horizon():
    result = generate_valid_forecast()

    start = datetime.fromisoformat(
        result["forecast_horizon"]["start"]
    )
    end = datetime.fromisoformat(
        result["forecast_horizon"]["end"]
    )

    assert end - start == timedelta(days=7)


# =========================================================
# Probability validation
# =========================================================

def test_probabilities_contain_all_directions():
    result = generate_valid_forecast()

    assert set(result["probabilities"]) == {
        BULLISH,
        BEARISH,
        NEUTRAL,
    }


def test_probabilities_total_exactly_100():
    result = generate_valid_forecast()

    assert sum(result["probabilities"].values()) == 100.0


def test_probabilities_are_within_valid_range():
    result = generate_valid_forecast()

    for probability in result["probabilities"].values():
        assert isinstance(probability, (int, float))
        assert 0 <= probability <= 100


def test_bullish_scenario_has_higher_bullish_probability():
    result = generate_valid_forecast()

    probabilities = result["probabilities"]

    assert probabilities[BULLISH] > probabilities[BEARISH]


def test_bearish_scenario_has_higher_bearish_probability():
    analysis = make_market_analysis(
        direction=BEARISH,
        score=-80.0,
        agreement_ratio=0.9,
    )

    result = generate_forecast(
        market_analysis=analysis,
        forecast_time=ANALYSIS_TIME,
    )

    probabilities = result["probabilities"]

    assert probabilities[BEARISH] > probabilities[BULLISH]


def test_neutral_scenario_has_higher_neutral_probability():
    analysis = make_market_analysis(
        direction=NEUTRAL,
        score=0.0,
        agreement_ratio=0.5,
    )

    result = generate_forecast(
        market_analysis=analysis,
        forecast_time=ANALYSIS_TIME,
    )

    probabilities = result["probabilities"]

    assert probabilities[NEUTRAL] > probabilities[BULLISH]
    assert probabilities[NEUTRAL] > probabilities[BEARISH]


@pytest.mark.parametrize(
    "direction,score,agreement",
    [
        (BULLISH, 80.0, 0.9),
        (BEARISH, -80.0, 0.9),
        (NEUTRAL, 0.0, 0.5),
    ],
)
def test_all_market_directions_are_supported(
    direction,
    score,
    agreement,
):
    analysis = make_market_analysis(
        direction=direction,
        score=score,
        agreement_ratio=agreement,
    )

    result = generate_forecast(
        market_analysis=analysis,
        forecast_time=ANALYSIS_TIME,
    )

    assert result["status"] == "SUCCESS"
    assert direction in result["probabilities"]


# =========================================================
# Confidence
# =========================================================

def test_confidence_score_is_valid():
    result = generate_valid_forecast()

    assert isinstance(result["confidence_score"], (int, float))
    assert 0 <= result["confidence_score"] <= 100


def test_confidence_increases_with_directional_conviction():
    weak_analysis = make_market_analysis(
        direction=BULLISH,
        score=20.0,
        agreement_ratio=0.5,
    )

    strong_analysis = make_market_analysis(
        direction=BULLISH,
        score=90.0,
        agreement_ratio=0.9,
    )

    weak_result = generate_forecast(
        market_analysis=weak_analysis,
        forecast_time=ANALYSIS_TIME,
    )

    strong_result = generate_forecast(
        market_analysis=strong_analysis,
        forecast_time=ANALYSIS_TIME,
    )

    assert (
        strong_result["confidence_score"]
        > weak_result["confidence_score"]
    )


# =========================================================
# Timestamps and UTC
# =========================================================

def test_forecast_timestamp_is_utc():
    result = generate_valid_forecast()

    timestamp = datetime.fromisoformat(
        result["forecast_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_analysis_timestamp_is_preserved():
    result = generate_valid_forecast()

    assert result["analysis_timestamp"] == ANALYSIS_TIME.isoformat()


def test_candle_close_timestamp_is_preserved():
    result = generate_valid_forecast()

    assert (
        result["last_candle_close_time"]
        == CANDLE_CLOSE_TIME.isoformat()
    )


def test_default_forecast_timestamp_is_utc():
    result = generate_forecast(
        market_analysis=make_market_analysis(),
    )

    timestamp = datetime.fromisoformat(
        result["forecast_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_forecast_time_before_analysis_is_rejected():
    earlier_time = ANALYSIS_TIME - timedelta(minutes=1)

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=make_market_analysis(),
            forecast_time=earlier_time,
        )


def test_future_analysis_timestamp_is_rejected():
    analysis = make_market_analysis()

    analysis["analysis_timestamp"] = (
        ANALYSIS_TIME + timedelta(minutes=1)
    )

    analysis["confluence"]["analysis_timestamp"] = (
        analysis["analysis_timestamp"]
    )

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


def test_candle_close_after_analysis_is_rejected():
    analysis = make_market_analysis()

    future_close = ANALYSIS_TIME + timedelta(minutes=1)

    analysis["last_candle_close_time"] = future_close
    analysis["confluence"]["last_candle_close_time"] = future_close

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


def test_naive_analysis_timestamp_is_rejected():
    analysis = make_market_analysis()

    analysis["analysis_timestamp"] = datetime(
        2026, 11, 14, 12, 0
    )

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


def test_non_utc_analysis_timestamp_is_rejected():
    analysis = make_market_analysis()

    analysis["analysis_timestamp"] = datetime(
        2026,
        11,
        14,
        15,
        0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


# =========================================================
# Forecast horizon
# =========================================================

@pytest.mark.parametrize(
    "horizon",
    [1, 3, 7, 14, 30, 90, 365],
)
def test_valid_horizons_are_supported(horizon):
    result = generate_valid_forecast(
        horizon_days=horizon
    )

    assert result["forecast_horizon"]["days"] == horizon

    start = datetime.fromisoformat(
        result["forecast_horizon"]["start"]
    )

    end = datetime.fromisoformat(
        result["forecast_horizon"]["end"]
    )

    assert end - start == timedelta(days=horizon)


@pytest.mark.parametrize(
    "horizon",
    [
        0,
        -1,
        -7,
        366,
        1.5,
        True,
        False,
        None,
        "7",
    ],
)
def test_invalid_horizons_are_rejected(horizon):
    with pytest.raises(ForecastEngineError):
        generate_valid_forecast(
            horizon_days=horizon
        )


# =========================================================
# Market analysis validation
# =========================================================

def test_non_dictionary_analysis_is_rejected():
    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=None,
            forecast_time=ANALYSIS_TIME,
        )


def test_failed_market_analysis_is_rejected():
    analysis = make_market_analysis()
    analysis["status"] = "FAILED"

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "field",
    [
        "analysis_timestamp",
        "confluence",
        "market_price",
        "source",
        "last_candle_close_time",
        "candles_analyzed",
    ],
)
def test_missing_required_fields_are_rejected(field):
    analysis = make_market_analysis()
    del analysis[field]

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "price",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "95000",
        True,
    ],
)
def test_invalid_market_prices_are_rejected(price):
    analysis = make_market_analysis()
    analysis["market_price"] = price

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "count",
    [
        0,
        -1,
        1.5,
        True,
        None,
        "200",
    ],
)
def test_invalid_candle_counts_are_rejected(count):
    analysis = make_market_analysis()
    analysis["candles_analyzed"] = count

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "source",
    [
        None,
        "",
        [],
        {},
        ["kraken", ""],
        [123],
    ],
)
def test_invalid_sources_are_rejected(source):
    analysis = make_market_analysis()
    analysis["source"] = source

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


# =========================================================
# Confluence validation
# =========================================================

def test_invalid_confluence_type_is_rejected():
    analysis = make_market_analysis()
    analysis["confluence"] = None

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


def test_failed_confluence_is_rejected():
    analysis = make_market_analysis()
    analysis["confluence"]["status"] = "FAILED"

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


def test_confluence_timestamp_mismatch_is_rejected():
    analysis = make_market_analysis()

    analysis["confluence"]["analysis_timestamp"] = (
        ANALYSIS_TIME - timedelta(minutes=5)
    )

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


def test_confluence_candle_timestamp_mismatch_is_rejected():
    analysis = make_market_analysis()

    analysis["confluence"]["last_candle_close_time"] = (
        CANDLE_CLOSE_TIME - timedelta(minutes=5)
    )

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "direction",
    [
        "SIDEWAYS",
        "INVALID",
        None,
        "",
    ],
)
def test_invalid_directions_are_rejected(direction):
    analysis = make_market_analysis()
    analysis["confluence"]["direction"] = direction

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "score",
    [
        -101,
        101,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "70",
        True,
    ],
)
def test_invalid_confluence_scores_are_rejected(score):
    analysis = make_market_analysis()
    analysis["confluence"]["confluence_score"] = score

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "agreement",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "0.8",
        True,
    ],
)
def test_invalid_agreement_ratios_are_rejected(agreement):
    analysis = make_market_analysis()
    analysis["confluence"]["agreement_ratio"] = agreement

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "component_scores",
    [
        None,
        [],
        "invalid",
    ],
)
def test_invalid_component_scores_are_rejected(component_scores):
    analysis = make_market_analysis()
    analysis["confluence"]["component_scores"] = component_scores

    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=analysis,
            forecast_time=ANALYSIS_TIME,
        )


# =========================================================
# Model validation
# =========================================================

@pytest.mark.parametrize(
    "model_version",
    [
        "",
        None,
        123,
        [],
        {},
    ],
)
def test_invalid_model_versions_are_rejected(model_version):
    with pytest.raises(ForecastEngineError):
        generate_forecast(
            market_analysis=make_market_analysis(),
            forecast_time=ANALYSIS_TIME,
            model_version=model_version,
        )


def test_custom_model_version_is_preserved():
    result = generate_forecast(
        market_analysis=make_market_analysis(),
        forecast_time=ANALYSIS_TIME,
        model_version="test-model-v2",
    )

    assert result["model"]["version"] == "test-model-v2"


def test_model_metadata_is_valid():
    result = generate_valid_forecast()

    assert isinstance(result["model"], dict)
    assert result["model"]["method"] == "weighted_confluence_heuristic"
    assert result["model"]["statistically_calibrated"] is False


# =========================================================
# Market condition and factors
# =========================================================

def test_market_condition_is_a_dictionary():
    result = generate_valid_forecast()

    assert isinstance(result["market_condition"], dict)


def test_market_condition_contains_direction():
    result = generate_valid_forecast()

    assert result["market_condition"]["direction"] == BULLISH


def test_unknown_regime_is_used_when_details_are_missing():
    result = generate_valid_forecast()

    assert result["market_condition"]["regime"] == "UNKNOWN"


def test_influencing_factors_are_a_list():
    result = generate_valid_forecast()

    assert isinstance(result["influencing_factors"], list)
    assert len(result["influencing_factors"]) > 0


def test_influencing_factors_contain_component_information():
    result = generate_valid_forecast()

    factor = result["influencing_factors"][0]

    assert "component" in factor
    assert "direction" in factor
    assert "score" in factor


def test_risk_factors_are_a_list():
    result = generate_valid_forecast()

    assert isinstance(result["risk_factors"], list)
    assert len(result["risk_factors"]) > 0


def test_low_agreement_generates_risk_factor():
    analysis = make_market_analysis(
        direction=BULLISH,
        score=70.0,
        agreement_ratio=0.3,
    )

    result = generate_forecast(
        market_analysis=analysis,
        forecast_time=ANALYSIS_TIME,
    )

    assert any(
        "Low directional agreement" in risk
        for risk in result["risk_factors"]
    )


def test_weak_score_generates_risk_factor():
    analysis = make_market_analysis(
        direction=NEUTRAL,
        score=10.0,
        agreement_ratio=0.8,
    )

    result = generate_forecast(
        market_analysis=analysis,
        forecast_time=ANALYSIS_TIME,
    )

    assert any(
        "Weak overall directional market bias" in risk
        for risk in result["risk_factors"]
    )


# =========================================================
# Input metadata and limitations
# =========================================================

def test_input_metadata_is_a_dictionary():
    result = generate_valid_forecast()

    assert isinstance(result["input_metadata"], dict)


def test_input_sources_are_preserved():
    result = generate_valid_forecast()

    assert result["input_metadata"]["sources"] == ["kraken"]


def test_candle_count_is_preserved():
    result = generate_valid_forecast()

    assert result["input_metadata"]["candles_analyzed"] == 200


def test_confluence_score_is_preserved():
    result = generate_valid_forecast()

    assert result["input_metadata"]["confluence_score"] == 70.0


def test_agreement_ratio_is_preserved():
    result = generate_valid_forecast()

    assert result["input_metadata"]["agreement_ratio"] == 0.8


def test_limitations_are_a_list():
    result = generate_valid_forecast()

    assert isinstance(result["limitations"], list)
    assert len(result["limitations"]) > 0


def test_limitations_disclose_uncalibrated_probabilities():
    result = generate_valid_forecast()

    assert any(
        "not statistically calibrated" in limitation
        for limitation in result["limitations"]
    )


def test_forecast_does_not_claim_statistical_calibration():
    result = generate_valid_forecast()

    assert result["model"]["statistically_calibrated"] is False


# =========================================================
# Input immutability
# =========================================================

def test_market_analysis_is_not_modified():
    analysis = make_market_analysis()
    original = deepcopy(analysis)

    generate_forecast(
        market_analysis=analysis,
        forecast_time=ANALYSIS_TIME,
    )

    assert analysis == original