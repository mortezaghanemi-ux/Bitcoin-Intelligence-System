"""Edge-case tests for the Bitcoin forecast engine."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from forecast.engine import (
    BULLISH,
    BEARISH,
    NEUTRAL,
    ForecastEngineError,
    _parse_utc_timestamp,
    _validate_horizon,
    _normalize_probabilities,
    _calculate_probabilities,
    _calculate_confidence,
    _market_condition,
    _build_influencing_factors,
    _build_risk_factors,
)


UTC = timezone.utc


# =========================================================
# UTC timestamp parsing
# =========================================================

@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "not-a-date",
        "2026-99-99T99:99:99Z",
        None,
        123,
        12.5,
        [],
        {},
        True,
    ],
)
def test_parse_utc_timestamp_rejects_invalid_values(value):
    with pytest.raises(ForecastEngineError):
        _parse_utc_timestamp(value, "test_timestamp")


def test_parse_utc_timestamp_accepts_utc_datetime():
    value = datetime(2026, 11, 16, 12, 30, tzinfo=UTC)

    result = _parse_utc_timestamp(value, "timestamp")

    assert result == value
    assert result.tzinfo == UTC


def test_parse_utc_timestamp_accepts_utc_iso_string():
    result = _parse_utc_timestamp(
        "2026-11-16T12:30:00Z",
        "timestamp",
    )

    assert result == datetime(
        2026, 11, 16, 12, 30, tzinfo=UTC
    )


def test_parse_utc_timestamp_accepts_zero_offset_string():
    result = _parse_utc_timestamp(
        "2026-11-16T12:30:00+00:00",
        "timestamp",
    )

    assert result.utcoffset() == timedelta(0)


def test_parse_utc_timestamp_rejects_naive_datetime():
    value = datetime(2026, 11, 16, 12, 30)

    with pytest.raises(ForecastEngineError):
        _parse_utc_timestamp(value, "timestamp")


def test_parse_utc_timestamp_rejects_non_utc_timezone():
    value = datetime(
        2026,
        11,
        16,
        15,
        30,
        tzinfo=timezone(timedelta(hours=3)),
    )

    with pytest.raises(ForecastEngineError):
        _parse_utc_timestamp(value, "timestamp")


def test_parse_utc_timestamp_normalizes_zero_offset_timezone():
    custom_utc = timezone(timedelta(0), name="CUSTOM_UTC")

    value = datetime(
        2026,
        11,
        16,
        12,
        30,
        tzinfo=custom_utc,
    )

    result = _parse_utc_timestamp(value, "timestamp")

    assert result.utcoffset() == timedelta(0)
    assert result.tzinfo == UTC


# =========================================================
# Forecast horizon validation
# =========================================================

@pytest.mark.parametrize(
    "horizon",
    [1, 2, 7, 30, 180, 365],
)
def test_validate_horizon_accepts_valid_values(horizon):
    assert _validate_horizon(horizon) == horizon


@pytest.mark.parametrize(
    "horizon",
    [
        0,
        -1,
        -365,
        366,
        1.5,
        7.0,
        True,
        False,
        None,
        "7",
        [],
        {},
    ],
)
def test_validate_horizon_rejects_invalid_values(horizon):
    with pytest.raises(ForecastEngineError):
        _validate_horizon(horizon)


# =========================================================
# Probability normalization
# =========================================================

def test_normalize_probabilities_accepts_decimal_values():
    result = _normalize_probabilities(
        {
            BULLISH: Decimal("60"),
            BEARISH: Decimal("25"),
            NEUTRAL: Decimal("15"),
        }
    )

    assert result[BULLISH] == 60.0
    assert result[BEARISH] == 25.0
    assert result[NEUTRAL] == 15.0
    assert sum(result.values()) == 100.0


def test_normalize_probabilities_accepts_integer_values():
    result = _normalize_probabilities(
        {
            BULLISH: 50,
            BEARISH: 30,
            NEUTRAL: 20,
        }
    )

    assert sum(result.values()) == 100.0


def test_normalize_probabilities_accepts_unequal_scale():
    result = _normalize_probabilities(
        {
            BULLISH: 600,
            BEARISH: 300,
            NEUTRAL: 100,
        }
    )

    assert result[BULLISH] == 60.0
    assert result[BEARISH] == 30.0
    assert result[NEUTRAL] == 10.0


def test_normalize_probabilities_handles_zero_probability():
    result = _normalize_probabilities(
        {
            BULLISH: 100,
            BEARISH: 0,
            NEUTRAL: 0,
        }
    )

    assert result[BULLISH] == 100.0
    assert result[BEARISH] == 0.0
    assert result[NEUTRAL] == 0.0
    assert sum(result.values()) == 100.0


def test_normalize_probabilities_always_totals_exactly_100():
    result = _normalize_probabilities(
        {
            BULLISH: 1,
            BEARISH: 1,
            NEUTRAL: 1,
        }
    )

    assert sum(result.values()) == 100.0


def test_normalize_probabilities_returns_all_directions():
    result = _normalize_probabilities(
        {
            BULLISH: 40,
            BEARISH: 35,
            NEUTRAL: 25,
        }
    )

    assert set(result) == {
        BULLISH,
        BEARISH,
        NEUTRAL,
    }


@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        [],
        (),
        "invalid",
        {},
    ],
)
def test_normalize_probabilities_rejects_invalid_containers(
    probabilities,
):
    with pytest.raises(ForecastEngineError):
        _normalize_probabilities(probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        {
            BULLISH: 50,
            BEARISH: 50,
        },
        {
            BULLISH: 40,
            BEARISH: 30,
            NEUTRAL: 20,
            "SIDEWAYS": 10,
        },
        {
            "INVALID": 50,
            BEARISH: 30,
            NEUTRAL: 20,
        },
    ],
)
def test_normalize_probabilities_rejects_invalid_directions(
    probabilities,
):
    with pytest.raises(ForecastEngineError):
        _normalize_probabilities(probabilities)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "50",
        [],
        {},
    ],
)
def test_normalize_probabilities_rejects_non_numeric_values(value):
    probabilities = {
        BULLISH: 50,
        BEARISH: 30,
        NEUTRAL: 20,
    }

    probabilities[BULLISH] = value

    with pytest.raises(ForecastEngineError):
        _normalize_probabilities(probabilities)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        -1,
        Decimal("-0.01"),
    ],
)
def test_normalize_probabilities_rejects_invalid_numeric_values(
    value,
):
    probabilities = {
        BULLISH: 50,
        BEARISH: 30,
        NEUTRAL: 20,
    }

    probabilities[BULLISH] = value

    with pytest.raises(ForecastEngineError):
        _normalize_probabilities(probabilities)


def test_normalize_probabilities_rejects_zero_total():
    with pytest.raises(ForecastEngineError):
        _normalize_probabilities(
            {
                BULLISH: 0,
                BEARISH: 0,
                NEUTRAL: 0,
            }
        )


# =========================================================
# Probability calculation
# =========================================================

@pytest.mark.parametrize(
    "direction",
    [BULLISH, BEARISH, NEUTRAL],
)
def test_calculate_probabilities_supports_all_directions(direction):
    result = _calculate_probabilities(
        direction=direction,
        score=70,
        agreement=0.8,
    )

    assert set(result) == {
        BULLISH,
        BEARISH,
        NEUTRAL,
    }

    assert sum(result.values()) == 100.0


def test_calculate_probabilities_bullish_direction():
    result = _calculate_probabilities(
        direction=BULLISH,
        score=90,
        agreement=0.9,
    )

    assert result[BULLISH] > result[BEARISH]


def test_calculate_probabilities_bearish_direction():
    result = _calculate_probabilities(
        direction=BEARISH,
        score=-90,
        agreement=0.9,
    )

    assert result[BEARISH] > result[BULLISH]


def test_calculate_probabilities_neutral_direction():
    result = _calculate_probabilities(
        direction=NEUTRAL,
        score=0,
        agreement=0.5,
    )

    assert result[NEUTRAL] > result[BULLISH]
    assert result[NEUTRAL] > result[BEARISH]


def test_calculate_probabilities_zero_conviction():
    result = _calculate_probabilities(
        direction=BULLISH,
        score=0,
        agreement=1,
    )

    assert result[BULLISH] == 33.0
    assert result[BEARISH] == 33.0
    assert result[NEUTRAL] == 34.0


def test_calculate_probabilities_full_conviction():
    result = _calculate_probabilities(
        direction=BULLISH,
        score=100,
        agreement=1,
    )

    assert result[BULLISH] == 75.0
    assert result[BEARISH] == 11.0
    assert result[NEUTRAL] == 14.0


# =========================================================
# Confidence calculation
# =========================================================

def test_calculate_confidence_at_zero_conviction():
    result = _calculate_confidence(
        score=0,
        agreement=1,
    )

    assert result == 35.0


def test_calculate_confidence_at_full_conviction():
    result = _calculate_confidence(
        score=100,
        agreement=1,
    )

    assert result == 100.0


def test_calculate_confidence_increases_with_score():
    low = _calculate_confidence(
        score=20,
        agreement=0.8,
    )

    high = _calculate_confidence(
        score=80,
        agreement=0.8,
    )

    assert high > low


def test_calculate_confidence_is_rounded_to_two_decimals():
    result = _calculate_confidence(
        score=73,
        agreement=0.83,
    )

    assert result == round(result, 2)


# =========================================================
# Market condition extraction
# =========================================================

def test_market_condition_uses_unknown_when_details_are_missing():
    result = _market_condition(
        {
            "direction": BULLISH,
        }
    )

    assert result == {
        "regime": "UNKNOWN",
        "direction": BULLISH,
        "trend_strength": "UNKNOWN",
    }


@pytest.mark.parametrize(
    "details",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_market_condition_handles_invalid_details(details):
    result = _market_condition(
        {
            "direction": BEARISH,
            "details": details,
        }
    )

    assert result["regime"] == "UNKNOWN"
    assert result["direction"] == BEARISH
    assert result["trend_strength"] == "UNKNOWN"


@pytest.mark.parametrize(
    "regime_details",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_market_condition_handles_invalid_regime_details(
    regime_details,
):
    result = _market_condition(
        {
            "direction": NEUTRAL,
            "details": {
                "regime": regime_details,
            },
        }
    )

    assert result["regime"] == "UNKNOWN"
    assert result["trend_strength"] == "UNKNOWN"


def test_market_condition_preserves_valid_regime():
    result = _market_condition(
        {
            "direction": BULLISH,
            "details": {
                "regime": {
                    "regime": "TRENDING",
                    "trend_strength": "STRONG",
                },
            },
        }
    )

    assert result["regime"] == "TRENDING"
    assert result["trend_strength"] == "STRONG"
    assert result["direction"] == BULLISH


def test_market_condition_rejects_empty_regime_as_unknown():
    result = _market_condition(
        {
            "direction": BULLISH,
            "details": {
                "regime": {
                    "regime": "   ",
                    "trend_strength": "STRONG",
                },
            },
        }
    )

    assert result["regime"] == "UNKNOWN"


def test_market_condition_rejects_non_string_regime():
    result = _market_condition(
        {
            "direction": BULLISH,
            "details": {
                "regime": {
                    "regime": 123,
                    "trend_strength": "STRONG",
                },
            },
        }
    )

    assert result["regime"] == "UNKNOWN"


def test_market_condition_rejects_non_string_trend_strength():
    result = _market_condition(
        {
            "direction": BEARISH,
            "details": {
                "regime": {
                    "regime": "TRENDING",
                    "trend_strength": 123,
                },
            },
        }
    )

    assert result["regime"] == "TRENDING"
    assert result["trend_strength"] == "UNKNOWN"


# =========================================================
# Influencing factors
# =========================================================

def test_build_influencing_factors_returns_empty_for_invalid_container():
    result = _build_influencing_factors(
        {
            "component_scores": None,
        }
    )

    assert result == []


@pytest.mark.parametrize(
    "component_scores",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_build_influencing_factors_rejects_non_dictionary_scores(
    component_scores,
):
    result = _build_influencing_factors(
        {
            "component_scores": component_scores,
        }
    )

    assert result == []


def test_build_influencing_factors_classifies_positive_scores():
    result = _build_influencing_factors(
        {
            "component_scores": {
                "trend": 75,
            },
        }
    )

    assert result == [
        {
            "component": "trend",
            "direction": BULLISH,
            "score": 75.0,
        }
    ]


def test_build_influencing_factors_classifies_negative_scores():
    result = _build_influencing_factors(
        {
            "component_scores": {
                "structure": -60,
            },
        }
    )

    assert result[0]["direction"] == BEARISH
    assert result[0]["score"] == -60.0


def test_build_influencing_factors_classifies_zero_as_neutral():
    result = _build_influencing_factors(
        {
            "component_scores": {
                "momentum": 0,
            },
        }
    )

    assert result[0]["direction"] == NEUTRAL
    assert result[0]["score"] == 0.0


def test_build_influencing_factors_sorts_by_absolute_score():
    result = _build_influencing_factors(
        {
            "component_scores": {
                "small": 10,
                "large_negative": -90,
                "medium": 40,
                "large_positive": 80,
            },
        }
    )

    assert [
        factor["component"]
        for factor in result
    ] == [
        "large_negative",
        "large_positive",
        "medium",
        "small",
    ]


def test_build_influencing_factors_ignores_invalid_scores():
    result = _build_influencing_factors(
        {
            "component_scores": {
                "valid_positive": 50,
                "valid_negative": -30,
                "boolean": True,
                "string": "80",
                "none": None,
                "nan": float("nan"),
                "positive_inf": float("inf"),
                "negative_inf": float("-inf"),
            },
        }
    )

    assert len(result) == 2

    assert {
        factor["component"]
        for factor in result
    } == {
        "valid_positive",
        "valid_negative",
    }


def test_build_influencing_factors_converts_component_names_to_strings():
    result = _build_influencing_factors(
        {
            "component_scores": {
                123: 50,
            },
        }
    )

    assert result[0]["component"] == "123"


def test_build_influencing_factors_rounds_scores_to_four_decimals():
    result = _build_influencing_factors(
        {
            "component_scores": {
                "precision": 12.345678,
            },
        }
    )

    assert result[0]["score"] == 12.3457


# =========================================================
# Risk factors
# =========================================================

def make_confluence(
    score=70,
    direction=BULLISH,
    regime="TRENDING",
):
    return {
        "direction": direction,
        "confluence_score": score,
        "details": {
            "regime": {
                "regime": regime,
                "trend_strength": "STRONG",
            },
        },
    }


def test_build_risk_factors_reports_low_agreement():
    result = _build_risk_factors(
        make_confluence(),
        agreement=0.3,
    )

    assert any(
        "Low directional agreement" in risk
        for risk in result
    )


def test_build_risk_factors_does_not_report_low_agreement_at_threshold():
    result = _build_risk_factors(
        make_confluence(),
        agreement=0.5,
    )

    assert not any(
        "Low directional agreement" in risk
        for risk in result
    )


def test_build_risk_factors_reports_weak_directional_bias():
    result = _build_risk_factors(
        make_confluence(score=10),
        agreement=0.8,
    )

    assert any(
        "Weak overall directional market bias" in risk
        for risk in result
    )


def test_build_risk_factors_does_not_report_weak_bias_at_threshold():
    result = _build_risk_factors(
        make_confluence(score=20),
        agreement=0.8,
    )

    assert not any(
        "Weak overall directional market bias" in risk
        for risk in result
    )


@pytest.mark.parametrize(
    "regime",
    [
        "RANGING",
        "TRANSITIONAL",
        "HIGH_VOLATILITY",
    ],
)
def test_build_risk_factors_reports_uncertain_market_regimes(regime):
    result = _build_risk_factors(
        make_confluence(regime=regime),
        agreement=0.8,
    )

    assert any(
        regime in risk
        for risk in result
    )


def test_build_risk_factors_uses_fallback_when_no_risks_exist():
    result = _build_risk_factors(
        make_confluence(
            score=80,
            regime="TRENDING",
        ),
        agreement=0.9,
    )

    assert result == [
        "Market conditions can change during the forecast horizon."
    ]


def test_build_risk_factors_can_report_multiple_risks():
    result = _build_risk_factors(
        make_confluence(
            score=10,
            regime="RANGING",
        ),
        agreement=0.2,
    )

    assert len(result) == 3

    assert any("Low directional agreement" in risk for risk in result)
    assert any("Weak overall directional market bias" in risk for risk in result)
    assert any("RANGING" in risk for risk in result)


def test_build_risk_factors_preserves_risk_order():
    result = _build_risk_factors(
        make_confluence(
            score=10,
            regime="HIGH_VOLATILITY",
        ),
        agreement=0.2,
    )

    assert "Low directional agreement" in result[0]
    assert "Weak overall directional market bias" in result[1]
    assert "HIGH_VOLATILITY" in result[2]