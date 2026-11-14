"""Bitcoin seven-day probabilistic forecast engine."""

from __future__ import annotations

import math

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
from typing import Any


BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

VALID_DIRECTIONS = {
    BULLISH,
    BEARISH,
    NEUTRAL,
}

DEFAULT_HORIZON_DAYS = 7
DEFAULT_MODEL_VERSION = "heuristic-v1"

PROBABILITY_SCALE = 100
PROBABILITY_DECIMALS = 2
PROBABILITY_UNITS = PROBABILITY_SCALE * (10 ** PROBABILITY_DECIMALS)


class ForecastEngineError(RuntimeError):
    """Raised when forecast generation fails."""


def _parse_utc_timestamp(
    value: Any,
    field_name: str,
) -> datetime:
    """Parse and validate a timezone-aware UTC timestamp."""

    if isinstance(value, datetime):
        timestamp = value

    elif isinstance(value, str) and value.strip():
        try:
            timestamp = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ForecastEngineError(
                f"Invalid timestamp: {field_name}."
            ) from exc

    else:
        raise ForecastEngineError(
            f"{field_name} must be a valid UTC timestamp."
        )

    if timestamp.tzinfo is None:
        raise ForecastEngineError(
            f"{field_name} must be timezone-aware."
        )

    try:
        offset = timestamp.utcoffset()
    except (ValueError, OverflowError) as exc:
        raise ForecastEngineError(
            f"Invalid timezone for {field_name}."
        ) from exc

    if offset != timedelta(0):
        raise ForecastEngineError(
            f"{field_name} must use UTC."
        )

    return timestamp.astimezone(timezone.utc)


def _normalize_forecast_time(
    forecast_time: datetime | None,
) -> datetime:
    """Return the forecast generation timestamp in UTC."""

    if forecast_time is None:
        return datetime.now(timezone.utc)

    return _parse_utc_timestamp(
        forecast_time,
        "forecast_time",
    )


def _validate_horizon(
    horizon_days: int,
) -> int:
    """Validate forecast horizon."""

    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or horizon_days <= 0
        or horizon_days > 365
    ):
        raise ForecastEngineError(
            "horizon_days must be an integer between 1 and 365."
        )

    return horizon_days


def _validate_market_analysis(
    market_analysis: dict[str, Any],
    forecast_time: datetime,
) -> dict[str, Any]:
    """Validate the integrated market analysis result."""

    if not isinstance(market_analysis, dict):
        raise ForecastEngineError(
            "market_analysis must be a dictionary."
        )

    if market_analysis.get("status") != "SUCCESS":
        raise ForecastEngineError(
            "Market analysis must have SUCCESS status."
        )

    required_fields = (
        "analysis_timestamp",
        "confluence",
        "market_price",
        "source",
        "last_candle_close_time",
        "candles_analyzed",
    )

    missing = [
        field
        for field in required_fields
        if field not in market_analysis
    ]

    if missing:
        raise ForecastEngineError(
            "Market analysis is missing required fields: "
            + ", ".join(missing)
        )

    analysis_timestamp = _parse_utc_timestamp(
        market_analysis["analysis_timestamp"],
        "analysis_timestamp",
    )

    candle_close_time = _parse_utc_timestamp(
        market_analysis["last_candle_close_time"],
        "last_candle_close_time",
    )

    if analysis_timestamp > forecast_time:
        raise ForecastEngineError(
            "Market analysis timestamp cannot be in the future."
        )

    if candle_close_time > analysis_timestamp:
        raise ForecastEngineError(
            "Market analysis contains a candle closing after its analysis time."
        )

    if candle_close_time > forecast_time:
        raise ForecastEngineError(
            "Future candle data is not allowed."
        )

    price = market_analysis["market_price"]

    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(price)
        or price <= 0
    ):
        raise ForecastEngineError(
            "market_price must be a finite positive number."
        )

    candle_count = market_analysis["candles_analyzed"]

    if (
        isinstance(candle_count, bool)
        or not isinstance(candle_count, int)
        or candle_count <= 0
    ):
        raise ForecastEngineError(
            "candles_analyzed must be a positive integer."
        )

    sources = market_analysis["source"]

    if (
        not isinstance(sources, list)
        or not sources
        or not all(
            isinstance(source, str) and source.strip()
            for source in sources
        )
    ):
        raise ForecastEngineError(
            "Market analysis must contain valid data sources."
        )

    confluence = market_analysis["confluence"]

    if not isinstance(confluence, dict):
        raise ForecastEngineError(
            "confluence must be a dictionary."
        )

    if confluence.get("status") != "SUCCESS":
        raise ForecastEngineError(
            "Confluence analysis must have SUCCESS status."
        )

    confluence_analysis_time = _parse_utc_timestamp(
        confluence.get("analysis_timestamp"),
        "confluence.analysis_timestamp",
    )

    confluence_candle_time = _parse_utc_timestamp(
        confluence.get("last_candle_close_time"),
        "confluence.last_candle_close_time",
    )

    if confluence_analysis_time != analysis_timestamp:
        raise ForecastEngineError(
            "Confluence and market analysis timestamps do not match."
        )

    if confluence_candle_time != candle_close_time:
        raise ForecastEngineError(
            "Confluence and market analysis candle timestamps do not match."
        )

    direction = confluence.get("direction")

    if direction not in VALID_DIRECTIONS:
        raise ForecastEngineError(
            "Confluence contains an invalid market direction."
        )

    score = confluence.get("confluence_score")

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not -100 <= score <= 100
    ):
        raise ForecastEngineError(
            "confluence_score must be between -100 and 100."
        )

    agreement = confluence.get("agreement_ratio")

    if (
        isinstance(agreement, bool)
        or not isinstance(agreement, (int, float))
        or not math.isfinite(agreement)
        or not 0 <= agreement <= 1
    ):
        raise ForecastEngineError(
            "agreement_ratio must be between 0 and 1."
        )

    component_scores = confluence.get("component_scores")

    if not isinstance(component_scores, dict):
        raise ForecastEngineError(
            "Confluence component_scores must be a dictionary."
        )

    return {
        "analysis_timestamp": analysis_timestamp,
        "last_candle_close_time": candle_close_time,
        "direction": direction,
        "score": float(score),
        "agreement": float(agreement),
        "market_price": float(price),
        "sources": sorted(set(sources)),
        "candles_analyzed": candle_count,
        "confluence": confluence,
    }


def _normalize_probabilities(
    probabilities: dict[str, float],
) -> dict[str, float]:
    """
    Normalize probabilities using integer hundredth-percent units.

    The largest remainder method guarantees:
    - Every probability is represented with two decimal places.
    - Every probability remains within 0..100.
    - The final total is exactly 100.00.
    """

    if not isinstance(probabilities, dict) or not probabilities:
        raise ForecastEngineError(
            "Invalid probability distribution."
        )

    if set(probabilities) != VALID_DIRECTIONS:
        raise ForecastEngineError(
            "Probability distribution contains invalid directions."
        )

    decimal_values: dict[str, Decimal] = {}

    for name, value in probabilities.items():

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, Decimal))
        ):
            raise ForecastEngineError(
                "Probabilities must be numeric."
            )

        try:
            decimal_value = Decimal(str(value))
        except Exception as exc:
            raise ForecastEngineError(
                "Invalid probability value."
            ) from exc

        if not decimal_value.is_finite() or decimal_value < 0:
            raise ForecastEngineError(
                "Probabilities must be finite and non-negative."
            )

        decimal_values[name] = decimal_value

    total = sum(
        decimal_values.values(),
        Decimal("0"),
    )

    if total <= 0:
        raise ForecastEngineError(
            "Invalid probability distribution."
        )

    exact_units = {
        name: (
            value / total * Decimal(PROBABILITY_UNITS)
        )
        for name, value in decimal_values.items()
    }

    integer_units = {
        name: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for name, value in exact_units.items()
    }

    remaining_units = (
        PROBABILITY_UNITS - sum(integer_units.values())
    )

    remainders = {
        name: exact_units[name] - Decimal(integer_units[name])
        for name in exact_units
    }

    remainder_order = sorted(
        remainders,
        key=lambda name: (
            remainders[name],
            exact_units[name],
            name == BULLISH,
        ),
        reverse=True,
    )

    for name in remainder_order[:remaining_units]:
        integer_units[name] += 1

    rounded = {
        name: integer_units[name] / (10 ** PROBABILITY_DECIMALS)
        for name in VALID_DIRECTIONS
    }

    if (
        any(
            not math.isfinite(value) or value < 0 or value > 100
            for value in rounded.values()
        )
        or sum(integer_units.values()) != PROBABILITY_UNITS
    ):
        raise ForecastEngineError(
            "Probability normalization failed."
        )

    return rounded


def _calculate_probabilities(
    direction: str,
    score: float,
    agreement: float,
) -> dict[str, float]:
    """
    Estimate three heuristic market scenarios.

    These values are heuristic estimates.
    They are not statistically calibrated probabilities.
    """

    conviction = (
        abs(score) / 100.0
    ) * agreement

    conviction = max(
        0.0,
        min(1.0, conviction),
    )

    if direction == BULLISH:
        probabilities = {
            BULLISH: 33.0 + 42.0 * conviction,
            BEARISH: 33.0 - 22.0 * conviction,
            NEUTRAL: 34.0 - 20.0 * conviction,
        }

    elif direction == BEARISH:
        probabilities = {
            BULLISH: 33.0 - 22.0 * conviction,
            BEARISH: 33.0 + 42.0 * conviction,
            NEUTRAL: 34.0 - 20.0 * conviction,
        }

    else:
        probabilities = {
            BULLISH: 33.0 - 15.0 * conviction,
            BEARISH: 33.0 - 15.0 * conviction,
            NEUTRAL: 34.0 + 30.0 * conviction,
        }

    return _normalize_probabilities(
        probabilities
    )


def _calculate_confidence(
    score: float,
    agreement: float,
) -> float:
    """Calculate a heuristic confidence score from 0 to 100."""

    conviction = (
        abs(score) / 100.0
    ) * agreement

    confidence = 35.0 + (
        65.0 * conviction
    )

    return round(
        max(0.0, min(100.0, confidence)),
        2,
    )


def _market_condition(
    confluence: dict[str, Any],
) -> dict[str, Any]:
    """Extract the market regime without inventing missing information."""

    details = confluence.get("details", {})

    if not isinstance(details, dict):
        details = {}

    regime_details = details.get("regime", {})

    if not isinstance(regime_details, dict):
        regime_details = {}

    regime = regime_details.get("regime")

    if not isinstance(regime, str) or not regime.strip():
        regime = "UNKNOWN"

    trend_strength = regime_details.get(
        "trend_strength",
        "UNKNOWN",
    )

    if not isinstance(trend_strength, str):
        trend_strength = "UNKNOWN"

    return {
        "regime": regime,
        "direction": confluence["direction"],
        "trend_strength": trend_strength,
    }


def _build_influencing_factors(
    confluence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a transparent list of directional influencing factors."""

    component_scores = confluence.get(
        "component_scores",
        {},
    )

    if not isinstance(component_scores, dict):
        return []

    factors = []

    for name, score in component_scores.items():

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            continue

        if score > 0:
            direction = BULLISH

        elif score < 0:
            direction = BEARISH

        else:
            direction = NEUTRAL

        factors.append(
            {
                "component": str(name),
                "direction": direction,
                "score": round(float(score), 4),
            }
        )

    factors.sort(
        key=lambda item: abs(item["score"]),
        reverse=True,
    )

    return factors


def _build_risk_factors(
    confluence: dict[str, Any],
    agreement: float,
) -> list[str]:
    """Describe uncertainty and potential forecast limitations."""

    risks = []

    if agreement < 0.5:
        risks.append(
            "Low directional agreement between market analysis components."
        )

    if abs(confluence["confluence_score"]) < 20:
        risks.append(
            "Weak overall directional market bias."
        )

    regime = _market_condition(
        confluence
    )["regime"]

    if regime in (
        "RANGING",
        "TRANSITIONAL",
        "HIGH_VOLATILITY",
    ):
        risks.append(
            f"Market regime may increase forecast uncertainty: {regime}."
        )

    if not risks:
        risks.append(
            "Market conditions can change during the forecast horizon."
        )

    return risks


def generate_forecast(
    market_analysis: dict[str, Any],
    forecast_time: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Any]:
    """
    Generate a probabilistic Bitcoin market forecast.

    The engine uses a transparent heuristic model.
    It does not claim statistical calibration or guaranteed accuracy.
    It does not persist records or execute trades.
    """

    timestamp = _normalize_forecast_time(
        forecast_time
    )

    horizon = _validate_horizon(
        horizon_days
    )

    if (
        not isinstance(model_version, str)
        or not model_version.strip()
    ):
        raise ForecastEngineError(
            "model_version must be a non-empty string."
        )

    validated = _validate_market_analysis(
        market_analysis,
        timestamp,
    )

    confluence = validated["confluence"]

    probabilities = _calculate_probabilities(
        direction=validated["direction"],
        score=validated["score"],
        agreement=validated["agreement"],
    )

    confidence = _calculate_confidence(
        score=validated["score"],
        agreement=validated["agreement"],
    )

    condition = _market_condition(
        confluence
    )

    influencing_factors = _build_influencing_factors(
        confluence
    )

    risk_factors = _build_risk_factors(
        confluence,
        validated["agreement"],
    )

    horizon_end = timestamp + timedelta(
        days=horizon
    )

    forecast_id = (
        f"BTC-{timestamp.strftime('%Y%m%dT%H%M%S')}"
        f"-{horizon}D-{model_version}"
    )

    return {
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "asset": "BTC",
        "forecast_timestamp": timestamp.isoformat(),
        "analysis_timestamp": (
            validated["analysis_timestamp"].isoformat()
        ),
        "last_candle_close_time": (
            validated["last_candle_close_time"].isoformat()
        ),
        "forecast_horizon": {
            "days": horizon,
            "start": timestamp.isoformat(),
            "end": horizon_end.isoformat(),
        },
        "market_price": validated["market_price"],
        "probabilities": probabilities,
        "confidence_score": confidence,
        "market_condition": condition,
        "influencing_factors": influencing_factors,
        "risk_factors": risk_factors,
        "model": {
            "name": "Bitcoin Heuristic Forecast Engine",
            "version": model_version,
            "method": "weighted_confluence_heuristic",
            "statistically_calibrated": False,
        },
        "input_metadata": {
            "sources": validated["sources"],
            "candles_analyzed": validated["candles_analyzed"],
            "confluence_score": validated["score"],
            "agreement_ratio": validated["agreement"],
        },
        "limitations": [
            "Probabilities are heuristic estimates, not statistically calibrated probabilities.",
            "Forecasts are estimates and do not guarantee future market outcomes.",
            "Forecast quality requires evaluation against completed market outcomes.",
            "This engine does not generate trading orders.",
            "This engine does not execute trades.",
            "This engine does not persist or overwrite forecast history.",
        ],
    }


if __name__ == "__main__":
    print(
        "Bitcoin seven-day forecast engine loaded."
    )