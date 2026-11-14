"""Bitcoin market confluence analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

SUCCESS = "SUCCESS"

DEFAULT_WEIGHTS = {
    "trend": 0.30,
    "structure": 0.25,
    "zones": 0.15,
    "liquidity": 0.15,
    "regime": 0.15,
}

REQUIRED_ANALYSES = (
    "trend",
    "structure",
    "zones",
    "liquidity",
    "regime",
)


class MarketConfluenceError(RuntimeError):
    """Raised when market confluence analysis fails."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> datetime:
    """Normalize the analysis timestamp to UTC."""

    if analysis_time is None:
        return datetime.now(timezone.utc)

    try:
        timestamp = analysis_time

        if not isinstance(timestamp, datetime):
            raise TypeError

        if timestamp.tzinfo is None:
            raise MarketConfluenceError(
                "analysis_time must be timezone-aware UTC."
            )

        if timestamp.utcoffset().total_seconds() != 0:
            raise MarketConfluenceError(
                "analysis_time must use UTC."
            )

        return timestamp.astimezone(timezone.utc)

    except MarketConfluenceError:
        raise

    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketConfluenceError(
            "Invalid analysis_time."
        ) from exc


def _validate_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    """Validate and normalize module weights."""

    if weights is None:
        return DEFAULT_WEIGHTS.copy()

    if not isinstance(weights, dict):
        raise MarketConfluenceError(
            "weights must be a dictionary."
        )

    if set(weights) != set(REQUIRED_ANALYSES):
        raise MarketConfluenceError(
            "weights must contain exactly these keys: "
            + ", ".join(REQUIRED_ANALYSES)
        )

    normalized = {}

    for name, value in weights.items():

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise MarketConfluenceError(
                f"Weight '{name}' must be a finite "
                "non-negative number."
            )

        normalized[name] = float(value)

    total = sum(normalized.values())

    if not np.isfinite(total) or total <= 0:
        raise MarketConfluenceError(
            "The sum of weights must be greater than zero."
        )

    return {
        name: value / total
        for name, value in normalized.items()
    }


def _parse_timestamp(
    value: Any,
    field_name: str,
) -> datetime:
    """Parse and validate a UTC timestamp."""

    if not isinstance(value, str) or not value.strip():
        raise MarketConfluenceError(
            f"{field_name} must be a non-empty ISO timestamp."
        )

    try:
        timestamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketConfluenceError(
            f"Invalid timestamp in {field_name}."
        ) from exc

    if timestamp.tzinfo is None:
        raise MarketConfluenceError(
            f"{field_name} must be timezone-aware."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise MarketConfluenceError(
            f"{field_name} must use UTC."
        )

    return timestamp.astimezone(timezone.utc)


def _validate_analyses(
    analyses: dict[str, Any],
    analysis_time: datetime,
) -> dict[str, dict[str, Any]]:
    """Validate all upstream analysis results."""

    if not isinstance(analyses, dict):
        raise MarketConfluenceError(
            "Analyses must be provided as a dictionary."
        )

    missing = [
        name
        for name in REQUIRED_ANALYSES
        if name not in analyses
    ]

    if missing:
        raise MarketConfluenceError(
            "Missing analysis results: "
            + ", ".join(missing)
        )

    validated = {}

    reference_close_time = None
    reference_candle_count = None

    for name in REQUIRED_ANALYSES:

        result = analyses[name]

        if not isinstance(result, dict):
            raise MarketConfluenceError(
                f"{name} analysis must be a dictionary."
            )

        if result.get("status") != SUCCESS:
            raise MarketConfluenceError(
                f"{name} analysis is not successful."
            )

        if "analysis_timestamp" not in result:
            raise MarketConfluenceError(
                f"{name} analysis is missing analysis_timestamp."
            )

        result_time = _parse_timestamp(
            result["analysis_timestamp"],
            f"{name}.analysis_timestamp",
        )

        if result_time > analysis_time:
            raise MarketConfluenceError(
                f"{name} analysis timestamp is in the future."
            )

        if "last_candle_close_time" not in result:
            raise MarketConfluenceError(
                f"{name} analysis is missing last_candle_close_time."
            )

        close_time = _parse_timestamp(
            result["last_candle_close_time"],
            f"{name}.last_candle_close_time",
        )

        if close_time > analysis_time:
            raise MarketConfluenceError(
                f"{name} contains a future candle."
            )

        if reference_close_time is None:
            reference_close_time = close_time

        elif close_time != reference_close_time:
            raise MarketConfluenceError(
                "All analyses must use the same last candle close time."
            )

        candle_count = result.get("candles_analyzed")

        if (
            isinstance(candle_count, bool)
            or not isinstance(candle_count, int)
            or candle_count <= 0
        ):
            raise MarketConfluenceError(
                f"{name}.candles_analyzed must be a positive integer."
            )

        if reference_candle_count is None:
            reference_candle_count = candle_count

        elif candle_count != reference_candle_count:
            raise MarketConfluenceError(
                "All analyses must use the same candle count."
            )

        sources = result.get("source")

        if (
            not isinstance(sources, list)
            or not sources
            or not all(
                isinstance(source, str) and source.strip()
                for source in sources
            )
        ):
            raise MarketConfluenceError(
                f"{name}.source must contain valid source names."
            )

        validated[name] = result

    return validated


def _clamp_score(
    value: float,
) -> float:
    """Clamp a directional score to the range -100 to 100."""

    if not np.isfinite(value):
        raise MarketConfluenceError(
            "A non-finite directional score was produced."
        )

    return float(
        np.clip(value, -100.0, 100.0)
    )


def _score_trend(
    result: dict[str, Any],
) -> float:
    """Extract the directional trend score."""

    value = result.get("trend_score")

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise MarketConfluenceError(
            "Trend analysis contains an invalid trend_score."
        )

    return _clamp_score(float(value))


def _score_structure(
    result: dict[str, Any],
) -> tuple[float, dict[str, int]]:
    """Score market structure direction and recent events."""

    direction = result.get("trend")

    if direction not in (BULLISH, BEARISH, NEUTRAL):
        raise MarketConfluenceError(
            "Structure analysis contains an invalid trend."
        )

    score = {
        BULLISH: 60.0,
        BEARISH: -60.0,
        NEUTRAL: 0.0,
    }[direction]

    events = result.get("structure_breaks", [])

    if not isinstance(events, list):
        raise MarketConfluenceError(
            "structure_breaks must be a list."
        )

    event_counts = {
        "BOS": 0,
        "CHoCH": 0,
    }

    for event in events:

        if not isinstance(event, dict):
            raise MarketConfluenceError(
                "Structure events must be dictionaries."
            )

        event_type = event.get("type")

        if event_type not in event_counts:
            continue

        event_counts[event_type] += 1

        event_direction = event.get("direction")

        if event_direction not in (BULLISH, BEARISH):
            continue

        adjustment = (
            15.0
            if event_type == "BOS"
            else 10.0
        )

        if event_direction == BULLISH:
            score += adjustment

        else:
            score -= adjustment

    return _clamp_score(score), event_counts


def _score_zones(
    result: dict[str, Any],
) -> tuple[float, dict[str, int]]:
    """Score active supply and demand zones."""

    zones = result.get("zones", [])

    if not isinstance(zones, list):
        raise MarketConfluenceError(
            "zones must be a list."
        )

    counts = {
        "active_demand": 0,
        "active_supply": 0,
        "mitigated": 0,
        "broken": 0,
    }

    for zone in zones:

        if not isinstance(zone, dict):
            raise MarketConfluenceError(
                "Every zone must be a dictionary."
            )

        zone_type = zone.get("type")
        status = zone.get("status")

        if status == "MITIGATED":
            counts["mitigated"] += 1
            continue

        if status == "BROKEN":
            counts["broken"] += 1
            continue

        if status != "ACTIVE":
            continue

        if zone_type == "DEMAND":
            counts["active_demand"] += 1

        elif zone_type == "SUPPLY":
            counts["active_supply"] += 1

    difference = (
        counts["active_demand"]
        - counts["active_supply"]
    )

    score = float(
        np.clip(difference * 20.0, -60.0, 60.0)
    )

    return score, counts


def _score_liquidity(
    result: dict[str, Any],
) -> tuple[float, dict[str, int]]:
    """Score liquidity sweeps."""

    sweeps = result.get("sweeps", [])

    if not isinstance(sweeps, list):
        raise MarketConfluenceError(
            "sweeps must be a list."
        )

    counts = {
        "buy_side_sweeps": 0,
        "sell_side_sweeps": 0,
        "unknown_sweeps": 0,
    }

    for sweep in sweeps:

        if not isinstance(sweep, dict):
            raise MarketConfluenceError(
                "Every liquidity sweep must be a dictionary."
            )

        side = sweep.get(
            "side",
            sweep.get("type"),
        )

        if side in (
            "BUY_SIDE",
            "BUY_SIDE_LIQUIDITY",
            "HIGH",
        ):
            counts["buy_side_sweeps"] += 1

        elif side in (
            "SELL_SIDE",
            "SELL_SIDE_LIQUIDITY",
            "LOW",
        ):
            counts["sell_side_sweeps"] += 1

        else:
            counts["unknown_sweeps"] += 1

    # A buy-side liquidity sweep can indicate
    # rejection above highs and bearish pressure.
    #
    # A sell-side liquidity sweep can indicate
    # rejection below lows and bullish pressure.

    score = (
        counts["sell_side_sweeps"]
        - counts["buy_side_sweeps"]
    ) * 20.0

    return _clamp_score(score), counts


def _score_regime(
    result: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Score market regime direction and conditions."""

    direction = result.get("direction")

    if direction not in (BULLISH, BEARISH, NEUTRAL):
        raise MarketConfluenceError(
            "Regime analysis contains an invalid direction."
        )

    regime = result.get("regime")

    valid_regimes = {
        "TRENDING",
        "RANGING",
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
        "TRANSITIONAL",
    }

    if regime not in valid_regimes:
        raise MarketConfluenceError(
            "Regime analysis contains an invalid regime."
        )

    score = {
        BULLISH: 50.0,
        BEARISH: -50.0,
        NEUTRAL: 0.0,
    }[direction]

    if regime == "TRENDING":
        score *= 1.0

    elif regime == "RANGING":
        score *= 0.35

    elif regime == "TRANSITIONAL":
        score *= 0.5

    elif regime in (
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
    ):
        score *= 0.25

    details = {
        "regime": regime,
        "direction": direction,
        "trend_strength": result.get("trend_strength"),
        "volatility": result.get("volatility", {}),
    }

    return _clamp_score(score), details


def _classify_confluence(
    score: float,
    bullish_threshold: float,
    bearish_threshold: float,
) -> str:
    """Classify the combined directional bias."""

    if score >= bullish_threshold:
        return BULLISH

    if score <= -bearish_threshold:
        return BEARISH

    return NEUTRAL


def analyze_market_confluence(
    trend_analysis: dict[str, Any],
    structure_analysis: dict[str, Any],
    zones_analysis: dict[str, Any],
    liquidity_analysis: dict[str, Any],
    regime_analysis: dict[str, Any],
    weights: dict[str, float] | None = None,
    bullish_threshold: float = 20.0,
    bearish_threshold: float = 20.0,
    analysis_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Combine five independent market analysis results.

    Inputs:
    - Trend analysis.
    - Price structure analysis.
    - Supply and demand zones.
    - Liquidity analysis.
    - Market regime analysis.

    Features:
    - Validates upstream analysis results.
    - Requires matching candle close timestamps.
    - Combines weighted directional scores.
    - Reports bullish and bearish evidence.
    - Provides a transparent component breakdown.
    - Preserves source information.
    - Uses UTC timestamps.
    - Does not generate trading signals.
    - Does not execute trades.
    """

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    normalized_weights = _validate_weights(
        weights
    )

    for name, value in (
        ("bullish_threshold", bullish_threshold),
        ("bearish_threshold", bearish_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            or value > 100
        ):
            raise MarketConfluenceError(
                f"{name} must be a finite number "
                "greater than 0 and at most 100."
            )

    analyses = {
        "trend": trend_analysis,
        "structure": structure_analysis,
        "zones": zones_analysis,
        "liquidity": liquidity_analysis,
        "regime": regime_analysis,
    }

    validated = _validate_analyses(
        analyses,
        timestamp,
    )

    component_scores = {}

    component_scores["trend"] = _score_trend(
        validated["trend"]
    )

    structure_score, structure_events = (
        _score_structure(
            validated["structure"]
        )
    )

    component_scores["structure"] = structure_score

    zones_score, zone_counts = _score_zones(
        validated["zones"]
    )

    component_scores["zones"] = zones_score

    liquidity_score, liquidity_counts = (
        _score_liquidity(
            validated["liquidity"]
        )
    )

    component_scores["liquidity"] = liquidity_score

    regime_score, regime_details = _score_regime(
        validated["regime"]
    )

    component_scores["regime"] = regime_score

    weighted_components = {
        name: component_scores[name] * normalized_weights[name]
        for name in REQUIRED_ANALYSES
    }

    final_score = _clamp_score(
        sum(weighted_components.values())
    )

    direction = _classify_confluence(
        final_score,
        float(bullish_threshold),
        float(bearish_threshold),
    )

    bullish_evidence = sum(
        value > 0
        for value in component_scores.values()
    )

    bearish_evidence = sum(
        value < 0
        for value in component_scores.values()
    )

    neutral_evidence = (
        len(component_scores)
        - bullish_evidence
        - bearish_evidence
    )

    sources = sorted(
        {
            source
            for result in validated.values()
            for source in result["source"]
        }
    )

    latest_close_time = _parse_timestamp(
        validated["trend"]["last_candle_close_time"],
        "trend.last_candle_close_time",
    )

    candle_count = validated["trend"]["candles_analyzed"]

    # This is a heuristic agreement measure.
    # It is not a statistical probability.

    agreement_ratio = max(
        bullish_evidence,
        bearish_evidence,
    ) / len(REQUIRED_ANALYSES)

    return {
        "analysis_timestamp": timestamp.isoformat(),
        "status": SUCCESS,
        "source": sources,
        "direction": direction,
        "confluence_score": round(final_score, 4),
        "agreement_ratio": round(
            agreement_ratio,
            4,
        ),
        "evidence": {
            "bullish": bullish_evidence,
            "bearish": bearish_evidence,
            "neutral": neutral_evidence,
            "total_components": len(component_scores),
        },
        "component_scores": {
            name: round(value, 4)
            for name, value in component_scores.items()
        },
        "weighted_scores": {
            name: round(value, 4)
            for name, value in weighted_components.items()
        },
        "weights": {
            name: round(value, 6)
            for name, value in normalized_weights.items()
        },
        "details": {
            "structure_events": structure_events,
            "zones": zone_counts,
            "liquidity": liquidity_counts,
            "regime": regime_details,
        },
        "market_price": validated["regime"].get(
            "market_price"
        ),
        "candles_analyzed": candle_count,
        "last_candle_close_time": latest_close_time.isoformat(),
        "parameters": {
            "bullish_threshold": float(bullish_threshold),
            "bearish_threshold": float(bearish_threshold),
        },
        "limitations": [
            "Confluence scores are heuristic, not probabilities.",
            "This analysis does not generate trading signals.",
            "This analysis does not execute trades.",
        ],
    }


if __name__ == "__main__":
    print(
        "Bitcoin market confluence analysis module loaded."
    )