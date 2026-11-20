
"""Bitcoin market analysis report builder."""

from __future__ import annotations

import math

from datetime import datetime, timedelta
from typing import Any

from market_analysis.engine import analyze_market
from reports.engine import (
    DEFAULT_REPORT_DIRECTORY,
    ReportEngine,
    ReportEngineError,
)


class MarketReportError(ValueError):
    """Raised when a market report cannot be generated."""


REQUIRED_COMPONENTS = (
    "trend",
    "structure",
    "zones",
    "liquidity",
    "regime",
)

VALID_DIRECTIONS = {
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
}

VALID_TRENDS = {
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
}

VALID_REGIMES = {
    "TRENDING",
    "RANGING",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    "TRANSITIONAL",
}


def _parse_utc_timestamp(
    value: Any,
    field_name: str,
) -> datetime:
    """Parse and validate an ISO UTC timestamp."""

    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketReportError(
                f"Invalid timestamp: {field_name}."
            ) from exc

    elif isinstance(value, datetime):
        timestamp = value

    else:
        raise MarketReportError(
            f"{field_name} must be a datetime or ISO timestamp."
        )

    if timestamp.tzinfo is None:
        raise MarketReportError(
            f"{field_name} must be timezone-aware UTC."
        )

    try:
        offset = timestamp.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketReportError(
            f"Invalid timezone in {field_name}."
        ) from exc

    if offset is None or offset != timedelta(0):
        raise MarketReportError(
            f"{field_name} must use UTC."
        )

    return timestamp.astimezone(
        datetime.now().astimezone().tzinfo
    ).astimezone(
        __import__("datetime").timezone.utc
    )


def _positive_number(
    value: Any,
    field_name: str,
) -> float:
    """Validate a finite positive number."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise MarketReportError(
            f"{field_name} must be a finite positive number."
        )

    return float(value)


def _non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    """Validate a non-negative integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise MarketReportError(
            f"{field_name} must be a non-negative integer."
        )

    return value


def _validate_sources(value: Any) -> list[str]:
    """Validate source identifiers."""

    if not isinstance(value, list) or not value:
        raise MarketReportError(
            "Market analysis must contain at least one data source."
        )

    if any(
        not isinstance(source, str) or not source.strip()
        for source in value
    ):
        raise MarketReportError(
            "Market analysis contains an invalid source."
        )

    return sorted(set(source.strip() for source in value))


def _validate_market_analysis(
    result: Any,
) -> dict[str, Any]:
    """Validate the integrated market analysis output."""

    if not isinstance(result, dict):
        raise MarketReportError(
            "Market analysis result must be a dictionary."
        )

    if result.get("status") != "SUCCESS":
        raise MarketReportError(
            "Only successful market analyses can be reported."
        )

    required_fields = (
        "analysis_timestamp",
        "analyses",
        "confluence",
        "market_price",
        "source",
        "candles_analyzed",
        "last_candle_close_time",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise MarketReportError(
            "Market analysis is missing required fields: "
            + ", ".join(missing)
        )

    timestamp = _parse_utc_timestamp(
        result["analysis_timestamp"],
        "analysis_timestamp",
    )

    last_close_time = _parse_utc_timestamp(
        result["last_candle_close_time"],
        "last_candle_close_time",
    )

    if last_close_time > timestamp:
        raise MarketReportError(
            "Last candle close time cannot be after analysis time."
        )

    market_price = _positive_number(
        result["market_price"],
        "market_price",
    )

    candle_count = _non_negative_integer(
        result["candles_analyzed"],
        "candles_analyzed",
    )

    if candle_count <= 0:
        raise MarketReportError(
            "Market analysis must contain at least one candle."
        )

    sources = _validate_sources(
        result["source"]
    )

    analyses = result["analyses"]

    if not isinstance(analyses, dict):
        raise MarketReportError(
            "Market component analyses must be a dictionary."
        )

    missing_components = [
        component
        for component in REQUIRED_COMPONENTS
        if component not in analyses
    ]

    if missing_components:
        raise MarketReportError(
            "Market analysis is missing components: "
            + ", ".join(missing_components)
        )

    for component in REQUIRED_COMPONENTS:
        component_result = analyses[component]

        if not isinstance(component_result, dict):
            raise MarketReportError(
                f"Invalid result for component: {component}."
            )

        if component_result.get("status") != "SUCCESS":
            raise MarketReportError(
                f"Component '{component}' did not succeed."
            )

        if (
            component_result.get("analysis_timestamp")
            != timestamp.isoformat()
        ):
            raise MarketReportError(
                f"Component '{component}' has a mismatched timestamp."
            )

        if (
            component_result.get("candles_analyzed")
            != candle_count
        ):
            raise MarketReportError(
                f"Component '{component}' analyzed a different candle count."
            )

        component_close_time = _parse_utc_timestamp(
            component_result.get("last_candle_close_time"),
            f"{component}.last_candle_close_time",
        )

        if component_close_time != last_close_time:
            raise MarketReportError(
                f"Component '{component}' analyzed a different last candle."
            )

    confluence = result["confluence"]

    if not isinstance(confluence, dict):
        raise MarketReportError(
            "Confluence result must be a dictionary."
        )

    if confluence.get("status") != "SUCCESS":
        raise MarketReportError(
            "Confluence analysis did not succeed."
        )

    if (
        confluence.get("analysis_timestamp")
        != timestamp.isoformat()
    ):
        raise MarketReportError(
            "Confluence timestamp does not match market analysis."
        )

    direction = confluence.get("direction")

    if direction not in VALID_DIRECTIONS:
        raise MarketReportError(
            "Confluence contains an invalid market direction."
        )

    score = confluence.get("confluence_score")

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not -100 <= score <= 100
    ):
        raise MarketReportError(
            "Confluence score must be between -100 and 100."
        )

    agreement = confluence.get("agreement_ratio")

    if (
        isinstance(agreement, bool)
        or not isinstance(agreement, (int, float))
        or not math.isfinite(agreement)
        or not 0 <= agreement <= 1
    ):
        raise MarketReportError(
            "Confluence agreement ratio must be between 0 and 1."
        )

    confluence_close_time = _parse_utc_timestamp(
        confluence.get("last_candle_close_time"),
        "confluence.last_candle_close_time",
    )

    if confluence_close_time != last_close_time:
        raise MarketReportError(
            "Confluence last candle does not match market analysis."
        )

    return {
        "timestamp": timestamp,
        "last_close_time": last_close_time,
        "market_price": market_price,
        "candles_analyzed": candle_count,
        "sources": sources,
        "analyses": analyses,
        "confluence": confluence,
        "original": result,
    }


def _build_market_summary(
    validated: dict[str, Any],
) -> dict[str, Any]:
    """Build a concise summary from verified component results."""

    analyses = validated["analyses"]
    trend = analyses["trend"]
    structure = analyses["structure"]
    regime = analyses["regime"]
    confluence = validated["confluence"]

    trend_direction = trend.get("trend")

    if trend_direction not in VALID_TRENDS:
        raise MarketReportError(
            "Trend analysis contains an invalid direction."
        )

    regime_name = regime.get("regime")

    if regime_name not in VALID_REGIMES:
        raise MarketReportError(
            "Market regime contains an invalid classification."
        )

    regime_direction = regime.get("direction")

    if regime_direction not in VALID_DIRECTIONS:
        raise MarketReportError(
            "Market regime contains an invalid direction."
        )

    moving_averages = trend.get(
        "moving_averages",
        {},
    )

    if not isinstance(moving_averages, dict):
        raise MarketReportError(
            "Trend moving averages must be a dictionary."
        )

    levels = structure.get(
        "levels",
        {},
    )

    if not isinstance(levels, dict):
        raise MarketReportError(
            "Price structure levels must be a dictionary."
        )

    return {
        "market_price": validated["market_price"],
        "trend": {
            "direction": trend_direction,
            "strength": trend.get("trend_strength"),
            "score": trend.get("trend_score"),
            "confidence": trend.get("confidence"),
            "adx": trend.get("adx"),
            "momentum_percent": trend.get("momentum_percent"),
            "moving_averages": moving_averages,
            "reversal_warning": trend.get("reversal_warning"),
        },
        "structure": {
            "direction": structure.get("trend"),
            "levels": levels,
            "summary": structure.get("summary", {}),
            "latest_structure_breaks": (
                structure.get("structure_breaks", [])[-5:]
            ),
            "developing_structures": (
                structure.get("developing_structures", [])
            ),
        },
        "regime": {
            "classification": regime_name,
            "direction": regime_direction,
            "trend_strength": regime.get("trend_strength"),
            "volatility": regime.get("volatility", {}),
            "volume": regime.get("volume", {}),
        },
        "confluence": {
            "direction": confluence["direction"],
            "score": confluence["confluence_score"],
            "agreement_ratio": confluence["agreement_ratio"],
            "evidence": confluence.get("evidence", {}),
            "component_scores": confluence.get(
                "component_scores",
                {},
            ),
        },
        "candles_analyzed": validated["candles_analyzed"],
        "last_candle_close_time": (
            validated["last_close_time"].isoformat()
        ),
        "sources": validated["sources"],
    }


def build_market_report(
    market_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a standardized market report.

    The input must be the successful output of
    market_analysis.engine.analyze_market.

    This function does not recalculate indicators,
    create trading signals, or execute trades.
    """

    validated = _validate_market_analysis(
        market_analysis
    )

    summary = _build_market_summary(
        validated
    )

    limitations = market_analysis.get(
        "limitations",
        [],
    )

    if not isinstance(limitations, list):
        raise MarketReportError(
            "Market analysis limitations must be a list."
        )

    if any(
        not isinstance(item, str) or not item.strip()
        for item in limitations
    ):
        raise MarketReportError(
            "Market analysis contains invalid limitation descriptions."
        )

    payload = {
        "asset": "BTC",
        "summary": summary,
        "components": validated["analyses"],
        "confluence": validated["confluence"],
        "limitations": limitations,
        "interpretation_notice": (
            "Market analysis is informational and is not a trading signal."
        ),
    }

    metadata = {
        "analysis_timestamp": validated["timestamp"].isoformat(),
        "last_candle_close_time": (
            validated["last_close_time"].isoformat()
        ),
        "candles_analyzed": validated["candles_analyzed"],
        "sources": validated["sources"],
        "market_analysis_status": "SUCCESS",
    }

    return {
        "report_type": "market",
        "generated_at": validated["timestamp"].isoformat(),
        "source": "market_analysis.engine",
        "metadata": metadata,
        "payload": payload,
    }


def generate_market_report(
    market_analysis: dict[str, Any] | None = None,
    *,
    candle_data: Any = None,
    analysis_time: datetime | None = None,
    output_directory=DEFAULT_REPORT_DIRECTORY,
    component_parameters: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    bullish_threshold: float = 20.0,
    bearish_threshold: float = 20.0,
) -> dict[str, Any]:
    """
    Generate and save a Bitcoin market report.

    Supply either:
    - market_analysis: an existing successful analyze_market result.
    - candle_data: a validated candle DataFrame to analyze first.

    Existing analysis results are preferred when available.
    """

    if market_analysis is not None and candle_data is not None:
        raise MarketReportError(
            "Provide market_analysis or candle_data, not both."
        )

    if market_analysis is None:
        if candle_data is None:
            raise MarketReportError(
                "market_analysis or candle_data is required."
            )

        try:
            market_analysis = analyze_market(
                data=candle_data,
                analysis_time=analysis_time,
                component_parameters=component_parameters,
                weights=weights,
                bullish_threshold=bullish_threshold,
                bearish_threshold=bearish_threshold,
            )

        except Exception as exc:
            raise MarketReportError(
                f"Unable to run market analysis: {exc}"
            ) from exc

    elif any(
        value is not None
        for value in (
            analysis_time,
            component_parameters,
            weights,
        )
    ):
        raise MarketReportError(
            "Analysis parameters cannot be supplied with an existing result."
        )

    report_document = build_market_report(
        market_analysis
    )

    engine = ReportEngine(
        output_directory=output_directory
    )

    try:
        return engine.generate(
            report_type=report_document["report_type"],
            payload=report_document["payload"],
            generated_at=report_document["generated_at"],
            source=report_document["source"],
            metadata=report_document["metadata"],
        )

    except ReportEngineError as exc:
        raise MarketReportError(
            f"Unable to save market report: {exc}"
        ) from exc


if __name__ == "__main__":
    print("Bitcoin Market Report module loaded.")
