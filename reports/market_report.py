
"""Bitcoin market analysis report builder."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

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

    return timestamp.astimezone(timezone.utc)


def _finite_number(
    value: Any,
    field_name: str,
) -> float:
    """Validate and normalize a finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(float(value))
    ):
        raise MarketReportError(
            f"{field_name} must be a finite number."
        )

    return float(value)


def _positive_number(
    value: Any,
    field_name: str,
) -> float:
    """Validate a finite positive number."""

    number = _finite_number(value, field_name)

    if number <= 0:
        raise MarketReportError(
            f"{field_name} must be greater than zero."
        )

    return number


def _non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    """Validate a non-negative integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise MarketReportError(
            f"{field_name} must be a non-negative integer."
        )

    return int(value)


def _validate_sources(value: Any) -> list[str]:
    """Validate source identifiers."""

    if not isinstance(value, (list, tuple)) or not value:
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


def _normalize_json_value(value: Any, path: str = "payload") -> Any:
    """Convert supported scientific Python values into strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise MarketReportError(
                f"{path} contains a non-finite number."
            )
        return number

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise MarketReportError(
                f"{path} contains a non-UTC datetime."
            )
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MarketReportError(
                    f"{path} contains a non-string dictionary key."
                )
            normalized[key] = _normalize_json_value(
                item,
                f"{path}.{key}",
            )
        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if isinstance(value, np.ndarray):
        return _normalize_json_value(value.tolist(), path)

    raise MarketReportError(
        f"{path} contains an unsupported JSON value: "
        f"{type(value).__name__}."
    )


def _validate_component_metadata(
    component: str,
    result: Any,
    timestamp: datetime,
    candle_count: int,
    last_close_time: datetime,
) -> dict[str, Any]:
    """Validate shared metadata returned by a market component."""

    if not isinstance(result, dict):
        raise MarketReportError(
            f"Invalid result for component: {component}."
        )

    if result.get("status") != "SUCCESS":
        raise MarketReportError(
            f"Component '{component}' did not succeed."
        )

    component_timestamp = _parse_utc_timestamp(
        result.get("analysis_timestamp"),
        f"{component}.analysis_timestamp",
    )

    if component_timestamp != timestamp:
        raise MarketReportError(
            f"Component '{component}' has a mismatched timestamp."
        )

    component_count = _non_negative_integer(
        result.get("candles_analyzed"),
        f"{component}.candles_analyzed",
    )

    if component_count != candle_count:
        raise MarketReportError(
            f"Component '{component}' analyzed a different candle count."
        )

    component_close_time = _parse_utc_timestamp(
        result.get("last_candle_close_time"),
        f"{component}.last_candle_close_time",
    )

    if component_close_time != last_close_time:
        raise MarketReportError(
            f"Component '{component}' analyzed a different last candle."
        )

    _validate_sources(result.get("source"))

    return result


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
        field for field in required_fields
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

    if candle_count == 0:
        raise MarketReportError(
            "Market analysis must contain at least one candle."
        )

    sources = _validate_sources(result["source"])

    analyses = result["analyses"]

    if not isinstance(analyses, dict):
        raise MarketReportError(
            "Market component analyses must be a dictionary."
        )

    missing_components = [
        component for component in REQUIRED_COMPONENTS
        if component not in analyses
    ]

    if missing_components:
        raise MarketReportError(
            "Market analysis is missing components: "
            + ", ".join(missing_components)
        )

    for component in REQUIRED_COMPONENTS:
        _validate_component_metadata(
            component=component,
            result=analyses[component],
            timestamp=timestamp,
            candle_count=candle_count,
            last_close_time=last_close_time,
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

    confluence_timestamp = _parse_utc_timestamp(
        confluence.get("analysis_timestamp"),
        "confluence.analysis_timestamp",
    )

    if confluence_timestamp != timestamp:
        raise MarketReportError(
            "Confluence timestamp does not match market analysis."
        )

    direction = confluence.get("direction")

    if direction not in VALID_DIRECTIONS:
        raise MarketReportError(
            "Confluence contains an invalid market direction."
        )

    score = _finite_number(
        confluence.get("confluence_score"),
        "confluence_score",
    )

    if not -100 <= score <= 100:
        raise MarketReportError(
            "Confluence score must be between -100 and 100."
        )

    agreement = _finite_number(
        confluence.get("agreement_ratio"),
        "agreement_ratio",
    )

    if not 0 <= agreement <= 1:
        raise MarketReportError(
            "Confluence agreement ratio must be between 0 and 1."
        )

    confluence_count = _non_negative_integer(
        confluence.get("candles_analyzed"),
        "confluence.candles_analyzed",
    )

    if confluence_count != candle_count:
        raise MarketReportError(
            "Confluence analyzed a different candle count."
        )

    confluence_close_time = _parse_utc_timestamp(
        confluence.get("last_candle_close_time"),
        "confluence.last_candle_close_time",
    )

    if confluence_close_time != last_close_time:
        raise MarketReportError(
            "Confluence last candle does not match market analysis."
        )

    confluence_sources = _validate_sources(
        confluence.get("source")
    )

    if confluence_sources != sources:
        raise MarketReportError(
            "Confluence sources do not match market analysis sources."
        )

    confluence_price = confluence.get("market_price")

    if confluence_price is not None:
        confluence_price = _positive_number(
            confluence_price,
            "confluence.market_price",
        )
        if not math.isclose(
            confluence_price,
            market_price,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise MarketReportError(
                "Confluence market price does not match market analysis."
            )

    return {
        "timestamp": timestamp,
        "last_close_time": last_close_time,
        "market_price": market_price,
        "candles_analyzed": candle_count,
        "sources": sources,
        "analyses": analyses,
        "confluence": confluence,
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

    if trend_direction not in VALID_DIRECTIONS:
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

    moving_averages = trend.get("moving_averages", {})

    if not isinstance(moving_averages, dict):
        raise MarketReportError(
            "Trend moving averages must be a dictionary."
        )

    levels = structure.get("levels", {})

    if not isinstance(levels, dict):
        raise MarketReportError(
            "Price structure levels must be a dictionary."
        )

    structure_breaks = structure.get("structure_breaks", [])
    developing_structures = structure.get(
        "developing_structures",
        [],
    )

    if not isinstance(structure_breaks, list):
        raise MarketReportError(
            "Structure breaks must be a list."
        )

    if not isinstance(developing_structures, list):
        raise MarketReportError(
            "Developing structures must be a list."
        )

    summary = {
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
            "latest_structure_breaks": structure_breaks[-5:],
            "developing_structures": developing_structures,
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

    return _normalize_json_value(summary)


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

    validated = _validate_market_analysis(market_analysis)

    summary = _build_market_summary(validated)

    limitations = market_analysis.get("limitations", [])

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

    payload = _normalize_json_value(payload)

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

        except (TypeError, ValueError, RuntimeError) as exc:
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
    ) or bullish_threshold != 20.0 or bearish_threshold != 20.0:
        raise MarketReportError(
            "Analysis parameters cannot be supplied with an existing result."
        )

    report_document = build_market_report(market_analysis)

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
