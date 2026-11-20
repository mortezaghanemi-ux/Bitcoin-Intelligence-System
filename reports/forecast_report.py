
"""Bitcoin forecast report builder."""

from __future__ import annotations

import math

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from forecast.engine import (
    VALID_DIRECTIONS,
)
from reports.engine import (
    DEFAULT_REPORT_DIRECTORY,
    ReportEngine,
    ReportEngineError,
)


class ForecastReportError(ValueError):
    """Raised when a forecast report cannot be generated."""


VALID_DIRECTIONS = {
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
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
            raise ForecastReportError(
                f"Invalid timestamp: {field_name}."
            ) from exc

    elif isinstance(value, datetime):
        timestamp = value

    else:
        raise ForecastReportError(
            f"{field_name} must be a datetime or ISO timestamp."
        )

    if timestamp.tzinfo is None:
        raise ForecastReportError(
            f"{field_name} must be timezone-aware UTC."
        )

    try:
        offset = timestamp.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForecastReportError(
            f"Invalid timezone in {field_name}."
        ) from exc

    if offset is None or offset != timedelta(0):
        raise ForecastReportError(
            f"{field_name} must use UTC."
        )

    return timestamp.astimezone(timezone.utc)


def _finite_number(
    value: Any,
    field_name: str,
) -> float:
    """Validate a finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ForecastReportError(
            f"{field_name} must be a finite number."
        )

    return float(value)


def _positive_number(
    value: Any,
    field_name: str,
) -> float:
    """Validate a positive numeric value."""

    number = _finite_number(value, field_name)

    if number <= 0:
        raise ForecastReportError(
            f"{field_name} must be greater than zero."
        )

    return number


def _validate_probability_distribution(
    probabilities: Any,
) -> dict[str, float]:
    """Validate the three forecast probabilities."""

    if not isinstance(probabilities, dict):
        raise ForecastReportError(
            "Forecast probabilities must be a dictionary."
        )

    if set(probabilities) != VALID_DIRECTIONS:
        raise ForecastReportError(
            "Forecast must contain BULLISH, BEARISH and NEUTRAL probabilities."
        )

    normalized = {}

    for direction in VALID_DIRECTIONS:
        value = _finite_number(
            probabilities[direction],
            f"probabilities.{direction}",
        )

        if not 0 <= value <= 100:
            raise ForecastReportError(
                f"Probability for {direction} must be between 0 and 100."
            )

        normalized[direction] = value

    total = sum(normalized.values())

    if not math.isclose(
        total,
        100.0,
        rel_tol=0.0,
        abs_tol=0.0001,
    ):
        raise ForecastReportError(
            "Forecast probabilities must total 100 percent."
        )

    return normalized


def _validate_string_list(
    value: Any,
    field_name: str,
) -> list[str]:
    """Validate a non-empty list of descriptive strings."""

    if not isinstance(value, list) or not value:
        raise ForecastReportError(
            f"{field_name} must be a non-empty list."
        )

    if any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        raise ForecastReportError(
            f"{field_name} contains an invalid item."
        )

    return [item.strip() for item in value]


def _validate_forecast(
    forecast: Any,
) -> dict[str, Any]:
    """Validate the output of forecast.engine.generate_forecast."""

    if not isinstance(forecast, dict):
        raise ForecastReportError(
            "Forecast must be a dictionary."
        )

    if forecast.get("status") != "SUCCESS":
        raise ForecastReportError(
            "Only successful forecasts can be reported."
        )

    required_fields = (
        "forecast_id",
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
    )

    missing = [
        field
        for field in required_fields
        if field not in forecast
    ]

    if missing:
        raise ForecastReportError(
            "Forecast is missing required fields: "
            + ", ".join(missing)
        )

    forecast_id = forecast["forecast_id"]

    if (
        not isinstance(forecast_id, str)
        or not forecast_id.strip()
    ):
        raise ForecastReportError(
            "forecast_id must be a non-empty string."
        )

    if forecast["asset"] != "BTC":
        raise ForecastReportError(
            "This report builder supports BTC forecasts only."
        )

    forecast_timestamp = _parse_utc_timestamp(
        forecast["forecast_timestamp"],
        "forecast_timestamp",
    )

    analysis_timestamp = _parse_utc_timestamp(
        forecast["analysis_timestamp"],
        "analysis_timestamp",
    )

    last_candle_close_time = _parse_utc_timestamp(
        forecast["last_candle_close_time"],
        "last_candle_close_time",
    )

    if analysis_timestamp > forecast_timestamp:
        raise ForecastReportError(
            "Analysis timestamp cannot be after forecast timestamp."
        )

    if last_candle_close_time > analysis_timestamp:
        raise ForecastReportError(
            "Last candle cannot close after analysis timestamp."
        )

    market_price = _positive_number(
        forecast["market_price"],
        "market_price",
    )

    horizon = forecast["forecast_horizon"]

    if not isinstance(horizon, dict):
        raise ForecastReportError(
            "forecast_horizon must be a dictionary."
        )

    horizon_days = horizon.get("days")

    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or not 1 <= horizon_days <= 365
    ):
        raise ForecastReportError(
            "Forecast horizon must be an integer between 1 and 365 days."
        )

    horizon_start = _parse_utc_timestamp(
        horizon.get("start"),
        "forecast_horizon.start",
    )

    horizon_end = _parse_utc_timestamp(
        horizon.get("end"),
        "forecast_horizon.end",
    )

    if horizon_start != forecast_timestamp:
        raise ForecastReportError(
            "Forecast horizon start does not match forecast timestamp."
        )

    expected_horizon_end = (
        forecast_timestamp + timedelta(days=horizon_days)
    )

    if horizon_end != expected_horizon_end:
        raise ForecastReportError(
            "Forecast horizon end does not match its duration."
        )

    probabilities = _validate_probability_distribution(
        forecast["probabilities"]
    )

    confidence = _finite_number(
        forecast["confidence_score"],
        "confidence_score",
    )

    if not 0 <= confidence <= 100:
        raise ForecastReportError(
            "confidence_score must be between 0 and 100."
        )

    condition = forecast["market_condition"]

    if not isinstance(condition, dict):
        raise ForecastReportError(
            "market_condition must be a dictionary."
        )

    regime = condition.get("regime")
    regime_direction = condition.get("direction")
    trend_strength = condition.get("trend_strength")

    if not isinstance(regime, str) or not regime.strip():
        raise ForecastReportError(
            "Market regime must be a non-empty string."
        )

    if regime_direction not in VALID_DIRECTIONS:
        raise ForecastReportError(
            "Market condition contains an invalid direction."
        )

    if not isinstance(trend_strength, str) or not trend_strength.strip():
        raise ForecastReportError(
            "Market trend strength must be a non-empty string."
        )

    factors = forecast["influencing_factors"]

    if not isinstance(factors, list):
        raise ForecastReportError(
            "influencing_factors must be a list."
        )

    normalized_factors = []

    for index, factor in enumerate(factors):
        if not isinstance(factor, dict):
            raise ForecastReportError(
                f"Influencing factor {index} must be a dictionary."
            )

        component = factor.get("component")
        direction = factor.get("direction")
        score = _finite_number(
            factor.get("score"),
            f"influencing_factors[{index}].score",
        )

        if not isinstance(component, str) or not component.strip():
            raise ForecastReportError(
                f"Influencing factor {index} has an invalid component."
            )

        if direction not in VALID_DIRECTIONS:
            raise ForecastReportError(
                f"Influencing factor {index} has an invalid direction."
            )

        normalized_factors.append({
            "component": component.strip(),
            "direction": direction,
            "score": score,
        })

    risk_factors = _validate_string_list(
        forecast["risk_factors"],
        "risk_factors",
    )

    limitations = _validate_string_list(
        forecast["limitations"],
        "limitations",
    )

    model = forecast["model"]

    if not isinstance(model, dict):
        raise ForecastReportError(
            "Forecast model metadata must be a dictionary."
        )

    for field in ("name", "version", "method"):
        value = model.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ForecastReportError(
                f"model.{field} must be a non-empty string."
            )

    if not isinstance(model.get("statistically_calibrated"), bool):
        raise ForecastReportError(
            "model.statistically_calibrated must be a boolean."
        )

    input_metadata = forecast["input_metadata"]

    if not isinstance(input_metadata, dict):
        raise ForecastReportError(
            "input_metadata must be a dictionary."
        )

    sources = _validate_string_list(
        input_metadata.get("sources"),
        "input_metadata.sources",
    )

    candles_analyzed = input_metadata.get("candles_analyzed")

    if (
        isinstance(candles_analyzed, bool)
        or not isinstance(candles_analyzed, int)
        or candles_analyzed <= 0
    ):
        raise ForecastReportError(
            "input_metadata.candles_analyzed must be a positive integer."
        )

    confluence_score = _finite_number(
        input_metadata.get("confluence_score"),
        "input_metadata.confluence_score",
    )

    if not -100 <= confluence_score <= 100:
        raise ForecastReportError(
            "Confluence score must be between -100 and 100."
        )

    agreement_ratio = _finite_number(
        input_metadata.get("agreement_ratio"),
        "input_metadata.agreement_ratio",
    )

    if not 0 <= agreement_ratio <= 1:
        raise ForecastReportError(
            "Agreement ratio must be between 0 and 1."
        )

    return {
        "forecast_id": forecast_id.strip(),
        "forecast_timestamp": forecast_timestamp,
        "analysis_timestamp": analysis_timestamp,
        "last_candle_close_time": last_candle_close_time,
        "horizon_days": horizon_days,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "market_price": market_price,
        "probabilities": probabilities,
        "confidence_score": confidence,
        "market_condition": {
            "regime": regime.strip(),
            "direction": regime_direction,
            "trend_strength": trend_strength.strip(),
        },
        "influencing_factors": normalized_factors,
        "risk_factors": risk_factors,
        "model": dict(model),
        "input_metadata": {
            "sources": sorted(set(sources)),
            "candles_analyzed": candles_analyzed,
            "confluence_score": confluence_score,
            "agreement_ratio": agreement_ratio,
        },
        "limitations": limitations,
    }


def build_forecast_report(
    forecast: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a standardized report from a successful BTC forecast.

    The forecast must be generated by forecast.engine.generate_forecast.

    This function does not recalculate forecasts, evaluate outcomes,
    generate trading signals, or execute trades.
    """

    validated = _validate_forecast(forecast)

    normalized_forecast = {
        "forecast_id": validated["forecast_id"],
        "status": "SUCCESS",
        "asset": "BTC",
        "forecast_timestamp": (
            validated["forecast_timestamp"].isoformat()
        ),
        "analysis_timestamp": (
            validated["analysis_timestamp"].isoformat()
        ),
        "last_candle_close_time": (
            validated["last_candle_close_time"].isoformat()
        ),
        "forecast_horizon": {
            "days": validated["horizon_days"],
            "start": validated["horizon_start"].isoformat(),
            "end": validated["horizon_end"].isoformat(),
        },
        "market_price": validated["market_price"],
        "probabilities": validated["probabilities"],
        "confidence_score": validated["confidence_score"],
        "market_condition": validated["market_condition"],
        "influencing_factors": validated["influencing_factors"],
        "risk_factors": validated["risk_factors"],
        "model": validated["model"],
        "input_metadata": validated["input_metadata"],
        "limitations": validated["limitations"],
    }

    payload = {
        "forecast": normalized_forecast,
        "interpretation_notice": (
            "Forecast probabilities are heuristic estimates and are "
            "not statistically calibrated unless explicitly established "
            "by independent evaluation."
        ),
    }

    metadata = {
        "forecast_id": validated["forecast_id"],
        "asset": "BTC",
        "forecast_timestamp": (
            validated["forecast_timestamp"].isoformat()
        ),
        "analysis_timestamp": (
            validated["analysis_timestamp"].isoformat()
        ),
        "forecast_horizon_days": validated["horizon_days"],
        "model_version": validated["model"]["version"],
        "statistically_calibrated": (
            validated["model"]["statistically_calibrated"]
        ),
        "sources": validated["input_metadata"]["sources"],
        "candles_analyzed": (
            validated["input_metadata"]["candles_analyzed"]
        ),
    }

    return {
        "report_type": "forecast",
        "generated_at": validated["forecast_timestamp"].isoformat(),
        "source": "forecast.engine",
        "metadata": metadata,
        "payload": payload,
    }


def generate_forecast_report(
    forecast: dict[str, Any],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> dict[str, Any]:
    """
    Generate and save a standardized forecast report.

    The supplied forecast is validated and preserved.
    Existing reports are never overwritten.
    """

    report_document = build_forecast_report(forecast)

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
        raise ForecastReportError(
            f"Unable to save forecast report: {exc}"
        ) from exc


if __name__ == "__main__":
    print("Bitcoin Forecast Report module loaded.")
