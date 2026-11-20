
"""Validated trading signal report generation.

Converts the output of the trading pipeline or trading runtime
into a standardized, append-only trading report.

This module does not generate signals, connect to exchanges,
or execute trades.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from reports.engine import (
    ReportEngine,
    ReportEngineError,
)


class TradingReportError(RuntimeError):
    """Raised when a trading report cannot be validated or generated."""


ALLOWED_SIGNALS = {"LONG", "SHORT", "NO_TRADE"}
ALLOWED_STATUSES = {"REQUIRES_MANUAL_REVIEW", "REJECTED"}
ALLOWED_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL"}

REQUIRED_EXECUTION_SAFETY = {
    "manual_review_only": True,
    "auto_trading": False,
    "exchange_connection": False,
    "order_execution": False,
}


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    """Parse a timezone-aware UTC timestamp."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise TradingReportError(
                f"{field} is not a valid ISO timestamp."
            ) from exc

    if not isinstance(value, datetime):
        raise TradingReportError(
            f"{field} must be a datetime or ISO timestamp."
        )

    if value.tzinfo is None:
        raise TradingReportError(
            f"{field} must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise TradingReportError(
            f"{field} has an invalid timezone."
        ) from exc

    if offset is None or offset != timedelta(0):
        raise TradingReportError(
            f"{field} must use UTC."
        )

    return value.astimezone(timezone.utc)


def _finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a finite numeric value."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TradingReportError(
            f"{field} must be a number."
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TradingReportError(
            f"{field} must be a finite number."
        ) from exc

    if not math.isfinite(number):
        raise TradingReportError(
            f"{field} must be a finite number."
        )

    if minimum is not None and number < minimum:
        raise TradingReportError(
            f"{field} must be at least {minimum}."
        )

    if maximum is not None and number > maximum:
        raise TradingReportError(
            f"{field} must not exceed {maximum}."
        )

    return number


def _required_string(value: Any, field: str) -> str:
    """Validate a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise TradingReportError(
            f"{field} must be a non-empty string."
        )

    return value.strip()


def _string_list(value: Any, field: str) -> list[str]:
    """Validate a list of non-empty strings."""

    if not isinstance(value, list):
        raise TradingReportError(
            f"{field} must be a list."
        )

    normalized = []

    for index, item in enumerate(value):
        normalized.append(
            _required_string(
                item,
                f"{field}[{index}]",
            )
        )

    return normalized


def _validate_execution(signal: dict[str, Any]) -> dict[str, bool]:
    """Verify that all trading execution safeguards are enabled."""

    execution = signal.get("execution")

    if not isinstance(execution, dict):
        raise TradingReportError(
            "Signal execution metadata is missing."
        )

    for field, expected in REQUIRED_EXECUTION_SAFETY.items():
        if execution.get(field) is not expected:
            raise TradingReportError(
                f"Unsafe execution setting: {field} must be {expected}."
            )

    return dict(REQUIRED_EXECUTION_SAFETY)


def _validate_data_quality(
    signal: dict[str, Any],
) -> dict[str, Any]:
    """Verify candle validation and look-ahead protection metadata."""

    quality = signal.get("data_quality")

    if not isinstance(quality, dict):
        raise TradingReportError(
            "Signal data_quality metadata is missing."
        )

    if quality.get("status") != "VALIDATED":
        raise TradingReportError(
            "Signal data quality status must be VALIDATED."
        )

    if quality.get("closed_candles_only") is not True:
        raise TradingReportError(
            "Trading report requires closed candles only."
        )

    if quality.get("future_data_used") is not False:
        raise TradingReportError(
            "Trading report cannot contain future-data usage."
        )

    timeframes = quality.get("validated_timeframes")

    if not isinstance(timeframes, list):
        raise TradingReportError(
            "Validated timeframes must be a list."
        )

    normalized_timeframes = [
        _required_string(
            timeframe,
            f"validated_timeframes[{index}]",
        )
        for index, timeframe in enumerate(timeframes)
    ]

    if set(normalized_timeframes) != {"5m", "15m", "1h", "4h"}:
        raise TradingReportError(
            "Trading report requires all four supported timeframes."
        )

    sources = _string_list(
        quality.get("sources"),
        "data_quality.sources",
    )

    if not sources:
        raise TradingReportError(
            "At least one validated data source is required."
        )

    close_times = quality.get("last_candle_close_times")

    if not isinstance(close_times, dict):
        raise TradingReportError(
            "Last candle close times must be a dictionary."
        )

    if set(close_times) != {"5m", "15m", "1h", "4h"}:
        raise TradingReportError(
            "Last candle close times must cover all four timeframes."
        )

    normalized_close_times = {
        timeframe: _parse_utc_timestamp(
            close_times[timeframe],
            f"last_candle_close_times.{timeframe}",
        ).isoformat()
        for timeframe in ("5m", "15m", "1h", "4h")
    }

    return {
        "status": "VALIDATED",
        "validated_timeframes": normalized_timeframes,
        "closed_candles_only": True,
        "future_data_used": False,
        "sources": sorted(set(sources)),
        "last_candle_close_times": normalized_close_times,
    }


def _validate_alignment(signal: dict[str, Any]) -> dict[str, Any]:
    """Validate multi-timeframe alignment metadata."""

    alignment = signal.get("alignment")

    if not isinstance(alignment, dict):
        raise TradingReportError(
            "Signal alignment metadata is missing."
        )

    direction = alignment.get("direction")

    if direction not in ALLOWED_DIRECTIONS:
        raise TradingReportError(
            "Alignment direction is invalid."
        )

    ratio = _finite_number(
        alignment.get("ratio"),
        "alignment.ratio",
        minimum=0,
        maximum=1,
    )

    conflict_count = alignment.get("conflict_count")

    if (
        isinstance(conflict_count, bool)
        or not isinstance(conflict_count, int)
        or not 0 <= conflict_count <= 4
    ):
        raise TradingReportError(
            "Alignment conflict_count must be an integer from 0 to 4."
        )

    timeframe_directions = alignment.get(
        "timeframe_directions"
    )

    if not isinstance(timeframe_directions, dict):
        raise TradingReportError(
            "Timeframe directions must be a dictionary."
        )

    return {
        "direction": direction,
        "ratio": ratio,
        "status": alignment.get("status"),
        "conflict_count": conflict_count,
        "timeframe_directions": dict(timeframe_directions),
    }


def _validate_trade_levels(
    signal: dict[str, Any],
    signal_type: str,
) -> dict[str, Any]:
    """Validate entry, stop-loss and take-profit levels."""

    entry_range = signal.get("entry_range")
    stop_loss = signal.get("stop_loss")
    take_profit = signal.get("take_profit")
    risk_reward = signal.get("risk_reward")
    risk_per_unit = signal.get("risk_per_unit")

    if signal_type == "NO_TRADE":
        if any(
            value is not None
            for value in (
                entry_range,
                stop_loss,
                risk_reward,
                risk_per_unit,
            )
        ):
            raise TradingReportError(
                "NO_TRADE signals cannot contain active trade levels."
            )

        if take_profit not in (None, []):
            raise TradingReportError(
                "NO_TRADE signals cannot contain take-profit targets."
            )

        return {
            "entry_range": None,
            "stop_loss": None,
            "take_profit": [],
            "risk_reward": None,
            "risk_per_unit": None,
        }

    if not isinstance(entry_range, dict):
        raise TradingReportError(
            "Actionable signals require an entry range."
        )

    lower = _finite_number(
        entry_range.get("lower"),
        "entry_range.lower",
        minimum=0,
    )

    upper = _finite_number(
        entry_range.get("upper"),
        "entry_range.upper",
        minimum=0,
    )

    if lower > upper:
        raise TradingReportError(
            "Entry range lower bound cannot exceed upper bound."
        )

    normalized_stop = _finite_number(
        stop_loss,
        "stop_loss",
        minimum=0,
    )

    normalized_rr = _finite_number(
        risk_reward,
        "risk_reward",
        minimum=0,
    )

    normalized_risk = _finite_number(
        risk_per_unit,
        "risk_per_unit",
        minimum=0,
    )

    if normalized_rr < 1.5:
        raise TradingReportError(
            "Actionable signal risk/reward is below the required minimum."
        )

    if normalized_risk <= 0:
        raise TradingReportError(
            "Actionable signal risk_per_unit must be greater than zero."
        )

    if signal_type == "LONG" and normalized_stop >= lower:
        raise TradingReportError(
            "LONG stop loss must be below the entry range."
        )

    if signal_type == "SHORT" and normalized_stop <= upper:
        raise TradingReportError(
            "SHORT stop loss must be above the entry range."
        )

    if not isinstance(take_profit, list) or not take_profit:
        raise TradingReportError(
            "Actionable signals require take-profit targets."
        )

    normalized_targets = []

    previous_level = 0
    previous_rr = 0.0

    for index, target in enumerate(take_profit):
        if not isinstance(target, dict):
            raise TradingReportError(
                f"take_profit[{index}] must be a dictionary."
            )

        level = target.get("level")

        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level <= previous_level
        ):
            raise TradingReportError(
                "Take-profit levels must have increasing positive integers."
            )

        price = _finite_number(
            target.get("price"),
            f"take_profit[{index}].price",
            minimum=0,
        )

        target_rr = _finite_number(
            target.get("risk_reward"),
            f"take_profit[{index}].risk_reward",
            minimum=0,
        )

        if target_rr <= previous_rr:
            raise TradingReportError(
                "Take-profit risk/reward values must increase."
            )

        if signal_type == "LONG" and price <= upper:
            raise TradingReportError(
                "LONG take-profit prices must be above the entry range."
            )

        if signal_type == "SHORT" and price >= lower:
            raise TradingReportError(
                "SHORT take-profit prices must be below the entry range."
            )

        normalized_targets.append({
            "level": level,
            "price": price,
            "risk_reward": target_rr,
        })

        previous_level = level
        previous_rr = target_rr

    if normalized_targets[-1]["risk_reward"] < 1.5:
        raise TradingReportError(
            "The final take-profit target does not meet minimum risk/reward."
        )

    return {
        "entry_range": {
            "lower": lower,
            "upper": upper,
        },
        "stop_loss": normalized_stop,
        "take_profit": normalized_targets,
        "risk_reward": normalized_rr,
        "risk_per_unit": normalized_risk,
    }


def _validate_risk_calculation(
    signal: dict[str, Any],
    signal_type: str,
) -> dict[str, Any] | None:
    """Validate position sizing and liquidation estimate."""

    risk = signal.get("risk_calculation")

    if signal_type == "NO_TRADE":
        if risk is not None:
            raise TradingReportError(
                "NO_TRADE signals cannot contain an active risk calculation."
            )
        return None

    if not isinstance(risk, dict):
        raise TradingReportError(
            "Actionable signals require a risk calculation."
        )

    position = risk.get("position")
    liquidation = risk.get("liquidation")

    if not isinstance(position, dict):
        raise TradingReportError(
            "Position calculation is missing."
        )

    if position.get("status") != "CALCULATED":
        raise TradingReportError(
            "Position sizing must have CALCULATED status."
        )

    position_fields = (
        "account_equity",
        "risk_fraction",
        "risk_amount",
        "quantity_btc",
        "notional_value",
        "required_margin",
        "leverage",
    )

    normalized_position = {
        field: _finite_number(
            position.get(field),
            f"risk_calculation.position.{field}",
            minimum=0,
        )
        for field in position_fields
    }

    if any(
        normalized_position[field] <= 0
        for field in position_fields
    ):
        raise TradingReportError(
            "Position calculation values must be greater than zero."
        )

    if normalized_position["leverage"] > 2:
        raise TradingReportError(
            "Trading report leverage cannot exceed 2."
        )

    if (
        normalized_position["required_margin"]
        > normalized_position["account_equity"]
    ):
        raise TradingReportError(
            "Required margin exceeds account equity."
        )

    expected_risk = (
        normalized_position["account_equity"]
        * normalized_position["risk_fraction"]
    )

    if not math.isclose(
        normalized_position["risk_amount"],
        expected_risk,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise TradingReportError(
            "Position risk amount is inconsistent with account equity."
        )

    if not isinstance(liquidation, dict):
        raise TradingReportError(
            "Liquidation estimate is missing."
        )

    if liquidation.get("status") != "ESTIMATE_ONLY":
        raise TradingReportError(
            "Liquidation must be explicitly marked as an estimate."
        )

    liquidation_price = _finite_number(
        liquidation.get("estimated_price"),
        "liquidation.estimated_price",
        minimum=0,
    )

    liquidation_leverage = _finite_number(
        liquidation.get("leverage"),
        "liquidation.leverage",
        minimum=1,
        maximum=2,
    )

    if not math.isclose(
        liquidation_leverage,
        normalized_position["leverage"],
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise TradingReportError(
            "Liquidation leverage does not match position leverage."
        )

    if signal_type == "LONG" and liquidation_price >= signal["current_price"]:
        raise TradingReportError(
            "LONG liquidation estimate must be below current price."
        )

    if signal_type == "SHORT" and liquidation_price <= signal["current_price"]:
        raise TradingReportError(
            "SHORT liquidation estimate must be above current price."
        )

    warning = _required_string(
        liquidation.get("warning"),
        "liquidation.warning",
    )

    return {
        "position": normalized_position,
        "liquidation": {
            "status": "ESTIMATE_ONLY",
            "estimated_price": liquidation_price,
            "leverage": liquidation_leverage,
            "maintenance_margin_rate": _finite_number(
                liquidation.get("maintenance_margin_rate"),
                "liquidation.maintenance_margin_rate",
                minimum=0,
            ),
            "warning": warning,
        },
    }


def _validate_signal(signal: Any) -> dict[str, Any]:
    """Validate and normalize the trading signal."""

    if not isinstance(signal, dict):
        raise TradingReportError(
            "Trading signal must be a dictionary."
        )

    signal_id = _required_string(
        signal.get("signal_id"),
        "signal_id",
    )

    signal_type = signal.get("signal")

    if signal_type not in ALLOWED_SIGNALS:
        raise TradingReportError(
            "Unsupported trading signal type."
        )

    status = signal.get("status")

    if status not in ALLOWED_STATUSES:
        raise TradingReportError(
            "Unsupported trading signal status."
        )

    if signal_type == "NO_TRADE" and status != "REJECTED":
        raise TradingReportError(
            "NO_TRADE signals must have REJECTED status."
        )

    if signal_type in {"LONG", "SHORT"} and status != "REQUIRES_MANUAL_REVIEW":
        raise TradingReportError(
            "LONG and SHORT signals must require manual review."
        )

    direction = signal.get("direction")

    if direction not in ALLOWED_DIRECTIONS:
        raise TradingReportError(
            "Trading signal direction is invalid."
        )

    if signal_type == "LONG" and direction != "BULLISH":
        raise TradingReportError(
            "LONG signal direction must be BULLISH."
        )

    if signal_type == "SHORT" and direction != "BEARISH":
        raise TradingReportError(
            "SHORT signal direction must be BEARISH."
        )

    if signal_type == "NO_TRADE" and direction not in ALLOWED_DIRECTIONS:
        raise TradingReportError(
            "NO_TRADE direction is invalid."
        )

    timeframe = _required_string(
        signal.get("timeframe"),
        "timeframe",
    )

    if timeframe not in {"5m", "15m", "1h", "4h"}:
        raise TradingReportError(
            "Unsupported signal timeframe."
        )

    analysis_timestamp = _parse_utc_timestamp(
        signal.get("analysis_timestamp"),
        "analysis_timestamp",
    )

    signal_timestamp = _parse_utc_timestamp(
        signal.get("signal_timestamp"),
        "signal_timestamp",
    )

    if signal_timestamp > analysis_timestamp:
        raise TradingReportError(
            "Signal timestamp cannot be later than analysis timestamp."
        )

    current_price = _finite_number(
        signal.get("current_price"),
        "current_price",
        minimum=0,
    )

    if current_price <= 0:
        raise TradingReportError(
            "Current price must be greater than zero."
        )

    confidence = _finite_number(
        signal.get("confidence"),
        "confidence",
        minimum=0,
        maximum=100,
    )

    confidence_threshold = _finite_number(
        signal.get("confidence_threshold"),
        "confidence_threshold",
        minimum=0,
        maximum=100,
    )

    if signal_type in {"LONG", "SHORT"} and confidence < confidence_threshold:
        raise TradingReportError(
            "Actionable signal confidence is below its threshold."
        )

    parameters = signal.get("parameters")

    if not isinstance(parameters, dict):
        raise TradingReportError(
            "Signal parameters are missing."
        )

    applied_leverage = _finite_number(
        parameters.get("applied_leverage"),
        "parameters.applied_leverage",
        minimum=1,
        maximum=2,
    )

    if _required_string(
        parameters.get("execution_timeframe"),
        "parameters.execution_timeframe",
    ) != timeframe:
        raise TradingReportError(
            "Signal timeframe does not match its execution timeframe."
        )

    reasons = _string_list(
        signal.get("reasons"),
        "reasons",
    )

    rejection_reasons = _string_list(
        signal.get("rejection_reasons"),
        "rejection_reasons",
    )

    limitations = _string_list(
        signal.get("limitations"),
        "limitations",
    )

    if not limitations:
        raise TradingReportError(
            "Trading signal limitations cannot be empty."
        )

    if signal_type == "NO_TRADE" and not rejection_reasons:
        raise TradingReportError(
            "Rejected signals must include rejection reasons."
        )

    if signal_type in {"LONG", "SHORT"} and rejection_reasons:
        raise TradingReportError(
            "Actionable signals cannot contain rejection reasons."
        )

    if not isinstance(signal.get("supporting_analysis"), dict):
        raise TradingReportError(
            "Supporting analysis metadata is missing."
        )

    execution = _validate_execution(signal)
    data_quality = _validate_data_quality(signal)
    alignment = _validate_alignment(signal)

    trade_levels = _validate_trade_levels(
        signal,
        signal_type,
    )

    risk_calculation = _validate_risk_calculation(
        signal,
        signal_type,
    )

    if signal_type in {"LONG", "SHORT"}:
        if risk_calculation["position"]["leverage"] != applied_leverage:
            raise TradingReportError(
                "Applied leverage does not match the risk calculation."
            )

    return {
        "signal_id": signal_id,
        "analysis_timestamp": analysis_timestamp.isoformat(),
        "signal_timestamp": signal_timestamp.isoformat(),
        "status": status,
        "signal": signal_type,
        "direction": direction,
        "timeframe": timeframe,
        "current_price": current_price,
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "alignment": alignment,
        "supporting_analysis": dict(signal["supporting_analysis"]),
        **trade_levels,
        "risk_calculation": risk_calculation,
        "reasons": reasons,
        "rejection_reasons": rejection_reasons,
        "data_quality": data_quality,
        "parameters": dict(parameters),
        "execution": execution,
        "limitations": limitations,
    }


def build_trading_report(
    trading_result: dict[str, Any],
    *,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a validated trading report without saving it."""

    if not isinstance(trading_result, dict):
        raise TradingReportError(
            "Trading result must be a dictionary."
        )

    pipeline_status = trading_result.get("pipeline_status")
    runtime_status = trading_result.get("runtime_status")

    if pipeline_status != "SUCCESS" and runtime_status != "SUCCESS":
        raise TradingReportError(
            "Trading pipeline or runtime did not report SUCCESS."
        )

    signal = _validate_signal(
        trading_result.get("signal")
    )

    execution = trading_result.get("execution")

    if not isinstance(execution, dict):
        raise TradingReportError(
            "Trading result execution metadata is missing."
        )

    for field, expected in REQUIRED_EXECUTION_SAFETY.items():
        if execution.get(field) is not expected:
            raise TradingReportError(
                f"Unsafe trading result execution setting: {field}."
            )

    persistence = trading_result.get("persistence", {})

    if not isinstance(persistence, dict):
        raise TradingReportError(
            "Trading persistence metadata must be a dictionary."
        )

    if pipeline_status == "SUCCESS":
        if persistence.get("history_saved") is not True:
            raise TradingReportError(
                "Trading signal history was not confirmed as saved."
            )

    payload = {
        "signal": signal,
        "persistence": dict(persistence),
        "execution": dict(REQUIRED_EXECUTION_SAFETY),
    }

    if runtime_status == "SUCCESS":
        pair = trading_result.get("pair")

        if pair is not None:
            payload["pair"] = _required_string(
                pair,
                "pair",
            )

        collection = trading_result.get("collection")

        if collection is not None:
            if not isinstance(collection, dict):
                raise TradingReportError(
                    "Runtime collection metadata must be a dictionary."
                )
            payload["collection"] = dict(collection)

        candle_counts = trading_result.get("candle_counts")

        if candle_counts is not None:
            if not isinstance(candle_counts, dict):
                raise TradingReportError(
                    "Runtime candle_counts must be a dictionary."
                )

            normalized_counts = {}

            for timeframe in ("5m", "15m", "1h", "4h"):
                count = candle_counts.get(timeframe)

                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise TradingReportError(
                        f"Invalid runtime candle count for {timeframe}."
                    )

                normalized_counts[timeframe] = count

            payload["candle_counts"] = normalized_counts

    report_time = (
        _parse_utc_timestamp(
            generated_at,
            "generated_at",
        )
        if generated_at is not None
        else _parse_utc_timestamp(
            signal["analysis_timestamp"],
            "analysis_timestamp",
        )
    )

    return {
        "report_type": "trading",
        "generated_at": report_time.isoformat(),
        "source": "Bitcoin-Intelligence-System",
        "metadata": {
            "signal_id": signal["signal_id"],
            "signal_type": signal["signal"],
            "signal_status": signal["status"],
            "timeframe": signal["timeframe"],
            "manual_review_only": True,
        },
        "payload": payload,
    }


def create_trading_report(
    trading_result: dict[str, Any],
    *,
    report_engine: ReportEngine | None = None,
    output_directory: str | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Validate, generate and persist a trading report."""

    report_data = build_trading_report(
        trading_result,
        generated_at=generated_at,
    )

    if report_engine is not None and output_directory is not None:
        raise TradingReportError(
            "Provide report_engine or output_directory, not both."
        )

    if report_engine is None:
        report_engine = (
            ReportEngine(output_directory)
            if output_directory is not None
            else ReportEngine()
        )

    try:
        return report_engine.generate(
            report_type=report_data["report_type"],
            payload=report_data["payload"],
            generated_at=report_data["generated_at"],
            source=report_data["source"],
            metadata=report_data["metadata"],
        )

    except ReportEngineError as exc:
        raise TradingReportError(
            f"Unable to generate trading report: {exc}"
        ) from exc


if __name__ == "__main__":
    print("Bitcoin trading report module loaded.")
    print("Manual review only. No exchange connection or order execution.")
