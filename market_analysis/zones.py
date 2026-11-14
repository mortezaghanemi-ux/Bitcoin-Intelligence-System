
"""Bitcoin supply and demand zone analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from validation.candles import validate_candles


REQUIRED_COLUMNS = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)

SUPPLY = "SUPPLY"
DEMAND = "DEMAND"

ACTIVE = "ACTIVE"
MITIGATED = "MITIGATED"
BROKEN = "BROKEN"


class ZoneAnalysisError(RuntimeError):
    """Raised when supply and demand analysis fails."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Normalize analysis timestamp to UTC."""

    if analysis_time is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        timestamp = pd.Timestamp(analysis_time)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ZoneAnalysisError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise ZoneAnalysisError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise ZoneAnalysisError(
            "analysis_time must be timezone-aware."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise ZoneAnalysisError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_parameters(
    atr_period: int,
    impulse_window: int,
    minimum_candles: int,
    atr_multiplier: float,
) -> None:
    """Validate analysis parameters."""

    integer_parameters = (
        ("atr_period", atr_period),
        ("impulse_window", impulse_window),
        ("minimum_candles", minimum_candles),
    )

    for name, value in integer_parameters:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ZoneAnalysisError(
                f"{name} must be a positive integer."
            )

    if minimum_candles < atr_period + 2:
        raise ZoneAnalysisError(
            "minimum_candles must be at least "
            "atr_period + 2."
        )

    if not isinstance(
        atr_multiplier,
        (int, float),
    ) or isinstance(atr_multiplier, bool):
        raise ZoneAnalysisError(
            "atr_multiplier must be numeric."
        )

    if (
        not np.isfinite(atr_multiplier)
        or atr_multiplier <= 0
    ):
        raise ZoneAnalysisError(
            "atr_multiplier must be positive and finite."
        )


def _validate_candles(
    data: pd.DataFrame,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate candle data using the central validator."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ZoneAnalysisError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ZoneAnalysisError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    try:
        accepted, rejected, report = validate_candles(
            data=data,
            validation_time=analysis_time.to_pydatetime(),
            require_closed=True,
        )

    except (TypeError, ValueError) as exc:
        raise ZoneAnalysisError(
            f"Candle validation failed: {exc}"
        ) from exc

    if not rejected.empty:
        raise ZoneAnalysisError(
            "Invalid candles detected. "
            f"Rejected: {len(rejected)}. "
            f"Validation errors: {report['errors']}."
        )

    if accepted.empty:
        raise ZoneAnalysisError(
            "No valid candles available."
        )

    result = accepted.copy(deep=True)

    if not result["open_time"].is_monotonic_increasing:
        raise ZoneAnalysisError(
            "Candles must be sorted by open_time."
        )

    if not result["close_time"].is_monotonic_increasing:
        raise ZoneAnalysisError(
            "Candles must be sorted by close_time."
        )

    for column in ("open_time", "close_time"):
        if not isinstance(
            result[column].dtype,
            pd.DatetimeTZDtype,
        ):
            raise ZoneAnalysisError(
                f"{column} must be timezone-aware."
            )

        if str(result[column].dt.tz) != "UTC":
            raise ZoneAnalysisError(
                f"{column} must use UTC."
            )

    if result["open_time"].duplicated().any():
        raise ZoneAnalysisError(
            "Duplicate candle open_time values detected."
        )

    if result["close_time"].duplicated().any():
        raise ZoneAnalysisError(
            "Duplicate candle close_time values detected."
        )

    valid_sources = result["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise ZoneAnalysisError(
            "Every candle must have a valid source."
        )

    return result.reset_index(drop=True)


def _calculate_atr(
    data: pd.DataFrame,
    period: int,
) -> pd.Series:
    """Calculate Average True Range."""

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()

    return atr


def _find_impulse(
    data: pd.DataFrame,
    atr: pd.Series,
    base_index: int,
    zone_type: str,
    impulse_window: int,
    atr_multiplier: float,
) -> dict[str, Any] | None:
    """
    Find the first qualifying displacement after a base candle.

    A supply zone requires a bullish base candle followed
    by a bearish displacement.

    A demand zone requires a bearish base candle followed
    by a bullish displacement.
    """

    base = data.iloc[base_index]

    base_open = float(base["open"])
    base_close = float(base["close"])

    if zone_type == SUPPLY:
        if base_close <= base_open:
            return None

    elif zone_type == DEMAND:
        if base_close >= base_open:
            return None

    else:
        return None

    base_atr = atr.iloc[base_index]

    if pd.isna(base_atr) or base_atr <= 0:
        return None

    threshold = float(base_atr) * atr_multiplier

    last_index = min(
        len(data) - 1,
        base_index + impulse_window,
    )

    for index in range(
        base_index + 1,
        last_index + 1,
    ):
        candle = data.iloc[index]
        close_price = float(candle["close"])

        if zone_type == SUPPLY:
            displacement = (
                float(base["low"]) - close_price
            )

            confirmed = (
                close_price < float(base["low"])
                and displacement >= threshold
            )

        else:
            displacement = (
                close_price - float(base["high"])
            )

            confirmed = (
                close_price > float(base["high"])
                and displacement >= threshold
            )

        if confirmed:
            return {
                "index": index,
                "time": candle["close_time"].isoformat(),
                "close": close_price,
                "displacement": float(displacement),
                "atr": float(base_atr),
                "strength": float(
                    displacement / base_atr
                ),
            }

    return None


def _evaluate_zone(
    data: pd.DataFrame,
    zone: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate zone status after its confirmation."""

    confirmation_index = zone["confirmation_index"]

    upper = zone["upper"]
    lower = zone["lower"]

    status = ACTIVE
    touches = 0
    first_touch_time = None
    last_touch_time = None

    for index in range(
        confirmation_index + 1,
        len(data),
    ):
        candle = data.iloc[index]

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        close_time = candle["close_time"].isoformat()

        if zone["type"] == SUPPLY:
            if close > upper:
                status = BROKEN
                break

        else:
            if close < lower:
                status = BROKEN
                break

        intersects = (
            high >= lower
            and low <= upper
        )

        if intersects:
            touches += 1

            if first_touch_time is None:
                first_touch_time = close_time

            last_touch_time = close_time

            status = MITIGATED

    result = dict(zone)

    result.update(
        {
            "status": status,
            "touch_count": touches,
            "first_touch_time": first_touch_time,
            "last_touch_time": last_touch_time,
        }
    )

    return result


def _detect_zones(
    data: pd.DataFrame,
    atr: pd.Series,
    impulse_window: int,
    atr_multiplier: float,
) -> list[dict[str, Any]]:
    """Detect confirmed supply and demand zones."""

    zones: list[dict[str, Any]] = []

    for base_index in range(
        len(data) - 1
    ):
        base = data.iloc[base_index]

        base_open = float(base["open"])
        base_close = float(base["close"])

        if base_close > base_open:
            zone_type = SUPPLY

        elif base_close < base_open:
            zone_type = DEMAND

        else:
            continue

        impulse = _find_impulse(
            data=data,
            atr=atr,
            base_index=base_index,
            zone_type=zone_type,
            impulse_window=impulse_window,
            atr_multiplier=atr_multiplier,
        )

        if impulse is None:
            continue

        upper = float(base["high"])
        lower = float(base["low"])

        if upper <= lower:
            continue

        zone = {
            "type": zone_type,
            "upper": upper,
            "lower": lower,
            "midpoint": (upper + lower) / 2,
            "base_index": base_index,
            "base_time": base["close_time"].isoformat(),
            "confirmation_index": impulse["index"],
            "confirmation_time": impulse["time"],
            "confirmation_price": impulse["close"],
            "displacement": impulse["displacement"],
            "atr_at_base": impulse["atr"],
            "strength_atr": impulse["strength"],
            "source": str(base["source"]),
        }

        evaluated = _evaluate_zone(
            data,
            zone,
        )

        zones.append(evaluated)

    zones.sort(
        key=lambda item: (
            item["confirmation_index"],
            item["type"],
        )
    )

    return zones


def analyze_supply_demand(
    data: pd.DataFrame,
    atr_period: int = 14,
    impulse_window: int = 5,
    atr_multiplier: float = 1.5,
    minimum_candles: int = 30,
    analysis_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Analyze Bitcoin supply and demand zones.

    Features:
    - Confirmed supply zones.
    - Confirmed demand zones.
    - ATR-based displacement confirmation.
    - Active, mitigated and broken zone states.
    - Zone touch counts.
    - Confirmation timestamps.
    - Source tracking.

    Only validated closed candles are accepted.
    No future candle information is used.
    Does not generate trading signals.
    Does not execute trades.
    Does not modify input data.
    """

    _validate_parameters(
        atr_period=atr_period,
        impulse_window=impulse_window,
        minimum_candles=minimum_candles,
        atr_multiplier=atr_multiplier,
    )

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    candles = _validate_candles(
        data,
        timestamp,
    )

    if len(candles) < minimum_candles:
        raise ZoneAnalysisError(
            f"At least {minimum_candles} candles are required. "
            f"Received: {len(candles)}."
        )

    atr = _calculate_atr(
        candles,
        atr_period,
    )

    zones = _detect_zones(
        data=candles,
        atr=atr,
        impulse_window=impulse_window,
        atr_multiplier=float(atr_multiplier),
    )

    supply_zones = [
        zone
        for zone in zones
        if zone["type"] == SUPPLY
    ]

    demand_zones = [
        zone
        for zone in zones
        if zone["type"] == DEMAND
    ]

    sources = sorted(
        candles["source"].astype(str).unique().tolist()
    )

    return {
        "analysis_timestamp": timestamp.isoformat(),
        "status": "SUCCESS",
        "source": sources,
        "candles_analyzed": len(candles),
        "last_candle_close_time": (
            candles["close_time"].iloc[-1].isoformat()
        ),
        "supply_zones": supply_zones,
        "demand_zones": demand_zones,
        "zones": zones,
        "summary": {
            "total_zones": len(zones),
            "supply_count": len(supply_zones),
            "demand_count": len(demand_zones),
            "active_count": sum(
                zone["status"] == ACTIVE
                for zone in zones
            ),
            "mitigated_count": sum(
                zone["status"] == MITIGATED
                for zone in zones
            ),
            "broken_count": sum(
                zone["status"] == BROKEN
                for zone in zones
            ),
        },
        "parameters": {
            "atr_period": atr_period,
            "impulse_window": impulse_window,
            "atr_multiplier": float(atr_multiplier),
            "minimum_candles": minimum_candles,
        },
    }


if __name__ == "__main__":
    print(
        "Bitcoin supply and demand analysis module loaded."
    )