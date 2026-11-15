"""Bitcoin order block and fair value gap detection engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any

import pandas as pd


SUPPORTED_TIMEFRAMES = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}

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

BULLISH = "BULLISH"
BEARISH = "BEARISH"


class OrderBlockDetectorError(RuntimeError):
    """Raised when order block detection fails."""


def _normalize_utc(
    value: datetime | None,
    field: str,
) -> pd.Timestamp:
    """Validate and normalize a timestamp to UTC."""

    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OrderBlockDetectorError(
            f"Invalid {field}."
        ) from exc

    if pd.isna(timestamp):
        raise OrderBlockDetectorError(
            f"{field} cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise OrderBlockDetectorError(
            f"{field} must be timezone-aware UTC."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise OrderBlockDetectorError(
            f"{field} must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_timeframe(timeframe: str) -> int:
    """Return the configured timeframe duration."""

    if not isinstance(timeframe, str):
        raise OrderBlockDetectorError(
            "timeframe must be a string."
        )

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise OrderBlockDetectorError(
            f"Unsupported timeframe: {timeframe}"
        )

    return SUPPORTED_TIMEFRAMES[timeframe]


def _validate_parameters(
    displacement_threshold: float,
    minimum_body_ratio: float,
    lookback: int,
    minimum_candles: int,
    mitigation_tolerance: float,
) -> None:
    """Validate detector configuration."""

    for name, value in (
        ("displacement_threshold", displacement_threshold),
        ("minimum_body_ratio", minimum_body_ratio),
        ("mitigation_tolerance", mitigation_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise OrderBlockDetectorError(
                f"{name} must be a finite non-negative number."
            )

    if displacement_threshold <= 0:
        raise OrderBlockDetectorError(
            "displacement_threshold must be greater than zero."
        )

    if not 0 < minimum_body_ratio <= 1:
        raise OrderBlockDetectorError(
            "minimum_body_ratio must be greater than zero "
            "and at most one."
        )

    if (
        isinstance(lookback, bool)
        or not isinstance(lookback, int)
        or lookback <= 0
    ):
        raise OrderBlockDetectorError(
            "lookback must be a positive integer."
        )

    if (
        isinstance(minimum_candles, bool)
        or not isinstance(minimum_candles, int)
        or minimum_candles < 3
    ):
        raise OrderBlockDetectorError(
            "minimum_candles must be an integer of at least 3."
        )

    if mitigation_tolerance >= 1:
        raise OrderBlockDetectorError(
            "mitigation_tolerance must be less than one."
        )


def _parse_utc_series(
    values: pd.Series,
    column: str,
) -> pd.Series:
    """Parse timestamps and require UTC timezone."""

    try:
        timestamps = pd.to_datetime(
            values,
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise OrderBlockDetectorError(
            f"Invalid timestamps in {column}."
        ) from exc

    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise OrderBlockDetectorError(
            f"{column} must contain timezone-aware timestamps."
        )

    if str(timestamps.dt.tz) != "UTC":
        raise OrderBlockDetectorError(
            f"{column} must use UTC."
        )

    if timestamps.isna().any():
        raise OrderBlockDetectorError(
            f"{column} contains missing timestamps."
        )

    return timestamps.dt.tz_convert("UTC")


def _validate_numeric_columns(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Validate finite numeric OHLCV values."""

    result = candles.copy(deep=True)

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        original = result[column]

        if original.map(
            lambda value: isinstance(value, bool)
        ).any():
            raise OrderBlockDetectorError(
                f"{column} cannot contain boolean values."
            )

        try:
            values = pd.to_numeric(
                original,
                errors="raise",
            ).astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OrderBlockDetectorError(
                f"{column} must contain numeric values."
            ) from exc

        if not values.map(math.isfinite).all():
            raise OrderBlockDetectorError(
                f"{column} contains non-finite values."
            )

        result[column] = values

    if (
        result[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise OrderBlockDetectorError(
            "OHLC prices must be greater than zero."
        )

    if (result["volume"] < 0).any():
        raise OrderBlockDetectorError(
            "Volume cannot be negative."
        )

    invalid_ohlc = (
        (result["high"] < result["low"])
        | (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():
        raise OrderBlockDetectorError(
            "Invalid OHLC relationships detected."
        )

    return result


def _validate_candles(
    data: pd.DataFrame,
    timeframe: str,
    analysis_time: pd.Timestamp,
    minimum_candles: int,
) -> pd.DataFrame:
    """Validate market candles before analysis."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise OrderBlockDetectorError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise OrderBlockDetectorError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if len(data) < minimum_candles:
        raise OrderBlockDetectorError(
            f"At least {minimum_candles} candles are required."
        )

    interval = _validate_timeframe(timeframe)

    candles = data.loc[
        :,
        REQUIRED_COLUMNS,
    ].copy(deep=True)

    candles["open_time"] = _parse_utc_series(
        candles["open_time"],
        "open_time",
    )

    candles["close_time"] = _parse_utc_series(
        candles["close_time"],
        "close_time",
    )

    if (
        candles["close_time"] <= candles["open_time"]
    ).any():
        raise OrderBlockDetectorError(
            "Every candle must close after it opens."
        )

    expected_duration = pd.Timedelta(seconds=interval)

    if not (
        candles["close_time"] - candles["open_time"]
    ).eq(expected_duration).all():
        raise OrderBlockDetectorError(
            f"Candle duration does not match {timeframe}."
        )

    if (candles["close_time"] > analysis_time).any():
        raise OrderBlockDetectorError(
            "Future or unclosed candles are not allowed."
        )

    if not candles["open_time"].is_monotonic_increasing:
        raise OrderBlockDetectorError(
            "Candles must be sorted by open_time."
        )

    if not candles["close_time"].is_monotonic_increasing:
        raise OrderBlockDetectorError(
            "Candles must be sorted by close_time."
        )

    if candles["open_time"].duplicated().any():
        raise OrderBlockDetectorError(
            "Duplicate open_time values detected."
        )

    if candles["close_time"].duplicated().any():
        raise OrderBlockDetectorError(
            "Duplicate close_time values detected."
        )

    valid_sources = candles["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise OrderBlockDetectorError(
            "Every candle must have a valid source."
        )

    return _validate_numeric_columns(
        candles
    ).reset_index(drop=True)


def _candle_direction(candle: pd.Series) -> str:
    """Return the direction of a candle body."""

    if candle["close"] > candle["open"]:
        return BULLISH

    if candle["close"] < candle["open"]:
        return BEARISH

    return "NEUTRAL"


def _candle_body_ratio(candle: pd.Series) -> float:
    """Calculate candle body as a fraction of its full range."""

    candle_range = float(
        candle["high"] - candle["low"]
    )

    if candle_range <= 0:
        return 0.0

    body = abs(
        float(candle["close"] - candle["open"])
    )

    return body / candle_range


def _build_zone_id(
    zone_type: str,
    timeframe: str,
    direction: str,
    origin_time: str,
    lower: float,
    upper: float,
) -> str:
    """Build a deterministic zone identifier."""

    payload = {
        "type": zone_type,
        "timeframe": timeframe,
        "direction": direction,
        "origin_time": origin_time,
        "lower": round(lower, 10),
        "upper": round(upper, 10),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _find_order_blocks(
    candles: pd.DataFrame,
    timeframe: str,
    analysis_time: pd.Timestamp,
    displacement_threshold: float,
    minimum_body_ratio: float,
    lookback: int,
) -> list[dict[str, Any]]:
    """Detect bullish and bearish order blocks."""

    zones = []

    for index in range(1, len(candles)):
        impulse = candles.iloc[index]

        impulse_direction = _candle_direction(impulse)

        if impulse_direction == "NEUTRAL":
            continue

        body_ratio = _candle_body_ratio(impulse)

        if body_ratio < minimum_body_ratio:
            continue

        previous_close = float(
            candles.iloc[index - 1]["close"]
        )

        if previous_close <= 0:
            continue

        displacement = abs(
            float(impulse["close"]) - previous_close
        ) / previous_close

        if displacement < displacement_threshold:
            continue

        if impulse_direction == BULLISH:
            opposite_direction = BEARISH
            zone_type = "BULLISH_ORDER_BLOCK"
        else:
            opposite_direction = BULLISH
            zone_type = "BEARISH_ORDER_BLOCK"

        start = max(0, index - lookback)

        origin_index = None

        for candidate_index in range(index - 1, start - 1, -1):
            candidate = candles.iloc[candidate_index]

            if _candle_direction(candidate) == opposite_direction:
                origin_index = candidate_index
                break

        if origin_index is None:
            continue

        origin = candles.iloc[origin_index]

        lower = float(origin["low"])
        upper = float(origin["high"])

        if lower >= upper:
            continue

        origin_time = origin["close_time"].isoformat()

        zone_id = _build_zone_id(
            zone_type,
            timeframe,
            impulse_direction,
            origin_time,
            lower,
            upper,
        )

        zones.append({
            "zone_id": zone_id,
            "type": zone_type,
            "direction": impulse_direction,
            "timeframe": timeframe,
            "lower": lower,
            "upper": upper,
            "midpoint": (lower + upper) / 2.0,
            "origin_index": origin_index,
            "origin_open_time": origin["open_time"].isoformat(),
            "origin_close_time": origin_time,
            "confirmation_time": impulse["close_time"].isoformat(),
            "displacement": displacement,
            "body_ratio": body_ratio,
            "status": "ACTIVE",
            "mitigated": False,
            "mitigation_time": None,
            "detection_time": analysis_time.isoformat(),
            "source": str(origin["source"]),
        })

    return zones


def _find_fair_value_gaps(
    candles: pd.DataFrame,
    timeframe: str,
    analysis_time: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Detect three-candle fair value gaps."""

    gaps = []

    for index in range(2, len(candles)):
        first = candles.iloc[index - 2]
        middle = candles.iloc[index - 1]
        third = candles.iloc[index]

        first_high = float(first["high"])
        first_low = float(first["low"])
        third_high = float(third["high"])
        third_low = float(third["low"])

        if (
            third_low > first_high
            and _candle_direction(middle) == BULLISH
        ):
            direction = BULLISH
            gap_type = "BULLISH_FVG"
            lower = first_high
            upper = third_low

        elif (
            third_high < first_low
            and _candle_direction(middle) == BEARISH
        ):
            direction = BEARISH
            gap_type = "BEARISH_FVG"
            lower = third_high
            upper = first_low

        else:
            continue

        if lower >= upper:
            continue

        midpoint = (lower + upper) / 2.0

        origin_time = middle["close_time"].isoformat()

        zone_id = _build_zone_id(
            gap_type,
            timeframe,
            direction,
            origin_time,
            lower,
            upper,
        )

        gaps.append({
            "zone_id": zone_id,
            "type": gap_type,
            "direction": direction,
            "timeframe": timeframe,
            "lower": lower,
            "upper": upper,
            "midpoint": midpoint,
            "consequent_encroachment": midpoint,
            "origin_index": index - 1,
            "origin_open_time": middle["open_time"].isoformat(),
            "origin_close_time": origin_time,
            "confirmation_time": third["close_time"].isoformat(),
            "status": "ACTIVE",
            "mitigated": False,
            "mitigation_time": None,
            "detection_time": analysis_time.isoformat(),
            "source": str(middle["source"]),
        })

    return gaps


def _apply_mitigation(
    zones: list[dict[str, Any]],
    candles: pd.DataFrame,
    mitigation_tolerance: float,
) -> list[dict[str, Any]]:
    """Determine whether a zone was mitigated after confirmation."""

    results = []

    for original_zone in zones:
        zone = dict(original_zone)

        lower = float(zone["lower"])
        upper = float(zone["upper"])
        direction = zone["direction"]

        confirmation_time = pd.Timestamp(
            zone["confirmation_time"]
        )

        subsequent = candles.loc[
            candles["close_time"] > confirmation_time
        ]

        mitigation_time = None
        mitigation_price = None

        for _, candle in subsequent.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])

            if direction == BULLISH:
                touched = low <= upper
                fully_mitigated = (
                    low <= lower * (1.0 + mitigation_tolerance)
                )
            else:
                touched = high >= lower
                fully_mitigated = (
                    high >= upper * (1.0 - mitigation_tolerance)
                )

            if not touched:
                continue

            if fully_mitigated:
                zone["status"] = "MITIGATED"
                zone["mitigated"] = True
                mitigation_time = candle["close_time"].isoformat()
                mitigation_price = close
                break

            zone["status"] = "PARTIALLY_MITIGATED"

        zone["mitigation_time"] = mitigation_time
        zone["mitigation_price"] = mitigation_price

        results.append(zone)

    return results


def _validate_zone(
    zone: dict[str, Any],
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """Validate that a detected zone has valid price boundaries."""

    lower = float(zone["lower"])
    upper = float(zone["upper"])

    valid = (
        math.isfinite(lower)
        and math.isfinite(upper)
        and lower > 0
        and upper > lower
    )

    latest_close = float(candles["close"].iloc[-1])

    if zone["direction"] == BULLISH:
        price_position = (
            "ABOVE_ZONE"
            if latest_close > upper
            else "INSIDE_ZONE"
            if latest_close >= lower
            else "BELOW_ZONE"
        )
    else:
        price_position = (
            "BELOW_ZONE"
            if latest_close < lower
            else "INSIDE_ZONE"
            if latest_close <= upper
            else "ABOVE_ZONE"
        )

    return {
        "valid": bool(valid),
        "price_position": price_position,
        "latest_close": latest_close,
        "zone_width": upper - lower,
        "zone_width_pct": (
            (upper - lower) / lower * 100.0
            if lower > 0
            else 0.0
        ),
    }


def _build_analysis_id(
    timeframe: str,
    analysis_timestamp: str,
    last_candle_close_time: str,
    sources: list[str],
) -> str:
    """Build a deterministic analysis identifier."""

    payload = {
        "timeframe": timeframe,
        "analysis_timestamp": analysis_timestamp,
        "last_candle_close_time": last_candle_close_time,
        "sources": sorted(sources),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


class OrderBlockDetector:
    """
    Detect and preserve order blocks and fair value gaps.

    Features:
    - Bullish and bearish order blocks.
    - Displacement confirmation.
    - Bullish and bearish fair value gaps.
    - Consequent encroachment.
    - Zone validation.
    - Partial and full mitigation.
    - Multiple supported timeframes.
    - UTC timestamps.
    - Closed candles only.
    - In-memory historical analysis preservation.

    This module does not generate or execute trading orders.
    """

    def __init__(
        self,
        timeframe: str,
        displacement_threshold: float = 0.003,
        minimum_body_ratio: float = 0.60,
        lookback: int = 10,
        minimum_candles: int = 30,
        mitigation_tolerance: float = 0.0,
    ) -> None:

        _validate_timeframe(timeframe)

        _validate_parameters(
            displacement_threshold,
            minimum_body_ratio,
            lookback,
            minimum_candles,
            mitigation_tolerance,
        )

        self.timeframe = timeframe
        self.displacement_threshold = float(
            displacement_threshold
        )
        self.minimum_body_ratio = float(
            minimum_body_ratio
        )
        self.lookback = lookback
        self.minimum_candles = minimum_candles
        self.mitigation_tolerance = float(
            mitigation_tolerance
        )

        self._history: list[dict[str, Any]] = []
        self._known_ids: set[str] = set()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a defensive copy of historical analyses."""

        return [
            json.loads(json.dumps(item))
            for item in self._history
        ]

    def clear(self) -> None:
        """Clear in-memory analysis history."""

        self._history.clear()
        self._known_ids.clear()

    def detect(
        self,
        data: pd.DataFrame,
        *,
        analysis_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Detect order blocks and fair value gaps."""

        timestamp = _normalize_utc(
            analysis_time,
            "analysis_time",
        )

        candles = _validate_candles(
            data,
            self.timeframe,
            timestamp,
            self.minimum_candles,
        )

        order_blocks = _find_order_blocks(
            candles,
            self.timeframe,
            timestamp,
            self.displacement_threshold,
            self.minimum_body_ratio,
            self.lookback,
        )

        fair_value_gaps = _find_fair_value_gaps(
            candles,
            self.timeframe,
            timestamp,
        )

        order_blocks = _apply_mitigation(
            order_blocks,
            candles,
            self.mitigation_tolerance,
        )

        fair_value_gaps = _apply_mitigation(
            fair_value_gaps,
            candles,
            self.mitigation_tolerance,
        )

        for zone in order_blocks:
            zone["validation"] = _validate_zone(
                zone,
                candles,
            )

        for gap in fair_value_gaps:
            gap["validation"] = _validate_zone(
                gap,
                candles,
            )

        sources = sorted(
            candles["source"].astype(str).unique().tolist()
        )

        analysis_timestamp = timestamp.isoformat()
        last_close_time = candles["close_time"].iloc[-1].isoformat()

        result = {
            "analysis_id": _build_analysis_id(
                self.timeframe,
                analysis_timestamp,
                last_close_time,
                sources,
            ),
            "analysis_timestamp": analysis_timestamp,
            "status": "SUCCESS",
            "timeframe": self.timeframe,
            "source": sources,
            "candles_analyzed": len(candles),
            "last_candle_close_time": last_close_time,
            "order_blocks": order_blocks,
            "fair_value_gaps": fair_value_gaps,
            "summary": {
                "order_block_count": len(order_blocks),
                "bullish_order_blocks": sum(
                    item["direction"] == BULLISH
                    for item in order_blocks
                ),
                "bearish_order_blocks": sum(
                    item["direction"] == BEARISH
                    for item in order_blocks
                ),
                "fair_value_gap_count": len(fair_value_gaps),
                "bullish_fvg_count": sum(
                    item["direction"] == BULLISH
                    for item in fair_value_gaps
                ),
                "bearish_fvg_count": sum(
                    item["direction"] == BEARISH
                    for item in fair_value_gaps
                ),
                "active_order_blocks": sum(
                    item["status"] == "ACTIVE"
                    for item in order_blocks
                ),
                "mitigated_order_blocks": sum(
                    item["mitigated"]
                    for item in order_blocks
                ),
                "active_fvgs": sum(
                    item["status"] == "ACTIVE"
                    for item in fair_value_gaps
                ),
                "mitigated_fvgs": sum(
                    item["mitigated"]
                    for item in fair_value_gaps
                ),
            },
            "parameters": {
                "displacement_threshold": self.displacement_threshold,
                "minimum_body_ratio": self.minimum_body_ratio,
                "lookback": self.lookback,
                "minimum_candles": self.minimum_candles,
                "mitigation_tolerance": self.mitigation_tolerance,
            },
            "limitations": [
                "Only validated closed candles are analyzed.",
                "Order blocks use the last opposite candle before a qualifying displacement.",
                "Fair value gaps use a three-candle price gap definition.",
                "Mitigation is evaluated only after zone confirmation.",
                "Consequent encroachment is the midpoint of a fair value gap.",
                "Zones are technical reference areas, not guaranteed reversal points.",
                "This module does not generate trading orders.",
                "This module does not execute trades.",
            ],
        }

        analysis_id = result["analysis_id"]

        if analysis_id not in self._known_ids:
            self._history.append(
                json.loads(json.dumps(result))
            )
            self._known_ids.add(analysis_id)

        return json.loads(json.dumps(result))


def detect_order_blocks(
    data: pd.DataFrame,
    timeframe: str,
    *,
    analysis_time: datetime | None = None,
    displacement_threshold: float = 0.003,
    minimum_body_ratio: float = 0.60,
    lookback: int = 10,
    minimum_candles: int = 30,
    mitigation_tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Convenience function for one-time order block and FVG detection.

    Use OrderBlockDetector when historical analysis preservation
    is required.
    """

    detector = OrderBlockDetector(
        timeframe=timeframe,
        displacement_threshold=displacement_threshold,
        minimum_body_ratio=minimum_body_ratio,
        lookback=lookback,
        minimum_candles=minimum_candles,
        mitigation_tolerance=mitigation_tolerance,
    )

    return detector.detect(
        data,
        analysis_time=analysis_time,
    )


if __name__ == "__main__":
    print(
        "Bitcoin order block and fair value gap detector loaded."
    )