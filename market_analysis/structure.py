
"""Bitcoin market price structure analysis."""

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

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

HH = "HH"
HL = "HL"
LH = "LH"
LL = "LL"

BOS = "BOS"
CHOCH = "CHoCH"


class PriceStructureError(RuntimeError):
    """Raised when price structure analysis fails."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Normalize analysis timestamp to UTC."""

    if analysis_time is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        timestamp = pd.Timestamp(analysis_time)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PriceStructureError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise PriceStructureError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise PriceStructureError(
            "analysis_time must be timezone-aware."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise PriceStructureError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_parameters(
    swing_window: int,
    minimum_candles: int,
) -> None:
    """Validate analysis parameters."""

    for name, value in (
        ("swing_window", swing_window),
        ("minimum_candles", minimum_candles),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise PriceStructureError(
                f"{name} must be a positive integer."
            )

    if minimum_candles < (2 * swing_window + 1):
        raise PriceStructureError(
            "minimum_candles must be at least "
            "2 * swing_window + 1."
        )


def _validate_candles(
    data: pd.DataFrame,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate OHLCV data using the central validator."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise PriceStructureError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise PriceStructureError(
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
        raise PriceStructureError(
            f"Candle validation failed: {exc}"
        ) from exc

    if not rejected.empty:
        raise PriceStructureError(
            "Invalid candles detected. "
            f"Rejected: {len(rejected)}. "
            f"Validation errors: {report['errors']}."
        )

    if accepted.empty:
        raise PriceStructureError(
            "No valid candles available."
        )

    result = accepted.copy(deep=True)

    if not result["open_time"].is_monotonic_increasing:
        raise PriceStructureError(
            "Candles must be sorted by open_time."
        )

    if not result["close_time"].is_monotonic_increasing:
        raise PriceStructureError(
            "Candles must be sorted by close_time."
        )

    if result["open_time"].duplicated().any():
        raise PriceStructureError(
            "Duplicate candle open_time values detected."
        )

    if result["close_time"].duplicated().any():
        raise PriceStructureError(
            "Duplicate candle close_time values detected."
        )

    for column in ("open_time", "close_time"):
        if not isinstance(
            result[column].dtype,
            pd.DatetimeTZDtype,
        ):
            raise PriceStructureError(
                f"{column} must be timezone-aware."
            )

        if str(result[column].dt.tz) != "UTC":
            raise PriceStructureError(
                f"{column} must use UTC."
            )

    valid_sources = result["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise PriceStructureError(
            "Every candle must have a valid source."
        )

    return result.reset_index(drop=True)


def _find_swings(
    data: pd.DataFrame,
    swing_window: int,
) -> list[dict[str, Any]]:
    """
    Detect confirmed swing highs and lows.

    A swing becomes available only after all right-side
    confirmation candles have closed.
    """

    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)

    swings: list[dict[str, Any]] = []

    for index in range(
        swing_window,
        len(data) - swing_window,
    ):
        left = index - swing_window
        right = index + swing_window + 1

        other_highs = np.concatenate(
            (
                highs[left:index],
                highs[index + 1:right],
            )
        )

        other_lows = np.concatenate(
            (
                lows[left:index],
                lows[index + 1:right],
            )
        )

        if highs[index] > other_highs.max():
            swings.append(
                {
                    "type": "HIGH",
                    "index": index,
                    "price": float(highs[index]),
                    "time": data["close_time"].iloc[
                        index
                    ].isoformat(),
                    "confirmation_index": (
                        index + swing_window
                    ),
                    "confirmation_time": (
                        data["close_time"].iloc[
                            index + swing_window
                        ].isoformat()
                    ),
                }
            )

        if lows[index] < other_lows.min():
            swings.append(
                {
                    "type": "LOW",
                    "index": index,
                    "price": float(lows[index]),
                    "time": data["close_time"].iloc[
                        index
                    ].isoformat(),
                    "confirmation_index": (
                        index + swing_window
                    ),
                    "confirmation_time": (
                        data["close_time"].iloc[
                            index + swing_window
                        ].isoformat()
                    ),
                }
            )

    swings.sort(
        key=lambda item: (
            item["confirmation_index"],
            item["index"],
            item["type"],
        )
    )

    return swings


def _classify_swing_points(
    swings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify confirmed swing points."""

    previous_high: float | None = None
    previous_low: float | None = None

    classified: list[dict[str, Any]] = []

    for swing in swings:
        item = dict(swing)

        if item["type"] == "HIGH":
            if previous_high is None:
                item["classification"] = None
            elif item["price"] > previous_high:
                item["classification"] = HH
            elif item["price"] < previous_high:
                item["classification"] = LH
            else:
                item["classification"] = "EQH"

            previous_high = item["price"]

        else:
            if previous_low is None:
                item["classification"] = None
            elif item["price"] > previous_low:
                item["classification"] = HL
            elif item["price"] < previous_low:
                item["classification"] = LL
            else:
                item["classification"] = "EQL"

            previous_low = item["price"]

        classified.append(item)

    return classified


def _detect_structure_breaks(
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """
    Detect BOS and CHoCH chronologically.

    Only swings already confirmed at the current candle
    can be used as break levels.

    The trend state is updated sequentially. No final
    trend information is used to classify earlier events.
    """

    swings_by_confirmation: dict[
        int,
        list[dict[str, Any]],
    ] = {}

    for swing in swings:
        confirmation_index = swing["confirmation_index"]

        swings_by_confirmation.setdefault(
            confirmation_index,
            [],
        ).append(swing)

    last_high: dict[str, Any] | None = None
    last_low: dict[str, Any] | None = None

    broken_high_indices: set[int] = set()
    broken_low_indices: set[int] = set()

    previous_high: float | None = None
    previous_low: float | None = None

    trend = NEUTRAL
    events: list[dict[str, Any]] = []

    for index in range(len(data)):
        # Activate only swings confirmed by this candle.
        for swing in swings_by_confirmation.get(index, []):
            if swing["type"] == "HIGH":
                last_high = swing
                previous_high = swing["price"]

            else:
                last_low = swing
                previous_low = swing["price"]

        close_price = float(
            data["close"].iloc[index]
        )

        close_time = data["close_time"].iloc[index]

        # A level cannot be broken before its confirmation.
        bullish_break = (
            last_high is not None
            and index > last_high["confirmation_index"]
            and last_high["index"] not in broken_high_indices
            and close_price > last_high["price"]
        )

        bearish_break = (
            last_low is not None
            and index > last_low["confirmation_index"]
            and last_low["index"] not in broken_low_indices
            and close_price < last_low["price"]
        )

        # Both directions on the same close are ambiguous.
        if bullish_break and bearish_break:
            continue

        if bullish_break:
            event_type = (
                CHOCH
                if trend == BEARISH
                else BOS
            )

            events.append(
                {
                    "type": event_type,
                    "direction": BULLISH,
                    "level": last_high["price"],
                    "level_time": last_high["time"],
                    "level_confirmation_time": (
                        last_high["confirmation_time"]
                    ),
                    "break_time": close_time.isoformat(),
                    "break_price": close_price,
                    "status": "CONFIRMED",
                }
            )

            broken_high_indices.add(
                last_high["index"]
            )

            trend = BULLISH

        elif bearish_break:
            event_type = (
                CHOCH
                if trend == BULLISH
                else BOS
            )

            events.append(
                {
                    "type": event_type,
                    "direction": BEARISH,
                    "level": last_low["price"],
                    "level_time": last_low["time"],
                    "level_confirmation_time": (
                        last_low["confirmation_time"]
                    ),
                    "break_time": close_time.isoformat(),
                    "break_price": close_price,
                    "status": "CONFIRMED",
                }
            )

            broken_low_indices.add(
                last_low["index"]
            )

            trend = BEARISH

    # Use the latest confirmed swing classifications
    # only when no directional break has established a trend.
    if trend == NEUTRAL:
        highs = [
            item
            for item in swings
            if item["type"] == "HIGH"
        ]

        lows = [
            item
            for item in swings
            if item["type"] == "LOW"
        ]

        if len(highs) >= 2 and len(lows) >= 2:
            high_direction = (
                highs[-1]["price"] - highs[-2]["price"]
            )

            low_direction = (
                lows[-1]["price"] - lows[-2]["price"]
            )

            if high_direction > 0 and low_direction > 0:
                trend = BULLISH

            elif high_direction < 0 and low_direction < 0:
                trend = BEARISH

    return events, trend


def _detect_developing_swings(
    data: pd.DataFrame,
    swing_window: int,
) -> list[dict[str, Any]]:
    """
    Detect possible swing candidates.

    Developing candidates are not confirmed swing points
    and are never used for BOS or CHoCH calculations.
    """

    if len(data) <= swing_window:
        return []

    index = len(data) - 1
    start = max(0, index - swing_window)

    previous = data.iloc[start:index]
    current = data.iloc[index]

    if previous.empty:
        return []

    developing: list[dict[str, Any]] = []

    if current["high"] > previous["high"].max():
        developing.append(
            {
                "type": "POTENTIAL_HIGH",
                "price": float(current["high"]),
                "time": current["close_time"].isoformat(),
                "status": "DEVELOPING",
            }
        )

    if current["low"] < previous["low"].min():
        developing.append(
            {
                "type": "POTENTIAL_LOW",
                "price": float(current["low"]),
                "time": current["close_time"].isoformat(),
                "status": "DEVELOPING",
            }
        )

    return developing


def analyze_price_structure(
    data: pd.DataFrame,
    swing_window: int = 3,
    minimum_candles: int = 30,
    analysis_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Analyze Bitcoin price structure.

    Features:
    - Swing highs and swing lows.
    - Higher highs and higher lows.
    - Lower highs and lower lows.
    - Break of Structure (BOS).
    - Change of Character (CHoCH).
    - Confirmed and developing structures.

    Uses validated closed candles only.
    Prevents look-ahead bias in structure break detection.
    Does not generate trading signals or execute trades.
    Does not modify input data.
    """

    _validate_parameters(
        swing_window,
        minimum_candles,
    )

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    candles = _validate_candles(
        data,
        timestamp,
    )

    if len(candles) < minimum_candles:
        raise PriceStructureError(
            f"At least {minimum_candles} candles are required. "
            f"Received: {len(candles)}."
        )

    raw_swings = _find_swings(
        candles,
        swing_window,
    )

    classified_swings = _classify_swing_points(
        raw_swings
    )

    events, structure_trend = _detect_structure_breaks(
        candles,
        raw_swings,
    )

    developing = _detect_developing_swings(
        candles,
        swing_window,
    )

    highs = [
        item
        for item in classified_swings
        if item["type"] == "HIGH"
    ]

    lows = [
        item
        for item in classified_swings
        if item["type"] == "LOW"
    ]

    sources = sorted(
        candles["source"].astype(str).unique().tolist()
    )

    latest_high = highs[-1] if highs else None
    latest_low = lows[-1] if lows else None

    return {
        "analysis_timestamp": timestamp.isoformat(),
        "status": "SUCCESS",
        "source": sources,
        "trend": structure_trend,
        "swing_window": swing_window,
        "candles_analyzed": len(candles),
        "last_candle_close_time": (
            candles["close_time"].iloc[-1].isoformat()
        ),
        "swing_highs": highs,
        "swing_lows": lows,
        "swings": classified_swings,
        "structure_breaks": events,
        "developing_structures": developing,
        "levels": {
            "resistance": (
                latest_high["price"]
                if latest_high is not None
                else None
            ),
            "resistance_time": (
                latest_high["time"]
                if latest_high is not None
                else None
            ),
            "support": (
                latest_low["price"]
                if latest_low is not None
                else None
            ),
            "support_time": (
                latest_low["time"]
                if latest_low is not None
                else None
            ),
        },
        "summary": {
            "swing_high_count": len(highs),
            "swing_low_count": len(lows),
            "bos_count": sum(
                event["type"] == BOS
                for event in events
            ),
            "choch_count": sum(
                event["type"] == CHOCH
                for event in events
            ),
        },
        "parameters": {
            "swing_window": swing_window,
            "minimum_candles": minimum_candles,
        },
    }


if __name__ == "__main__":
    print(
        "Bitcoin price structure analysis module loaded."
    )