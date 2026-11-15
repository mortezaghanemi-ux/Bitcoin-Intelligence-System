"""Bitcoin trader market structure analysis engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any

import pandas as pd

from market_analysis.structure import (
    BEARISH,
    BULLISH,
    PriceStructureError,
    analyze_price_structure,
)


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

FIBONACCI_LEVELS = (
    0.0,
    0.236,
    0.382,
    0.5,
    0.618,
    0.786,
    1.0,
)


class MarketStructureAnalyzerError(RuntimeError):
    """Raised when trader market structure analysis fails."""


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
        raise MarketStructureAnalyzerError(
            f"Invalid {field}."
        ) from exc

    if pd.isna(timestamp):
        raise MarketStructureAnalyzerError(
            f"{field} cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise MarketStructureAnalyzerError(
            f"{field} must be timezone-aware UTC."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise MarketStructureAnalyzerError(
            f"{field} must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_timeframe(timeframe: str) -> int:
    """Return the configured timeframe duration."""

    if not isinstance(timeframe, str):
        raise MarketStructureAnalyzerError(
            "timeframe must be a string."
        )

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise MarketStructureAnalyzerError(
            f"Unsupported timeframe: {timeframe}"
        )

    return SUPPORTED_TIMEFRAMES[timeframe]


def _validate_parameters(
    swing_window: int,
    minimum_candles: int,
    liquidity_tolerance: float,
) -> None:
    """Validate structure analysis parameters."""

    if (
        isinstance(swing_window, bool)
        or not isinstance(swing_window, int)
        or swing_window <= 0
    ):
        raise MarketStructureAnalyzerError(
            "swing_window must be a positive integer."
        )

    if (
        isinstance(minimum_candles, bool)
        or not isinstance(minimum_candles, int)
        or minimum_candles <= 0
    ):
        raise MarketStructureAnalyzerError(
            "minimum_candles must be a positive integer."
        )

    if minimum_candles < (2 * swing_window + 1):
        raise MarketStructureAnalyzerError(
            "minimum_candles must be at least "
            "2 * swing_window + 1."
        )

    if (
        isinstance(liquidity_tolerance, bool)
        or not isinstance(liquidity_tolerance, (int, float))
        or not math.isfinite(liquidity_tolerance)
        or liquidity_tolerance < 0
    ):
        raise MarketStructureAnalyzerError(
            "liquidity_tolerance must be a finite "
            "non-negative number."
        )


def _validate_timeframe_candles(
    data: pd.DataFrame,
    timeframe: str,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate candle timestamps, duration and chronology."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise MarketStructureAnalyzerError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise MarketStructureAnalyzerError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    interval_seconds = _validate_timeframe(timeframe)

    candles = data.loc[:, REQUIRED_COLUMNS].copy(deep=True)

    for column in ("open_time", "close_time"):
        try:
            timestamps = pd.to_datetime(
                candles[column],
                errors="raise",
            )

        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketStructureAnalyzerError(
                f"Invalid timestamps in {column}."
            ) from exc

        if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
            raise MarketStructureAnalyzerError(
                f"{column} must contain timezone-aware timestamps."
            )

        if str(timestamps.dt.tz) != "UTC":
            raise MarketStructureAnalyzerError(
                f"{column} must use UTC."
            )

        if timestamps.isna().any():
            raise MarketStructureAnalyzerError(
                f"{column} contains missing timestamps."
            )

        candles[column] = timestamps

    if (
        candles["close_time"] <= candles["open_time"]
    ).any():
        raise MarketStructureAnalyzerError(
            "Every candle must close after it opens."
        )

    durations = candles["close_time"] - candles["open_time"]

    expected_duration = pd.Timedelta(
        seconds=interval_seconds
    )

    if not durations.eq(expected_duration).all():
        raise MarketStructureAnalyzerError(
            f"Candle duration does not match timeframe '{timeframe}'."
        )

    if (candles["close_time"] > analysis_time).any():
        raise MarketStructureAnalyzerError(
            "Future or unclosed candles are not allowed."
        )

    if not candles["open_time"].is_monotonic_increasing:
        raise MarketStructureAnalyzerError(
            "Candles must be sorted by open_time."
        )

    if not candles["close_time"].is_monotonic_increasing:
        raise MarketStructureAnalyzerError(
            "Candles must be sorted by close_time."
        )

    if candles["open_time"].duplicated().any():
        raise MarketStructureAnalyzerError(
            "Duplicate open_time values detected."
        )

    if candles["close_time"].duplicated().any():
        raise MarketStructureAnalyzerError(
            "Duplicate close_time values detected."
        )

    return candles.reset_index(drop=True)


def _build_liquidity_zones(
    structure: dict[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Build liquidity zones from equal confirmed swing levels."""

    zones = []

    for swing in structure.get("swings", []):
        classification = swing.get("classification")

        if classification not in ("EQH", "EQL"):
            continue

        price = float(swing["price"])

        zone = {
            "type": (
                "BUY_SIDE_LIQUIDITY"
                if classification == "EQH"
                else "SELL_SIDE_LIQUIDITY"
            ),
            "classification": classification,
            "price": price,
            "lower_bound": price * (1 - tolerance),
            "upper_bound": price * (1 + tolerance),
            "time": swing["time"],
            "confirmation_time": swing["confirmation_time"],
            "status": "CONFIRMED",
        }

        zones.append(zone)

    return zones


def _detect_liquidity_sweeps(
    candles: pd.DataFrame,
    structure: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Detect confirmed liquidity sweeps.

    A sweep requires:
    - A previously confirmed swing level.
    - A later candle wick beyond that level.
    - The candle close back inside the level.

    Unconfirmed swing levels are never used.
    """

    sweeps = []

    for swing in structure.get("swings", []):
        swing_type = swing.get("type")

        if swing_type not in ("HIGH", "LOW"):
            continue

        confirmation_index = swing.get("confirmation_index")

        if (
            isinstance(confirmation_index, bool)
            or not isinstance(confirmation_index, int)
            or confirmation_index < 0
            or confirmation_index >= len(candles)
        ):
            continue

        level = float(swing["price"])

        for index in range(confirmation_index + 1, len(candles)):
            candle = candles.iloc[index]

            if swing_type == "HIGH":
                swept = (
                    float(candle["high"]) > level
                    and float(candle["close"]) < level
                )
                sweep_type = "BUY_SIDE_SWEEP"
                direction = BEARISH

            else:
                swept = (
                    float(candle["low"]) < level
                    and float(candle["close"]) > level
                )
                sweep_type = "SELL_SIDE_SWEEP"
                direction = BULLISH

            if not swept:
                continue

            sweeps.append(
                {
                    "type": sweep_type,
                    "direction": direction,
                    "level": level,
                    "swing_time": swing["time"],
                    "confirmation_time": swing["confirmation_time"],
                    "sweep_time": candle["close_time"].isoformat(),
                    "sweep_high": float(candle["high"]),
                    "sweep_low": float(candle["low"]),
                    "close_price": float(candle["close"]),
                    "status": "CONFIRMED",
                }
            )

            # Only the first sweep of a swing level is recorded.
            break

    sweeps.sort(key=lambda item: item["sweep_time"])

    return sweeps


def _calculate_fibonacci(
    structure: dict[str, Any],
) -> dict[str, Any] | None:
    """Calculate retracement levels from the latest confirmed swing range."""

    swings = structure.get("swings", [])

    highs = [
        swing for swing in swings
        if swing.get("type") == "HIGH"
    ]

    lows = [
        swing for swing in swings
        if swing.get("type") == "LOW"
    ]

    if not highs or not lows:
        return None

    latest_high = highs[-1]
    latest_low = lows[-1]

    high_price = float(latest_high["price"])
    low_price = float(latest_low["price"])

    if high_price <= low_price:
        return None

    high_time = pd.Timestamp(latest_high["time"])
    low_time = pd.Timestamp(latest_low["time"])

    bullish_leg = low_time < high_time
    price_range = high_price - low_price

    levels = {}

    for ratio in FIBONACCI_LEVELS:
        if bullish_leg:
            price = high_price - (price_range * ratio)
        else:
            price = low_price + (price_range * ratio)

        levels[f"{ratio:.3f}"] = round(float(price), 8)

    return {
        "direction": BULLISH if bullish_leg else BEARISH,
        "swing_high": high_price,
        "swing_high_time": latest_high["time"],
        "swing_low": low_price,
        "swing_low_time": latest_low["time"],
        "range": price_range,
        "levels": levels,
        "status": "CALCULATED",
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


class MarketStructureAnalyzer:
    """
    Analyze and preserve trader market structure.

    Features:
    - Confirmed swing highs and lows.
    - Higher highs and higher lows.
    - Lower highs and lower lows.
    - BOS and CHoCH.
    - Trend identification.
    - Support and resistance.
    - Equal-high and equal-low liquidity zones.
    - Confirmed liquidity sweeps.
    - Fibonacci retracement levels.
    - UTC timestamps.
    - Closed candles only.
    - In-memory historical analysis preservation.

    This module does not generate or execute trading orders.
    """

    def __init__(
        self,
        timeframe: str,
        swing_window: int = 3,
        minimum_candles: int = 30,
        liquidity_tolerance: float = 0.001,
    ) -> None:

        _validate_timeframe(timeframe)

        _validate_parameters(
            swing_window,
            minimum_candles,
            liquidity_tolerance,
        )

        self.timeframe = timeframe
        self.swing_window = swing_window
        self.minimum_candles = minimum_candles
        self.liquidity_tolerance = float(liquidity_tolerance)

        self._history: list[dict[str, Any]] = []
        self._known_ids: set[str] = set()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a defensive copy of analysis history."""

        return [
            json.loads(json.dumps(item))
            for item in self._history
        ]

    def clear(self) -> None:
        """Clear stored in-memory analysis history."""

        self._history.clear()
        self._known_ids.clear()

    def analyze(
        self,
        data: pd.DataFrame,
        *,
        analysis_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Run market structure analysis for the configured timeframe."""

        timestamp = _normalize_utc(
            analysis_time,
            "analysis_time",
        )

        candles = _validate_timeframe_candles(
            data,
            self.timeframe,
            timestamp,
        )

        if len(candles) < self.minimum_candles:
            raise MarketStructureAnalyzerError(
                f"At least {self.minimum_candles} candles are required. "
                f"Received: {len(candles)}."
            )

        try:
            structure = analyze_price_structure(
                data=candles.copy(deep=True),
                swing_window=self.swing_window,
                minimum_candles=self.minimum_candles,
                analysis_time=timestamp.to_pydatetime(),
            )

        except (
            PriceStructureError,
            TypeError,
            ValueError,
        ) as exc:
            raise MarketStructureAnalyzerError(
                f"Price structure analysis failed: {exc}"
            ) from exc

        sources = sorted(
            candles["source"].astype(str).unique().tolist()
        )

        analysis_timestamp = timestamp.isoformat()

        last_close_time = candles["close_time"].iloc[-1].isoformat()

        liquidity_zones = _build_liquidity_zones(
            structure,
            self.liquidity_tolerance,
        )

        liquidity_sweeps = _detect_liquidity_sweeps(
            candles,
            structure,
        )

        fibonacci = _calculate_fibonacci(structure)

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
            "trend": structure["trend"],
            "candles_analyzed": len(candles),
            "last_candle_close_time": last_close_time,
            "swing_highs": structure["swing_highs"],
            "swing_lows": structure["swing_lows"],
            "swings": structure["swings"],
            "structure_breaks": structure["structure_breaks"],
            "developing_structures": structure["developing_structures"],
            "liquidity_zones": liquidity_zones,
            "liquidity_sweeps": liquidity_sweeps,
            "fibonacci": fibonacci,
            "levels": structure["levels"],
            "summary": {
                **structure["summary"],
                "liquidity_zone_count": len(liquidity_zones),
                "liquidity_sweep_count": len(liquidity_sweeps),
                "fibonacci_available": fibonacci is not None,
            },
            "parameters": {
                "swing_window": self.swing_window,
                "minimum_candles": self.minimum_candles,
                "liquidity_tolerance": self.liquidity_tolerance,
            },
            "limitations": [
                "Only validated closed candles are analyzed.",
                "Swing points require right-side confirmation.",
                "BOS and CHoCH use confirmed swing levels only.",
                "Liquidity sweeps require a wick beyond a confirmed level and a close back inside.",
                "Fibonacci levels are technical reference levels, not guaranteed reversal prices.",
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


def analyze_market_structure(
    data: pd.DataFrame,
    timeframe: str,
    *,
    analysis_time: datetime | None = None,
    swing_window: int = 3,
    minimum_candles: int = 30,
    liquidity_tolerance: float = 0.001,
) -> dict[str, Any]:
    """
    Convenience function for one-time structure analysis.

    Use MarketStructureAnalyzer when historical
    analysis preservation is required.
    """

    analyzer = MarketStructureAnalyzer(
        timeframe=timeframe,
        swing_window=swing_window,
        minimum_candles=minimum_candles,
        liquidity_tolerance=liquidity_tolerance,
    )

    return analyzer.analyze(
        data,
        analysis_time=analysis_time,
    )


if __name__ == "__main__":
    print(
        "Bitcoin trader market structure analyzer loaded."
    )