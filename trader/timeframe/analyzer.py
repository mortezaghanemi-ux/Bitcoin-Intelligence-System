"""Bitcoin multi-timeframe trading analysis engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_TIMEFRAMES = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}

TIMEFRAME_ORDER = (
    "5m",
    "15m",
    "1h",
    "4h",
)

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

OUTPUT_TIMEFRAMES = TIMEFRAME_ORDER

DIRECTIONS = (
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
)


class MultiTimeframeAnalysisError(RuntimeError):
    """Raised when multi-timeframe analysis fails."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Validate and normalize the analysis timestamp to UTC."""

    if analysis_time is None:
        return pd.Timestamp(
            datetime.now(timezone.utc)
        )

    try:
        timestamp = pd.Timestamp(analysis_time)

    except (TypeError, ValueError, OverflowError) as exc:
        raise MultiTimeframeAnalysisError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise MultiTimeframeAnalysisError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise MultiTimeframeAnalysisError(
            "analysis_time must be timezone-aware UTC."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise MultiTimeframeAnalysisError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_timeframe(
    timeframe: str,
) -> int:
    """Return the duration of a supported timeframe."""

    if not isinstance(timeframe, str):
        raise MultiTimeframeAnalysisError(
            "Timeframe must be a string."
        )

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise MultiTimeframeAnalysisError(
            f"Unsupported timeframe: {timeframe}"
        )

    return SUPPORTED_TIMEFRAMES[timeframe]


def _parse_utc_series(
    values: pd.Series,
    column: str,
) -> pd.Series:
    """Parse timestamps and require timezone-aware UTC values."""

    try:
        timestamps = pd.to_datetime(
            values,
            errors="raise",
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise MultiTimeframeAnalysisError(
            f"Invalid timestamps in {column}."
        ) from exc

    if not isinstance(
        timestamps.dtype,
        pd.DatetimeTZDtype,
    ):
        raise MultiTimeframeAnalysisError(
            f"{column} must contain timezone-aware timestamps."
        )

    if str(timestamps.dt.tz) != "UTC":
        raise MultiTimeframeAnalysisError(
            f"{column} must use UTC timezone."
        )

    if timestamps.isna().any():
        raise MultiTimeframeAnalysisError(
            f"{column} contains missing timestamps."
        )

    return timestamps.dt.tz_convert("UTC")


def _validate_numeric_columns(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Validate finite numeric OHLCV values."""

    result = candles.copy(deep=True)

    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    for column in numeric_columns:

        original = result[column]

        if original.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).any():
            raise MultiTimeframeAnalysisError(
                f"{column} cannot contain boolean values."
            )

        try:
            values = pd.to_numeric(
                original,
                errors="raise",
            ).astype(float)

        except (TypeError, ValueError, OverflowError) as exc:
            raise MultiTimeframeAnalysisError(
                f"{column} must contain numeric values."
            ) from exc

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise MultiTimeframeAnalysisError(
                f"{column} contains non-finite values."
            )

        result[column] = values

    if (
        result[
            ["open", "high", "low", "close"]
        ] <= 0
    ).any().any():
        raise MultiTimeframeAnalysisError(
            "OHLC prices must be greater than zero."
        )

    if (result["volume"] < 0).any():
        raise MultiTimeframeAnalysisError(
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
        raise MultiTimeframeAnalysisError(
            "Invalid OHLC relationships detected."
        )

    return result


def _validate_candles(
    data: pd.DataFrame,
    timeframe: str,
    analysis_time: pd.Timestamp,
    minimum_candles: int,
) -> pd.DataFrame:
    """Validate one timeframe's candles."""

    if not isinstance(data, pd.DataFrame):
        raise MultiTimeframeAnalysisError(
            f"{timeframe} data must be a pandas DataFrame."
        )

    if data.empty:
        raise MultiTimeframeAnalysisError(
            f"{timeframe} candle data is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise MultiTimeframeAnalysisError(
            f"{timeframe} is missing columns: "
            + ", ".join(missing)
        )

    if len(data) < minimum_candles:
        raise MultiTimeframeAnalysisError(
            f"{timeframe} requires at least "
            f"{minimum_candles} candles."
        )

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
        candles["close_time"]
        <= candles["open_time"]
    ).any():
        raise MultiTimeframeAnalysisError(
            f"{timeframe} contains invalid candle durations."
        )

    interval = pd.Timedelta(
        seconds=SUPPORTED_TIMEFRAMES[timeframe]
    )

    if (
        candles["close_time"]
        - candles["open_time"]
        != interval
    ).any():
        raise MultiTimeframeAnalysisError(
            f"{timeframe} contains incorrect candle durations."
        )

    if (
        candles["close_time"]
        > analysis_time
    ).any():
        raise MultiTimeframeAnalysisError(
            f"{timeframe} contains future or unclosed candles."
        )

    if not candles["open_time"].is_monotonic_increasing:
        raise MultiTimeframeAnalysisError(
            f"{timeframe} candles must be sorted by open_time."
        )

    if not candles["close_time"].is_monotonic_increasing:
        raise MultiTimeframeAnalysisError(
            f"{timeframe} close_time values must be sorted."
        )

    if candles["open_time"].duplicated().any():
        raise MultiTimeframeAnalysisError(
            f"{timeframe} contains duplicate open_time values."
        )

    if candles["close_time"].duplicated().any():
        raise MultiTimeframeAnalysisError(
            f"{timeframe} contains duplicate close_time values."
        )

    valid_sources = candles["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise MultiTimeframeAnalysisError(
            f"{timeframe} contains invalid source values."
        )

    candles = _validate_numeric_columns(
        candles
    )

    return candles.reset_index(drop=True)


def _classify_direction(
    value: float,
    threshold: float = 0.0001,
) -> str:
    """Classify a numeric movement as bullish, bearish or neutral."""

    if value > threshold:
        return "BULLISH"

    if value < -threshold:
        return "BEARISH"

    return "NEUTRAL"


def _calculate_trend(
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate trend direction using EMA alignment and price movement."""

    close = candles["close"]

    ema_fast = close.ewm(
        span=20,
        adjust=False,
        min_periods=1,
    ).mean()

    ema_slow = close.ewm(
        span=50,
        adjust=False,
        min_periods=1,
    ).mean()

    latest_close = float(close.iloc[-1])
    fast_value = float(ema_fast.iloc[-1])
    slow_value = float(ema_slow.iloc[-1])

    lookback = min(5, len(close) - 1)

    previous_close = float(
        close.iloc[-1 - lookback]
    )

    if previous_close <= 0:
        raise MultiTimeframeAnalysisError(
            "Cannot calculate trend from invalid reference price."
        )

    price_change = (
        latest_close - previous_close
    ) / previous_close

    ema_direction = _classify_direction(
        (fast_value - slow_value)
        / max(abs(slow_value), 1e-12)
    )

    price_direction = _classify_direction(
        price_change
    )

    if (
        ema_direction == price_direction
        and ema_direction != "NEUTRAL"
    ):
        direction = ema_direction

    elif ema_direction == "NEUTRAL":
        direction = price_direction

    elif price_direction == "NEUTRAL":
        direction = ema_direction

    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "ema_fast": fast_value,
        "ema_slow": slow_value,
        "price_change_pct": price_change * 100.0,
        "ema_alignment": ema_direction,
        "price_direction": price_direction,
    }


def _calculate_momentum(
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate momentum from recent price changes."""

    close = candles["close"]

    periods = min(5, len(close) - 1)

    previous = float(
        close.iloc[-1 - periods]
    )

    current = float(
        close.iloc[-1]
    )

    change = (
        (current - previous) / previous
        if previous > 0
        else 0.0
    )

    recent_returns = (
        close.pct_change()
        .dropna()
        .tail(14)
    )

    average_return = (
        float(recent_returns.mean())
        if not recent_returns.empty
        else 0.0
    )

    direction = _classify_direction(
        change
    )

    return {
        "direction": direction,
        "change_pct": change * 100.0,
        "average_return_pct": average_return * 100.0,
    }


def _calculate_structure(
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """Identify basic higher-high and lower-low structure."""

    window = min(5, len(candles))

    recent = candles.tail(window)

    previous = candles.iloc[-window:-1]

    latest_high = float(
        recent["high"].iloc[-1]
    )

    latest_low = float(
        recent["low"].iloc[-1]
    )

    previous_high = float(
        previous["high"].max()
    )

    previous_low = float(
        previous["low"].min()
    )

    higher_high = latest_high > previous_high
    lower_high = latest_high < previous_high

    higher_low = latest_low > previous_low
    lower_low = latest_low < previous_low

    if higher_high and higher_low:
        direction = "BULLISH"
        label = "HH_HL"

    elif lower_high and lower_low:
        direction = "BEARISH"
        label = "LH_LL"

    elif higher_high or higher_low:
        direction = "BULLISH"
        label = "MIXED_BULLISH"

    elif lower_high or lower_low:
        direction = "BEARISH"
        label = "MIXED_BEARISH"

    else:
        direction = "NEUTRAL"
        label = "RANGE"

    return {
        "direction": direction,
        "label": label,
        "higher_high": bool(higher_high),
        "higher_low": bool(higher_low),
        "lower_high": bool(lower_high),
        "lower_low": bool(lower_low),
        "latest_high": latest_high,
        "latest_low": latest_low,
    }


def _calculate_support_resistance(
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate rolling support and resistance reference levels."""

    window = min(20, len(candles))

    recent = candles.tail(window)

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    current_price = float(
        candles["close"].iloc[-1]
    )

    support_distance = (
        (current_price - support)
        / current_price
        * 100.0
    )

    resistance_distance = (
        (resistance - current_price)
        / current_price
        * 100.0
    )

    if support_distance <= 0.5:
        position = "NEAR_SUPPORT"

    elif resistance_distance <= 0.5:
        position = "NEAR_RESISTANCE"

    else:
        position = "BETWEEN_LEVELS"

    return {
        "support": support,
        "resistance": resistance,
        "support_distance_pct": support_distance,
        "resistance_distance_pct": resistance_distance,
        "position": position,
    }


def _calculate_volume_confirmation(
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """Compare the latest volume with recent average volume."""

    volume = candles["volume"]

    window = min(20, len(volume))

    average_volume = float(
        volume.tail(window).mean()
    )

    latest_volume = float(
        volume.iloc[-1]
    )

    if average_volume > 0:
        ratio = latest_volume / average_volume
    else:
        ratio = 0.0

    price_change = (
        float(candles["close"].iloc[-1])
        - float(candles["open"].iloc[-1])
    )

    if ratio >= 1.2 and price_change > 0:
        direction = "BULLISH"

    elif ratio >= 1.2 and price_change < 0:
        direction = "BEARISH"

    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "latest_volume": latest_volume,
        "average_volume": average_volume,
        "volume_ratio": ratio,
        "confirmed": bool(ratio >= 1.2),
    }


def _calculate_timeframe_analysis(
    candles: pd.DataFrame,
    timeframe: str,
) -> dict[str, Any]:
    """Calculate all analysis components for one timeframe."""

    trend = _calculate_trend(candles)

    momentum = _calculate_momentum(candles)

    structure = _calculate_structure(candles)

    levels = _calculate_support_resistance(
        candles
    )

    volume = _calculate_volume_confirmation(
        candles
    )

    directions = [
        trend["direction"],
        momentum["direction"],
        structure["direction"],
    ]

    bullish_votes = directions.count(
        "BULLISH"
    )

    bearish_votes = directions.count(
        "BEARISH"
    )

    if bullish_votes > bearish_votes:
        signal_direction = "BULLISH"

    elif bearish_votes > bullish_votes:
        signal_direction = "BEARISH"

    else:
        signal_direction = "NEUTRAL"

    agreement = (
        max(bullish_votes, bearish_votes)
        / len(directions)
    )

    return {
        "timeframe": timeframe,
        "direction": signal_direction,
        "trend": trend,
        "momentum": momentum,
        "structure": structure,
        "support_resistance": levels,
        "volume": volume,
        "component_directions": {
            "trend": trend["direction"],
            "momentum": momentum["direction"],
            "structure": structure["direction"],
            "volume": volume["direction"],
        },
        "component_agreement": round(
            agreement,
            4,
        ),
        "current_price": float(
            candles["close"].iloc[-1]
        ),
        "candles_analyzed": len(candles),
        "last_candle_close_time": (
            candles["close_time"].iloc[-1].isoformat()
        ),
        "source": sorted(
            candles["source"].unique().tolist()
        ),
    }


def _build_alignment(
    analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate directional alignment across available timeframes."""

    directions = {
        timeframe: result["direction"]
        for timeframe, result in analyses.items()
    }

    bullish = sum(
        direction == "BULLISH"
        for direction in directions.values()
    )

    bearish = sum(
        direction == "BEARISH"
        for direction in directions.values()
    )

    neutral = sum(
        direction == "NEUTRAL"
        for direction in directions.values()
    )

    total = len(directions)

    if bullish > bearish:
        overall_direction = "BULLISH"

    elif bearish > bullish:
        overall_direction = "BEARISH"

    else:
        overall_direction = "NEUTRAL"

    if overall_direction == "BULLISH":
        aligned_count = bullish

    elif overall_direction == "BEARISH":
        aligned_count = bearish

    else:
        aligned_count = neutral

    alignment_ratio = (
        aligned_count / total
        if total
        else 0.0
    )

    conflicts = []

    available = [
        timeframe
        for timeframe in TIMEFRAME_ORDER
        if timeframe in directions
    ]

    for index, timeframe in enumerate(available):

        for higher_timeframe in available[index + 1:]:

            first = directions[timeframe]
            second = directions[higher_timeframe]

            if (
                first != "NEUTRAL"
                and second != "NEUTRAL"
                and first != second
            ):
                conflicts.append({
                    "timeframe": timeframe,
                    "higher_timeframe": higher_timeframe,
                    "direction": first,
                    "higher_timeframe_direction": second,
                    "type": "DIRECTION_CONFLICT",
                })

    if conflicts:
        alignment_status = "CONFLICTING"

    elif alignment_ratio == 1.0:
        alignment_status = "FULL_ALIGNMENT"

    elif alignment_ratio >= 0.75:
        alignment_status = "STRONG_ALIGNMENT"

    elif alignment_ratio >= 0.5:
        alignment_status = "PARTIAL_ALIGNMENT"

    else:
        alignment_status = "WEAK_ALIGNMENT"

    return {
        "direction": overall_direction,
        "status": alignment_status,
        "alignment_ratio": round(
            alignment_ratio,
            4,
        ),
        "timeframe_count": total,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "directions": directions,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
    }


def _build_higher_timeframe_validation(
    analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare lower timeframe directions with higher timeframe directions."""

    comparisons = []

    for index, timeframe in enumerate(TIMEFRAME_ORDER):

        if timeframe not in analyses:
            continue

        higher_timeframes = [
            candidate
            for candidate in TIMEFRAME_ORDER[index + 1:]
            if candidate in analyses
        ]

        for higher in higher_timeframes:

            lower_direction = analyses[
                timeframe
            ]["direction"]

            higher_direction = analyses[
                higher
            ]["direction"]

            if (
                lower_direction == "NEUTRAL"
                or higher_direction == "NEUTRAL"
            ):
                status = "INCONCLUSIVE"

            elif lower_direction == higher_direction:
                status = "CONFIRMED"

            else:
                status = "CONFLICT"

            comparisons.append({
                "timeframe": timeframe,
                "higher_timeframe": higher,
                "lower_direction": lower_direction,
                "higher_direction": higher_direction,
                "status": status,
            })

    confirmed = sum(
        item["status"] == "CONFIRMED"
        for item in comparisons
    )

    conflicts = sum(
        item["status"] == "CONFLICT"
        for item in comparisons
    )

    inconclusive = sum(
        item["status"] == "INCONCLUSIVE"
        for item in comparisons
    )

    return {
        "comparisons": comparisons,
        "confirmed_count": confirmed,
        "conflict_count": conflicts,
        "inconclusive_count": inconclusive,
        "status": (
            "CONFLICTING"
            if conflicts
            else "CONFIRMED"
            if confirmed
            else "INCONCLUSIVE"
        ),
    }


def _build_analysis_id(
    analysis_timestamp: str,
    analyses: dict[str, dict[str, Any]],
) -> str:
    """Build a deterministic identifier for an analysis."""

    identity = {
        "analysis_timestamp": analysis_timestamp,
        "timeframes": {
            timeframe: {
                "last_candle_close_time": result[
                    "last_candle_close_time"
                ],
                "source": result["source"],
                "candles_analyzed": result[
                    "candles_analyzed"
                ],
                "direction": result["direction"],
            }
            for timeframe, result in analyses.items()
        },
    }

    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


class MultiTimeframeAnalyzer:
    """
    Analyze Bitcoin market conditions across multiple timeframes.

    Features:
    - Supports 5m, 15m, 1h and 4h.
    - Uses closed candles only.
    - Rejects future and invalid candle data.
    - Calculates trend, momentum and market structure.
    - Evaluates support, resistance and volume confirmation.
    - Detects timeframe alignment and directional conflicts.
    - Validates lower timeframe direction against higher timeframes.
    - Preserves historical analysis results in memory.
    - Does not generate orders or execute trades.
    """

    def __init__(
        self,
        minimum_candles: int = 20,
        required_timeframes: tuple[str, ...] | None = None,
    ) -> None:

        if (
            isinstance(minimum_candles, bool)
            or not isinstance(minimum_candles, int)
            or minimum_candles < 5
        ):
            raise MultiTimeframeAnalysisError(
                "minimum_candles must be an integer of at least 5."
            )

        if required_timeframes is None:
            required_timeframes = TIMEFRAME_ORDER

        if (
            not isinstance(required_timeframes, (tuple, list))
            or not required_timeframes
        ):
            raise MultiTimeframeAnalysisError(
                "required_timeframes must be a non-empty tuple or list."
            )

        normalized = []

        for timeframe in required_timeframes:

            _validate_timeframe(timeframe)

            if timeframe in normalized:
                raise MultiTimeframeAnalysisError(
                    f"Duplicate required timeframe: {timeframe}"
                )

            normalized.append(timeframe)

        self.minimum_candles = minimum_candles

        self.required_timeframes = tuple(
            timeframe
            for timeframe in TIMEFRAME_ORDER
            if timeframe in normalized
        )

        self._history: list[dict[str, Any]] = []

        self._known_ids: set[str] = set()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a defensive copy of historical analyses."""

        return json.loads(
            json.dumps(self._history)
        )

    def clear_history(self) -> None:
        """Clear stored analysis history."""

        self._history.clear()
        self._known_ids.clear()

    def analyze(
        self,
        data: dict[str, pd.DataFrame],
        *,
        analysis_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Run multi-timeframe analysis on validated closed candles."""

        timestamp = _normalize_analysis_time(
            analysis_time
        )

        if not isinstance(data, dict):
            raise MultiTimeframeAnalysisError(
                "data must be a dictionary of timeframe DataFrames."
            )

        if not data:
            raise MultiTimeframeAnalysisError(
                "No timeframe data was provided."
            )

        unknown = set(data) - set(SUPPORTED_TIMEFRAMES)

        if unknown:
            raise MultiTimeframeAnalysisError(
                "Unsupported timeframes: "
                + ", ".join(sorted(map(str, unknown)))
            )

        missing = [
            timeframe
            for timeframe in self.required_timeframes
            if timeframe not in data
        ]

        if missing:
            raise MultiTimeframeAnalysisError(
                "Missing required timeframes: "
                + ", ".join(missing)
            )

        analyses = {}

        for timeframe in TIMEFRAME_ORDER:

            if timeframe not in data:
                continue

            candles = _validate_candles(
                data[timeframe],
                timeframe,
                timestamp,
                self.minimum_candles,
            )

            analyses[timeframe] = (
                _calculate_timeframe_analysis(
                    candles,
                    timeframe,
                )
            )

        alignment = _build_alignment(
            analyses
        )

        higher_timeframe_validation = (
            _build_higher_timeframe_validation(
                analyses
            )
        )

        analysis_timestamp = timestamp.isoformat()

        analysis_id = _build_analysis_id(
            analysis_timestamp,
            analyses,
        )

        all_sources = sorted({
            source
            for result in analyses.values()
            for source in result["source"]
        })

        result = {
            "analysis_id": analysis_id,
            "analysis_timestamp": analysis_timestamp,
            "status": "SUCCESS",
            "overall_direction": alignment["direction"],
            "alignment": alignment,
            "higher_timeframe_validation": (
                higher_timeframe_validation
            ),
            "timeframes": analyses,
            "source": all_sources,
            "timeframes_analyzed": list(
                analyses.keys()
            ),
            "parameters": {
                "minimum_candles": self.minimum_candles,
                "required_timeframes": list(
                    self.required_timeframes
                ),
            },
            "limitations": [
                "Trend and momentum calculations are technical estimates.",
                "Support and resistance are rolling reference levels.",
                "Market structure uses recent candle extremes, not confirmed swing-point analysis.",
                "Volume confirmation compares the latest candle with recent average volume.",
                "Timeframe alignment is not a probability of future price movement.",
                "Conflicting timeframes require additional review.",
                "Only validated closed candles are analyzed.",
                "This module does not generate trading orders.",
                "This module does not execute trades.",
            ],
        }

        if analysis_id not in self._known_ids:

            self._history.append(
                json.loads(
                    json.dumps(result)
                )
            )

            self._known_ids.add(
                analysis_id
            )

        return json.loads(
            json.dumps(result)
        )


def analyze_multiple_timeframes(
    data: dict[str, pd.DataFrame],
    *,
    analysis_time: datetime | None = None,
    minimum_candles: int = 20,
    required_timeframes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Convenience function for one-time multi-timeframe analysis."""

    analyzer = MultiTimeframeAnalyzer(
        minimum_candles=minimum_candles,
        required_timeframes=required_timeframes,
    )

    return analyzer.analyze(
        data,
        analysis_time=analysis_time,
    )


if __name__ == "__main__":
    print(
        "Bitcoin multi-timeframe analyzer loaded."
    )