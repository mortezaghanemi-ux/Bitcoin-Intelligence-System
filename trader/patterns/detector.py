"""Bitcoin candlestick pattern detection engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import math

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

OUTPUT_COLUMNS = (
    "pattern_id",
    "pattern",
    "direction",
    "confidence",
    "timeframe",
    "open_time",
    "close_time",
    "source",
    "detection_time",
)


class PatternDetectionError(RuntimeError):
    """Raised when candlestick pattern detection fails."""


def _validate_utc(value: datetime, field: str) -> None:
    """Validate a timezone-aware UTC datetime."""

    if not isinstance(value, datetime):
        raise ValueError(
            f"{field} must be a datetime instance."
        )

    if value.tzinfo is None:
        raise ValueError(
            f"{field} must be timezone-aware."
        )

    if value.utcoffset() != timedelta(0):
        raise ValueError(
            f"{field} must use UTC timezone."
        )


def _timestamp(value) -> pd.Timestamp:
    """Parse a timestamp and require UTC timezone."""

    try:
        result = pd.Timestamp(value)

        if pd.isna(result):
            raise ValueError("Timestamp is missing.")

        if result.tzinfo is None:
            raise ValueError(
                "Timestamp must be timezone-aware UTC."
            )

        if result.utcoffset() != timedelta(0):
            raise ValueError(
                "Timestamp must use UTC timezone."
            )

        return result.tz_convert("UTC")

    except (TypeError, ValueError, OverflowError) as exc:
        raise PatternDetectionError(
            f"Invalid UTC timestamp: {value}"
        ) from exc


def _validate_timeframe(timeframe: str) -> int:
    """Return timeframe duration in seconds."""

    if not isinstance(timeframe, str):
        raise ValueError(
            "timeframe must be a string."
        )

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    return SUPPORTED_TIMEFRAMES[timeframe]


def _validate_candles(
    data: pd.DataFrame,
    timeframe: str,
    detection_time: datetime,
) -> pd.DataFrame:
    """Validate candle structure, timestamps and OHLC integrity."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise PatternDetectionError(
            "Cannot detect patterns from empty data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise PatternDetectionError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    interval_seconds = _validate_timeframe(
        timeframe
    )

    checked_at = _timestamp(
        detection_time
    )

    result = data.loc[
        :,
        REQUIRED_COLUMNS,
    ].copy(deep=True)

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):

        if (
            not pd.api.types.is_numeric_dtype(
                result[column]
            )
            or pd.api.types.is_bool_dtype(
                result[column]
            )
        ):
            raise PatternDetectionError(
                f"{column} must contain numeric values."
            )

        values = result[column].to_numpy(
            dtype=float
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise PatternDetectionError(
                f"{column} contains invalid numeric values."
            )

    if (result["open"] <= 0).any():
        raise PatternDetectionError(
            "Open prices must be positive."
        )

    if (result["high"] <= 0).any():
        raise PatternDetectionError(
            "High prices must be positive."
        )

    if (result["low"] <= 0).any():
        raise PatternDetectionError(
            "Low prices must be positive."
        )

    if (result["close"] <= 0).any():
        raise PatternDetectionError(
            "Close prices must be positive."
        )

    if (result["volume"] < 0).any():
        raise PatternDetectionError(
            "Volume cannot be negative."
        )

    if (
        result["high"]
        < result[["open", "close", "low"]].max(axis=1)
    ).any():
        raise PatternDetectionError(
            "Invalid candle high price."
        )

    if (
        result["low"]
        > result[["open", "close", "high"]].min(axis=1)
    ).any():
        raise PatternDetectionError(
            "Invalid candle low price."
        )

    result["open_time"] = result[
        "open_time"
    ].map(_timestamp)

    result["close_time"] = result[
        "close_time"
    ].map(_timestamp)

    if (result["close_time"] <= result["open_time"]).any():
        raise PatternDetectionError(
            "Candle close_time must be after open_time."
        )

    durations = (
        result["close_time"]
        - result["open_time"]
    )

    expected_duration = pd.Timedelta(
        seconds=interval_seconds
    )

    if (durations != expected_duration).any():
        raise PatternDetectionError(
            "Candle duration does not match timeframe."
        )

    if (result["close_time"] > checked_at).any():
        raise PatternDetectionError(
            "Unclosed candles are not allowed."
        )

    if result["source"].isna().any():
        raise PatternDetectionError(
            "Candle source cannot be missing."
        )

    result["source"] = (
        result["source"]
        .astype(str)
        .str.strip()
    )

    if (result["source"] == "").any():
        raise PatternDetectionError(
            "Candle source cannot be empty."
        )

    result = result.sort_values(
        "open_time",
        kind="stable",
    ).reset_index(drop=True)

    if result["open_time"].duplicated().any():
        raise PatternDetectionError(
            "Duplicate candle open_time detected."
        )

    return result


def _body(candle) -> float:
    return abs(
        float(candle["close"])
        - float(candle["open"])
    )


def _range(candle) -> float:
    return (
        float(candle["high"])
        - float(candle["low"])
    )


def _upper_wick(candle) -> float:
    return (
        float(candle["high"])
        - max(
            float(candle["open"]),
            float(candle["close"]),
        )
    )


def _lower_wick(candle) -> float:
    return (
        min(
            float(candle["open"]),
            float(candle["close"]),
        )
        - float(candle["low"])
    )


def _bullish(candle) -> bool:
    return (
        float(candle["close"])
        > float(candle["open"])
    )


def _bearish(candle) -> bool:
    return (
        float(candle["close"])
        < float(candle["open"])
    )


def _small_body(candle, threshold: float = 0.10) -> bool:
    candle_range = _range(candle)

    if candle_range <= 0:
        return False

    return (
        _body(candle) / candle_range
        <= threshold
    )


def _add_pattern(
    patterns: list,
    *,
    name: str,
    direction: str,
    confidence: float,
    timeframe: str,
    candle,
    detection_time: pd.Timestamp,
) -> None:
    """Append a validated pattern detection."""

    if direction not in {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }:
        raise PatternDetectionError(
            "Invalid pattern direction."
        )

    if not 0 <= confidence <= 100:
        raise PatternDetectionError(
            "Confidence must be between 0 and 100."
        )

    opened = candle["open_time"]
    closed = candle["close_time"]
    source = str(candle["source"])

    identity = (
        f"{name}|{direction}|{timeframe}|"
        f"{opened.isoformat()}|{source}"
    )

    pattern_id = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    patterns.append({
        "pattern_id": pattern_id,
        "pattern": name,
        "direction": direction,
        "confidence": round(
            float(confidence),
            2,
        ),
        "timeframe": timeframe,
        "open_time": opened,
        "close_time": closed,
        "source": source,
        "detection_time": detection_time,
    })


def _detect_single_candle(
    candles: pd.DataFrame,
    index: int,
    patterns: list,
    timeframe: str,
    detection_time: pd.Timestamp,
) -> None:
    """Detect single-candle patterns."""

    candle = candles.iloc[index]

    body = _body(candle)
    candle_range = _range(candle)

    if candle_range <= 0:
        return

    upper = _upper_wick(candle)
    lower = _lower_wick(candle)

    body_ratio = body / candle_range

    # Doji

    if body_ratio <= 0.10:

        _add_pattern(
            patterns,
            name="Doji",
            direction="NEUTRAL",
            confidence=75,
            timeframe=timeframe,
            candle=candle,
            detection_time=detection_time,
        )

    # Marubozu

    if body_ratio >= 0.90:

        direction = (
            "BULLISH"
            if _bullish(candle)
            else "BEARISH"
        )

        if body > 0:

            _add_pattern(
                patterns,
                name="Marubozu",
                direction=direction,
                confidence=85,
                timeframe=timeframe,
                candle=candle,
                detection_time=detection_time,
            )

    # Hammer

    if (
        lower >= body * 2
        and upper <= body * 0.5
        and body_ratio <= 0.40
    ):

        _add_pattern(
            patterns,
            name="Hammer",
            direction="BULLISH",
            confidence=80,
            timeframe=timeframe,
            candle=candle,
            detection_time=detection_time,
        )

    # Hanging Man

    if (
        index >= 1
        and lower >= body * 2
        and upper <= body * 0.5
        and body_ratio <= 0.40
        and candles.iloc[index - 1]["close"]
        > candles.iloc[index - 1]["open"]
    ):

        _add_pattern(
            patterns,
            name="Hanging Man",
            direction="BEARISH",
            confidence=75,
            timeframe=timeframe,
            candle=candle,
            detection_time=detection_time,
        )

    # Shooting Star / Bearish Pin Bar

    if (
        upper >= body * 2
        and lower <= body * 0.5
        and body_ratio <= 0.40
    ):

        _add_pattern(
            patterns,
            name="Shooting Star",
            direction="BEARISH",
            confidence=80,
            timeframe=timeframe,
            candle=candle,
            detection_time=detection_time,
        )

    # Bullish Pin Bar

    if (
        lower >= body * 2
        and upper <= body * 0.5
        and body_ratio <= 0.40
    ):

        _add_pattern(
            patterns,
            name="Bullish Pin Bar",
            direction="BULLISH",
            confidence=80,
            timeframe=timeframe,
            candle=candle,
            detection_time=detection_time,
        )


def _detect_two_candle(
    candles: pd.DataFrame,
    index: int,
    patterns: list,
    timeframe: str,
    detection_time: pd.Timestamp,
) -> None:
    """Detect two-candle patterns."""

    if index < 1:
        return

    previous = candles.iloc[index - 1]
    current = candles.iloc[index]

    previous_open = float(previous["open"])
    previous_close = float(previous["close"])

    current_open = float(current["open"])
    current_close = float(current["close"])

    previous_body = _body(previous)
    current_body = _body(current)

    # Bullish Engulfing

    if (
        _bearish(previous)
        and _bullish(current)
        and current_open <= previous_close
        and current_close >= previous_open
        and current_body > previous_body
    ):

        _add_pattern(
            patterns,
            name="Bullish Engulfing",
            direction="BULLISH",
            confidence=85,
            timeframe=timeframe,
            candle=current,
            detection_time=detection_time,
        )

    # Bearish Engulfing

    if (
        _bullish(previous)
        and _bearish(current)
        and current_open >= previous_close
        and current_close <= previous_open
        and current_body > previous_body
    ):

        _add_pattern(
            patterns,
            name="Bearish Engulfing",
            direction="BEARISH",
            confidence=85,
            timeframe=timeframe,
            candle=current,
            detection_time=detection_time,
        )

    # Bullish Harami

    if (
        _bearish(previous)
        and _bullish(current)
        and current_open >= min(previous_open, previous_close)
        and current_close <= max(previous_open, previous_close)
        and current_body < previous_body
    ):

        _add_pattern(
            patterns,
            name="Bullish Harami",
            direction="BULLISH",
            confidence=75,
            timeframe=timeframe,
            candle=current,
            detection_time=detection_time,
        )

    # Bearish Harami

    if (
        _bullish(previous)
        and _bearish(current)
        and current_open >= min(previous_open, previous_close)
        and current_close <= max(previous_open, previous_close)
        and current_body < previous_body
    ):

        _add_pattern(
            patterns,
            name="Bearish Harami",
            direction="BEARISH",
            confidence=75,
            timeframe=timeframe,
            candle=current,
            detection_time=detection_time,
        )

    # Tweezer Bottom

    previous_low = float(previous["low"])
    current_low = float(current["low"])

    tolerance = max(
        previous_low,
        current_low,
    ) * 0.001

    if (
        _bearish(previous)
        and _bullish(current)
        and abs(previous_low - current_low) <= tolerance
    ):

        _add_pattern(
            patterns,
            name="Tweezer Bottom",
            direction="BULLISH",
            confidence=78,
            timeframe=timeframe,
            candle=current,
            detection_time=detection_time,
        )

    # Tweezer Top

    previous_high = float(previous["high"])
    current_high = float(current["high"])

    tolerance = max(
        previous_high,
        current_high,
    ) * 0.001

    if (
        _bullish(previous)
        and _bearish(current)
        and abs(previous_high - current_high) <= tolerance
    ):

        _add_pattern(
            patterns,
            name="Tweezer Top",
            direction="BEARISH",
            confidence=78,
            timeframe=timeframe,
            candle=current,
            detection_time=detection_time,
        )


def _detect_three_candle(
    candles: pd.DataFrame,
    index: int,
    patterns: list,
    timeframe: str,
    detection_time: pd.Timestamp,
) -> None:
    """Detect three-candle patterns."""

    if index < 2:
        return

    first = candles.iloc[index - 2]
    second = candles.iloc[index - 1]
    third = candles.iloc[index]

    first_midpoint = (
        float(first["open"])
        + float(first["close"])
    ) / 2

    # Morning Star

    if (
        _bearish(first)
        and _small_body(second, 0.35)
        and _bullish(third)
        and float(third["close"]) > first_midpoint
    ):

        _add_pattern(
            patterns,
            name="Morning Star",
            direction="BULLISH",
            confidence=85,
            timeframe=timeframe,
            candle=third,
            detection_time=detection_time,
        )

    # Evening Star

    if (
        _bullish(first)
        and _small_body(second, 0.35)
        and _bearish(third)
        and float(third["close"]) < first_midpoint
    ):

        _add_pattern(
            patterns,
            name="Evening Star",
            direction="BEARISH",
            confidence=85,
            timeframe=timeframe,
            candle=third,
            detection_time=detection_time,
        )

    # Three White Soldiers

    soldiers = [
        first,
        second,
        third,
    ]

    if (
        all(_bullish(item) for item in soldiers)
        and float(second["close"]) > float(first["close"])
        and float(third["close"]) > float(second["close"])
        and float(second["open"]) > float(first["open"])
        and float(third["open"]) > float(second["open"])
        and all(
            _upper_wick(item) <= _body(item) * 0.35
            for item in soldiers
        )
    ):

        _add_pattern(
            patterns,
            name="Three White Soldiers",
            direction="BULLISH",
            confidence=90,
            timeframe=timeframe,
            candle=third,
            detection_time=detection_time,
        )

    # Three Black Crows

    if (
        all(_bearish(item) for item in soldiers)
        and float(second["close"]) < float(first["close"])
        and float(third["close"]) < float(second["close"])
        and float(second["open"]) < float(first["open"])
        and float(third["open"]) < float(second["open"])
        and all(
            _lower_wick(item) <= _body(item) * 0.35
            for item in soldiers
        )
    ):

        _add_pattern(
            patterns,
            name="Three Black Crows",
            direction="BEARISH",
            confidence=90,
            timeframe=timeframe,
            candle=third,
            detection_time=detection_time,
        )


def _detect_five_candle(
    candles: pd.DataFrame,
    index: int,
    patterns: list,
    timeframe: str,
    detection_time: pd.Timestamp,
) -> None:
    """Detect five-candle continuation patterns."""

    if index < 4:
        return

    group = [
        candles.iloc[position]
        for position in range(index - 4, index + 1)
    ]

    first, second, third, fourth, fifth = group

    middle_candles = [
        second,
        third,
        fourth,
    ]

    # Rising Three Methods

    if (
        _bullish(first)
        and _bullish(fifth)
        and _body(fifth) > _body(first) * 0.5
        and all(
            _bearish(item)
            for item in middle_candles
        )
        and all(
            float(item["high"]) <= float(first["high"])
            and float(item["low"]) >= float(first["low"])
            for item in middle_candles
        )
        and float(fifth["close"]) > float(first["high"])
    ):

        _add_pattern(
            patterns,
            name="Rising Three Methods",
            direction="BULLISH",
            confidence=88,
            timeframe=timeframe,
            candle=fifth,
            detection_time=detection_time,
        )

    # Falling Three Methods

    if (
        _bearish(first)
        and _bearish(fifth)
        and _body(fifth) > _body(first) * 0.5
        and all(
            _bullish(item)
            for item in middle_candles
        )
        and all(
            float(item["high"]) <= float(first["high"])
            and float(item["low"]) >= float(first["low"])
            for item in middle_candles
        )
        and float(fifth["close"]) < float(first["low"])
    ):

        _add_pattern(
            patterns,
            name="Falling Three Methods",
            direction="BEARISH",
            confidence=88,
            timeframe=timeframe,
            candle=fifth,
            detection_time=detection_time,
        )


class CandlestickPatternDetector:
    """
    Detect candlestick patterns using closed OHLCV candles.

    Detection history is preserved in memory.
    Existing pattern IDs are never inserted twice.
    """

    def __init__(self, timeframe: str):
        self.timeframe = timeframe

        _validate_timeframe(timeframe)

        self._history = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    @property
    def history(self) -> pd.DataFrame:
        """Return a copy of detection history."""

        return self._history.copy(
            deep=True
        )

    def clear(self) -> None:
        """Clear in-memory detection history."""

        self._history = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    def detect(
        self,
        data: pd.DataFrame,
        *,
        detection_time: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Detect patterns across supplied closed candles.

        Each pattern is identified by its final candle.
        No future candle information is used.
        """

        checked_at = (
            detection_time
            if detection_time is not None
            else datetime.now(timezone.utc)
        )

        _validate_utc(
            checked_at,
            "detection_time",
        )

        checked_at = _timestamp(
            checked_at
        )

        candles = _validate_candles(
            data,
            self.timeframe,
            checked_at.to_pydatetime(),
        )

        patterns = []

        for index in range(len(candles)):

            _detect_single_candle(
                candles,
                index,
                patterns,
                self.timeframe,
                checked_at,
            )

            _detect_two_candle(
                candles,
                index,
                patterns,
                self.timeframe,
                checked_at,
            )

            _detect_three_candle(
                candles,
                index,
                patterns,
                self.timeframe,
                checked_at,
            )

            _detect_five_candle(
                candles,
                index,
                patterns,
                self.timeframe,
                checked_at,
            )

        detected = pd.DataFrame(
            patterns,
            columns=OUTPUT_COLUMNS,
        )

        if detected.empty:
            return detected

        # Remove duplicate IDs within this detection batch.

        detected = detected.drop_duplicates(
            subset=["pattern_id"],
            keep="first",
        )

        # Preserve historical detections.

        if not self._history.empty:

            known_ids = set(
                self._history["pattern_id"]
            )

            detected = detected.loc[
                ~detected["pattern_id"].isin(
                    known_ids
                )
            ]

        if not detected.empty:

            self._history = pd.concat(
                [
                    self._history,
                    detected,
                ],
                ignore_index=True,
            )

        return detected.reset_index(
            drop=True
        )


def detect_patterns(
    data: pd.DataFrame,
    timeframe: str,
    *,
    detection_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Convenience function for one-time pattern detection.

    For persistent in-memory detection history,
    use CandlestickPatternDetector.
    """

    detector = CandlestickPatternDetector(
        timeframe
    )

    return detector.detect(
        data,
        detection_time=detection_time,
    )