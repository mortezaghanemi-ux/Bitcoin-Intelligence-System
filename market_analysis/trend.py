"""Bitcoin market trend analysis."""

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

TREND_BULLISH = "BULLISH"
TREND_BEARISH = "BEARISH"
TREND_NEUTRAL = "NEUTRAL"

STRENGTH_STRONG = "STRONG"
STRENGTH_MODERATE = "MODERATE"
STRENGTH_WEAK = "WEAK"


class TrendAnalysisError(RuntimeError):
    """Raised when trend analysis cannot be completed."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Validate and normalize the UTC analysis timestamp."""

    if analysis_time is None:
        return pd.Timestamp(
            datetime.now(timezone.utc)
        )

    try:
        timestamp = pd.Timestamp(analysis_time)

    except (TypeError, ValueError, OverflowError) as exc:
        raise TrendAnalysisError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise TrendAnalysisError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise TrendAnalysisError(
            "analysis_time must be timezone-aware UTC."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise TrendAnalysisError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_parameters(
    fast_period: int,
    slow_period: int,
    adx_period: int,
    minimum_candles: int,
) -> None:
    """Validate trend analysis parameters."""

    parameters = {
        "fast_period": fast_period,
        "slow_period": slow_period,
        "adx_period": adx_period,
        "minimum_candles": minimum_candles,
    }

    for name, value in parameters.items():

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise TrendAnalysisError(
                f"{name} must be a positive integer."
            )

    if fast_period >= slow_period:
        raise TrendAnalysisError(
            "fast_period must be smaller than slow_period."
        )

    if minimum_candles < slow_period:
        raise TrendAnalysisError(
            "minimum_candles cannot be smaller than slow_period."
        )

    if minimum_candles < adx_period * 2:
        raise TrendAnalysisError(
            "minimum_candles must be at least twice adx_period."
        )


def _validate_candles(
    data: pd.DataFrame,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate candles using the project's central validator."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise TrendAnalysisError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise TrendAnalysisError(
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
        raise TrendAnalysisError(
            f"Candle validation failed: {exc}"
        ) from exc

    if not rejected.empty:
        raise TrendAnalysisError(
            "Invalid candles detected. "
            f"Rejected records: {len(rejected)}. "
            f"Validation errors: {report['errors']}."
        )

    if accepted.empty:
        raise TrendAnalysisError(
            "No valid candles available for analysis."
        )

    result = accepted.copy(deep=True)

    # Ensure chronological ordering.
    if not result["open_time"].is_monotonic_increasing:
        raise TrendAnalysisError(
            "Candles must be sorted by open_time."
        )

    if not result["close_time"].is_monotonic_increasing:
        raise TrendAnalysisError(
            "Candle close_time values must be sorted."
        )

    # A single chronological candle series cannot contain
    # multiple records for the same opening or closing time.
    if result["open_time"].duplicated().any():
        raise TrendAnalysisError(
            "Duplicate candle open_time values detected."
        )

    if result["close_time"].duplicated().any():
        raise TrendAnalysisError(
            "Duplicate candle close_time values detected."
        )

    # Validate source identifiers.
    valid_sources = result["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise TrendAnalysisError(
            "Every candle must have a valid source."
        )

    # Ensure all timestamps are UTC.
    for column in ("open_time", "close_time"):

        if not isinstance(
            result[column].dtype,
            pd.DatetimeTZDtype,
        ):
            raise TrendAnalysisError(
                f"{column} must contain timezone-aware timestamps."
            )

        if str(result[column].dt.tz) != "UTC":
            raise TrendAnalysisError(
                f"{column} timestamps must use UTC."
            )

    return result.reset_index(drop=True)


def _calculate_adx(
    data: pd.DataFrame,
    period: int,
) -> pd.Series:
    """Calculate Average Directional Index."""

    upward = data["high"].diff()
    downward = -data["low"].diff()

    positive_dm = upward.where(
        (upward > downward) & (upward > 0),
        0.0,
    )

    negative_dm = downward.where(
        (downward > upward) & (downward > 0),
        0.0,
    )

    previous_close = data["close"].shift(1)

    ranges = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    true_range = ranges.max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    positive_smoothed = positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    negative_smoothed = negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    positive_di = (
        100
        * positive_smoothed
        / atr.replace(0, np.nan)
    )

    negative_di = (
        100
        * negative_smoothed
        / atr.replace(0, np.nan)
    )

    denominator = (
        positive_di + negative_di
    ).replace(0, np.nan)

    dx = (
        100
        * (positive_di - negative_di).abs()
        / denominator
    )

    # A valid market with no directional movement
    # has zero directional strength.
    no_direction = (
        atr.notna()
        & positive_di.fillna(0).eq(0)
        & negative_di.fillna(0).eq(0)
    )

    dx = dx.mask(
        no_direction,
        0.0,
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx


def _calculate_momentum(
    close: pd.Series,
    period: int = 5,
) -> float:
    """Calculate recent price momentum percentage."""

    if len(close) <= period:
        raise TrendAnalysisError(
            "Insufficient candles for momentum calculation."
        )

    previous = float(
        close.iloc[-period - 1]
    )

    current = float(
        close.iloc[-1]
    )

    if previous <= 0:
        raise TrendAnalysisError(
            "Invalid previous price for momentum calculation."
        )

    return float(
        ((current - previous) / previous) * 100
    )


def _calculate_ma_structure(
    data: pd.DataFrame,
    fast_period: int,
    slow_period: int,
) -> dict[str, Any]:
    """Calculate exponential moving average structure."""

    close = data["close"]

    fast_ma = close.ewm(
        span=fast_period,
        adjust=False,
        min_periods=fast_period,
    ).mean()

    slow_ma = close.ewm(
        span=slow_period,
        adjust=False,
        min_periods=slow_period,
    ).mean()

    current_price = float(
        close.iloc[-1]
    )

    fast_value = float(
        fast_ma.iloc[-1]
    )

    slow_value = float(
        slow_ma.iloc[-1]
    )

    if not np.isfinite(fast_value):
        raise TrendAnalysisError(
            "Insufficient data for fast EMA."
        )

    if not np.isfinite(slow_value):
        raise TrendAnalysisError(
            "Insufficient data for slow EMA."
        )

    if fast_value > slow_value:
        alignment = TREND_BULLISH

    elif fast_value < slow_value:
        alignment = TREND_BEARISH

    else:
        alignment = TREND_NEUTRAL

    return {
        "fast_ma": fast_value,
        "slow_ma": slow_value,
        "price": current_price,
        "alignment": alignment,
        "price_above_fast": current_price > fast_value,
        "price_above_slow": current_price > slow_value,
        "fast_above_slow": fast_value > slow_value,
    }


def _classify_strength(
    adx: float,
) -> str:
    """Classify trend strength using ADX."""

    if adx >= 25:
        return STRENGTH_STRONG

    if adx >= 20:
        return STRENGTH_MODERATE

    return STRENGTH_WEAK


def _calculate_trend_score(
    structure: dict[str, Any],
    momentum: float,
    adx: float,
) -> float:
    """Calculate directional trend score."""

    score = 0.0

    if structure["price_above_fast"]:
        score += 20
    else:
        score -= 20

    if structure["price_above_slow"]:
        score += 25
    else:
        score -= 25

    if structure["fast_above_slow"]:
        score += 25
    else:
        score -= 25

    if momentum > 0:
        score += min(
            abs(momentum) * 5,
            20,
        )

    elif momentum < 0:
        score -= min(
            abs(momentum) * 5,
            20,
        )

    if adx < 20:
        score *= 0.7

    return float(
        np.clip(score, -100, 100)
    )


def _classify_trend(
    score: float,
    threshold: float = 20.0,
) -> str:
    """Classify the overall market trend."""

    if score >= threshold:
        return TREND_BULLISH

    if score <= -threshold:
        return TREND_BEARISH

    return TREND_NEUTRAL


def _calculate_confidence(
    score: float,
    adx: float,
) -> float:
    """Calculate a heuristic trend confidence score."""

    directional_strength = abs(score)

    adx_strength = min(
        max(adx, 0.0),
        50.0,
    )

    confidence = (
        directional_strength * 0.7
        + adx_strength * 0.6
    )

    return float(
        np.clip(confidence, 0, 100)
    )


def _detect_reversal_warning(
    data: pd.DataFrame,
    fast_period: int,
    slow_period: int,
) -> dict[str, Any]:
    """Detect a moving-average crossover warning."""

    close = data["close"]

    fast_ma = close.ewm(
        span=fast_period,
        adjust=False,
        min_periods=fast_period,
    ).mean()

    slow_ma = close.ewm(
        span=slow_period,
        adjust=False,
        min_periods=slow_period,
    ).mean()

    if len(data) < 2:
        return {
            "detected": False,
            "direction": None,
        }

    values = (
        fast_ma.iloc[-2],
        slow_ma.iloc[-2],
        fast_ma.iloc[-1],
        slow_ma.iloc[-1],
    )

    if not all(
        np.isfinite(value)
        for value in values
    ):
        return {
            "detected": False,
            "direction": None,
        }

    (
        previous_fast,
        previous_slow,
        current_fast,
        current_slow,
    ) = values

    bearish_cross = (
        previous_fast >= previous_slow
        and current_fast < current_slow
    )

    bullish_cross = (
        previous_fast <= previous_slow
        and current_fast > current_slow
    )

    if bearish_cross:
        return {
            "detected": True,
            "direction": TREND_BEARISH,
        }

    if bullish_cross:
        return {
            "detected": True,
            "direction": TREND_BULLISH,
        }

    return {
        "detected": False,
        "direction": None,
    }


def analyze_trend(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    adx_period: int = 14,
    minimum_candles: int = 50,
    analysis_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Analyze Bitcoin market trend.

    Parameters
    ----------
    data:
        OHLCV candle data with UTC timestamps and source.
    fast_period:
        Fast EMA period.
    slow_period:
        Slow EMA period.
    adx_period:
        ADX calculation period.
    minimum_candles:
        Minimum number of candles required.
    analysis_time:
        Optional timezone-aware UTC analysis timestamp.

    Returns
    -------
    dict
        Trend classification, strength, confidence,
        momentum, moving averages and reversal warnings.

    Notes
    -----
    - Uses the central project candle validator.
    - Uses closed candles only.
    - Uses historical information only.
    - Does not generate trading signals.
    - Does not execute trades.
    - Does not modify the original DataFrame.
    """

    _validate_parameters(
        fast_period=fast_period,
        slow_period=slow_period,
        adx_period=adx_period,
        minimum_candles=minimum_candles,
    )

    analysis_timestamp = _normalize_analysis_time(
        analysis_time
    )

    candles = _validate_candles(
        data,
        analysis_timestamp,
    )

    if len(candles) < minimum_candles:
        raise TrendAnalysisError(
            f"At least {minimum_candles} valid candles are required. "
            f"Received: {len(candles)}."
        )

    structure = _calculate_ma_structure(
        candles,
        fast_period,
        slow_period,
    )

    adx_series = _calculate_adx(
        candles,
        adx_period,
    )

    adx_value = float(
        adx_series.iloc[-1]
    )

    if not np.isfinite(adx_value):
        raise TrendAnalysisError(
            "Insufficient data for ADX calculation."
        )

    momentum = _calculate_momentum(
        candles["close"]
    )

    score = _calculate_trend_score(
        structure=structure,
        momentum=momentum,
        adx=adx_value,
    )

    trend = _classify_trend(score)

    strength = _classify_strength(
        adx_value
    )

    confidence = _calculate_confidence(
        score,
        adx_value,
    )

    reversal = _detect_reversal_warning(
        candles,
        fast_period,
        slow_period,
    )

    sources = sorted(
        candles["source"].astype(str).unique().tolist()
    )

    latest_close_time = candles["close_time"].iloc[-1]

    return {
        "analysis_timestamp": analysis_timestamp.isoformat(),
        "source": sources,
        "status": "SUCCESS",
        "trend": trend,
        "trend_strength": strength,
        "trend_score": round(score, 4),
        "confidence": round(confidence, 4),
        "adx": round(adx_value, 4),
        "momentum_percent": round(momentum, 6),
        "moving_averages": {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "fast_value": round(
                structure["fast_ma"],
                8,
            ),
            "slow_value": round(
                structure["slow_ma"],
                8,
            ),
            "alignment": structure["alignment"],
            "price_above_fast": structure["price_above_fast"],
            "price_above_slow": structure["price_above_slow"],
            "fast_above_slow": structure["fast_above_slow"],
        },
        "reversal_warning": reversal,
        "market_price": structure["price"],
        "last_candle_close_time": latest_close_time.isoformat(),
        "candles_analyzed": len(candles),
        "parameters": {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "adx_period": adx_period,
            "minimum_candles": minimum_candles,
        },
    }


if __name__ == "__main__":
    print(
        "Bitcoin market trend analysis module loaded."
    )