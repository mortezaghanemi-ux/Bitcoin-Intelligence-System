
"""Bitcoin market regime detection."""

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

TRENDING = "TRENDING"
RANGING = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"
TRANSITIONAL = "TRANSITIONAL"

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"


class MarketRegimeError(RuntimeError):
    """Raised when market regime analysis fails."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Normalize analysis timestamp to UTC."""

    if analysis_time is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        timestamp = pd.Timestamp(analysis_time)

    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketRegimeError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise MarketRegimeError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise MarketRegimeError(
            "analysis_time must be timezone-aware UTC."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise MarketRegimeError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_parameters(
    adx_period: int,
    atr_period: int,
    volatility_lookback: int,
    bollinger_period: int,
    minimum_candles: int,
    trending_threshold: float,
    ranging_threshold: float,
    high_volatility_ratio: float,
    low_volatility_ratio: float,
) -> None:
    """Validate regime analysis parameters."""

    integer_parameters = {
        "adx_period": adx_period,
        "atr_period": atr_period,
        "volatility_lookback": volatility_lookback,
        "bollinger_period": bollinger_period,
        "minimum_candles": minimum_candles,
    }

    for name, value in integer_parameters.items():

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise MarketRegimeError(
                f"{name} must be a positive integer."
            )

    numeric_parameters = {
        "trending_threshold": trending_threshold,
        "ranging_threshold": ranging_threshold,
        "high_volatility_ratio": high_volatility_ratio,
        "low_volatility_ratio": low_volatility_ratio,
    }

    for name, value in numeric_parameters.items():

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(value)
        ):
            raise MarketRegimeError(
                f"{name} must be a finite numeric value."
            )

    minimum_required = max(
        adx_period * 2,
        atr_period + volatility_lookback,
        bollinger_period,
    )

    if minimum_candles < minimum_required:
        raise MarketRegimeError(
            "minimum_candles must be at least "
            f"{minimum_required}."
        )

    if not 0 < trending_threshold <= 100:
        raise MarketRegimeError(
            "trending_threshold must be between 0 and 100."
        )

    if (
        ranging_threshold < 0
        or ranging_threshold >= trending_threshold
    ):
        raise MarketRegimeError(
            "ranging_threshold must be non-negative "
            "and smaller than trending_threshold."
        )

    if high_volatility_ratio <= 1:
        raise MarketRegimeError(
            "high_volatility_ratio must be greater than 1."
        )

    if not 0 < low_volatility_ratio < 1:
        raise MarketRegimeError(
            "low_volatility_ratio must be between 0 and 1."
        )

    if low_volatility_ratio >= high_volatility_ratio:
        raise MarketRegimeError(
            "Low volatility ratio must be smaller "
            "than high volatility ratio."
        )


def _validate_candles(
    data: pd.DataFrame,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate closed candles using the central validator."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise MarketRegimeError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise MarketRegimeError(
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
        raise MarketRegimeError(
            f"Candle validation failed: {exc}"
        ) from exc

    if not rejected.empty:
        raise MarketRegimeError(
            "Invalid candles detected. "
            f"Rejected records: {len(rejected)}. "
            f"Validation errors: {report['errors']}."
        )

    if accepted.empty:
        raise MarketRegimeError(
            "No valid candles available for analysis."
        )

    result = accepted.copy(deep=True)

    if not result["open_time"].is_monotonic_increasing:
        raise MarketRegimeError(
            "Candles must be sorted by open_time."
        )

    if not result["close_time"].is_monotonic_increasing:
        raise MarketRegimeError(
            "Candle close_time values must be sorted."
        )

    if result["open_time"].duplicated().any():
        raise MarketRegimeError(
            "Duplicate candle open_time values detected."
        )

    if result["close_time"].duplicated().any():
        raise MarketRegimeError(
            "Duplicate candle close_time values detected."
        )

    valid_sources = result["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise MarketRegimeError(
            "Every candle must have a valid source."
        )

    for column in ("open_time", "close_time"):

        if not isinstance(
            result[column].dtype,
            pd.DatetimeTZDtype,
        ):
            raise MarketRegimeError(
                f"{column} must contain timezone-aware timestamps."
            )

        if str(result[column].dt.tz) != "UTC":
            raise MarketRegimeError(
                f"{column} timestamps must use UTC."
            )

    return result.reset_index(drop=True)


def _calculate_true_range(
    data: pd.DataFrame,
) -> pd.Series:
    """Calculate True Range."""

    previous_close = data["close"].shift(1)

    ranges = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def _calculate_atr(
    data: pd.DataFrame,
    period: int,
) -> pd.Series:
    """Calculate Wilder-style Average True Range."""

    true_range = _calculate_true_range(data)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _calculate_adx(
    data: pd.DataFrame,
    period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, positive DI and negative DI."""

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

    atr = _calculate_atr(
        data,
        period,
    )

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

    safe_atr = atr.replace(0, np.nan)

    positive_di = (
        100 * positive_smoothed / safe_atr
    )

    negative_di = (
        100 * negative_smoothed / safe_atr
    )

    denominator = (
        positive_di + negative_di
    ).replace(0, np.nan)

    dx = (
        100
        * (positive_di - negative_di).abs()
        / denominator
    )

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

    return (
        adx.clip(lower=0, upper=100),
        positive_di.clip(lower=0, upper=100),
        negative_di.clip(lower=0, upper=100),
    )


def _calculate_bollinger_width(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate normalized Bollinger Band width."""

    middle = close.rolling(
        period,
        min_periods=period,
    ).mean()

    deviation = close.rolling(
        period,
        min_periods=period,
    ).std(ddof=0)

    upper = middle + (2 * deviation)
    lower = middle - (2 * deviation)

    safe_middle = middle.replace(0, np.nan)

    return (
        (upper - lower) / safe_middle
    ) * 100


def _classify_volatility(
    current_atr_percent: float,
    historical_median: float,
    high_ratio: float,
    low_ratio: float,
) -> tuple[str, float]:
    """Classify volatility relative to recent history."""

    if (
        not np.isfinite(current_atr_percent)
        or not np.isfinite(historical_median)
        or historical_median <= 0
    ):
        raise MarketRegimeError(
            "Insufficient data for volatility classification."
        )

    ratio = current_atr_percent / historical_median

    if ratio >= high_ratio:
        condition = HIGH_VOLATILITY

    elif ratio <= low_ratio:
        condition = LOW_VOLATILITY

    else:
        condition = "NORMAL_VOLATILITY"

    return condition, float(ratio)


def _classify_regime(
    adx: float,
    volatility_condition: str,
    trending_threshold: float,
    ranging_threshold: float,
) -> str:
    """Classify the primary market regime."""

    if volatility_condition == HIGH_VOLATILITY:
        return HIGH_VOLATILITY

    if volatility_condition == LOW_VOLATILITY:
        return LOW_VOLATILITY

    if adx >= trending_threshold:
        return TRENDING

    if adx <= ranging_threshold:
        return RANGING

    return TRANSITIONAL


def _classify_direction(
    positive_di: float,
    negative_di: float,
) -> str:
    """Determine directional market bias."""

    if positive_di > negative_di:
        return BULLISH

    if negative_di > positive_di:
        return BEARISH

    return NEUTRAL


def analyze_market_regime(
    data: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    volatility_lookback: int = 50,
    bollinger_period: int = 20,
    minimum_candles: int = 64,
    trending_threshold: float = 25.0,
    ranging_threshold: float = 20.0,
    high_volatility_ratio: float = 1.5,
    low_volatility_ratio: float = 0.7,
    analysis_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Detect the current Bitcoin market regime.

    Regimes:
    - TRENDING
    - RANGING
    - HIGH_VOLATILITY
    - LOW_VOLATILITY
    - TRANSITIONAL

    Only validated, closed candles are used.

    Historical volatility is calculated using
    available past and current candles only.

    This module does not generate trading signals.
    It does not execute trades.
    It does not modify input data.
    """

    _validate_parameters(
        adx_period=adx_period,
        atr_period=atr_period,
        volatility_lookback=volatility_lookback,
        bollinger_period=bollinger_period,
        minimum_candles=minimum_candles,
        trending_threshold=trending_threshold,
        ranging_threshold=ranging_threshold,
        high_volatility_ratio=high_volatility_ratio,
        low_volatility_ratio=low_volatility_ratio,
    )

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    candles = _validate_candles(
        data,
        timestamp,
    )

    if len(candles) < minimum_candles:
        raise MarketRegimeError(
            f"At least {minimum_candles} candles are required. "
            f"Received: {len(candles)}."
        )

    # Trend calculations.

    adx_series, positive_di, negative_di = _calculate_adx(
        candles,
        adx_period,
    )

    adx_value = float(adx_series.iloc[-1])
    positive_di_value = float(positive_di.iloc[-1])
    negative_di_value = float(negative_di.iloc[-1])

    if not np.isfinite(adx_value):
        raise MarketRegimeError(
            "Unable to calculate a valid ADX value."
        )

    if (
        not np.isfinite(positive_di_value)
        or not np.isfinite(negative_di_value)
    ):
        raise MarketRegimeError(
            "Unable to calculate valid directional indicators."
        )

    # Volatility calculations.

    atr_series = _calculate_atr(
        candles,
        atr_period,
    )

    atr_value = float(atr_series.iloc[-1])

    current_price = float(
        candles["close"].iloc[-1]
    )

    if (
        not np.isfinite(current_price)
        or current_price <= 0
    ):
        raise MarketRegimeError(
            "Latest closing price must be positive and finite."
        )

    if not np.isfinite(atr_value):
        raise MarketRegimeError(
            "Unable to calculate a valid ATR value."
        )

    atr_percent = (
        atr_value / current_price
    ) * 100

    atr_percent_history = (
        atr_series / candles["close"]
    ) * 100

    historical_window = atr_percent_history.tail(
        volatility_lookback
    )

    historical_median = float(
        historical_window.median()
    )

    volatility_condition, volatility_ratio = (
        _classify_volatility(
            current_atr_percent=atr_percent,
            historical_median=historical_median,
            high_ratio=high_volatility_ratio,
            low_ratio=low_volatility_ratio,
        )
    )

    # Bollinger Band calculations.

    bollinger_width_series = _calculate_bollinger_width(
        candles["close"],
        bollinger_period,
    )

    bollinger_width = float(
        bollinger_width_series.iloc[-1]
    )

    if not np.isfinite(bollinger_width):
        raise MarketRegimeError(
            "Unable to calculate Bollinger Band width."
        )

    # Volume calculations.

    volume_average = float(
        candles["volume"]
        .rolling(
            volatility_lookback,
            min_periods=volatility_lookback,
        )
        .mean()
        .iloc[-1]
    )

    current_volume = float(
        candles["volume"].iloc[-1]
    )

    if volume_average > 0:
        volume_ratio = current_volume / volume_average

    elif current_volume == 0:
        volume_ratio = 1.0

    else:
        volume_ratio = 0.0

    # Final classification.

    direction = _classify_direction(
        positive_di_value,
        negative_di_value,
    )

    regime = _classify_regime(
        adx=adx_value,
        volatility_condition=volatility_condition,
        trending_threshold=trending_threshold,
        ranging_threshold=ranging_threshold,
    )

    sources = sorted(
        candles["source"].astype(str).unique().tolist()
    )

    latest_close_time = candles["close_time"].iloc[-1]

    trend_strength = (
        "STRONG"
        if adx_value >= trending_threshold
        else (
            "WEAK"
            if adx_value <= ranging_threshold
            else "MODERATE"
        )
    )

    return {
        "analysis_timestamp": timestamp.isoformat(),
        "status": "SUCCESS",
        "source": sources,
        "regime": regime,
        "direction": direction,
        "trend_strength": trend_strength,
        "volatility": {
            "condition": volatility_condition,
            "atr": round(atr_value, 8),
            "atr_percent": round(atr_percent, 6),
            "historical_median_atr_percent": round(
                historical_median,
                6,
            ),
            "relative_volatility": round(
                volatility_ratio,
                6,
            ),
            "bollinger_width_percent": round(
                bollinger_width,
                6,
            ),
        },
        "trend": {
            "adx": round(adx_value, 6),
            "positive_di": round(
                positive_di_value,
                6,
            ),
            "negative_di": round(
                negative_di_value,
                6,
            ),
        },
        "volume": {
            "current": current_volume,
            "average": round(volume_average, 8),
            "relative_volume": round(
                volume_ratio,
                6,
            ),
            "above_average": current_volume > volume_average,
        },
        "market_price": current_price,
        "candles_analyzed": len(candles),
        "last_candle_close_time": (
            latest_close_time.isoformat()
        ),
        "parameters": {
            "adx_period": adx_period,
            "atr_period": atr_period,
            "volatility_lookback": volatility_lookback,
            "bollinger_period": bollinger_period,
            "minimum_candles": minimum_candles,
            "trending_threshold": trending_threshold,
            "ranging_threshold": ranging_threshold,
            "high_volatility_ratio": high_volatility_ratio,
            "low_volatility_ratio": low_volatility_ratio,
        },
    }


if __name__ == "__main__":
    print(
        "Bitcoin market regime analysis module loaded."
    )