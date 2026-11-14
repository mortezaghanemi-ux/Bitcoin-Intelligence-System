"""Technical indicator calculations for validated Bitcoin market data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "indicators.yaml"

REQUIRED_COLUMNS = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class IndicatorCalculationError(RuntimeError):
    """Raised when indicator calculation cannot be completed."""


def _load_config(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load indicator configuration."""

    if config is not None:

        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        if "indicators" in config:
            config = config["indicators"]

        if not isinstance(config, dict):
            raise IndicatorCalculationError(
                "Invalid indicator configuration."
            )

        return config

    try:

        with DEFAULT_CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            loaded = yaml.safe_load(file)

    except (OSError, yaml.YAMLError) as exc:

        raise IndicatorCalculationError(
            f"Cannot load indicator configuration: {exc}"
        ) from exc

    if not isinstance(loaded, dict):

        raise IndicatorCalculationError(
            "Indicator configuration must be a mapping."
        )

    indicators = loaded.get("indicators")

    if not isinstance(indicators, dict):

        raise IndicatorCalculationError(
            "Missing indicators configuration."
        )

    return indicators


def _validate_candles(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the structure and integrity of candle data."""

    if not isinstance(data, pd.DataFrame):

        raise TypeError(
            "data must be a pandas DataFrame"
        )

    if data.empty:

        raise IndicatorCalculationError(
            "Cannot calculate indicators from empty data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:

        raise IndicatorCalculationError(
            "Missing candle columns: "
            + ", ".join(missing)
        )

    result = data.copy(deep=True)

    for column in NUMERIC_COLUMNS:

        if (
            not pd.api.types.is_numeric_dtype(
                result[column]
            )
            or pd.api.types.is_bool_dtype(
                result[column]
            )
        ):

            raise IndicatorCalculationError(
                f"{column} must contain numeric values."
            )

        values = result[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():

            raise IndicatorCalculationError(
                f"{column} contains non-finite values."
            )

    if (result["volume"] < 0).any():

        raise IndicatorCalculationError(
            "Volume cannot be negative."
        )

    if (
        (result["open"] <= 0).any()
        or (result["high"] <= 0).any()
        or (result["low"] <= 0).any()
        or (result["close"] <= 0).any()
    ):

        raise IndicatorCalculationError(
            "Prices must be greater than zero."
        )

    invalid_ohlc = (
        (result["high"] < result["low"])
        | (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():

        raise IndicatorCalculationError(
            "Invalid OHLC candle relationships detected."
        )

    for column in ("open_time", "close_time"):

        try:

            timestamps = pd.to_datetime(
                result[column],
                errors="raise",
                utc=False,
            )

        except (TypeError, ValueError) as exc:

            raise IndicatorCalculationError(
                f"Invalid {column} timestamps."
            ) from exc

        if not isinstance(
            timestamps.dtype,
            pd.DatetimeTZDtype,
        ):

            raise IndicatorCalculationError(
                f"{column} must contain timezone-aware timestamps."
            )

        if str(timestamps.dt.tz) != "UTC":

            raise IndicatorCalculationError(
                f"{column} timestamps must use UTC."
            )

        result[column] = timestamps

    if (
        result["close_time"] <= result["open_time"]
    ).any():

        raise IndicatorCalculationError(
            "Candle close_time must be after open_time."
        )

    if result["open_time"].duplicated().any():

        raise IndicatorCalculationError(
            "Duplicate candle timestamps detected."
        )

    if not result["open_time"].is_monotonic_increasing:

        raise IndicatorCalculationError(
            "Candles must be sorted by open_time."
        )

    return result.reset_index(drop=True)


def _period(value: Any, name: str) -> int:
    """Validate a positive integer indicator period."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):

        raise IndicatorCalculationError(
            f"{name} must be a positive integer."
        )

    return value


def _enabled(section: dict, name: str) -> bool:
    """Return whether an indicator is enabled."""

    settings = section.get(name, {})

    if not isinstance(settings, dict):

        raise IndicatorCalculationError(
            f"Invalid configuration for {name}."
        )

    enabled = settings.get("enabled", True)

    if not isinstance(enabled, bool):

        raise IndicatorCalculationError(
            f"{name}.enabled must be boolean."
        )

    return enabled


def _settings(section: dict, name: str) -> dict:
    """Get and validate an indicator configuration section."""

    value = section.get(name, {})

    if not isinstance(value, dict):

        raise IndicatorCalculationError(
            f"Invalid configuration for {name}."
        )

    return value


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate exponential moving average."""

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate simple moving average."""

    return series.rolling(
        window=period,
        min_periods=period,
    ).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index."""

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain / average_loss.replace(0, np.nan)
    )

    result = 100 - (
        100 / (1 + relative_strength)
    )

    result = result.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )

    result = result.mask(
        (average_gain == 0) & (average_loss > 0),
        0.0,
    )

    result = result.mask(
        (average_gain == 0) & (average_loss == 0),
        50.0,
    )

    return result


def _true_range(data: pd.DataFrame) -> pd.Series:
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


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range."""

    return _true_range(data).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _adx(
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

    atr = _atr(data, period)

    smoothed_positive = positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_negative = negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    positive_di = (
        100 * smoothed_positive / atr.replace(0, np.nan)
    )

    negative_di = (
        100 * smoothed_negative / atr.replace(0, np.nan)
    )

    denominator = (
        positive_di + negative_di
    ).replace(0, np.nan)

    dx = (
        100
        * (positive_di - negative_di).abs()
        / denominator
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx, positive_di, negative_di


def _macd(
    series: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, signal and histogram."""

    if fast >= slow:

        raise IndicatorCalculationError(
            "MACD fast period must be smaller than slow period."
        )

    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal,
    ).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def _stochastic(
    data: pd.DataFrame,
    k_period: int,
    d_period: int,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""

    lowest = data["low"].rolling(
        k_period,
        min_periods=k_period,
    ).min()

    highest = data["high"].rolling(
        k_period,
        min_periods=k_period,
    ).max()

    denominator = (highest - lowest).replace(
        0,
        np.nan,
    )

    k_line = (
        100
        * (data["close"] - lowest)
        / denominator
    )

    d_line = k_line.rolling(
        d_period,
        min_periods=d_period,
    ).mean()

    return k_line, d_line


def _cci(
    data: pd.DataFrame,
    period: int,
) -> pd.Series:
    """Calculate Commodity Channel Index."""

    typical_price = (
        data["high"]
        + data["low"]
        + data["close"]
    ) / 3

    average = typical_price.rolling(
        period,
        min_periods=period,
    ).mean()

    mean_deviation = typical_price.rolling(
        period,
        min_periods=period,
    ).apply(
        lambda values: np.mean(
            np.abs(values - np.mean(values))
        ),
        raw=True,
    )

    return (
        (typical_price - average)
        / (0.015 * mean_deviation.replace(0, np.nan))
    )


def _williams_r(
    data: pd.DataFrame,
    period: int,
) -> pd.Series:
    """Calculate Williams %R."""

    highest = data["high"].rolling(
        period,
        min_periods=period,
    ).max()

    lowest = data["low"].rolling(
        period,
        min_periods=period,
    ).min()

    denominator = (highest - lowest).replace(
        0,
        np.nan,
    )

    return (
        -100
        * (highest - data["close"])
        / denominator
    )


def _roc(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate Rate of Change."""

    previous = series.shift(period).replace(
        0,
        np.nan,
    )

    return (
        (series - previous)
        / previous
        * 100
    )


def _bollinger(
    series: pd.Series,
    period: int,
    deviations: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""

    middle = _sma(series, period)

    standard_deviation = series.rolling(
        period,
        min_periods=period,
    ).std(ddof=0)

    upper = middle + deviations * standard_deviation
    lower = middle - deviations * standard_deviation

    return upper, middle, lower


def _obv(data: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""

    direction = np.sign(
        data["close"].diff()
    ).fillna(0)

    return (
        direction * data["volume"]
    ).cumsum()


def _mfi(
    data: pd.DataFrame,
    period: int,
) -> pd.Series:
    """Calculate Money Flow Index."""

    typical_price = (
        data["high"]
        + data["low"]
        + data["close"]
    ) / 3

    raw_money_flow = (
        typical_price * data["volume"]
    )

    direction = typical_price.diff()

    positive_flow = raw_money_flow.where(
        direction > 0,
        0.0,
    )

    negative_flow = raw_money_flow.where(
        direction < 0,
        0.0,
    )

    positive_sum = positive_flow.rolling(
        period,
        min_periods=period,
    ).sum()

    negative_sum = negative_flow.rolling(
        period,
        min_periods=period,
    ).sum()

    ratio = (
        positive_sum
        / negative_sum.replace(0, np.nan)
    )

    result = 100 - (
        100 / (1 + ratio)
    )

    result = result.mask(
        (negative_sum == 0) & (positive_sum > 0),
        100.0,
    )

    result = result.mask(
        (positive_sum == 0) & (negative_sum > 0),
        0.0,
    )

    result = result.mask(
        (positive_sum == 0) & (negative_sum == 0),
        50.0,
    )

    return result


def _vwap(data: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP for the supplied dataset."""

    typical_price = (
        data["high"]
        + data["low"]
        + data["close"]
    ) / 3

    money_flow = (
        typical_price * data["volume"]
    )

    cumulative_volume = data["volume"].cumsum()

    return (
        money_flow.cumsum()
        / cumulative_volume.replace(0, np.nan)
    )


def _parabolic_sar(
    data: pd.DataFrame,
    step: float,
    maximum: float,
) -> pd.Series:
    """Calculate Parabolic SAR without future candle data."""

    if not 0 < step <= maximum <= 1:

        raise IndicatorCalculationError(
            "Invalid Parabolic SAR parameters."
        )

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)

    size = len(data)

    result = np.full(size, np.nan)

    if size < 2:
        return pd.Series(result, index=data.index)

    bullish = data["close"].iloc[1] >= data["close"].iloc[0]

    extreme_point = high[0] if bullish else low[0]

    sar = low[0] if bullish else high[0]

    acceleration = step

    for index in range(1, size):

        current_sar = sar + acceleration * (
            extreme_point - sar
        )

        if bullish:

            current_sar = min(
                current_sar,
                low[index - 1],
            )

            if index >= 2:

                current_sar = min(
                    current_sar,
                    low[index - 2],
                )

            if low[index] < current_sar:

                bullish = False
                current_sar = extreme_point
                extreme_point = low[index]
                acceleration = step

            else:

                if high[index] > extreme_point:

                    extreme_point = high[index]

                    acceleration = min(
                        acceleration + step,
                        maximum,
                    )

        else:

            current_sar = max(
                current_sar,
                high[index - 1],
            )

            if index >= 2:

                current_sar = max(
                    current_sar,
                    high[index - 2],
                )

            if high[index] > current_sar:

                bullish = True
                current_sar = extreme_point
                extreme_point = high[index]
                acceleration = step

            else:

                if low[index] < extreme_point:

                    extreme_point = low[index]

                    acceleration = min(
                        acceleration + step,
                        maximum,
                    )

        sar = current_sar
        result[index] = sar

    return pd.Series(
        result,
        index=data.index,
    )


def calculate_indicators(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate configured technical indicators.

    Parameters
    ----------
    data:
        Validated, UTC, chronological OHLCV candle data.
    config:
        Optional indicator configuration mapping.
        When omitted, config/indicators.yaml is loaded.

    Returns
    -------
    pandas.DataFrame
        A copy of the input data with calculated indicator columns.

    Notes
    -----
    - Input data is never modified.
    - Rolling warm-up periods produce NaN values.
    - No future candle information is used.
    - Ichimoku spans are not shifted into the future.
    - VWAP is cumulative over the supplied dataset.
    """

    candles = _validate_candles(data)

    settings = _load_config(config)

    if not isinstance(settings.get("enabled"), bool):

        raise IndicatorCalculationError(
            "indicators.enabled must be boolean."
        )

    if not settings["enabled"]:
        return candles

    result = candles.copy(deep=True)

    close = candles["close"]

    high = candles["high"]

    low = candles["low"]

    # Trend indicators

    trend = _settings(settings, "trend")

    if _enabled(trend, "ema"):

        ema_settings = _settings(trend, "ema")

        periods = ema_settings.get(
            "periods",
            [20, 50, 100, 200],
        )

        if not isinstance(periods, list) or not periods:

            raise IndicatorCalculationError(
                "EMA periods must be a non-empty list."
            )

        for value in periods:

            period = _period(value, "EMA period")

            result[f"ema_{period}"] = _ema(
                close,
                period,
            )

    if _enabled(trend, "sma"):

        sma_settings = _settings(trend, "sma")

        periods = sma_settings.get(
            "periods",
            [20, 50, 100, 200],
        )

        if not isinstance(periods, list) or not periods:

            raise IndicatorCalculationError(
                "SMA periods must be a non-empty list."
            )

        for value in periods:

            period = _period(value, "SMA period")

            result[f"sma_{period}"] = _sma(
                close,
                period,
            )

    if _enabled(trend, "adx"):

        period = _period(
            _settings(trend, "adx").get("period", 14),
            "ADX period",
        )

        adx, positive_di, negative_di = _adx(
            candles,
            period,
        )

        result[f"adx_{period}"] = adx
        result[f"plus_di_{period}"] = positive_di
        result[f"minus_di_{period}"] = negative_di

    if _enabled(trend, "ichimoku"):

        ichimoku = _settings(trend, "ichimoku")

        tenkan_period = _period(
            ichimoku.get("tenkan", 9),
            "Ichimoku tenkan",
        )

        kijun_period = _period(
            ichimoku.get("kijun", 26),
            "Ichimoku kijun",
        )

        senkou_period = _period(
            ichimoku.get("senkou_b", 52),
            "Ichimoku senkou_b",
        )

        tenkan_high = high.rolling(
            tenkan_period,
            min_periods=tenkan_period,
        ).max()

        tenkan_low = low.rolling(
            tenkan_period,
            min_periods=tenkan_period,
        ).min()

        kijun_high = high.rolling(
            kijun_period,
            min_periods=kijun_period,
        ).max()

        kijun_low = low.rolling(
            kijun_period,
            min_periods=kijun_period,
        ).min()

        senkou_high = high.rolling(
            senkou_period,
            min_periods=senkou_period,
        ).max()

        senkou_low = low.rolling(
            senkou_period,
            min_periods=senkou_period,
        ).min()

        result["ichimoku_tenkan"] = (
            tenkan_high + tenkan_low
        ) / 2

        result["ichimoku_kijun"] = (
            kijun_high + kijun_low
        ) / 2

        result["ichimoku_senkou_a"] = (
            result["ichimoku_tenkan"]
            + result["ichimoku_kijun"]
        ) / 2

        result["ichimoku_senkou_b"] = (
            senkou_high + senkou_low
        ) / 2

    if _enabled(trend, "parabolic_sar"):

        sar_settings = _settings(
            trend,
            "parabolic_sar",
        )

        step = sar_settings.get("step", 0.02)

        maximum = sar_settings.get("maximum", 0.2)

        if (
            isinstance(step, bool)
            or isinstance(maximum, bool)
            or not isinstance(step, (int, float))
            or not isinstance(maximum, (int, float))
            or not np.isfinite(step)
            or not np.isfinite(maximum)
        ):

            raise IndicatorCalculationError(
                "Invalid Parabolic SAR parameters."
            )

        result["parabolic_sar"] = _parabolic_sar(
            candles,
            float(step),
            float(maximum),
        )

    # Momentum indicators

    momentum = _settings(settings, "momentum")

    if _enabled(momentum, "rsi"):

        period = _period(
            _settings(momentum, "rsi").get("period", 14),
            "RSI period",
        )

        result[f"rsi_{period}"] = _rsi(
            close,
            period,
        )

    if _enabled(momentum, "macd"):

        macd_settings = _settings(momentum, "macd")

        fast = _period(
            macd_settings.get("fast", 12),
            "MACD fast",
        )

        slow = _period(
            macd_settings.get("slow", 26),
            "MACD slow",
        )

        signal = _period(
            macd_settings.get("signal", 9),
            "MACD signal",
        )

        macd_line, signal_line, histogram = _macd(
            close,
            fast,
            slow,
            signal,
        )

        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram

    if _enabled(momentum, "stochastic"):

        stochastic = _settings(
            momentum,
            "stochastic",
        )

        k_period = _period(
            stochastic.get("k_period", 14),
            "Stochastic K period",
        )

        d_period = _period(
            stochastic.get("d_period", 3),
            "Stochastic D period",
        )

        k_line, d_line = _stochastic(
            candles,
            k_period,
            d_period,
        )

        result["stochastic_k"] = k_line
        result["stochastic_d"] = d_line

    if _enabled(momentum, "cci"):

        period = _period(
            _settings(momentum, "cci").get("period", 20),
            "CCI period",
        )

        result[f"cci_{period}"] = _cci(
            candles,
            period,
        )

    if _enabled(momentum, "williams_r"):

        period = _period(
            _settings(momentum, "williams_r").get("period", 14),
            "Williams %R period",
        )

        result[f"williams_r_{period}"] = _williams_r(
            candles,
            period,
        )

    if _enabled(momentum, "roc"):

        period = _period(
            _settings(momentum, "roc").get("period", 12),
            "ROC period",
        )

        result[f"roc_{period}"] = _roc(
            close,
            period,
        )

    # Volatility indicators

    volatility = _settings(settings, "volatility")

    if _enabled(volatility, "atr"):

        period = _period(
            _settings(volatility, "atr").get("period", 14),
            "ATR period",
        )

        result[f"atr_{period}"] = _atr(
            candles,
            period,
        )

    if _enabled(volatility, "bollinger_bands"):

        bands = _settings(
            volatility,
            "bollinger_bands",
        )

        period = _period(
            bands.get("period", 20),
            "Bollinger period",
        )

        deviations = bands.get(
            "standard_deviation",
            2,
        )

        if (
            isinstance(deviations, bool)
            or not isinstance(deviations, (int, float))
            or not np.isfinite(deviations)
            or deviations <= 0
        ):

            raise IndicatorCalculationError(
                "Bollinger standard deviation must be positive."
            )

        upper, middle, lower = _bollinger(
            close,
            period,
            float(deviations),
        )

        result["bollinger_upper"] = upper
        result["bollinger_middle"] = middle
        result["bollinger_lower"] = lower

    # Volume indicators

    volume = _settings(settings, "volume")

    if _enabled(volume, "volume_sma"):

        period = _period(
            _settings(volume, "volume_sma").get(
                "period",
                20,
            ),
            "Volume SMA period",
        )

        result[f"volume_sma_{period}"] = _sma(
            candles["volume"],
            period,
        )

    if _enabled(volume, "obv"):

        result["obv"] = _obv(candles)

    if _enabled(volume, "mfi"):

        period = _period(
            _settings(volume, "mfi").get("period", 14),
            "MFI period",
        )

        result[f"mfi_{period}"] = _mfi(
            candles,
            period,
        )

    if _enabled(volume, "vwap"):

        result["vwap"] = _vwap(candles)

    return result


if __name__ == "__main__":

    print(
        "Technical indicator calculation module loaded."
    )