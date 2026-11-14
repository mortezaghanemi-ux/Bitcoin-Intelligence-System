"""Tests for technical indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.calculations import (
    IndicatorCalculationError,
    calculate_indicators,
)


@pytest.fixture
def candles():
    """Create deterministic UTC OHLCV candle data."""

    size = 100

    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=size,
        freq="5min",
        tz="UTC",
    )

    close = (
        95000
        + np.arange(size) * 25
        + np.sin(np.arange(size) / 3) * 100
    )

    return pd.DataFrame(
        {
            "open_time": timestamps,
            "close_time": timestamps + pd.Timedelta(minutes=5),
            "open": close - 20,
            "high": close + 100,
            "low": close - 100,
            "close": close,
            "volume": np.linspace(100, 200, size),
            "source": "kraken",
        }
    )


@pytest.fixture
def small_config():
    """Return a compact configuration for fast indicator tests."""

    return {
        "indicators": {
            "enabled": True,
            "trend": {
                "ema": {
                    "enabled": True,
                    "periods": [5, 10],
                },
                "sma": {
                    "enabled": True,
                    "periods": [5, 10],
                },
                "adx": {
                    "enabled": True,
                    "period": 5,
                },
                "ichimoku": {
                    "enabled": True,
                    "tenkan": 5,
                    "kijun": 10,
                    "senkou_b": 20,
                },
                "parabolic_sar": {
                    "enabled": True,
                    "step": 0.02,
                    "maximum": 0.2,
                },
            },
            "momentum": {
                "rsi": {
                    "enabled": True,
                    "period": 5,
                },
                "macd": {
                    "enabled": True,
                    "fast": 3,
                    "slow": 6,
                    "signal": 3,
                },
                "stochastic": {
                    "enabled": True,
                    "k_period": 5,
                    "d_period": 3,
                },
                "cci": {
                    "enabled": True,
                    "period": 5,
                },
                "williams_r": {
                    "enabled": True,
                    "period": 5,
                },
                "roc": {
                    "enabled": True,
                    "period": 5,
                },
            },
            "volatility": {
                "atr": {
                    "enabled": True,
                    "period": 5,
                },
                "bollinger_bands": {
                    "enabled": True,
                    "period": 5,
                    "standard_deviation": 2,
                },
            },
            "volume": {
                "volume_sma": {
                    "enabled": True,
                    "period": 5,
                },
                "obv": {
                    "enabled": True,
                },
                "mfi": {
                    "enabled": True,
                    "period": 5,
                },
                "vwap": {
                    "enabled": True,
                },
            },
        },
    }


def test_calculate_all_indicators(candles, small_config):
    """All configured indicators should be calculated."""

    result = calculate_indicators(candles, small_config)

    expected_columns = {
        "ema_5",
        "ema_10",
        "sma_5",
        "sma_10",
        "adx_5",
        "plus_di_5",
        "minus_di_5",
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "parabolic_sar",
        "rsi_5",
        "macd",
        "macd_signal",
        "macd_histogram",
        "stochastic_k",
        "stochastic_d",
        "cci_5",
        "williams_r_5",
        "roc_5",
        "atr_5",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "volume_sma_5",
        "obv",
        "mfi_5",
        "vwap",
    }

    assert expected_columns.issubset(result.columns)


def test_input_data_is_not_modified(candles, small_config):
    """Indicator calculation must not modify original data."""

    original = candles.copy(deep=True)

    calculate_indicators(candles, small_config)

    pd.testing.assert_frame_equal(candles, original)


def test_result_preserves_original_candle_columns(
    candles,
    small_config,
):
    """Original candle columns should remain unchanged."""

    result = calculate_indicators(candles, small_config)

    for column in candles.columns:
        pd.testing.assert_series_equal(
            result[column],
            candles[column],
        )


def test_global_indicators_disabled(candles):
    """Disabled indicators should return candles without calculations."""

    result = calculate_indicators(
        candles,
        {"indicators": {"enabled": False}},
    )

    assert list(result.columns) == list(candles.columns)

    pd.testing.assert_frame_equal(
        result,
        candles.reset_index(drop=True),
    )


def test_non_dataframe_input_is_rejected():
    """Non-DataFrame input should raise TypeError."""

    with pytest.raises(TypeError):
        calculate_indicators([])


def test_empty_dataframe_is_rejected():
    """Empty candle data should be rejected."""

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(pd.DataFrame())


def test_missing_required_column_is_rejected(candles):
    """Missing required candle columns should be rejected."""

    invalid = candles.drop(columns=["volume"])

    with pytest.raises(
        IndicatorCalculationError,
        match="Missing candle columns",
    ):
        calculate_indicators(invalid)


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close", "volume"],
)
def test_non_numeric_values_are_rejected(candles, column):
    """Non-numeric OHLCV columns should be rejected."""

    invalid = candles.copy()
    invalid[column] = "invalid"

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(invalid)


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close", "volume"],
)
def test_non_finite_values_are_rejected(candles, column):
    """NaN values should be rejected."""

    invalid = candles.copy()
    invalid.loc[10, column] = np.nan

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(invalid)


def test_boolean_numeric_column_is_rejected(candles):
    """Boolean values must not be accepted as numeric prices."""

    invalid = candles.copy()
    invalid["volume"] = invalid["volume"].astype(bool)

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(invalid)


def test_negative_volume_is_rejected(candles):
    """Negative volume should be rejected."""

    invalid = candles.copy()
    invalid.loc[10, "volume"] = -1

    with pytest.raises(
        IndicatorCalculationError,
        match="Volume cannot be negative",
    ):
        calculate_indicators(invalid)


def test_zero_price_is_rejected(candles):
    """Zero prices should be rejected."""

    invalid = candles.copy()
    invalid.loc[10, "close"] = 0

    with pytest.raises(
        IndicatorCalculationError,
        match="Prices must be greater than zero",
    ):
        calculate_indicators(invalid)


def test_invalid_ohlc_relationship_is_rejected(candles):
    """Invalid OHLC relationships should be rejected."""

    invalid = candles.copy()
    invalid.loc[10, "high"] = invalid.loc[10, "low"] - 1

    with pytest.raises(
        IndicatorCalculationError,
        match="Invalid OHLC",
    ):
        calculate_indicators(invalid)


@pytest.mark.parametrize(
    "column",
    ["open_time", "close_time"],
)
def test_naive_timestamps_are_rejected(candles, column):
    """Naive timestamps should be rejected."""

    invalid = candles.copy()
    invalid[column] = invalid[column].dt.tz_localize(None)

    with pytest.raises(
        IndicatorCalculationError,
        match="timezone-aware",
    ):
        calculate_indicators(invalid)


@pytest.mark.parametrize(
    "column",
    ["open_time", "close_time"],
)
def test_non_utc_timestamps_are_rejected(candles, column):
    """Non-UTC timestamps should be rejected."""

    invalid = candles.copy()
    invalid[column] = invalid[column].dt.tz_convert(
        "Europe/Berlin"
    )

    with pytest.raises(
        IndicatorCalculationError,
        match="UTC",
    ):
        calculate_indicators(invalid)


def test_invalid_timestamp_string_is_rejected(candles):
    """Unparseable timestamps should be rejected."""

    invalid = candles.copy()
    invalid["open_time"] = invalid["open_time"].astype(object)
    invalid.loc[5, "open_time"] = "not-a-timestamp"

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(invalid)


def test_close_time_must_follow_open_time(candles):
    """Candle close time must be later than open time."""

    invalid = candles.copy()
    invalid.loc[10, "close_time"] = invalid.loc[10, "open_time"]

    with pytest.raises(
        IndicatorCalculationError,
        match="close_time must be after open_time",
    ):
        calculate_indicators(invalid)


def test_duplicate_timestamps_are_rejected(candles):
    """Duplicate candle timestamps should be rejected."""

    invalid = candles.copy()
    invalid.loc[10, "open_time"] = invalid.loc[9, "open_time"]

    with pytest.raises(
        IndicatorCalculationError,
        match="Duplicate candle timestamps",
    ):
        calculate_indicators(invalid)


def test_unsorted_candles_are_rejected(candles):
    """Candles must be chronologically sorted."""

    invalid = candles.iloc[::-1].reset_index(drop=True)

    with pytest.raises(
        IndicatorCalculationError,
        match="sorted by open_time",
    ):
        calculate_indicators(invalid)


def test_ema_warmup_period(candles):
    """EMA should remain NaN during its warm-up period."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "trend": {
                    "ema": {
                        "enabled": True,
                        "periods": [5],
                    },
                },
            },
        },
    )

    assert result["ema_5"].iloc[:4].isna().all()
    assert pd.notna(result["ema_5"].iloc[4])


def test_sma_calculation(candles):
    """SMA should equal the rolling arithmetic mean."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "trend": {
                    "ema": {"enabled": False},
                    "sma": {
                        "enabled": True,
                        "periods": [5],
                    },
                },
            },
        },
    )

    expected = candles["close"].rolling(5).mean()

    pd.testing.assert_series_equal(
        result["sma_5"],
        expected,
        check_names=False,
    )


def test_rsi_values_are_bounded(candles):
    """RSI values should remain between zero and one hundred."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "momentum": {
                    "rsi": {
                        "enabled": True,
                        "period": 5,
                    },
                },
            },
        },
    )

    valid = result["rsi_5"].dropna()

    assert not valid.empty
    assert valid.between(0, 100).all()


def test_rsi_constant_prices_returns_fifty(candles):
    """RSI should return 50 when gains and losses are both zero."""

    candles["open"] = 95000.0
    candles["high"] = 95100.0
    candles["low"] = 94900.0
    candles["close"] = 95000.0

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "momentum": {
                    "rsi": {
                        "enabled": True,
                        "period": 5,
                    },
                },
            },
        },
    )

    valid = result["rsi_5"].dropna()

    assert not valid.empty
    assert (valid == 50).all()


def test_roc_calculation(candles):
    """ROC should represent percentage price change."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "momentum": {
                    "roc": {
                        "enabled": True,
                        "period": 5,
                    },
                },
            },
        },
    )

    expected = candles["close"].pct_change(periods=5) * 100

    pd.testing.assert_series_equal(
        result["roc_5"],
        expected,
        check_names=False,
    )


def test_bollinger_band_order(candles):
    """Upper band must be above middle and lower band."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "volatility": {
                    "bollinger_bands": {
                        "enabled": True,
                        "period": 5,
                        "standard_deviation": 2,
                    },
                },
            },
        },
    )

    valid = result.dropna(
        subset=[
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
        ]
    )

    assert (
        valid["bollinger_upper"] >= valid["bollinger_middle"]
    ).all()

    assert (
        valid["bollinger_middle"] >= valid["bollinger_lower"]
    ).all()


def test_obv_is_calculated(candles):
    """OBV should produce a cumulative volume series."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "volume": {
                    "obv": {"enabled": True},
                },
            },
        },
    )

    assert "obv" in result.columns
    assert np.isfinite(result["obv"]).all()


def test_vwap_is_calculated(candles):
    """VWAP should contain finite values for positive volume."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "volume": {
                    "vwap": {"enabled": True},
                },
            },
        },
    )

    assert "vwap" in result.columns
    assert np.isfinite(result["vwap"]).all()


def test_disabled_indicator_is_not_created(candles):
    """Disabled indicators should not produce output columns."""

    result = calculate_indicators(
        candles,
        {
            "indicators": {
                "enabled": True,
                "trend": {
                    "ema": {"enabled": False},
                    "sma": {"enabled": False},
                },
                "momentum": {
                    "rsi": {"enabled": False},
                },
            },
        },
    )

    assert "ema_20" not in result.columns
    assert "sma_20" not in result.columns
    assert "rsi_14" not in result.columns


def test_invalid_global_enabled_type_is_rejected(candles):
    """Global enabled setting must be boolean."""

    with pytest.raises(
        IndicatorCalculationError,
        match="indicators.enabled must be boolean",
    ):
        calculate_indicators(
            candles,
            {"indicators": {"enabled": "yes"}},
        )


def test_invalid_config_type_is_rejected(candles):
    """Configuration must be a dictionary."""

    with pytest.raises(TypeError):
        calculate_indicators(candles, config="invalid")


def test_invalid_indicator_section_is_rejected(candles):
    """Indicator sections must be dictionaries."""

    with pytest.raises(
        IndicatorCalculationError,
        match="Invalid configuration for trend",
    ):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "trend": [],
                },
            },
        )


def test_invalid_indicator_enabled_type_is_rejected(candles):
    """Individual enabled settings must be boolean."""

    with pytest.raises(
        IndicatorCalculationError,
        match="ema.enabled must be boolean",
    ):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "trend": {
                        "ema": {
                            "enabled": "true",
                        },
                    },
                },
            },
        )


@pytest.mark.parametrize(
    "period",
    [0, -1, True, 2.5, "5"],
)
def test_invalid_ema_period_is_rejected(candles, period):
    """EMA periods must be positive integers."""

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "trend": {
                        "ema": {
                            "enabled": True,
                            "periods": [period],
                        },
                    },
                },
            },
        )


def test_empty_ema_periods_are_rejected(candles):
    """EMA requires a non-empty period list."""

    with pytest.raises(
        IndicatorCalculationError,
        match="EMA periods",
    ):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "trend": {
                        "ema": {
                            "enabled": True,
                            "periods": [],
                        },
                    },
                },
            },
        )


def test_macd_fast_period_must_be_smaller_than_slow(candles):
    """MACD fast period must be smaller than slow period."""

    with pytest.raises(
        IndicatorCalculationError,
        match="MACD fast period",
    ):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "momentum": {
                        "macd": {
                            "enabled": True,
                            "fast": 10,
                            "slow": 5,
                            "signal": 3,
                        },
                    },
                },
            },
        )


@pytest.mark.parametrize(
    "step,maximum",
    [
        (0, 0.2),
        (-0.01, 0.2),
        (0.3, 0.2),
        (0.02, 0),
        (True, 0.2),
        (0.02, float("inf")),
    ],
)
def test_invalid_parabolic_sar_parameters_are_rejected(
    candles,
    step,
    maximum,
):
    """Invalid Parabolic SAR parameters should be rejected."""

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "trend": {
                        "parabolic_sar": {
                            "enabled": True,
                            "step": step,
                            "maximum": maximum,
                        },
                    },
                },
            },
        )


@pytest.mark.parametrize(
    "deviations",
    [0, -1, True, float("inf"), "2"],
)
def test_invalid_bollinger_deviations_are_rejected(
    candles,
    deviations,
):
    """Bollinger deviations must be a positive finite number."""

    with pytest.raises(IndicatorCalculationError):
        calculate_indicators(
            candles,
            {
                "indicators": {
                    "enabled": True,
                    "volatility": {
                        "bollinger_bands": {
                            "enabled": True,
                            "period": 5,
                            "standard_deviation": deviations,
                        },
                    },
                },
            },
        )


def test_single_candle_parabolic_sar(candles):
    """Parabolic SAR should handle a single candle safely."""

    single = candles.iloc[:1].copy()

    result = calculate_indicators(
        single,
        {
            "indicators": {
                "enabled": True,
                "trend": {
                    "parabolic_sar": {
                        "enabled": True,
                        "step": 0.02,
                        "maximum": 0.2,
                    },
                },
            },
        },
    )

    assert len(result) == 1
    assert pd.isna(result.loc[0, "parabolic_sar"])


def test_output_has_reset_index(candles, small_config):
    """Output index should be reset consistently."""

    custom = candles.copy()
    custom.index = range(100, 200)

    result = calculate_indicators(custom, small_config)

    assert list(result.index) == list(range(len(result)))


def test_indicator_calculation_does_not_use_future_candles(
    candles,
    small_config,
):
    """Earlier results must not change when future candles are appended."""

    full_result = calculate_indicators(candles, small_config)

    cutoff = 60

    partial_result = calculate_indicators(
        candles.iloc[:cutoff].copy(),
        small_config,
    )

    columns = [
        column
        for column in full_result.columns
        if column not in candles.columns
    ]

    for column in columns:
        pd.testing.assert_series_equal(
            full_result[column].iloc[:cutoff].reset_index(drop=True),
            partial_result[column].reset_index(drop=True),
            check_names=False,
        )