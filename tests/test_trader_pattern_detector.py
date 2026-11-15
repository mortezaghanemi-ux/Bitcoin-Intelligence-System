"""Tests for the trader candlestick pattern detector."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.patterns.detector import (
    OUTPUT_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    CandlestickPatternDetector,
    PatternDetectionError,
    detect_patterns,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026, 11, 13, 12, 0, tzinfo=UTC
)

BASE_OPEN_TIME = datetime(
    2026, 11, 13, 11, 55, tzinfo=UTC
)


def make_candle(
    *,
    open_price=100.0,
    high=102.0,
    low=98.0,
    close=101.0,
    volume=100.0,
    source="kraken",
    open_time=BASE_OPEN_TIME,
    close_time=None,
):
    """Create a valid five-minute OHLCV candle."""

    if close_time is None:
        close_time = open_time + timedelta(minutes=5)

    return {
        "open_time": open_time,
        "close_time": close_time,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "source": source,
    }


def make_frame(candles):
    """Create a DataFrame from candle dictionaries."""

    return pd.DataFrame(candles)


def make_sequence(rows, start_time=BASE_OPEN_TIME):
    """
    Create sequential five-minute candles.

    Each row contains:
    open, high, low, close, volume.
    """

    candles = []

    for index, row in enumerate(rows):
        opened = start_time + timedelta(
            minutes=5 * index
        )

        candles.append(
            make_candle(
                open_price=row[0],
                high=row[1],
                low=row[2],
                close=row[3],
                volume=row[4] if len(row) > 4 else 100,
                open_time=opened,
            )
        )

    return make_frame(candles)


def detected_names(result):
    """Return detected pattern names."""

    return set(result["pattern"].tolist())


class TestInputValidation:

    def test_rejects_non_dataframe(self):
        detector = CandlestickPatternDetector("5m")

        with pytest.raises(TypeError):
            detector.detect(
                [],
                detection_time=CHECK_TIME,
            )

    def test_rejects_empty_dataframe(self):
        detector = CandlestickPatternDetector("5m")

        with pytest.raises(PatternDetectionError):
            detector.detect(
                pd.DataFrame(),
                detection_time=CHECK_TIME,
            )

    def test_rejects_missing_required_columns(self):
        detector = CandlestickPatternDetector("5m")

        data = pd.DataFrame({
            "open": [100.0],
            "close": [101.0],
        })

        with pytest.raises(PatternDetectionError):
            detector.detect(
                data,
                detection_time=CHECK_TIME,
            )

    def test_rejects_non_utc_timestamp(self):
        detector = CandlestickPatternDetector("5m")

        non_utc = datetime(
            2026, 11, 13, 13, 0,
            tzinfo=timezone(timedelta(hours=1)),
        )

        candle = make_candle(
            open_time=non_utc,
            close_time=non_utc + timedelta(minutes=5),
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_future_open_time(self):
        detector = CandlestickPatternDetector("5m")

        future_open = CHECK_TIME

        candle = make_candle(
            open_time=future_open,
            close_time=future_open + timedelta(minutes=5),
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_unclosed_candle(self):
        detector = CandlestickPatternDetector("5m")

        opened = CHECK_TIME

        candle = make_candle(
            open_time=opened,
            close_time=opened + timedelta(minutes=5),
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_invalid_high(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            open_price=100,
            high=98,
            low=97,
            close=101,
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_invalid_low(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            open_price=100,
            high=102,
            low=103,
            close=101,
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_zero_open_price(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            open_price=0,
            high=102,
            low=0,
            close=101,
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_zero_high_price(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            open_price=0,
            high=0,
            low=0,
            close=0,
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_non_numeric_price(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle()
        candle["close"] = "invalid"

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_infinite_price(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            close=float("inf"),
            high=float("inf"),
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_nan_volume(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            volume=float("nan"),
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_negative_volume(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            volume=-1,
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_naive_open_timestamp(self):
        detector = CandlestickPatternDetector("5m")

        naive_open = datetime(2026, 11, 13, 11, 55)

        candle = make_candle(
            open_time=naive_open,
            close_time=naive_open + timedelta(minutes=5),
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_duplicate_open_times(self):
        detector = CandlestickPatternDetector("5m")

        first = make_candle()

        second = make_candle(
            open_time=BASE_OPEN_TIME,
        )

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([first, second]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_missing_source(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(source=None)

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_empty_source(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(source="   ")

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_non_utc_detection_time(self):
        detector = CandlestickPatternDetector("5m")

        non_utc = datetime(
            2026, 11, 13, 13, 0,
            tzinfo=timezone(timedelta(hours=1)),
        )

        with pytest.raises(ValueError):
            detector.detect(
                make_frame([make_candle()]),
                detection_time=non_utc,
            )

    def test_rejects_naive_detection_time(self):
        detector = CandlestickPatternDetector("5m")

        naive = datetime(2026, 11, 13, 12, 0)

        with pytest.raises(ValueError):
            detector.detect(
                make_frame([make_candle()]),
                detection_time=naive,
            )

    def test_rejects_invalid_timeframe_duration(self):
        detector = CandlestickPatternDetector("15m")

        candle = make_candle()

        with pytest.raises(PatternDetectionError):
            detector.detect(
                make_frame([candle]),
                detection_time=CHECK_TIME,
            )

    def test_rejects_unsupported_timeframe(self):
        with pytest.raises(ValueError):
            CandlestickPatternDetector("1d")

    def test_rejects_non_string_timeframe(self):
        with pytest.raises(ValueError):
            CandlestickPatternDetector(5)


class TestSingleCandlePatterns:

    def test_detects_doji(self):
        candle = make_candle(
            open_price=100,
            high=102,
            low=98,
            close=100.05,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Doji" in detected_names(result)

    def test_detects_marubozu(self):
        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Marubozu" in detected_names(result)

    def test_detects_hammer(self):
        candle = make_candle(
            open_price=100,
            high=101,
            low=94,
            close=100.8,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Hammer" in detected_names(result)

    def test_detects_shooting_star(self):
        candle = make_candle(
            open_price=100,
            high=106,
            low=99.8,
            close=100.5,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Shooting Star" in detected_names(result)

    def test_detects_bullish_pin_bar(self):
        candle = make_candle(
            open_price=100,
            high=101,
            low=94,
            close=100.8,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Bullish Pin Bar" in detected_names(result)

    def test_single_candle_patterns_have_valid_directions(self):
        candle = make_candle(
            open_price=100,
            high=102,
            low=98,
            close=100.05,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert set(result["direction"]).issubset({
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        })


class TestTwoCandlePatterns:

    def test_detects_bullish_engulfing(self):
        data = make_sequence([
            (105, 106, 99, 100),
            (99, 107, 98, 106),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Bullish Engulfing" in detected_names(result)

    def test_detects_bearish_engulfing(self):
        data = make_sequence([
            (100, 106, 99, 105),
            (106, 107, 98, 99),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Bearish Engulfing" in detected_names(result)

    def test_detects_bullish_harami(self):
        data = make_sequence([
            (110, 111, 99, 100),
            (102, 104, 101, 103),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Bullish Harami" in detected_names(result)

    def test_detects_bearish_harami(self):
        data = make_sequence([
            (100, 111, 99, 110),
            (108, 109, 106, 107),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Bearish Harami" in detected_names(result)

    def test_detects_tweezer_bottom(self):
        data = make_sequence([
            (105, 106, 99, 100),
            (100, 104, 99.05, 103),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Tweezer Bottom" in detected_names(result)

    def test_detects_tweezer_top(self):
        data = make_sequence([
            (100, 110, 99, 108),
            (108, 110.05, 105, 106),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Tweezer Top" in detected_names(result)


class TestThreeCandlePatterns:

    def test_detects_morning_star(self):
        data = make_sequence([
            (110, 111, 99, 100),
            (100, 101, 99, 100.2),
            (100, 108, 99, 107),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Morning Star" in detected_names(result)

    def test_detects_evening_star(self):
        data = make_sequence([
            (100, 111, 99, 110),
            (110, 111, 109, 109.8),
            (109, 110, 102, 103),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Evening Star" in detected_names(result)

    def test_detects_three_white_soldiers(self):
        data = make_sequence([
            (100, 106, 99, 105),
            (102, 108, 101, 107),
            (104, 110, 103, 109),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Three White Soldiers" in detected_names(result)

    def test_detects_three_black_crows(self):
        data = make_sequence([
            (110, 111, 104, 105),
            (108, 109, 102, 103),
            (106, 107, 100, 101),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Three Black Crows" in detected_names(result)


class TestFiveCandlePatterns:

    def test_detects_rising_three_methods(self):
        data = make_sequence([
            (100, 110, 99, 109),
            (108, 109, 105, 106),
            (107, 108, 104, 105),
            (106, 107, 103, 104),
            (108, 116, 107, 115),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Rising Three Methods" in detected_names(result)

    def test_detects_falling_three_methods(self):
        data = make_sequence([
            (110, 111, 100, 101),
            (102, 106, 101, 105),
            (103, 107, 102, 106),
            (104, 108, 103, 107),
            (102, 103, 94, 95),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert "Falling Three Methods" in detected_names(result)


class TestDetectionOutput:

    def test_output_contains_required_columns(self):
        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        assert tuple(result.columns) == OUTPUT_COLUMNS

    def test_output_pattern_ids_are_unique(self):
        data = make_sequence([
            (100, 110, 100, 109.5),
            (100, 110, 100, 109.5),
        ])

        result = detect_patterns(
            data,
            "5m",
            detection_time=CHECK_TIME,
        )

        assert result["pattern_id"].is_unique

    def test_output_confidence_is_between_zero_and_100(self):
        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        if not result.empty:
            assert result["confidence"].between(
                0, 100
            ).all()

    def test_output_uses_requested_timeframe(self):
        detector = CandlestickPatternDetector("15m")

        opened = datetime(
            2026, 11, 13, 10, 0, tzinfo=UTC
        )

        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
            open_time=opened,
            close_time=opened + timedelta(minutes=15),
        )

        result = detector.detect(
            make_frame([candle]),
            detection_time=CHECK_TIME,
        )

        assert set(result["timeframe"]) == {"15m"}

    def test_output_timestamps_are_utc(self):
        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
        )

        result = detect_patterns(
            make_frame([candle]),
            "5m",
            detection_time=CHECK_TIME,
        )

        for column in (
            "open_time",
            "close_time",
            "detection_time",
        ):
            if not result.empty:
                assert str(
                    result[column].dt.tz
                ) == "UTC"


class TestDetectionHistory:

    def test_history_preserves_previous_detections(self):
        detector = CandlestickPatternDetector("5m")

        first = make_frame([
            make_candle(
                open_price=100,
                high=110,
                low=100,
                close=109.5,
            )
        ])

        second_open = BASE_OPEN_TIME + timedelta(
            minutes=5
        )

        second = make_frame([
            make_candle(
                open_price=100,
                high=110,
                low=100,
                close=109.5,
                open_time=second_open,
            )
        ])

        first_result = detector.detect(
            first,
            detection_time=CHECK_TIME,
        )

        second_result = detector.detect(
            second,
            detection_time=CHECK_TIME,
        )

        assert not first_result.empty
        assert not second_result.empty
        assert len(detector.history) >= (
            len(first_result) + len(second_result)
        )

    def test_history_returns_a_copy(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
        )

        detector.detect(
            make_frame([candle]),
            detection_time=CHECK_TIME,
        )

        history = detector.history
        original_length = len(detector.history)

        history.drop(
            history.index,
            inplace=True,
        )

        assert len(detector.history) == original_length

    def test_clear_removes_history(self):
        detector = CandlestickPatternDetector("5m")

        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
        )

        detector.detect(
            make_frame([candle]),
            detection_time=CHECK_TIME,
        )

        assert not detector.history.empty

        detector.clear()

        assert detector.history.empty
        assert tuple(detector.history.columns) == OUTPUT_COLUMNS

    def test_repeated_detection_does_not_duplicate_history(self):
        detector = CandlestickPatternDetector("5m")

        data = make_frame([
            make_candle(
                open_price=100,
                high=110,
                low=100,
                close=109.5,
            )
        ])

        first = detector.detect(
            data,
            detection_time=CHECK_TIME,
        )

        second = detector.detect(
            data,
            detection_time=CHECK_TIME,
        )

        assert not first.empty
        assert second.empty
        assert len(detector.history) == len(first)

    def test_history_is_not_modified_by_input_changes(self):
        detector = CandlestickPatternDetector("5m")

        data = make_frame([
            make_candle(
                open_price=100,
                high=110,
                low=100,
                close=109.5,
            )
        ])

        detector.detect(
            data,
            detection_time=CHECK_TIME,
        )

        original_history = detector.history.copy(
            deep=True
        )

        data.loc[0, "close"] = 1

        pd.testing.assert_frame_equal(
            detector.history,
            original_history,
        )


class TestSupportedTimeframes:

    @pytest.mark.parametrize(
        "timeframe",
        ["5m", "15m", "1h", "4h"],
    )
    def test_supported_timeframes_are_accepted(self, timeframe):
        detector = CandlestickPatternDetector(timeframe)

        assert detector.timeframe == timeframe

    def test_supported_timeframes_have_expected_durations(self):
        assert SUPPORTED_TIMEFRAMES == {
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
        }

    @pytest.mark.parametrize(
        ("timeframe", "minutes"),
        [
            ("5m", 5),
            ("15m", 15),
            ("1h", 60),
            ("4h", 240),
        ],
    )
    def test_detect_accepts_matching_candle_duration(
        self,
        timeframe,
        minutes,
    ):
        detector = CandlestickPatternDetector(timeframe)

        opened = datetime(
            2026, 11, 13, 10, 0, tzinfo=UTC
        )

        candle = make_candle(
            open_price=100,
            high=110,
            low=100,
            close=109.5,
            open_time=opened,
            close_time=opened + timedelta(minutes=minutes),
        )

        result = detector.detect(
            make_frame([candle]),
            detection_time=CHECK_TIME,
        )

        assert isinstance(result, pd.DataFrame)
        assert tuple(result.columns) == OUTPUT_COLUMNS