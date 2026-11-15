"""Tests for the Trader Candle Manager."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.candles.manager import (
    CandleManager,
    CandleManagerError,
    SUPPORTED_TIMEFRAMES,
    timeframe_to_seconds,
)


UTC = timezone.utc

CHECK_TIME = datetime(
    2026, 11, 15, 12, 0,
    tzinfo=UTC,
)


def make_candle(
    open_time="2026-11-15 11:00:00+00:00",
    *,
    close_time=None,
    open_price=95000,
    high=96000,
    low=94000,
    close=95500,
    volume=120.5,
    source="kraken",
):
    opened = pd.Timestamp(open_time)

    if close_time is None:
        close_time = (
            opened + pd.Timedelta(minutes=5)
        )

    return {
        "open_time": opened,
        "close_time": pd.Timestamp(close_time),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "source": source,
    }


def make_frame(*candles):
    return pd.DataFrame(
        list(candles)
    )


class TestTimeframes:

    @pytest.mark.parametrize(
        "timeframe,seconds",
        [
            ("5m", 300),
            ("15m", 900),
            ("1h", 3600),
            ("4h", 14400),
        ],
    )
    def test_supported_timeframes(
        self,
        timeframe,
        seconds,
    ):
        assert timeframe_to_seconds(
            timeframe
        ) == seconds

    def test_supported_timeframes_constant(self):
        assert SUPPORTED_TIMEFRAMES == {
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
        }

    @pytest.mark.parametrize(
        "timeframe",
        [
            "1m",
            "30m",
            "1d",
            "",
            None,
            5,
        ],
    )
    def test_unsupported_timeframe_raises(
        self,
        timeframe,
    ):
        with pytest.raises(ValueError):
            timeframe_to_seconds(timeframe)

    def test_manager_rejects_unsupported_timeframe(self):
        with pytest.raises(ValueError):
            CandleManager("1d")


class TestCandleManagerInitialization:

    def test_initialization(self):
        manager = CandleManager("5m")

        assert manager.timeframe == "5m"
        assert manager.interval_seconds == 300
        assert manager.candles.empty

    def test_candles_property_returns_copy(self):
        manager = CandleManager("5m")

        manager.ingest(
            make_frame(make_candle()),
            validation_time=CHECK_TIME,
        )

        returned = manager.candles
        returned.loc[0, "close"] = 1

        assert manager.candles.loc[0, "close"] == 95500

    def test_clear_removes_stored_candles(self):
        manager = CandleManager("5m")

        manager.ingest(
            make_frame(make_candle()),
            validation_time=CHECK_TIME,
        )

        manager.clear()

        assert manager.candles.empty


class TestCandleIngestion:

    def test_ingest_valid_closed_candle(self):
        manager = CandleManager("5m")

        result = manager.ingest(
            make_frame(make_candle()),
            validation_time=CHECK_TIME,
        )

        assert result["status"] == "PASS"
        assert len(result["accepted"]) == 1
        assert result["rejected"].empty
        assert len(manager.candles) == 1

    def test_ingest_returns_report(self):
        manager = CandleManager("5m")

        result = manager.ingest(
            make_frame(make_candle()),
            validation_time=CHECK_TIME,
        )

        report = result["report"]

        assert report["timeframe"] == "5m"
        assert report["accepted_records"] == 1
        assert report["rejected_records"] == 0
        assert report["stored_records"] == 1

    def test_rejects_empty_dataset(self):
        manager = CandleManager("5m")

        with pytest.raises(CandleManagerError):
            manager.ingest(
                pd.DataFrame(),
                validation_time=CHECK_TIME,
            )

    def test_rejects_non_dataframe(self):
        manager = CandleManager("5m")

        with pytest.raises(TypeError):
            manager.ingest(
                [],
                validation_time=CHECK_TIME,
            )

    def test_rejects_missing_columns(self):
        manager = CandleManager("5m")

        with pytest.raises(CandleManagerError):
            manager.ingest(
                pd.DataFrame({
                    "open": [95000],
                    "close": [95500],
                }),
                validation_time=CHECK_TIME,
            )

    def test_rejects_naive_validation_time(self):
        manager = CandleManager("5m")

        naive_time = datetime(
            2026, 11, 15, 12, 0
        )

        with pytest.raises(ValueError):
            manager.ingest(
                make_frame(make_candle()),
                validation_time=naive_time,
            )

    def test_rejects_non_utc_validation_time(self):
        manager = CandleManager("5m")

        non_utc = datetime(
            2026, 11, 15, 14, 0,
            tzinfo=timezone(
                timedelta(hours=2)
            ),
        )

        with pytest.raises(ValueError):
            manager.ingest(
                make_frame(make_candle()),
                validation_time=non_utc,
            )

    def test_rejects_unclosed_candle(self):
        manager = CandleManager("5m")

        candle = make_candle(
            close_time=(
                CHECK_TIME + timedelta(minutes=5)
            )
        )

        result = manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        assert result["status"] == "FAIL"
        assert result["accepted"].empty
        assert len(result["rejected"]) == 1

    def test_rejects_invalid_ohlc(self):
        manager = CandleManager("5m")

        candle = make_candle(
            high=94000,
        )

        result = manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        assert result["status"] == "FAIL"
        assert result["accepted"].empty

    def test_rejects_negative_volume(self):
        manager = CandleManager("5m")

        candle = make_candle(
            volume=-1,
        )

        result = manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        assert result["status"] == "FAIL"
        assert result["accepted"].empty

    def test_rejects_wrong_timeframe_duration(self):
        manager = CandleManager("5m")

        candle = make_candle(
            close_time=(
                pd.Timestamp(
                    "2026-11-15 11:10:00+00:00"
                )
            ),
        )

        result = manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        assert result["status"] == "FAIL"
        assert result["accepted"].empty
        assert len(result["rejected"]) == 1

        assert (
            "invalid_timeframe_duration"
            in result["rejected"].iloc[0][
                "validation_errors"
            ]
        )

    def test_rejects_naive_candle_timestamp(self):
        manager = CandleManager("5m")

        candle = make_candle()
        candle["open_time"] = pd.Timestamp(
            "2026-11-15 11:00:00"
        )

        result = manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        assert result["accepted"].empty
        assert len(result["rejected"]) == 1

    def test_rejects_stale_candle(self):
        manager = CandleManager("5m")

        result = manager.ingest(
            make_frame(make_candle()),
            validation_time=CHECK_TIME,
            max_age_minutes=1,
        )

        assert result["accepted"].empty
        assert len(result["rejected"]) == 1

    def test_rejects_duplicate_candles_in_same_batch(self):
        manager = CandleManager("5m")

        candle = make_candle()

        result = manager.ingest(
            make_frame(candle, candle),
            validation_time=CHECK_TIME,
        )

        assert len(result["accepted"]) == 1
        assert len(result["rejected"]) == 1
        assert len(manager.candles) == 1

    def test_accepts_different_sources_same_timestamp(self):
        manager = CandleManager("5m")

        first = make_candle(
            source="kraken"
        )

        second = make_candle(
            source="coingecko"
        )

        result = manager.ingest(
            make_frame(first, second),
            validation_time=CHECK_TIME,
        )

        assert len(result["accepted"]) == 2
        assert len(manager.candles) == 2

    def test_accepts_multiple_timeframes_independently(self):
        manager_5m = CandleManager("5m")
        manager_15m = CandleManager("15m")

        candle_5m = make_candle()

        candle_15m = make_candle(
            close_time=pd.Timestamp(
                "2026-11-15 11:15:00+00:00"
            )
        )

        result_5m = manager_5m.ingest(
            make_frame(candle_5m),
            validation_time=CHECK_TIME,
        )

        result_15m = manager_15m.ingest(
            make_frame(candle_15m),
            validation_time=CHECK_TIME,
        )

        assert result_5m["status"] == "PASS"
        assert result_15m["status"] == "PASS"


class TestHistoricalProtection:

    def test_identical_historical_candle_is_not_duplicated(self):
        manager = CandleManager("5m")

        candle = make_candle()

        manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        result = manager.ingest(
            make_frame(candle),
            validation_time=CHECK_TIME,
        )

        assert result["accepted"].empty
        assert len(manager.candles) == 1
        assert result["rejected"].empty

    def test_conflicting_historical_candle_is_rejected(self):
        manager = CandleManager("5m")

        original = make_candle()

        manager.ingest(
            make_frame(original),
            validation_time=CHECK_TIME,
        )

        conflicting = make_candle(
            close=97000,
        )

        result = manager.ingest(
            make_frame(conflicting),
            validation_time=CHECK_TIME,
        )

        assert result["accepted"].empty
        assert len(result["rejected"]) == 1

        assert (
            "historical_candle_conflict"
            in result["rejected"].iloc[0][
                "validation_errors"
            ]
        )

    def test_conflict_does_not_modify_original_record(self):
        manager = CandleManager("5m")

        original = make_candle()

        manager.ingest(
            make_frame(original),
            validation_time=CHECK_TIME,
        )

        manager.ingest(
            make_frame(
                make_candle(close=97000)
            ),
            validation_time=CHECK_TIME,
        )

        stored = manager.candles.iloc[0]

        assert stored["close"] == 95500

    def test_new_candle_does_not_modify_old_candle(self):
        manager = CandleManager("5m")

        first = make_candle()

        second = make_candle(
            open_time="2026-11-15 11:05:00+00:00",
        )

        manager.ingest(
            make_frame(first),
            validation_time=CHECK_TIME,
        )

        manager.ingest(
            make_frame(second),
            validation_time=CHECK_TIME,
        )

        stored = manager.candles

        assert len(stored) == 2
        assert stored.iloc[0]["close"] == 95500

    def test_ingestion_is_sorted_by_source_and_time(self):
        manager = CandleManager("5m")

        later = make_candle(
            open_time="2026-11-15 11:10:00+00:00",
        )

        earlier = make_candle(
            open_time="2026-11-15 11:00:00+00:00",
        )

        manager.ingest(
            make_frame(later, earlier),
            validation_time=CHECK_TIME,
        )

        stored = manager.candles

        assert stored["open_time"].is_monotonic_increasing


class TestMissingCandles:

    def test_detects_missing_interval(self):
        manager = CandleManager("5m")

        first = make_candle(
            open_time="2026-11-15 11:00:00+00:00",
        )

        third = make_candle(
            open_time="2026-11-15 11:10:00+00:00",
        )

        manager.ingest(
            make_frame(first, third),
            validation_time=CHECK_TIME,
        )

        missing = manager.get_missing_candles()

        assert len(missing) == 1

        assert missing.iloc[0][
            "missing_open_time"
        ] == pd.Timestamp(
            "2026-11-15 11:05:00+00:00"
        )

    def test_no_missing_intervals_for_consecutive_candles(self):
        manager = CandleManager("5m")

        first = make_candle(
            open_time="2026-11-15 11:00:00+00:00",
        )

        second = make_candle(
            open_time="2026-11-15 11:05:00+00:00",
        )

        manager.ingest(
            make_frame(first, second),
            validation_time=CHECK_TIME,
        )

        assert manager.get_missing_candles().empty

    def test_missing_intervals_are_source_specific(self):
        manager = CandleManager("5m")

        candles = [
            make_candle(
                open_time="2026-11-15 11:00:00+00:00",
                source="kraken",
            ),
            make_candle(
                open_time="2026-11-15 11:10:00+00:00",
                source="kraken",
            ),
            make_candle(
                open_time="2026-11-15 11:00:00+00:00",
                source="coingecko",
            ),
            make_candle(
                open_time="2026-11-15 11:05:00+00:00",
                source="coingecko",
            ),
        ]

        manager.ingest(
            make_frame(*candles),
            validation_time=CHECK_TIME,
        )

        missing = manager.get_missing_candles()

        assert len(missing) == 1
        assert missing.iloc[0]["source"] == "kraken"

    def test_empty_dataset_returns_empty_missing_dataframe(self):
        manager = CandleManager("5m")

        missing = manager.get_missing_candles()

        assert missing.empty
        assert list(missing.columns) == [
            "source",
            "missing_open_time",
        ]

    def test_detect_missing_rejects_non_dataframe(self):
        manager = CandleManager("5m")

        with pytest.raises(TypeError):
            manager.detect_missing_candles([])


class TestCandleRetrieval:

    def test_get_candles_returns_all_records(self):
        manager = CandleManager("5m")

        candles = [
            make_candle(
                open_time="2026-11-15 11:00:00+00:00"
            ),
            make_candle(
                open_time="2026-11-15 11:05:00+00:00"
            ),
        ]

        manager.ingest(
            make_frame(*candles),
            validation_time=CHECK_TIME,
        )

        assert len(manager.get_candles()) == 2

    def test_get_candles_filters_source(self):
        manager = CandleManager("5m")

        manager.ingest(
            make_frame(
                make_candle(source="kraken"),
                make_candle(source="coingecko"),
            ),
            validation_time=CHECK_TIME,
        )

        result = manager.get_candles(
            source="kraken"
        )

        assert len(result) == 1
        assert result.iloc[0]["source"] == "kraken"

    def test_get_candles_applies_limit(self):
        manager = CandleManager("5m")

        candles = [
            make_candle(
                open_time=f"2026-11-15 11:{minute:02d}:00+00:00"
            )
            for minute in (0, 5, 10)
        ]

        manager.ingest(
            make_frame(*candles),
            validation_time=CHECK_TIME,
        )

        result = manager.get_candles(
            limit=2
        )

        assert len(result) == 2

    @pytest.mark.parametrize(
        "limit",
        [
            0,
            -1,
            1.5,
            True,
            "2",
        ],
    )
    def test_invalid_limit_raises(self, limit):
        manager = CandleManager("5m")

        with pytest.raises(ValueError):
            manager.get_candles(
                limit=limit
            )

    def test_latest_closed_candle(self):
        manager = CandleManager("5m")

        candles = [
            make_candle(
                open_time="2026-11-15 11:00:00+00:00"
            ),
            make_candle(
                open_time="2026-11-15 11:05:00+00:00"
            ),
        ]

        manager.ingest(
            make_frame(*candles),
            validation_time=CHECK_TIME,
        )

        latest = manager.get_latest_closed_candle()

        assert latest is not None

        assert latest["open_time"] == pd.Timestamp(
            "2026-11-15 11:05:00+00:00"
        )

    def test_latest_candle_returns_none_when_empty(self):
        manager = CandleManager("5m")

        assert manager.get_latest_closed_candle() is None

    def test_latest_candle_can_filter_source(self):
        manager = CandleManager("5m")

        manager.ingest(
            make_frame(
                make_candle(
                    source="kraken"
                ),
                make_candle(
                    source="coingecko"
                ),
            ),
            validation_time=CHECK_TIME,
        )

        latest = manager.get_latest_closed_candle(
            source="kraken"
        )

        assert latest["source"] == "kraken"