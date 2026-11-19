"""Tests for the persisted Kraken trading data loader."""

import json

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.data_loader import (
    TRADING_TIMEFRAMES,
    TradingDataLoaderError,
    load_timeframe_history,
    load_trading_candles,
)


PAIR = "XBTUSD"

BASE_TIME = datetime(
    2026,
    1,
    1,
    0,
       0,
    tzinfo=timezone.utc,
)


def make_record(
    timeframe="5m",
    open_time=None,
    *,
    pair=PAIR,
    source="kraken",
    opened="100",
    high="110",
    low="90",
    closed="105",
    volume="12.5",
):
    """Create one valid persisted Kraken candle record."""

    from collectors.kraken_ohlcv import TIMEFRAME_MINUTES

    if open_time is None:
        open_time = BASE_TIME

    close_time = open_time + timedelta(
        minutes=TIMEFRAME_MINUTES[timeframe]
    )

    return {
        "schema_version": 1,
        "source": source,
        "pair": pair,
        "timeframe": timeframe,
        "open_time": open_time.isoformat().replace(
            "+00:00",
            "Z",
        ),
        "close_time": close_time.isoformat().replace(
            "+00:00",
            "Z",
        ),
        "open": opened,
        "high": high,
        "low": low,
        "close": closed,
        "volume": volume,
        "collected_at": "2026-01-01T00:10:00Z",
    }


def write_records(
    directory,
    timeframe,
    records,
    pair=PAIR,
):
    """Write test records to the expected JSONL file."""

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = directory / (
        f"kraken_{pair}_{timeframe}.jsonl"
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(record) + "\n"
            )

    return file_path


# --------------------------------------------------
# Valid history tests
# --------------------------------------------------

def test_loads_valid_timeframe_history(tmp_path):

    record = make_record()

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    result = load_timeframe_history(
        "5m",
        data_dir=tmp_path,
    )

    assert len(result) == 1

    assert result.loc[0, "source"] == "kraken"

    assert result.loc[0, "open"] == 100.0
    assert result.loc[0, "high"] == 110.0
    assert result.loc[0, "low"] == 90.0
    assert result.loc[0, "close"] == 105.0
    assert result.loc[0, "volume"] == 12.5

    assert pd.api.types.is_numeric_dtype(
        result["open"]
    )

    assert pd.api.types.is_numeric_dtype(
        result["volume"]
    )


def test_sorts_history_by_open_time(tmp_path):

    second_time = BASE_TIME + timedelta(minutes=5)

    records = [
        make_record(
            open_time=second_time,
        ),
        make_record(
            open_time=BASE_TIME,
        ),
    ]

    write_records(
        tmp_path,
        "5m",
        records,
    )

    result = load_timeframe_history(
        "5m",
        data_dir=tmp_path,
    )

    assert result["open_time"].is_monotonic_increasing

    assert (
        result.loc[0, "open_time"]
        < result.loc[1, "open_time"]
    )


def test_loads_all_trading_timeframes(tmp_path):

    for timeframe in TRADING_TIMEFRAMES:

        write_records(
            tmp_path,
            timeframe,
            [
                make_record(
                    timeframe=timeframe,
                )
            ],
        )

    result = load_trading_candles(
        data_dir=tmp_path,
    )

    assert set(result) == set(TRADING_TIMEFRAMES)

    for timeframe in TRADING_TIMEFRAMES:

        assert isinstance(
            result[timeframe],
            pd.DataFrame,
        )

        assert len(result[timeframe]) == 1


def test_does_not_modify_raw_jsonl_file(tmp_path):

    record = make_record()

    file_path = write_records(
        tmp_path,
        "5m",
        [record],
    )

    original_content = file_path.read_text(
        encoding="utf-8"
    )

    load_timeframe_history(
        "5m",
        data_dir=tmp_path,
    )

    updated_content = file_path.read_text(
        encoding="utf-8"
    )

    assert updated_content == original_content


# --------------------------------------------------
# Numeric validation tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "field,value",
    [
        ("open", "invalid"),
        ("high", "invalid"),
        ("low", "invalid"),
        ("close", "invalid"),
        ("volume", "invalid"),
        ("open", "NaN"),
        ("close", "inf"),
        ("volume", "-inf"),
    ],
)
def test_rejects_invalid_numeric_values(
    tmp_path,
    field,
    value,
):

    record = make_record()
    record[field] = value

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(TradingDataLoaderError):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", "0"),
        ("high", "0"),
        ("low", "0"),
        ("close", "0"),
        ("volume", "-1"),
    ],
)
def test_rejects_non_positive_prices_and_negative_volume(
    tmp_path,
    field,
    value,
):

    record = make_record()
    record[field] = value

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(TradingDataLoaderError):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_high_below_open(tmp_path):

    record = make_record(
        opened="100",
        high="99",
        low="90",
        closed="95",
    )

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="invalid high price",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_low_above_close(tmp_path):

    record = make_record(
        opened="100",
        high="120",
        low="106",
        closed="105",
    )

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="invalid low price",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


# --------------------------------------------------
# Timestamp validation tests
# --------------------------------------------------

def test_rejects_timezone_naive_timestamp(tmp_path):

    record = make_record()

    record["open_time"] = "2026-01-01T00:00:00"

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="UTC timezone",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_non_utc_timestamp(tmp_path):

    record = make_record()

    record["open_time"] = "2026-01-01T03:00:00+03:00"

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="must use UTC",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_invalid_candle_duration(tmp_path):

    record = make_record()

    record["close_time"] = "2026-01-01T00:06:00Z"

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Invalid candle duration",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_misaligned_candle_timestamp(tmp_path):

    misaligned_time = BASE_TIME + timedelta(
        minutes=1
    )

    record = make_record(
        open_time=misaligned_time,
    )

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Misaligned candle timestamp",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


# --------------------------------------------------
# Record integrity tests
# --------------------------------------------------

def test_rejects_duplicate_open_time(tmp_path):

    record = make_record()

    write_records(
        tmp_path,
        "5m",
        [
            record,
            record,
        ],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="duplicate candle timestamps",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_missing_required_metadata(tmp_path):

    record = make_record()

    del record["pair"]

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Missing fields",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_unsupported_schema_version(tmp_path):

    record = make_record()
    record["schema_version"] = 2

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Unsupported schema_version",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_boolean_schema_version(tmp_path):

    record = make_record()
    record["schema_version"] = True

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Unsupported schema_version",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_unexpected_pair(tmp_path):

    record = make_record(
        pair="ETHUSD",
    )

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Unexpected trading pair",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_unexpected_source(tmp_path):

    record = make_record(
        source="coinbase",
    )

    write_records(
        tmp_path,
        "5m",
        [record],
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Unexpected data source",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_corrupted_jsonl(tmp_path):

    file_path = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    file_path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="Corrupted JSONL",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_empty_history_file(tmp_path):

    file_path = (
        tmp_path / "kraken_XBTUSD_5m.jsonl"
    )

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        TradingDataLoaderError,
        match="No candle records",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_missing_history_file(tmp_path):

    with pytest.raises(
        TradingDataLoaderError,
        match="does not exist",
    ):

        load_timeframe_history(
            "5m",
            data_dir=tmp_path,
        )


def test_rejects_unsupported_timeframe(tmp_path):

    with pytest.raises(ValueError):

        load_timeframe_history(
            "2m",
            data_dir=tmp_path,
        )


def test_rejects_invalid_pair(tmp_path):

    with pytest.raises(ValueError):

        load_timeframe_history(
            "5m",
            pair="BTC/USD",
            data_dir=tmp_path,
        )


def test_rejects_missing_timeframe_during_full_load(tmp_path):

    for timeframe in TRADING_TIMEFRAMES:

        if timeframe == "4h":
            continue

        write_records(
            tmp_path,
            timeframe,
            [
                make_record(
                    timeframe=timeframe,
                )
            ],
        )

    with pytest.raises(
        TradingDataLoaderError,
        match="does not exist",
    ):

        load_trading_candles(
            data_dir=tmp_path,
        )