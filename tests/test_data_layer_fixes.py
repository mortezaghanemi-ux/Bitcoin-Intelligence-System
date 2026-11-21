"""Tests for freshness scope, candle persistence and FRED dedup."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from collectors.fred import _load_latest_values
from collectors.pipeline import _persist_accepted_candles
from validation.candles import validate_candles

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def make_candles(count, step_minutes=5):
    rows = []
    last_close = NOW - timedelta(minutes=1)

    for index in range(count):
        close_time = last_close - timedelta(
            minutes=step_minutes * (count - 1 - index)
        )
        open_time = close_time - timedelta(minutes=step_minutes)

        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "source": "kraken",
            }
        )

    return pd.DataFrame(rows)


def test_freshness_all_rejects_old_history():
    data = make_candles(5)

    accepted, rejected, _ = validate_candles(
        data,
        validation_time=NOW,
        max_age_minutes=15,
    )

    assert len(rejected) > 0
    assert "stale_candle" in rejected.iloc[0]["validation_errors"]


def test_freshness_latest_accepts_old_history():
    data = make_candles(5)

    accepted, rejected, _ = validate_candles(
        data,
        validation_time=NOW,
        max_age_minutes=15,
        freshness_scope="latest",
    )

    assert len(accepted) == 5
    assert rejected.empty


def test_freshness_latest_still_rejects_stale_newest():
    data = make_candles(3)
    data["open_time"] = data["open_time"] - timedelta(hours=2)
    data["close_time"] = data["close_time"] - timedelta(hours=2)

    accepted, rejected, _ = validate_candles(
        data,
        validation_time=NOW,
        max_age_minutes=15,
        freshness_scope="latest",
    )

    assert len(rejected) == 1
    assert "stale_candle" in rejected.iloc[0]["validation_errors"]
    assert len(accepted) == 2


def test_freshness_scope_rejects_unknown_value():
    with pytest.raises(ValueError):
        validate_candles(
            make_candles(1),
            validation_time=NOW,
            freshness_scope="nope",
        )


def test_persist_accepted_candles_deduplicates(tmp_path):
    data = make_candles(3)

    first = _persist_accepted_candles(
        data, "XBTUSD", "5m", tmp_path
    )
    second = _persist_accepted_candles(
        data, "XBTUSD", "5m", tmp_path
    )

    lines = (
        (tmp_path / "kraken_XBTUSD_5m.jsonl")
        .read_text()
        .strip()
        .splitlines()
    )

    assert first == 3
    assert second == 0
    assert len(lines) == 3
    assert json.loads(lines[0])["source"] == "kraken"


def test_fred_latest_values_keep_last_value(tmp_path):
    path = tmp_path / "fred.jsonl"

    records = [
        {"series_id": "DGS10", "observation_date": "2026-01-02", "value": 4.1},
        {"series_id": "DGS10", "observation_date": "2026-01-02", "value": 4.2},
        {"series_id": "FEDFUNDS", "observation_date": "2026-01-01", "value": 3.5},
    ]
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n"
    )

    latest = _load_latest_values(path)

    assert latest[("DGS10", "2026-01-02")] == 4.2
    assert latest[("FEDFUNDS", "2026-01-01")] == 3.5
    assert _load_latest_values(tmp_path / "missing.jsonl") == {}
