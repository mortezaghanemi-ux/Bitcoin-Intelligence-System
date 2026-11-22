"""Regression tests for the first-collection batch of candles.

On a first collection Kraken returns up to 720 candles, almost all of
them historical. The freshness limit must apply to the newest candle
only. Applying it to every candle rejected 717 of 720 four-hour candles
(only the 3 closed within 720 minutes survived), the pipeline reported
FAIL, and the scheduler cycle stopped.
"""

from datetime import datetime, timedelta, timezone

import pytest

from collectors import kraken_ohlcv
from collectors.pipeline import run_kraken_pipeline

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}

# The per-timeframe limits from config/data_quality.yaml.
CONFIGURED_LIMITS = {"5m": 15, "15m": 45, "1h": 180, "4h": 720}

BATCH_SIZE = 720


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def kraken_payload(timeframe, now, batch=BATCH_SIZE):
    """Build a response shaped like Kraken's real OHLC answer.

    The final entry is the candle that is still forming, exactly as
    Kraken returns it.
    """

    minutes = TIMEFRAME_MINUTES[timeframe]
    step = minutes * 60
    now_epoch = int(now.timestamp())

    forming_open = now_epoch - (now_epoch % step)

    rows = []

    for index in range(batch):
        open_epoch = forming_open - step * (batch - 1 - index)
        price = 90000.0 + index * 0.5

        rows.append(
            [
                open_epoch,
                str(price),
                str(price + 50),
                str(price - 50),
                str(price + 10),
                str(price),
                "12.5",
                30,
            ]
        )

    return {
        "error": [],
        "result": {"XXBTZUSD": rows, "last": forming_open},
    }


def run_batch(tmp_path, monkeypatch, timeframe, now):
    monkeypatch.setattr(
        kraken_ohlcv.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            kraken_payload(timeframe, now)
        ),
    )

    return run_kraken_pipeline(
        pair="XBTUSD",
        timeframe=timeframe,
        output_dir=tmp_path / "raw",
        report_path=tmp_path / "report.csv",
        collection_time=now,
        processed_dir=tmp_path / "processed",
    )


# 07:30 UTC on a normal day: a few minutes after a 5m, 15m, 1h and 4h
# boundary is not guaranteed, so pick a time inside every limit.
RUN_TIME = datetime(2026, 11, 22, 4, 7, tzinfo=timezone.utc)


@pytest.mark.parametrize("timeframe", list(TIMEFRAME_MINUTES))
def test_first_collection_batch_passes_for_every_timeframe(
    tmp_path, monkeypatch, timeframe
):
    result = run_batch(tmp_path, monkeypatch, timeframe, RUN_TIME)

    # 720 returned, the forming one is dropped, 719 are closed.
    assert result["collected_records"] == BATCH_SIZE - 1
    assert result["status"] == "PASS", result["rejected"].head()
    assert len(result["accepted"]) == BATCH_SIZE - 1
    assert result["rejected"].empty
    assert result["max_age_minutes"] == CONFIGURED_LIMITS[timeframe]


@pytest.mark.parametrize("timeframe", list(TIMEFRAME_MINUTES))
def test_stale_newest_candle_is_still_rejected(
    tmp_path, monkeypatch, timeframe
):
    """The freshness rule still protects against dead data feeds."""

    stale = RUN_TIME + timedelta(days=3)

    monkeypatch.setattr(
        kraken_ohlcv.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            kraken_payload(timeframe, RUN_TIME)
        ),
    )

    result = run_kraken_pipeline(
        pair="XBTUSD",
        timeframe=timeframe,
        output_dir=tmp_path / "raw",
        report_path=tmp_path / "report.csv",
        collection_time=stale,
    )

    assert result["status"] == "FAIL"
    assert not result["rejected"].empty


def test_validated_history_is_persisted_once(tmp_path, monkeypatch):
    run_batch(tmp_path, monkeypatch, "4h", RUN_TIME)
    run_batch(tmp_path, monkeypatch, "4h", RUN_TIME)

    stored = (
        tmp_path / "processed" / "kraken_XBTUSD_4h.jsonl"
    ).read_text().splitlines()

    assert len(stored) == BATCH_SIZE - 1
