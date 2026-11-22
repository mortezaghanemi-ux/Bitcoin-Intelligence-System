"""Optional data sources must not block the trading signal.

A failure in the Kraken Futures, ETF, on-chain or macro source used to
stop the whole scheduler cycle, so analysis and the trader never ran,
even though none of them uses that data. Candle collection stays
critical.
"""

from datetime import datetime, timezone

import pytest

from collectors import kraken_futures
from scheduler.engine import SchedulerEngine, SchedulerError

from tests.test_kraken_futures import (
    COLLECTION_TIME,
    collect,
    make_payload,
    make_ticker,
    mock_response,
)
from tests.test_scheduler import make_config, make_engine, make_tasks, NOW


def config_with_sources(optional):
    config = make_config()
    config["derivatives"]["enabled"] = True
    config["macro"]["enabled"] = True
    config["execution"]["allow_retry_on_failure"] = False

    if optional is not None:
        config["execution"]["optional_collections"] = optional

    return config


def failing(events, name):
    def run():
        events.append(name)
        raise RuntimeError(f"{name} unavailable")

    return run


# ---------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------

def test_optional_source_failure_does_not_stop_the_cycle(tmp_path):
    events = []
    tasks = make_tasks(events)
    tasks["derivatives"] = failing(events, "derivatives")

    engine = make_engine(
        tmp_path,
        config=config_with_sources(["derivatives", "macro"]),
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "SUCCESS"
    assert result["degraded_sources"] == ["derivatives"]
    assert result["failed_tasks"] == 1

    # The chain that produces the signal still ran, in order.
    assert events[-4:] == [
        "validation",
        "market_analysis",
        "forecast",
        "trader",
    ]


def test_optional_failure_is_reported_for_every_failed_source(tmp_path):
    events = []
    tasks = make_tasks(events)
    tasks["derivatives"] = failing(events, "derivatives")
    tasks["macro"] = failing(events, "macro")

    engine = make_engine(
        tmp_path,
        config=config_with_sources(["derivatives", "macro"]),
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "SUCCESS"
    assert sorted(result["degraded_sources"]) == ["derivatives", "macro"]
    assert "trader" in events


def test_same_failure_still_stops_when_source_is_not_optional(tmp_path):
    events = []
    tasks = make_tasks(events)
    tasks["derivatives"] = failing(events, "derivatives")

    engine = make_engine(
        tmp_path,
        config=config_with_sources(None),
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "FAILED"
    assert result["degraded_sources"] == []
    assert "trader" not in events
    assert "validation" not in events


def test_candle_failure_is_always_critical(tmp_path):
    events = []
    tasks = make_tasks(events)
    tasks["market_data"] = failing(events, "market_data")

    engine = make_engine(
        tmp_path,
        config=config_with_sources(["derivatives", "macro"]),
        tasks=tasks,
    )

    result = engine.run_cycle(now=NOW)

    assert result["status"] == "FAILED"
    assert "trader" not in events


@pytest.mark.parametrize(
    "value",
    [
        ["market_data"],
        ["candles"],
        ["trader"],
        "derivatives",
        [1],
    ],
)
def test_critical_or_invalid_names_cannot_be_optional(tmp_path, value):
    config = make_config()
    config["execution"]["optional_collections"] = value

    with pytest.raises(SchedulerError):
        make_engine(tmp_path, config=config)


def test_shipped_configuration_marks_only_side_sources_optional():
    from config.loader import load_config

    execution = load_config()["schedules"]["schedules"]["execution"]

    assert execution["optional_collections"] == [
        "derivatives",
        "etf_flows",
        "onchain",
        "macro",
    ]


# ---------------------------------------------------------------
# Kraken Futures collector
# ---------------------------------------------------------------

STALE_TIME = "2026-11-19T04:00:00Z"


def test_stale_contract_is_skipped_and_fresh_ones_are_kept(
    tmp_path, monkeypatch
):
    mock_response(
        monkeypatch,
        make_payload(
            [
                make_ticker(symbol="FI_XBTUSD_261225", lastTime=STALE_TIME),
                make_ticker(symbol="PF_XBTUSD"),
            ]
        ),
    )

    records = collect(tmp_path)

    assert [record["symbol"] for record in records] == ["PF_XBTUSD"]


def test_malformed_contract_is_skipped_too(tmp_path, monkeypatch):
    broken = make_ticker(symbol="FI_XBTUSD_270326")
    del broken["lastTime"]

    mock_response(
        monkeypatch,
        make_payload([broken, make_ticker(symbol="PF_XBTUSD")]),
    )

    assert [r["symbol"] for r in collect(tmp_path)] == ["PF_XBTUSD"]


def test_all_contracts_stale_is_still_an_error(tmp_path, monkeypatch):
    mock_response(
        monkeypatch,
        make_payload(
            [
                make_ticker(symbol="PF_XBTUSD", lastTime=STALE_TIME),
                make_ticker(symbol="FI_XBTUSD_261225", lastTime=STALE_TIME),
            ]
        ),
    )

    with pytest.raises(
        kraken_futures.KrakenFuturesCollectorError,
        match="stale",
    ):
        collect(tmp_path)

    assert not (tmp_path / "kraken_futures_BTC.jsonl").exists()


def test_non_bitcoin_tickers_do_not_count_as_examined(
    tmp_path, monkeypatch
):
    mock_response(
        monkeypatch,
        make_payload([make_ticker(symbol="PF_ETHUSD", pair="ETH:USD")]),
    )

    assert collect(tmp_path) == []
