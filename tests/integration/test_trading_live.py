"""Live integration test for the complete Bitcoin trading pipeline.

This test fetches real OHLCV candles from Kraken and passes them
through the complete trading pipeline.

No exchange account, API key or order execution is used.

To enable:
    RUN_KRAKEN_LIVE_TESTS=1 pytest -v \
        tests/integration/test_trading_live.py
"""

from __future__ import annotations

import os

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from collectors.kraken_ohlcv import (
    collect_kraken_ohlcv,
)

from trader.pipeline import (
    TradingPipeline,
)


RUN_LIVE_TESTS = (
    os.getenv("RUN_KRAKEN_LIVE_TESTS", "").strip() == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_LIVE_TESTS,
    reason=(
        "Live Kraken trading integration tests are disabled. "
        "Set RUN_KRAKEN_LIVE_TESTS=1 to enable them."
    ),
)


TIMEFRAMES = (
    "5m",
    "15m",
    "1h",
    "4h",
)

REQUIRED_COLUMNS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
]

MINIMUM_CANDLES = 30


def collect_live_timeframe(
    timeframe: str,
    output_dir: Path,
    collection_time: datetime,
) -> pd.DataFrame:
    """Fetch real closed Kraken candles for one timeframe."""

    records = collect_kraken_ohlcv(
        pair="XBTUSD",
        timeframe=timeframe,
        output_dir=output_dir,
        collection_time=collection_time,
        timeout=20,
    )

    assert records, (
        f"Kraken returned no new closed candles "
        f"for timeframe {timeframe}."
    )

    candles = pd.DataFrame(records)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in candles.columns
    ]

    assert not missing_columns, (
        f"{timeframe} data is missing columns: "
        f"{missing_columns}"
    )

    candles = candles.loc[
        :,
        REQUIRED_COLUMNS,
    ].copy(deep=True)

    candles["open_time"] = pd.to_datetime(
        candles["open_time"],
        utc=True,
        errors="raise",
    )

    candles["close_time"] = pd.to_datetime(
        candles["close_time"],
        utc=True,
        errors="raise",
    )

    candles = candles.sort_values(
        "open_time",
        kind="stable",
    ).reset_index(drop=True)

    assert len(candles) >= MINIMUM_CANDLES, (
        f"{timeframe} requires at least "
        f"{MINIMUM_CANDLES} candles. "
        f"Received: {len(candles)}."
    )

    assert not candles["open_time"].duplicated().any(), (
        f"{timeframe} contains duplicate open timestamps."
    )

    assert not candles["close_time"].duplicated().any(), (
        f"{timeframe} contains duplicate close timestamps."
    )

    assert (
        candles["close_time"] <= pd.Timestamp(collection_time)
    ).all(), (
        f"{timeframe} contains an unclosed or future candle."
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        candles[column] = pd.to_numeric(
            candles[column],
            errors="raise",
        )

        assert candles[column].map(
            lambda value: pd.notna(value) and abs(float(value)) != float("inf")
        ).all(), (
            f"{timeframe} contains invalid {column} values."
        )

    assert (
        candles[["open", "high", "low", "close"]] > 0
    ).all().all(), (
        f"{timeframe} contains non-positive prices."
    )

    assert (candles["volume"] >= 0).all(), (
        f"{timeframe} contains negative volume."
    )

    invalid_ohlc = (
        (candles["high"] < candles["low"])
        | (candles["high"] < candles["open"])
        | (candles["high"] < candles["close"])
        | (candles["low"] > candles["open"])
        | (candles["low"] > candles["close"])
    )

    assert not invalid_ohlc.any(), (
        f"{timeframe} contains invalid OHLC relationships."
    )

    assert candles["source"].eq("kraken").all(), (
        f"{timeframe} contains an unexpected data source."
    )

    return candles


def test_live_kraken_data_runs_complete_trading_pipeline(
    tmp_path: Path,
):
    """Test real Kraken data through the complete trading pipeline."""

    # Use one consistent UTC timestamp for all timeframes.
    analysis_time = datetime.now(timezone.utc)

    raw_output = tmp_path / "raw"

    raw_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 1: Fetch real market data from Kraken.

    market_data = {}

    for timeframe in TIMEFRAMES:

        candles = collect_live_timeframe(
            timeframe=timeframe,
            output_dir=raw_output,
            collection_time=analysis_time,
        )

        market_data[timeframe] = candles

    # Step 2: Confirm all four timeframes were collected.

    assert set(market_data) == set(TIMEFRAMES)

    for timeframe in TIMEFRAMES:

        assert len(market_data[timeframe]) >= MINIMUM_CANDLES

    # Step 3: Create isolated signal history storage.

    history_path = (
        tmp_path
        / "history"
        / "signals.jsonl"
    )

    rejection_path = (
        tmp_path
        / "history"
        / "rejections.jsonl"
    )

    # Step 4: Initialize the actual trading pipeline.

    pipeline = TradingPipeline(
        history_path=history_path,
        rejection_path=rejection_path,
    )

    # Step 5: Run the complete trading analysis.

    result = pipeline.run(
        market_data,
        analysis_time=analysis_time,
        account_equity=10000.0,
        leverage=1.0,
    )

    # Step 6: Verify successful pipeline execution.

    assert isinstance(result, dict)

    assert result["pipeline_status"] == "SUCCESS"

    assert set(
        result["validated_timeframes"]
    ) == set(TIMEFRAMES)

    assert result["candle_counts"].keys() == market_data.keys()

    for timeframe in TIMEFRAMES:

        assert (
            result["candle_counts"][timeframe]
            >= MINIMUM_CANDLES
        )

    # Step 7: Verify signal generation.

    signal = result["signal"]

    assert isinstance(signal, dict)

    assert isinstance(
        signal.get("signal_id"),
        str,
    )

    assert signal["signal_id"]

    assert signal.get("signal") in (
        "LONG",
        "SHORT",
        "NO_TRADE",
    )

    assert signal.get("data_quality", {}).get(
        "future_data_used"
    ) is False

    assert signal.get("data_quality", {}).get(
        "closed_candles_only"
    ) is True

    # Step 8: Verify signal persistence.

    persistence = result["persistence"]

    assert persistence["status"] in (
        "SAVED",
        "DUPLICATE",
    )

    assert persistence["history_saved"] is True

    assert history_path.is_file(), (
        "Signal history file was not created."
    )

    # Step 9: Verify execution safety.

    execution = result["execution"]

    assert execution["manual_review_only"] is True
    assert execution["auto_trading"] is False
    assert execution["exchange_connection"] is False
    assert execution["order_execution"] is False

    # Step 10: Verify that no exchange order was executed.

    assert signal.get("execution", {}).get(
        "manual_review_only"
    ) is True

    assert signal.get("execution", {}).get(
        "auto_trading"
    ) is False

    assert signal.get("execution", {}).get(
        "exchange_connection"
    ) is False

    assert signal.get("execution", {}).get(
        "order_execution"
    ) is False