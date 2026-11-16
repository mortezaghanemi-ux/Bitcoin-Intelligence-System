"""Integration tests for the Bitcoin trading pipeline.

These tests use synthetic, validated OHLCV data.
They do not require an internet connection, exchange API key,
or real order execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trader.pipeline import (
    TradingPipeline,
    TradingPipelineError,
)


TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

MINIMUM_CANDLES = 35
SIGNAL_ID = "a" * 64


class FakeSignalEngine:
    """Deterministic signal engine for pipeline integration tests."""

    def __init__(
        self,
        signal_type: str = "LONG",
    ) -> None:
        self.signal_type = signal_type
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        *,
        analysis_time: datetime,
        account_equity: float | None = None,
        leverage: float = 1.0,
    ) -> dict[str, Any]:
        """Return a deterministic manual-review-only signal."""

        assert set(data) == set(TIMEFRAMES)

        for timeframe in TIMEFRAMES:
            assert len(data[timeframe]) >= MINIMUM_CANDLES

        self.calls.append({
            "analysis_time": analysis_time,
            "account_equity": account_equity,
            "leverage": leverage,
        })

        rejected = self.signal_type == "NO_TRADE"

        result = {
            "signal_id": SIGNAL_ID,
            "signal": self.signal_type,
            "status": (
                "REJECTED"
                if rejected
                else "REQUIRES_MANUAL_REVIEW"
            ),
            "analysis_timestamp": analysis_time.isoformat(),
            "confidence": 75.0,
            "data_quality": {
                "future_data_used": False,
                "closed_candles_only": True,
            },
            "execution": {
                "manual_review_only": True,
                "auto_trading": False,
                "exchange_connection": False,
                "order_execution": False,
            },
            "reasons": [
                "Synthetic integration-test signal."
            ],
        }

        if rejected:
            result["rejection_reasons"] = [
                "insufficient_market_confirmation"
            ]

        return result


def build_candles(
    timeframe: str,
    analysis_time: datetime,
    count: int = MINIMUM_CANDLES,
) -> pd.DataFrame:
    """Create chronological, closed UTC candles."""

    interval_minutes = TIMEFRAMES[timeframe]
    interval = timedelta(minutes=interval_minutes)

    current_timestamp = int(
        analysis_time.timestamp()
    )

    interval_seconds = interval_minutes * 60

    latest_close_timestamp = (
        current_timestamp // interval_seconds
    ) * interval_seconds

    latest_close = datetime.fromtimestamp(
        latest_close_timestamp,
        tz=timezone.utc,
    )

    rows = []
    base_price = 95000.0

    for index in range(count):
        close_time = (
            latest_close
            - interval * (count - index - 1)
        )

        open_time = close_time - interval

        open_price = base_price + index * 2.0
        close_price = open_price + 1.0

        rows.append({
            "open_time": open_time,
            "close_time": close_time,
            "open": open_price,
            "high": close_price + 5.0,
            "low": open_price - 5.0,
            "close": close_price,
            "volume": 10.0 + index,
            "source": "test",
        })

    return pd.DataFrame(rows)


def build_market_data(
    analysis_time: datetime,
) -> dict[str, pd.DataFrame]:
    """Build synthetic OHLCV data for all supported timeframes."""

    return {
        timeframe: build_candles(
            timeframe,
            analysis_time,
        )
        for timeframe in TIMEFRAMES
    }


def build_pipeline(
    tmp_path: Path,
    *,
    signal_type: str = "LONG",
) -> TradingPipeline:
    """Create a pipeline with isolated persistence files."""

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

    return TradingPipeline(
        history_path=history_path,
        rejection_path=rejection_path,
        engine=FakeSignalEngine(signal_type),
    )


# ---------------------------------------------------------
# Successful pipeline execution
# ---------------------------------------------------------


def test_pipeline_accepts_all_supported_timeframes(
    tmp_path: Path,
):
    """All four supported timeframes must pass validation."""

    analysis_time = datetime.now(timezone.utc)

    market_data = build_market_data(
        analysis_time
    )

    pipeline = build_pipeline(tmp_path)

    result = pipeline.run(
        market_data,
        analysis_time=analysis_time,
        account_equity=10000.0,
        leverage=1.0,
    )

    assert result["pipeline_status"] == "SUCCESS"

    assert set(
        result["validated_timeframes"]
    ) == set(TIMEFRAMES)

    for timeframe in TIMEFRAMES:
        assert (
            result["candle_counts"][timeframe]
            == MINIMUM_CANDLES
        )


def test_pipeline_generates_valid_signal(
    tmp_path: Path,
):
    """The pipeline must return a supported signal type."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(tmp_path)

    result = pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    signal = result["signal"]

    assert signal["signal_id"] == SIGNAL_ID

    assert signal["signal"] in (
        "LONG",
        "SHORT",
        "NO_TRADE",
    )

    assert signal["status"] in (
        "REJECTED",
        "REQUIRES_MANUAL_REVIEW",
    )

    assert signal["analysis_timestamp"] == (
        analysis_time.isoformat()
    )


def test_pipeline_confirms_closed_candles_and_no_future_data(
    tmp_path: Path,
):
    """Only closed candles and no future data may be used."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(tmp_path)

    result = pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    quality = result["signal"]["data_quality"]

    assert quality["future_data_used"] is False
    assert quality["closed_candles_only"] is True


# ---------------------------------------------------------
# Signal history persistence
# ---------------------------------------------------------


def test_pipeline_persists_signal_history(
    tmp_path: Path,
):
    """A generated signal must be saved to persistent history."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(tmp_path)

    result = pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    assert result["persistence"]["status"] == "SAVED"
    assert result["persistence"]["history_saved"] is True
    assert result["persistence"]["rejection_saved"] is False

    assert pipeline.history_store.count() == 1

    stored = pipeline.history_store.get(
        SIGNAL_ID
    )

    assert stored is not None
    assert stored["signal_id"] == SIGNAL_ID


def test_pipeline_prevents_duplicate_signal_records(
    tmp_path: Path,
):
    """Repeated signal identifiers must not create duplicates."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(tmp_path)

    market_data = build_market_data(
        analysis_time
    )

    first_result = pipeline.run(
        market_data,
        analysis_time=analysis_time,
    )

    second_result = pipeline.run(
        market_data,
        analysis_time=analysis_time,
    )

    assert first_result["persistence"]["status"] == "SAVED"

    assert second_result["persistence"]["status"] == "DUPLICATE"

    assert pipeline.history_store.count() == 1


def test_pipeline_restores_history_after_restart(
    tmp_path: Path,
):
    """Persisted signal history must survive pipeline recreation."""

    analysis_time = datetime.now(timezone.utc)

    first_pipeline = build_pipeline(tmp_path)

    first_pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    second_pipeline = build_pipeline(tmp_path)

    assert second_pipeline.history_store.count() == 1

    stored = second_pipeline.history_store.get(
        SIGNAL_ID
    )

    assert stored is not None
    assert stored["signal_id"] == SIGNAL_ID


# ---------------------------------------------------------
# Rejection persistence
# ---------------------------------------------------------


def test_pipeline_persists_rejected_no_trade_signal(
    tmp_path: Path,
):
    """Rejected NO_TRADE signals must enter rejection history."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(
        tmp_path,
        signal_type="NO_TRADE",
    )

    result = pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    assert result["signal"]["signal"] == "NO_TRADE"
    assert result["signal"]["status"] == "REJECTED"

    assert result["persistence"]["status"] == "SAVED"
    assert result["persistence"]["history_saved"] is True
    assert result["persistence"]["rejection_saved"] is True

    assert pipeline.history_store.count() == 1
    assert pipeline.rejection_store.count() == 1

    rejection = pipeline.rejection_store.get(
        SIGNAL_ID
    )

    assert rejection is not None
    assert rejection["signal"] == "NO_TRADE"
    assert rejection["status"] == "REJECTED"


def test_pipeline_prevents_duplicate_rejection_records(
    tmp_path: Path,
):
    """Repeated rejected signals must not duplicate rejection history."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(
        tmp_path,
        signal_type="NO_TRADE",
    )

    market_data = build_market_data(
        analysis_time
    )

    first_result = pipeline.run(
        market_data,
        analysis_time=analysis_time,
    )

    second_result = pipeline.run(
        market_data,
        analysis_time=analysis_time,
    )

    assert first_result["persistence"]["status"] == "SAVED"
    assert second_result["persistence"]["status"] == "DUPLICATE"

    assert pipeline.history_store.count() == 1
    assert pipeline.rejection_store.count() == 1

    assert second_result["persistence"]["rejection_saved"] is True


def test_pipeline_restores_rejections_after_restart(
    tmp_path: Path,
):
    """Rejection history must survive pipeline recreation."""

    analysis_time = datetime.now(timezone.utc)

    first_pipeline = build_pipeline(
        tmp_path,
        signal_type="NO_TRADE",
    )

    first_pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    second_pipeline = build_pipeline(
        tmp_path,
        signal_type="NO_TRADE",
    )

    assert second_pipeline.history_store.count() == 1
    assert second_pipeline.rejection_store.count() == 1

    rejection = second_pipeline.rejection_store.get(
        SIGNAL_ID
    )

    assert rejection is not None
    assert rejection["status"] == "REJECTED"


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------


def test_pipeline_rejects_missing_timeframe(
    tmp_path: Path,
):
    """Missing timeframe data must stop pipeline execution."""

    analysis_time = datetime.now(timezone.utc)

    market_data = build_market_data(
        analysis_time
    )

    del market_data["4h"]

    pipeline = build_pipeline(tmp_path)

    with pytest.raises(
        TradingPipelineError,
        match="Invalid timeframe collection",
    ):
        pipeline.run(
            market_data,
            analysis_time=analysis_time,
        )


def test_pipeline_rejects_unknown_timeframe(
    tmp_path: Path,
):
    """Unsupported timeframe data must be rejected."""

    analysis_time = datetime.now(timezone.utc)

    market_data = build_market_data(
        analysis_time
    )

    market_data["1d"] = build_candles(
        "1h",
        analysis_time,
    )

    pipeline = build_pipeline(tmp_path)

    with pytest.raises(
        TradingPipelineError,
        match="Invalid timeframe collection",
    ):
        pipeline.run(
            market_data,
            analysis_time=analysis_time,
        )


def test_pipeline_rejects_future_candles(
    tmp_path: Path,
):
    """A candle closing after analysis time must be rejected."""

    analysis_time = datetime.now(timezone.utc)

    market_data = build_market_data(
        analysis_time
    )

    market_data["5m"].loc[
        market_data["5m"].index[-1],
        "close_time",
    ] = (
        pd.Timestamp(analysis_time)
        + pd.Timedelta(minutes=5)
    )

    pipeline = build_pipeline(tmp_path)

    with pytest.raises(
        TradingPipelineError,
        match="candle validation failed",
    ):
        pipeline.run(
            market_data,
            analysis_time=analysis_time,
        )


def test_pipeline_rejects_invalid_ohlc_data(
    tmp_path: Path,
):
    """Invalid OHLC relationships must stop pipeline execution."""

    analysis_time = datetime.now(timezone.utc)

    market_data = build_market_data(
        analysis_time
    )

    last_index = market_data["5m"].index[-1]

    market_data["5m"].loc[
        last_index,
        "high",
    ] = 1.0

    pipeline = build_pipeline(tmp_path)

    with pytest.raises(
        TradingPipelineError,
        match="candle validation failed",
    ):
        pipeline.run(
            market_data,
            analysis_time=analysis_time,
        )


def test_pipeline_rejects_naive_analysis_time(
    tmp_path: Path,
):
    """Analysis timestamps must be timezone-aware UTC."""

    pipeline = build_pipeline(tmp_path)

    naive_time = datetime.now()

    with pytest.raises(
        TradingPipelineError,
        match="timezone-aware UTC",
    ):
        pipeline.run(
            {},
            analysis_time=naive_time,
        )


# ---------------------------------------------------------
# Risk parameter propagation
# ---------------------------------------------------------


def test_pipeline_passes_risk_parameters_to_engine(
    tmp_path: Path,
):
    """Equity and leverage must reach the signal engine unchanged."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(tmp_path)

    pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
        account_equity=10000.0,
        leverage=2.0,
    )

    assert len(pipeline.engine.calls) == 1

    call = pipeline.engine.calls[0]

    assert call["analysis_time"] == analysis_time
    assert call["account_equity"] == 10000.0
    assert call["leverage"] == 2.0


# ---------------------------------------------------------
# Execution safety
# ---------------------------------------------------------


def test_pipeline_enforces_execution_safety(
    tmp_path: Path,
):
    """The pipeline must never enable automatic trading or execution."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(tmp_path)

    result = pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    execution = result["execution"]

    assert execution["manual_review_only"] is True
    assert execution["auto_trading"] is False
    assert execution["exchange_connection"] is False
    assert execution["order_execution"] is False

    signal_execution = result["signal"]["execution"]

    assert signal_execution["manual_review_only"] is True
    assert signal_execution["auto_trading"] is False
    assert signal_execution["exchange_connection"] is False
    assert signal_execution["order_execution"] is False


# ---------------------------------------------------------
# Data integrity
# ---------------------------------------------------------


def test_pipeline_does_not_modify_input_data(
    tmp_path: Path,
):
    """Pipeline processing must not mutate caller-owned DataFrames."""

    analysis_time = datetime.now(timezone.utc)

    market_data = build_market_data(
        analysis_time
    )

    original_data = {
        timeframe: candles.copy(deep=True)
        for timeframe, candles in market_data.items()
    }

    pipeline = build_pipeline(tmp_path)

    pipeline.run(
        market_data,
        analysis_time=analysis_time,
    )

    for timeframe in TIMEFRAMES:
        pd.testing.assert_frame_equal(
            market_data[timeframe],
            original_data[timeframe],
        )


def test_pipeline_keeps_history_and_rejection_stores_separate(
    tmp_path: Path,
):
    """Accepted and rejected signal records must remain separated."""

    analysis_time = datetime.now(timezone.utc)

    pipeline = build_pipeline(
        tmp_path,
        signal_type="LONG",
    )

    pipeline.run(
        build_market_data(analysis_time),
        analysis_time=analysis_time,
    )

    assert pipeline.history_store.count() == 1
    assert pipeline.rejection_store.count() == 0