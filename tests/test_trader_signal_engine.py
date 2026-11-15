"""Tests for the Bitcoin trading signal engine."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import trader.signal_engine.engine as signal_module

from trader.signal_engine.engine import (
    TradingSignalEngine,
    TradingSignalEngineError,
    generate_trading_signal,
    _calculate_atr,
    _calculate_confidence,
    _calculate_position_risk,
    _direction,
    _select_order_block_direction,
    _select_pattern_direction,
)


UTC = timezone.utc
CHECK_TIME = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}

TIMEFRAMES = ("5m", "15m", "1h", "4h")


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


def make_candles(
    timeframe="5m",
    count=40,
    start_price=100.0,
):
    interval = TIMEFRAME_SECONDS[timeframe]

    start = CHECK_TIME - timedelta(
        seconds=interval * (count + 2)
    )

    rows = []

    for index in range(count):
        open_time = start + timedelta(
            seconds=index * interval
        )

        close_time = open_time + timedelta(
            seconds=interval
        )

        opened = start_price + index * 0.1
        closed = opened + 0.05

        rows.append({
            "open_time": open_time,
            "close_time": close_time,
            "open": opened,
            "high": closed + 0.5,
            "low": opened - 0.5,
            "close": closed,
            "volume": 10.0 + index,
            "source": "kraken",
        })

    return pd.DataFrame(rows)


def make_market_data():
    return {
        timeframe: make_candles(timeframe)
        for timeframe in TIMEFRAMES
    }


def install_mock_analyzers(
    monkeypatch,
    direction="BULLISH",
    structure_direction="BULLISH",
    conflict_count=0,
    alignment_ratio=1.0,
    volume_direction="BULLISH",
):
    class FakeMultiTimeframeAnalyzer:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def analyze(self, data, analysis_time=None):
            return {
                "overall_direction": direction,
                "alignment": {
                    "alignment_ratio": alignment_ratio,
                    "conflict_count": conflict_count,
                    "status": "ALIGNED",
                    "directions": {
                        timeframe: direction
                        for timeframe in TIMEFRAMES
                    },
                },
                "timeframes": {
                    timeframe: {
                        "volume": {
                            "direction": volume_direction
                        }
                    }
                    for timeframe in TIMEFRAMES
                },
            }

    class FakeMarketStructureAnalyzer:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def analyze(self, candles, analysis_time=None):
            return {
                "trend": {
                    "direction": structure_direction
                }
            }

    class FakeOrderBlockDetector:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def detect(self, candles, analysis_time=None):
            return {
                "order_blocks": [],
                "fair_value_gaps": [],
                "summary": {
                    "active_order_blocks": 0
                },
            }

    class FakePatternDetector:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def detect(self, candles, detection_time=None):
            return pd.DataFrame(
                columns=[
                    "pattern_id",
                    "pattern",
                    "direction",
                    "confidence",
                ]
            )

    monkeypatch.setattr(
        signal_module,
        "MultiTimeframeAnalyzer",
        FakeMultiTimeframeAnalyzer,
    )

    monkeypatch.setattr(
        signal_module,
        "MarketStructureAnalyzer",
        FakeMarketStructureAnalyzer,
    )

    monkeypatch.setattr(
        signal_module,
        "OrderBlockDetector",
        FakeOrderBlockDetector,
    )

    monkeypatch.setattr(
        signal_module,
        "CandlestickPatternDetector",
        FakePatternDetector,
    )


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


def test_default_configuration():
    engine = TradingSignalEngine()

    assert engine.minimum_candles == 30
    assert engine.minimum_confidence == 60.0
    assert engine.minimum_risk_reward == 1.5
    assert engine.maximum_leverage == 2.0
    assert engine.swing_window == 3
    assert engine.execution_timeframe == "5m"


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum_candles": 0},
        {"minimum_candles": 4},
        {"minimum_candles": True},
        {"minimum_confidence": -1},
        {"minimum_confidence": 101},
        {"minimum_confidence": float("nan")},
        {"minimum_confidence": True},
        {"minimum_risk_reward": 0},
        {"minimum_risk_reward": -1},
        {"minimum_risk_reward": float("inf")},
        {"maximum_leverage": 0},
        {"maximum_leverage": 3},
        {"maximum_leverage": float("nan")},
        {"maximum_leverage": True},
        {"swing_window": 0},
        {"swing_window": -1},
        {"swing_window": True},
        {"risk_fraction": 0},
        {"risk_fraction": 0.1},
        {"risk_fraction": float("nan")},
        {"maintenance_margin_rate": -0.1},
        {"maintenance_margin_rate": 0.1},
    ],
)
def test_invalid_configuration_is_rejected(parameters):
    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine(**parameters)


def test_minimum_candles_must_match_swing_window():
    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine(
            minimum_candles=10,
            swing_window=5,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["1m", "1d", "", None],
)
def test_invalid_execution_timeframe_is_rejected(timeframe):
    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine(
            execution_timeframe=timeframe
        )


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_all_supported_execution_timeframes(timeframe):
    engine = TradingSignalEngine(
        execution_timeframe=timeframe
    )

    assert engine.execution_timeframe == timeframe


# ---------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------


def test_analysis_time_must_be_timezone_aware(monkeypatch):
    install_mock_analyzers(monkeypatch)

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            make_market_data(),
            analysis_time=datetime(2026, 11, 15, 12, 0),
        )


def test_analysis_time_must_use_utc(monkeypatch):
    install_mock_analyzers(monkeypatch)

    non_utc = datetime(
        2026,
        11,
        15,
        14,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            make_market_data(),
            analysis_time=non_utc,
        )


def test_missing_analysis_time_is_supported(monkeypatch):
    install_mock_analyzers(monkeypatch)

    result = TradingSignalEngine().generate(
        make_market_data()
    )

    assert result["analysis_timestamp"]
    assert result["data_quality"]["status"] == "VALIDATED"


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------


def test_non_dictionary_input_is_rejected():
    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            [],
            analysis_time=CHECK_TIME,
        )


def test_missing_timeframe_is_rejected():
    data = make_market_data()
    del data["4h"]

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_unknown_timeframe_is_rejected():
    data = make_market_data()
    data["1d"] = make_candles("1h")

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_non_dataframe_input_is_rejected(timeframe):
    data = make_market_data()
    data[timeframe] = []

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_empty_dataframe_is_rejected(timeframe):
    data = make_market_data()
    data[timeframe] = pd.DataFrame()

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_missing_required_column_is_rejected():
    data = make_market_data()
    data["5m"] = data["5m"].drop(columns=["volume"])

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_insufficient_candles_are_rejected():
    data = make_market_data()
    data["5m"] = make_candles("5m", count=10)

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


@pytest.mark.parametrize(
    "column",
    ["open_time", "close_time"],
)
def test_naive_timestamps_are_rejected(column):
    data = make_market_data()

    data["5m"][column] = (
        data["5m"][column].dt.tz_localize(None)
    )

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_non_utc_timestamps_are_rejected():
    data = make_market_data()

    data["5m"]["open_time"] = (
        data["5m"]["open_time"]
        .dt.tz_convert("Europe/Berlin")
    )

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_future_candles_are_rejected():
    data = make_market_data()

    future_open = CHECK_TIME + timedelta(minutes=5)

    data["5m"].loc[
        data["5m"].index[-1],
        "open_time",
    ] = future_open

    data["5m"].loc[
        data["5m"].index[-1],
        "close_time",
    ] = future_open + timedelta(minutes=5)

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_unsorted_candles_are_rejected():
    data = make_market_data()

    data["5m"] = data["5m"].iloc[::-1].reset_index(drop=True)

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_duplicate_candles_are_rejected():
    data = make_market_data()

    data["5m"].loc[1, "open_time"] = (
        data["5m"].loc[0, "open_time"]
    )

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close", "volume"],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        "invalid",
    ],
)
def test_invalid_numeric_values_are_rejected(column, bad_value):
    data = make_market_data()

    if isinstance(bad_value, str):
        data["5m"][column] = (
            data["5m"][column].astype(object)
        )

    data["5m"].loc[0, column] = bad_value

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


@pytest.mark.parametrize(
    "column",
    ["open", "high", "low", "close"],
)
def test_non_positive_prices_are_rejected(column):
    data = make_market_data()
    data["5m"].loc[0, column] = 0

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_negative_volume_is_rejected():
    data = make_market_data()
    data["5m"].loc[0, "volume"] = -1

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_boolean_price_is_rejected():
    data = make_market_data()

    data["5m"]["close"] = (
        data["5m"]["close"].astype(object)
    )

    data["5m"].loc[0, "close"] = True

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_invalid_ohlc_relationship_is_rejected():
    data = make_market_data()
    data["5m"].loc[0, "high"] = 1

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


def test_invalid_source_is_rejected():
    data = make_market_data()
    data["5m"].loc[0, "source"] = ""

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            data,
            analysis_time=CHECK_TIME,
        )


# ---------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("BULLISH", "BULLISH"),
        ("BEARISH", "BEARISH"),
        ("NEUTRAL", "NEUTRAL"),
        ("LONG", "BULLISH"),
        ("SHORT", "BEARISH"),
        (" bullish ", "BULLISH"),
        ("invalid", "NEUTRAL"),
        (None, "NEUTRAL"),
        (123, "NEUTRAL"),
    ],
)
def test_direction_normalization(value, expected):
    assert _direction(value) == expected


# ---------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------


def test_confidence_calculation():
    score, reasons = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=1.0,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=0,
    )

    assert score == 100.0
    assert len(reasons) > 0


def test_confidence_penalizes_conflicts():
    score, reasons = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=0.5,
        structure_direction="BEARISH",
        order_block_direction="NEUTRAL",
        pattern_direction="NEUTRAL",
        volume_direction="NEUTRAL",
        conflict_count=2,
    )

    assert score < 40
    assert any("conflicts" in reason for reason in reasons)


def test_confidence_is_limited_to_zero_and_one_hundred():
    score, _ = _calculate_confidence(
        overall_direction="BULLISH",
        alignment_ratio=10.0,
        structure_direction="BULLISH",
        order_block_direction="BULLISH",
        pattern_direction="BULLISH",
        volume_direction="BULLISH",
        conflict_count=0,
    )

    assert 0 <= score <= 100


# ---------------------------------------------------------
# Order block and pattern helpers
# ---------------------------------------------------------


def test_order_block_direction_is_selected():
    results = {
        "5m": {
            "order_blocks": [
                {
                    "status": "ACTIVE",
                    "direction": "BULLISH",
                    "lower": 99,
                    "upper": 101,
                }
            ]
        }
    }

    assert _select_order_block_direction(
        results,
        100,
    ) == "BULLISH"


def test_nearest_order_block_is_selected():
    results = {
        "5m": {
            "order_blocks": [
                {
                    "status": "ACTIVE",
                    "direction": "BEARISH",
                    "lower": 105,
                    "upper": 106,
                }
            ]
        },
        "1h": {
            "order_blocks": [
                {
                    "status": "ACTIVE",
                    "direction": "BULLISH",
                    "lower": 99,
                    "upper": 101,
                }
            ]
        },
    }

    assert _select_order_block_direction(
        results,
        100,
    ) == "BULLISH"


def test_inactive_order_blocks_are_ignored():
    results = {
        "5m": {
            "order_blocks": [
                {
                    "status": "MITIGATED",
                    "direction": "BULLISH",
                    "lower": 99,
                    "upper": 101,
                }
            ]
        }
    }

    assert _select_order_block_direction(
        results,
        100,
    ) == "NEUTRAL"


def test_conflicting_nearest_order_blocks_return_neutral():
    results = {
        "5m": {
            "order_blocks": [
                {
                    "status": "ACTIVE",
                    "direction": "BULLISH",
                    "lower": 99,
                    "upper": 101,
                }
            ]
        },
        "1h": {
            "order_blocks": [
                {
                    "status": "ACTIVE",
                    "direction": "BEARISH",
                    "lower": 99,
                    "upper": 101,
                }
            ]
        },
    }

    assert _select_order_block_direction(
        results,
        100,
    ) == "NEUTRAL"


def test_pattern_direction_bullish():
    results = {
        "5m": pd.DataFrame({
            "direction": ["BULLISH", "BULLISH", "BEARISH"]
        })
    }

    assert _select_pattern_direction(results) == "BULLISH"


def test_pattern_direction_bearish():
    results = {
        "5m": pd.DataFrame({
            "direction": ["BEARISH", "BEARISH", "BULLISH"]
        })
    }

    assert _select_pattern_direction(results) == "BEARISH"


def test_equal_patterns_return_neutral():
    results = {
        "5m": pd.DataFrame({
            "direction": ["BULLISH", "BEARISH"]
        })
    }

    assert _select_pattern_direction(results) == "NEUTRAL"


def test_empty_patterns_return_neutral():
    results = {
        "5m": pd.DataFrame(columns=["direction"])
    }

    assert _select_pattern_direction(results) == "NEUTRAL"


# ---------------------------------------------------------
# ATR and position risk
# ---------------------------------------------------------


def test_atr_is_positive():
    atr = _calculate_atr(make_candles())

    assert atr > 0


def test_position_risk_without_equity():
    result = _calculate_position_risk(
        direction="BULLISH",
        entry_price=100,
        stop_loss=95,
        account_equity=None,
        risk_fraction=0.005,
        leverage=1,
        maintenance_margin_rate=0.005,
    )

    assert result["position"]["status"] == "NOT_CALCULATED"
    assert result["position"]["quantity_btc"] is None
    assert result["liquidation"]["status"] == "ESTIMATE_ONLY"


def test_position_risk_with_equity():
    result = _calculate_position_risk(
        direction="BULLISH",
        entry_price=100,
        stop_loss=95,
        account_equity=10000,
        risk_fraction=0.005,
        leverage=2,
        maintenance_margin_rate=0.005,
    )

    assert result["position"]["status"] == "CALCULATED"
    assert result["position"]["risk_amount"] == 50
    assert result["position"]["quantity_btc"] == 10
    assert result["position"]["leverage"] == 2


@pytest.mark.parametrize(
    "equity",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        True,
        "invalid",
    ],
)
def test_invalid_account_equity_is_rejected(equity):
    with pytest.raises(TradingSignalEngineError):
        _calculate_position_risk(
            direction="BULLISH",
            entry_price=100,
            stop_loss=95,
            account_equity=equity,
            risk_fraction=0.005,
            leverage=1,
            maintenance_margin_rate=0.005,
        )


# ---------------------------------------------------------
# Signal generation
# ---------------------------------------------------------


def test_generates_bullish_signal(monkeypatch):
    install_mock_analyzers(monkeypatch)

    engine = TradingSignalEngine()

    result = engine.generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
        account_equity=10000,
        leverage=2,
    )

    assert result["signal"] == "LONG"
    assert result["direction"] == "BULLISH"
    assert result["status"] == "REQUIRES_MANUAL_REVIEW"
    assert result["confidence"] >= 60

    assert result["entry_range"] is not None
    assert result["stop_loss"] > 0
    assert len(result["take_profit"]) == 3
    assert result["risk_reward"] >= 1.5

    assert result["risk_calculation"]["position"]["status"] == "CALCULATED"


def test_generates_bearish_signal(monkeypatch):
    install_mock_analyzers(
        monkeypatch,
        direction="BEARISH",
        structure_direction="BEARISH",
        volume_direction="BEARISH",
    )

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["signal"] == "SHORT"
    assert result["direction"] == "BEARISH"
    assert result["status"] == "REQUIRES_MANUAL_REVIEW"


def test_neutral_direction_produces_no_trade(monkeypatch):
    install_mock_analyzers(
        monkeypatch,
        direction="NEUTRAL",
        structure_direction="NEUTRAL",
        alignment_ratio=0.0,
        volume_direction="NEUTRAL",
    )

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["signal"] == "NO_TRADE"
    assert result["status"] == "REJECTED"
    assert result["entry_range"] is None
    assert result["stop_loss"] is None
    assert result["rejection_reasons"]


def test_conflicting_timeframes_produce_no_trade(monkeypatch):
    install_mock_analyzers(
        monkeypatch,
        direction="BULLISH",
        structure_direction="BULLISH",
        conflict_count=2,
    )

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["signal"] == "NO_TRADE"
    assert any(
        "Conflicting timeframe" in reason
        for reason in result["rejection_reasons"]
    )


def test_structure_conflict_produces_no_trade(monkeypatch):
    install_mock_analyzers(
        monkeypatch,
        direction="BULLISH",
        structure_direction="BEARISH",
    )

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["signal"] == "NO_TRADE"
    assert any(
        "structure conflicts" in reason.lower()
        for reason in result["rejection_reasons"]
    )


def test_low_confidence_produces_no_trade(monkeypatch):
    install_mock_analyzers(
        monkeypatch,
        direction="BULLISH",
        structure_direction="NEUTRAL",
        alignment_ratio=0.1,
        volume_direction="NEUTRAL",
    )

    result = TradingSignalEngine(
        minimum_confidence=90,
    ).generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["signal"] == "NO_TRADE"
    assert any(
        "Confidence is below" in reason
        for reason in result["rejection_reasons"]
    )


# ---------------------------------------------------------
# Safety and data quality
# ---------------------------------------------------------


def test_signal_never_enables_automatic_trading(monkeypatch):
    install_mock_analyzers(monkeypatch)

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["execution"]["manual_review_only"] is True
    assert result["execution"]["auto_trading"] is False
    assert result["execution"]["exchange_connection"] is False
    assert result["execution"]["order_execution"] is False


def test_signal_reports_closed_candles_only(monkeypatch):
    install_mock_analyzers(monkeypatch)

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    quality = result["data_quality"]

    assert quality["status"] == "VALIDATED"
    assert quality["closed_candles_only"] is True
    assert quality["future_data_used"] is False
    assert quality["validated_timeframes"] == list(TIMEFRAMES)


def test_signal_contains_all_data_sources(monkeypatch):
    install_mock_analyzers(monkeypatch)

    result = TradingSignalEngine().generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert result["data_quality"]["sources"] == ["kraken"]


# ---------------------------------------------------------
# Leverage validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "leverage",
    [
        0,
        -1,
        2.1,
        float("nan"),
        float("inf"),
        True,
        "invalid",
    ],
)
def test_invalid_leverage_is_rejected(monkeypatch, leverage):
    install_mock_analyzers(monkeypatch)

    with pytest.raises(TradingSignalEngineError):
        TradingSignalEngine().generate(
            make_market_data(),
            analysis_time=CHECK_TIME,
            leverage=leverage,
        )


# ---------------------------------------------------------
# History and duplicate prevention
# ---------------------------------------------------------


def test_signal_is_saved_in_history(monkeypatch):
    install_mock_analyzers(monkeypatch)

    engine = TradingSignalEngine()

    result = engine.generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert len(engine.history) == 1
    assert engine.history[0]["signal_id"] == result["signal_id"]


def test_duplicate_signal_is_not_saved_twice(monkeypatch):
    install_mock_analyzers(monkeypatch)

    engine = TradingSignalEngine()
    data = make_market_data()

    first = engine.generate(
        data,
        analysis_time=CHECK_TIME,
    )

    second = engine.generate(
        data,
        analysis_time=CHECK_TIME,
    )

    assert len(engine.history) == 1
    assert first["signal_id"] == second["signal_id"]
    assert second["duplicate"] is True


def test_history_returns_defensive_copy(monkeypatch):
    install_mock_analyzers(monkeypatch)

    engine = TradingSignalEngine()

    engine.generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    history = engine.history
    history[0]["signal"] = "MODIFIED"

    assert engine.history[0]["signal"] != "MODIFIED"


def test_clear_history_removes_previous_signals(monkeypatch):
    install_mock_analyzers(monkeypatch)

    engine = TradingSignalEngine()

    engine.generate(
        make_market_data(),
        analysis_time=CHECK_TIME,
    )

    assert len(engine.history) == 1

    engine.clear_history()

    assert engine.history == []


def test_changed_market_data_creates_new_signal(monkeypatch):
    install_mock_analyzers(monkeypatch)

    engine = TradingSignalEngine()
    data = make_market_data()

    first = engine.generate(
        data,
        analysis_time=CHECK_TIME,
    )

    changed = make_market_data()
    changed["5m"].loc[
        changed["5m"].index[-1],
        "close",
    ] += 0.2

    changed["5m"].loc[
        changed["5m"].index[-1],
        "high",
    ] += 0.2

    second = engine.generate(
        changed,
        analysis_time=CHECK_TIME,
    )

    assert first["signal_id"] != second["signal_id"]
    assert len(engine.history) == 2


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------


def test_generate_trading_signal_convenience_function(monkeypatch):
    install_mock_analyzers(monkeypatch)

    result = generate_trading_signal(
        make_market_data(),
        analysis_time=CHECK_TIME,
        minimum_candles=30,
    )

    assert result["signal"] in ("LONG", "SHORT", "NO_TRADE")
    assert result["signal_id"]
    assert result["data_quality"]["status"] == "VALIDATED"