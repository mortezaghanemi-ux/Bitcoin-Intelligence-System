"""Tests for trading signal rejection records."""

from copy import deepcopy

import pytest

from trader.signal_engine.rejections import (
    SignalRejectionError,
    SignalRejectionManager,
)


SIGNAL_ID = "a" * 64

VALID_REJECTED_SIGNAL = {
    "signal_id": SIGNAL_ID,
    "analysis_timestamp": "2026-11-15T12:00:00+00:00",
    "status": "REJECTED",
    "signal": "NO_TRADE",
    "direction": "NEUTRAL",
    "rejection_reasons": [
        "Multi-timeframe direction is neutral."
    ],
    "execution": {
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
        "order_execution": False,
    },
}


def make_signal(**updates):
    signal = deepcopy(VALID_REJECTED_SIGNAL)
    signal.update(updates)
    return signal


def test_manager_initializes_with_empty_history():
    manager = SignalRejectionManager()

    assert manager.read_all() == []
    assert manager.count() == 0


def test_rejected_signal_is_recorded():
    manager = SignalRejectionManager()

    result = manager.record(make_signal())

    assert result["signal_id"] == SIGNAL_ID
    assert result["status"] == "REJECTED"
    assert manager.count() == 1


def test_rejection_reasons_are_preserved():
    manager = SignalRejectionManager()

    signal = make_signal()
    result = manager.record(signal)

    assert result["rejection_reasons"] == signal["rejection_reasons"]


def test_duplicate_rejection_is_not_recorded_twice():
    manager = SignalRejectionManager()

    first = manager.record(make_signal())
    second = manager.record(make_signal())

    assert manager.count() == 1
    assert first["signal_id"] == second["signal_id"]
    assert second["duplicate"] is True


def test_record_does_not_modify_original_signal():
    manager = SignalRejectionManager()
    signal = make_signal()
    original = deepcopy(signal)

    manager.record(signal)

    assert signal == original


def test_read_all_returns_defensive_copy():
    manager = SignalRejectionManager()
    manager.record(make_signal())

    records = manager.read_all()
    records[0]["status"] = "MODIFIED"

    assert manager.read_all()[0]["status"] == "REJECTED"


def test_non_rejected_signal_is_rejected():
    manager = SignalRejectionManager()

    signal = make_signal(
        status="REQUIRES_MANUAL_REVIEW",
        signal="LONG",
    )

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_missing_signal_id_is_rejected():
    manager = SignalRejectionManager()
    signal = make_signal(signal_id=None)

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_invalid_signal_id_is_rejected():
    manager = SignalRejectionManager()
    signal = make_signal(signal_id="invalid")

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_missing_analysis_timestamp_is_rejected():
    manager = SignalRejectionManager()
    signal = make_signal(analysis_timestamp=None)

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_non_utc_timestamp_is_rejected():
    manager = SignalRejectionManager()

    signal = make_signal(
        analysis_timestamp="2026-11-15T12:00:00+03:30"
    )

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_empty_rejection_reasons_are_rejected():
    manager = SignalRejectionManager()
    signal = make_signal(rejection_reasons=[])

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_missing_execution_safety_fields_are_rejected():
    manager = SignalRejectionManager()
    signal = make_signal(
        execution={
            "manual_review_only": True,
            "auto_trading": False,
        }
    )

    with pytest.raises(SignalRejectionError):
        manager.record(signal)


def test_unsafe_execution_configuration_is_rejected():
    manager = SignalRejectionManager()

    signal = make_signal(
        execution={
            "manual_review_only": False,
            "auto_trading": True,
            "exchange_connection": True,
            "order_execution": True,
        }
    )

    with pytest.raises(SignalRejectionError):
        manager.record(signal)