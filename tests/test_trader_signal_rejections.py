"""Tests for trading signal rejection storage."""

import json
from copy import deepcopy

import pytest

from trader.signal_engine.rejections import (
    SignalRejectionError,
    SignalRejectionStore,
    record_rejection,
    load_rejections,
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


def make_store(tmp_path):
    return SignalRejectionStore(
        tmp_path / "signal_rejections.jsonl"
    )


def test_store_initializes_with_empty_history(tmp_path):
    store = make_store(tmp_path)

    assert store.read_all() == []
    assert store.count() == 0


def test_rejected_signal_is_recorded(tmp_path):
    store = make_store(tmp_path)

    result = store.append(make_signal())

    assert result["signal_id"] == SIGNAL_ID
    assert result["status"] == "REJECTED"
    assert store.count() == 1


def test_rejection_is_persisted_as_jsonl(tmp_path):
    store = make_store(tmp_path)

    store.append(make_signal())

    assert store.rejection_path.exists()

    lines = store.rejection_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["signal_id"] == SIGNAL_ID


def test_rejection_reasons_are_preserved(tmp_path):
    store = make_store(tmp_path)
    signal = make_signal()

    result = store.append(signal)

    assert result["rejection_reasons"] == signal["rejection_reasons"]


def test_duplicate_rejection_is_not_recorded_twice(tmp_path):
    store = make_store(tmp_path)

    first = store.append(make_signal())
    second = store.append(make_signal())

    assert store.count() == 1
    assert first["signal_id"] == second["signal_id"]
    assert second["duplicate"] is True


def test_append_does_not_modify_original_signal(tmp_path):
    store = make_store(tmp_path)
    signal = make_signal()
    original = deepcopy(signal)

    store.append(signal)

    assert signal == original


def test_read_all_returns_defensive_copy(tmp_path):
    store = make_store(tmp_path)
    store.append(make_signal())

    records = store.read_all()
    records[0]["status"] = "MODIFIED"

    assert store.read_all()[0]["status"] == "REJECTED"


def test_get_returns_existing_rejection(tmp_path):
    store = make_store(tmp_path)
    store.append(make_signal())

    result = store.get(SIGNAL_ID)

    assert result is not None
    assert result["signal_id"] == SIGNAL_ID


def test_get_returns_none_for_unknown_signal(tmp_path):
    store = make_store(tmp_path)

    assert store.get("b" * 64) is None


def test_contains_detects_existing_signal(tmp_path):
    store = make_store(tmp_path)
    store.append(make_signal())

    assert store.contains(SIGNAL_ID) is True
    assert store.contains("b" * 64) is False


def test_count_returns_number_of_rejections(tmp_path):
    store = make_store(tmp_path)

    store.append(make_signal())

    second = make_signal(signal_id="b" * 64)
    store.append(second)

    assert store.count() == 2


def test_count_by_reason_groups_rejections(tmp_path):
    store = make_store(tmp_path)

    first = make_signal()
    second = make_signal(
        signal_id="b" * 64,
        rejection_reasons=[
            "Multi-timeframe direction is neutral.",
            "Risk reward is insufficient.",
        ],
    )

    store.append(first)
    store.append(second)

    counts = store.count_by_reason()

    assert counts["Multi-timeframe direction is neutral."] == 2
    assert counts["Risk reward is insufficient."] == 1


def test_verify_integrity_returns_valid_result(tmp_path):
    store = make_store(tmp_path)
    store.append(make_signal())

    result = store.verify_integrity()

    assert result["valid"] is True
    assert result["total_records"] == 1
    assert result["unique_records"] == 1
    assert result["duplicate_signal_ids"] == []
    assert result["storage_format"] == "JSONL"
    assert result["append_only"] is True
    assert result["rejected_signals_only"] is True


def test_non_rejected_signal_is_rejected(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal(
        status="REQUIRES_MANUAL_REVIEW",
    )

    with pytest.raises(SignalRejectionError):
        store.append(signal)


def test_non_no_trade_signal_is_rejected(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal(signal="LONG")

    with pytest.raises(SignalRejectionError):
        store.append(signal)


@pytest.mark.parametrize(
    "signal_id",
    [
        None,
        "",
        "invalid",
        "g" * 64,
        "a" * 63,
    ],
)
def test_invalid_signal_id_is_rejected(tmp_path, signal_id):
    store = make_store(tmp_path)

    with pytest.raises(SignalRejectionError):
        store.append(make_signal(signal_id=signal_id))


def test_signal_id_is_normalized_to_lowercase(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal(signal_id="A" * 64)
    result = store.append(signal)

    assert result["signal_id"] == "a" * 64


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        "invalid",
        "2026-11-15T12:00:00",
        "2026-11-15T12:00:00+03:30",
    ],
)
def test_invalid_timestamp_is_rejected(tmp_path, timestamp):
    store = make_store(tmp_path)

    with pytest.raises(SignalRejectionError):
        store.append(
            make_signal(analysis_timestamp=timestamp)
        )


def test_utc_z_timestamp_is_accepted_and_normalized(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal(
        analysis_timestamp="2026-11-15T12:00:00Z"
    )

    result = store.append(signal)

    assert result["analysis_timestamp"] == (
        "2026-11-15T12:00:00+00:00"
    )


@pytest.mark.parametrize(
    "reasons",
    [
        None,
        [],
        [""],
        ["   "],
        [None],
    ],
)
def test_invalid_rejection_reasons_are_rejected(tmp_path, reasons):
    store = make_store(tmp_path)

    with pytest.raises(SignalRejectionError):
        store.append(
            make_signal(rejection_reasons=reasons)
        )


def test_missing_execution_metadata_is_rejected(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal(execution=None)

    with pytest.raises(SignalRejectionError):
        store.append(signal)


@pytest.mark.parametrize(
    "execution",
    [
        {
            "manual_review_only": False,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
        {
            "manual_review_only": True,
            "auto_trading": True,
            "exchange_connection": False,
            "order_execution": False,
        },
        {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": True,
            "order_execution": False,
        },
        {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": True,
        },
    ],
)
def test_unsafe_execution_configuration_is_rejected(
    tmp_path,
    execution,
):
    store = make_store(tmp_path)

    with pytest.raises(SignalRejectionError):
        store.append(
            make_signal(execution=execution)
        )


def test_missing_execution_safety_fields_are_rejected(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal(
        execution={
            "manual_review_only": True,
            "auto_trading": False,
        }
    )

    with pytest.raises(SignalRejectionError):
        store.append(signal)


def test_non_json_compatible_signal_is_rejected(tmp_path):
    store = make_store(tmp_path)

    signal = make_signal()
    signal["extra"] = object()

    with pytest.raises(SignalRejectionError):
        store.append(signal)


def test_load_rejections_convenience_function(tmp_path):
    path = tmp_path / "rejections.jsonl"

    record_rejection(
        make_signal(),
        rejection_path=path,
    )

    records = load_rejections(
        rejection_path=path,
    )

    assert len(records) == 1
    assert records[0]["signal_id"] == SIGNAL_ID


def test_corrupted_jsonl_history_is_rejected(tmp_path):
    store = make_store(tmp_path)

    store.rejection_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    store.rejection_path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(SignalRejectionError):
        store.read_all()


def test_non_object_jsonl_record_is_rejected(tmp_path):
    store = make_store(tmp_path)

    store.rejection_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    store.rejection_path.write_text(
        '"not-an-object"\n',
        encoding="utf-8",
    )

    with pytest.raises(SignalRejectionError):
        store.read_all()


def test_invalid_existing_rejection_record_is_rejected(tmp_path):
    store = make_store(tmp_path)

    store.rejection_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    store.rejection_path.write_text(
        json.dumps({"signal_id": SIGNAL_ID}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SignalRejectionError):
        store.read_all()


def test_empty_lines_in_history_are_ignored(tmp_path):
    store = make_store(tmp_path)

    store.rejection_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = json.dumps(VALID_REJECTED_SIGNAL)

    store.rejection_path.write_text(
        "\n" + record + "\n\n",
        encoding="utf-8",
    )

    assert store.count() == 1