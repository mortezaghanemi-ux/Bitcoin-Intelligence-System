"""Tests for the append-only trading signal history store."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from trader.signal_engine.history import (
    SignalHistoryError,
    SignalHistoryStore,
    load_signal_history,
    save_signal,
)


SIGNAL_ID = "a" * 64

CHECK_TIME = datetime(
    2026, 11, 15, 12, 0,
    tzinfo=timezone.utc,
).isoformat()


def make_signal(
    signal_id=SIGNAL_ID,
    status="REQUIRES_MANUAL_REVIEW",
    signal="LONG",
):
    return {
        "signal_id": signal_id,
        "analysis_timestamp": CHECK_TIME,
        "signal": signal,
        "status": status,
        "direction": "BULLISH",
        "confidence": 85.0,
        "execution": {
            "manual_review_only": True,
            "auto_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
    }


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------


def test_store_initializes_with_custom_path(tmp_path):
    path = tmp_path / "history.jsonl"

    store = SignalHistoryStore(path)

    assert store.history_path == path
    assert store.count() == 0


def test_missing_history_file_returns_empty_list(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "missing.jsonl"
    )

    assert store.read_all() == []


# ---------------------------------------------------------
# Signal validation
# ---------------------------------------------------------


def test_valid_signal_is_accepted(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    result = store.append(make_signal())

    assert result["signal_id"] == SIGNAL_ID
    assert result["signal"] == "LONG"
    assert result["status"] == "REQUIRES_MANUAL_REVIEW"


def test_non_dictionary_signal_is_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    with pytest.raises(SignalHistoryError):
        store.append("invalid")


@pytest.mark.parametrize(
    "signal_id",
    [
        None,
        "",
        "123",
        "g" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_invalid_signal_id_is_rejected(tmp_path, signal_id):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    with pytest.raises(SignalHistoryError):
        store.append(make_signal(signal_id=signal_id))


def test_missing_signal_id_is_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    del signal["signal_id"]

    with pytest.raises(SignalHistoryError):
        store.append(signal)


def test_missing_analysis_timestamp_is_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    del signal["analysis_timestamp"]

    with pytest.raises(SignalHistoryError):
        store.append(signal)


@pytest.mark.parametrize(
    "timestamp",
    [
        "invalid",
        "",
        None,
        "2026-11-15T12:00:00",
        "2026-11-15T12:00:00+02:00",
    ],
)
def test_invalid_timestamp_is_rejected(tmp_path, timestamp):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    signal["analysis_timestamp"] = timestamp

    with pytest.raises(SignalHistoryError):
        store.append(signal)


def test_utc_timestamp_with_z_suffix_is_accepted(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    signal["analysis_timestamp"] = "2026-11-15T12:00:00Z"

    result = store.append(signal)

    assert result["analysis_timestamp"].endswith("+00:00")


@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "APPROVED",
        "EXECUTED",
        "PENDING",
    ],
)
def test_invalid_status_is_rejected(tmp_path, status):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    with pytest.raises(SignalHistoryError):
        store.append(make_signal(status=status))


@pytest.mark.parametrize(
    "signal_type",
    [
        None,
        "",
        "BUY",
        "SELL",
        "INVALID",
    ],
)
def test_invalid_signal_type_is_rejected(tmp_path, signal_type):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    with pytest.raises(SignalHistoryError):
        store.append(make_signal(signal=signal_type))


def test_no_trade_must_be_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal(
        signal="NO_TRADE",
        status="REQUIRES_MANUAL_REVIEW",
    )

    with pytest.raises(SignalHistoryError):
        store.append(signal)


@pytest.mark.parametrize(
    "execution",
    [
        None,
        {},
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
def test_unsafe_execution_settings_are_rejected(
    tmp_path,
    execution,
):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    signal["execution"] = execution

    with pytest.raises(SignalHistoryError):
        store.append(signal)


def test_non_json_value_is_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    signal["extra"] = object()

    with pytest.raises(SignalHistoryError):
        store.append(signal)


def test_nan_value_is_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    signal = make_signal()
    signal["confidence"] = float("nan")

    with pytest.raises(SignalHistoryError):
        store.append(signal)


# ---------------------------------------------------------
# Append-only storage
# ---------------------------------------------------------


def test_append_creates_history_file(tmp_path):
    path = tmp_path / "nested" / "history.jsonl"
    store = SignalHistoryStore(path)

    store.append(make_signal())

    assert path.exists()


def test_append_stores_one_json_line(tmp_path):
    path = tmp_path / "history.jsonl"
    store = SignalHistoryStore(path)

    store.append(make_signal())

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["signal_id"] == SIGNAL_ID


def test_multiple_signals_are_appended(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    second = make_signal(
        signal_id="b" * 64,
        signal="SHORT",
    )
    second["direction"] = "BEARISH"

    store.append(second)

    assert store.count() == 2


def test_existing_records_are_not_overwritten(tmp_path):
    path = tmp_path / "history.jsonl"
    store = SignalHistoryStore(path)

    store.append(make_signal())

    original_content = path.read_text(
        encoding="utf-8"
    )

    store.append(
        make_signal(
            signal_id="b" * 64,
            signal="SHORT",
        )
    )

    updated_content = path.read_text(
        encoding="utf-8"
    )

    assert updated_content.startswith(original_content)


def test_duplicate_signal_is_not_appended_twice(tmp_path):
    path = tmp_path / "history.jsonl"
    store = SignalHistoryStore(path)

    store.append(make_signal())

    duplicate = store.append(make_signal())

    assert duplicate["duplicate"] is True
    assert store.count() == 1


def test_duplicate_returns_original_record(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    original = make_signal()
    store.append(original)

    modified = make_signal()
    modified["confidence"] = 10

    result = store.append(modified)

    assert result["confidence"] == 85.0
    assert result["duplicate"] is True


def test_invalid_duplicate_is_still_rejected(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    invalid = make_signal()
    invalid["execution"]["auto_trading"] = True

    with pytest.raises(SignalHistoryError):
        store.append(invalid)


# ---------------------------------------------------------
# Reading and lookup
# ---------------------------------------------------------


def test_read_all_returns_all_records(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    store.append(
        make_signal(
            signal_id="b" * 64,
            signal="SHORT",
        )
    )

    records = store.read_all()

    assert len(records) == 2


def test_get_returns_existing_signal(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    result = store.get(SIGNAL_ID)

    assert result is not None
    assert result["signal_id"] == SIGNAL_ID


def test_get_returns_none_for_missing_signal(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    assert store.get("b" * 64) is None


def test_contains_returns_true_for_existing_signal(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    assert store.contains(SIGNAL_ID) is True


def test_contains_returns_false_for_missing_signal(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    assert store.contains("b" * 64) is False


def test_read_all_returns_defensive_copy(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    records = store.read_all()
    records[0]["signal"] = "MODIFIED"

    assert store.read_all()[0]["signal"] == "LONG"


def test_get_returns_defensive_copy(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    record = store.get(SIGNAL_ID)
    record["signal"] = "MODIFIED"

    assert store.get(SIGNAL_ID)["signal"] == "LONG"


# ---------------------------------------------------------
# Status statistics
# ---------------------------------------------------------


def test_count_returns_total_records(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    store.append(
        make_signal(
            signal_id="b" * 64,
            signal="NO_TRADE",
            status="REJECTED",
        )
    )

    assert store.count() == 2


def test_count_by_status(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    store.append(
        make_signal(
            signal_id="b" * 64,
            signal="NO_TRADE",
            status="REJECTED",
        )
    )

    counts = store.count_by_status()

    assert counts == {
        "REJECTED": 1,
        "REQUIRES_MANUAL_REVIEW": 1,
    }


# ---------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------


def test_empty_history_integrity(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    result = store.verify_integrity()

    assert result["valid"] is True
    assert result["total_records"] == 0
    assert result["unique_records"] == 0
    assert result["duplicate_signal_ids"] == []
    assert result["storage_format"] == "JSONL"
    assert result["append_only"] is True


def test_valid_history_integrity(tmp_path):
    store = SignalHistoryStore(
        tmp_path / "history.jsonl"
    )

    store.append(make_signal())

    result = store.verify_integrity()

    assert result["valid"] is True
    assert result["total_records"] == 1
    assert result["unique_records"] == 1


def test_duplicate_ids_are_detected(tmp_path):
    path = tmp_path / "history.jsonl"

    record = make_signal()

    path.write_text(
        json.dumps(record) + "\n"
        + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    store = SignalHistoryStore(path)

    result = store.verify_integrity()

    assert result["valid"] is False
    assert result["total_records"] == 2
    assert result["unique_records"] == 1
    assert result["duplicate_signal_ids"] == [SIGNAL_ID]


def test_corrupted_json_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"

    path.write_text(
        '{"signal_id": invalid}\n',
        encoding="utf-8",
    )

    store = SignalHistoryStore(path)

    with pytest.raises(SignalHistoryError):
        store.read_all()


def test_non_object_json_record_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"

    path.write_text(
        '["invalid"]\n',
        encoding="utf-8",
    )

    store = SignalHistoryStore(path)

    with pytest.raises(SignalHistoryError):
        store.read_all()


def test_record_without_signal_id_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"

    path.write_text(
        '{"status": "REJECTED"}\n',
        encoding="utf-8",
    )

    store = SignalHistoryStore(path)

    with pytest.raises(SignalHistoryError):
        store.read_all()


# ---------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------


def test_save_signal_convenience_function(tmp_path):
    path = tmp_path / "history.jsonl"

    result = save_signal(
        make_signal(),
        history_path=path,
    )

    assert result["signal_id"] == SIGNAL_ID
    assert path.exists()


def test_load_signal_history_convenience_function(tmp_path):
    path = tmp_path / "history.jsonl"

    save_signal(
        make_signal(),
        history_path=path,
    )

    records = load_signal_history(path)

    assert len(records) == 1
    assert records[0]["signal_id"] == SIGNAL_ID


def test_load_signal_history_returns_empty_list_for_missing_file(
    tmp_path,
):
    records = load_signal_history(
        tmp_path / "missing.jsonl"
    )

    assert records == []