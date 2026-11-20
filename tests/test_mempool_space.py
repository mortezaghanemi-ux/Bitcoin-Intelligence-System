
"""Tests for the Mempool.space on-chain data collector."""

from __future__ import annotations

import json

from datetime import datetime, timezone

import pytest
import requests

from collectors import mempool_space


class FakeResponse:
    """Mock HTTP response."""

    def __init__(
        self,
        payload,
        *,
        status_error=None,
        json_error=None,
    ):
        self.payload = payload
        self.status_error = status_error
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        if self.json_error:
            raise self.json_error

        return self.payload


class FakeSession:
    """Mock requests session."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(
        self,
        url,
        *,
        headers,
        timeout,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )

        endpoint = url.replace(
            mempool_space.BASE_URL,
            "",
        )

        response = self.responses[endpoint]

        if isinstance(response, Exception):
            raise response

        return response


@pytest.fixture
def valid_responses():
    """Return valid responses for all public endpoints."""

    return {
        "/mempool": FakeResponse(
            {
                "count": 1200,
                "vsize": 2500000,
                "total_fee": 150000,
            }
        ),
        "/v1/fees/recommended": FakeResponse(
            {
                "fastestFee": 25,
                "halfHourFee": 20,
                "hourFee": 15,
                "economyFee": 10,
                "minimumFee": 5,
            }
        ),
        "/blocks/tip/height": FakeResponse(
            "870000"
        ),
        "/v1/difficulty-adjustment": FakeResponse(
            {
                "progressPercent": 50.0,
                "difficultyChange": 2.5,
                "estimatedRetargetDate": 1730000000000,
                "remainingBlocks": 2016,
                "remainingTime": 604800000,
                "previousRetarget": 1.2,
                "nextRetargetHeight": 872000,
                "timeAvg": 600000,
                "timeOffset": 0,
                "expectedBlocks": 2016,
            }
        ),
    }


@pytest.fixture
def fixed_collection_time():
    """Return a deterministic UTC collection timestamp."""

    return datetime(
        2026,
        1,
        15,
        12,
        30,
        tzinfo=timezone.utc,
    )


def install_fake_session(
    monkeypatch,
    responses,
):
    """Install a fake requests session."""

    session = FakeSession(responses)

    monkeypatch.setattr(
        mempool_space.requests,
        "Session",
        lambda: session,
    )

    return session


def read_jsonl(path):
    """Read JSONL records from disk."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def test_collects_all_mempool_space_datasets(
    tmp_path,
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    session = install_fake_session(
        monkeypatch,
        valid_responses,
    )

    result = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=fixed_collection_time,
    )

    assert result is not None

    assert result["schema_version"] == 1
    assert result["source"] == "mempool_space"
    assert result["asset"] == "BTC"

    assert (
        result["timestamp_basis"]
        == "collection_observation"
    )

    assert (
        result["collected_at"]
        == "2026-01-15T12:30:00Z"
    )

    assert result["provider_timestamp"] is None

    assert result["mempool"]["count"] == 1200
    assert result["mempool"]["vsize"] == 2500000.0
    assert result["mempool"]["total_fee"] == 150000.0

    assert result["fees"]["fastestFee"] == 25.0
    assert result["fees"]["halfHourFee"] == 20.0
    assert result["fees"]["hourFee"] == 15.0
    assert result["fees"]["economyFee"] == 10.0
    assert result["fees"]["minimumFee"] == 5.0
    assert result["fees"]["unit"] == "sat/vB"

    assert (
        result["blockchain"]["tip_height"]
        == 870000
    )

    assert (
        result["difficulty_adjustment"]["progressPercent"]
        == 50.0
    )

    assert len(session.calls) == 4

    expected_endpoints = {
        "/mempool",
        "/v1/fees/recommended",
        "/blocks/tip/height",
        "/v1/difficulty-adjustment",
    }

    actual_endpoints = {
        call["url"].replace(
            mempool_space.BASE_URL,
            "",
        )
        for call in session.calls
    }

    assert actual_endpoints == expected_endpoints

    for call in session.calls:
        assert call["headers"]["accept"] == "application/json"
        assert call["timeout"] == mempool_space.DEFAULT_TIMEOUT

    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    assert output_file.exists()

    records = read_jsonl(output_file)

    assert len(records) == 1
    assert records[0] == result


def test_uses_custom_timeout(
    tmp_path,
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    session = install_fake_session(
        monkeypatch,
        valid_responses,
    )

    mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=fixed_collection_time,
        timeout=30,
    )

    assert len(session.calls) == 4

    assert all(
        call["timeout"] == 30.0
        for call in session.calls
    )


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        False,
        "15",
        None,
    ],
)
def test_rejects_invalid_timeout(
    timeout,
):
    with pytest.raises(ValueError):
        mempool_space.collect_mempool_space(
            timeout=timeout,
        )


def test_rejects_naive_collection_time(
    tmp_path,
):
    collection_time = datetime(
        2026,
        1,
        15,
        12,
        30,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware UTC",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_rejects_non_utc_collection_time(
    tmp_path,
):
    from datetime import timedelta

    collection_time = datetime(
        2026,
        1,
        15,
        15,
        30,
        tzinfo=timezone(
            timedelta(hours=3)
        ),
    )

    with pytest.raises(
        ValueError,
        match="must use UTC",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
            collection_time=collection_time,
        )


def test_accepts_utc_collection_time(
    tmp_path,
    monkeypatch,
    valid_responses,
):
    install_fake_session(
        monkeypatch,
        valid_responses,
    )

    collection_time = datetime(
        2026,
        1,
        15,
        12,
        30,
        tzinfo=timezone.utc,
    )

    result = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=collection_time,
    )

    assert (
        result["collected_at"]
        == "2026-01-15T12:30:00Z"
    )


def test_skips_duplicate_collection_timestamp(
    tmp_path,
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    session = install_fake_session(
        monkeypatch,
        valid_responses,
    )

    first_result = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=fixed_collection_time,
    )

    second_result = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=fixed_collection_time,
    )

    assert first_result is not None
    assert second_result is None

    assert len(session.calls) == 4

    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    assert len(read_jsonl(output_file)) == 1


def test_preserves_previous_records(
    tmp_path,
    monkeypatch,
    valid_responses,
):
    install_fake_session(
        monkeypatch,
        valid_responses,
    )

    first_time = datetime(
        2026,
        1,
        15,
        12,
        30,
        tzinfo=timezone.utc,
    )

    second_time = datetime(
        2026,
        1,
        15,
        12,
        35,
        tzinfo=timezone.utc,
    )

    first = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=first_time,
    )

    second = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=second_time,
    )

    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    records = read_jsonl(output_file)

    assert len(records) == 2
    assert records[0] == first
    assert records[1] == second

    assert (
        records[0]["collected_at"]
        != records[1]["collected_at"]
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "/mempool",
        "/v1/fees/recommended",
        "/blocks/tip/height",
        "/v1/difficulty-adjustment",
    ],
)
def test_handles_http_request_failure(
    tmp_path,
    monkeypatch,
    valid_responses,
    endpoint,
):
    responses = dict(valid_responses)

    responses[endpoint] = requests.ConnectionError(
        "connection failed"
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="request failed",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )

    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    assert not output_file.exists()


def test_handles_invalid_json_response(
    tmp_path,
    monkeypatch,
    valid_responses,
):
    responses = dict(valid_responses)

    responses["/mempool"] = FakeResponse(
        None,
        json_error=ValueError(
            "invalid JSON"
        ),
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="invalid JSON",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_rejects_invalid_mempool_response_type(
    tmp_path,
    monkeypatch,
    valid_responses,
    payload,
):
    responses = dict(valid_responses)

    responses["/mempool"] = FakeResponse(
        payload
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Mempool response",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    [
        "count",
        "vsize",
        "total_fee",
    ],
)
def test_rejects_missing_mempool_fields(
    tmp_path,
    monkeypatch,
    valid_responses,
    field,
):
    responses = dict(valid_responses)

    payload = {
        "count": 100,
        "vsize": 2000,
        "total_fee": 300,
    }

    del payload[field]

    responses["/mempool"] = FakeResponse(
        payload
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="missing",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        1.5,
        True,
        "100",
        None,
        float("inf"),
        float("nan"),
    ],
)
def test_rejects_invalid_mempool_count(
    tmp_path,
    monkeypatch,
    valid_responses,
    value,
):
    responses = dict(valid_responses)

    responses["/mempool"] = FakeResponse(
        {
            "count": value,
            "vsize": 2000,
            "total_fee": 300,
        }
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    [
        "vsize",
        "total_fee",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        "100",
        None,
        float("inf"),
        float("nan"),
    ],
)
def test_rejects_invalid_mempool_numeric_fields(
    tmp_path,
    monkeypatch,
    valid_responses,
    field,
    value,
):
    responses = dict(valid_responses)

    payload = {
        "count": 100,
        "vsize": 2000,
        "total_fee": 300,
    }

    payload[field] = value

    responses["/mempool"] = FakeResponse(
        payload
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    [
        "fastestFee",
        "halfHourFee",
        "hourFee",
        "economyFee",
        "minimumFee",
    ],
)
def test_rejects_missing_fee_fields(
    tmp_path,
    monkeypatch,
    valid_responses,
    field,
):
    responses = dict(valid_responses)

    payload = {
        "fastestFee": 25,
        "halfHourFee": 20,
        "hourFee": 15,
        "economyFee": 10,
        "minimumFee": 5,
    }

    del payload[field]

    responses["/v1/fees/recommended"] = FakeResponse(
        payload
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="missing",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        "25",
        None,
        float("inf"),
        float("nan"),
    ],
)
def test_rejects_invalid_fee_values(
    tmp_path,
    monkeypatch,
    valid_responses,
    value,
):
    responses = dict(valid_responses)

    responses["/v1/fees/recommended"] = FakeResponse(
        {
            "fastestFee": value,
            "halfHourFee": 20,
            "hourFee": 15,
            "economyFee": 10,
            "minimumFee": 5,
        }
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "payload",
    [
        "0",
        "-1",
        "abc",
        "",
        None,
        True,
        0,
        -1,
        1.5,
        {},
    ],
)
def test_rejects_invalid_block_tip_height(
    tmp_path,
    monkeypatch,
    valid_responses,
    payload,
):
    responses = dict(valid_responses)

    responses["/blocks/tip/height"] = FakeResponse(
        payload
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="height",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "invalid",
        123,
        {},
    ],
)
def test_rejects_invalid_difficulty_adjustment(
    tmp_path,
    monkeypatch,
    valid_responses,
    payload,
):
    responses = dict(valid_responses)

    responses[
        "/v1/difficulty-adjustment"
    ] = FakeResponse(payload)

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Difficulty adjustment",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


def test_rejects_non_finite_nested_json_values(
    tmp_path,
    monkeypatch,
    valid_responses,
):
    responses = dict(valid_responses)

    responses["/v1/difficulty-adjustment"] = FakeResponse(
        {
            "progressPercent": float("nan"),
        }
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Non-finite",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


def test_rejects_unsupported_nested_json_values(
    tmp_path,
    monkeypatch,
    valid_responses,
):
    responses = dict(valid_responses)

    responses["/v1/difficulty-adjustment"] = FakeResponse(
        {
            "unexpected": object(),
        }
    )

    install_fake_session(
        monkeypatch,
        responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Unsupported response value",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
        )


def test_rejects_invalid_existing_jsonl(
    tmp_path,
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        '{"collected_at": "broken"}\n'
        "{invalid json}\n",
        encoding="utf-8",
    )

    install_fake_session(
        monkeypatch,
        valid_responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Invalid JSONL",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
            collection_time=fixed_collection_time,
        )


def test_rejects_existing_record_without_timestamp(
    tmp_path,
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        '{"source": "mempool_space"}\n',
        encoding="utf-8",
    )

    install_fake_session(
        monkeypatch,
        valid_responses,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Missing collection timestamp",
    ):
        mempool_space.collect_mempool_space(
            output_dir=tmp_path,
            collection_time=fixed_collection_time,
        )


def test_ignores_blank_lines_in_existing_jsonl(
    tmp_path,
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    output_file = (
        tmp_path
        / "mempool_space_BTC.jsonl"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        '\n{"collected_at": "2026-01-14T12:30:00Z"}\n\n',
        encoding="utf-8",
    )

    install_fake_session(
        monkeypatch,
        valid_responses,
    )

    result = mempool_space.collect_mempool_space(
        output_dir=tmp_path,
        collection_time=fixed_collection_time,
    )

    assert result is not None

    records = read_jsonl(output_file)

    assert len(records) == 2


def test_default_output_directory_is_used(
    monkeypatch,
    valid_responses,
    fixed_collection_time,
):
    session = install_fake_session(
        monkeypatch,
        valid_responses,
    )

    captured = {}

    original_open = mempool_space.Path.open

    def tracking_open(path, *args, **kwargs):
        captured["path"] = path
        return original_open(
            path,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        mempool_space.Path,
        "open",
        tracking_open,
    )

    monkeypatch.setattr(
        mempool_space,
        "DEFAULT_OUTPUT_DIR",
        mempool_space.Path(
            "/tmp/mempool_space_test_output"
        ),
    )

    monkeypatch.setattr(
        mempool_space.Path,
        "mkdir",
        lambda *args, **kwargs: None,
    )

    def fake_file_open(path, *args, **kwargs):
        captured["output_path"] = path
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        mempool_space.Path,
        "open",
        fake_file_open,
    )

    with pytest.raises(
        mempool_space.MempoolSpaceCollectorError,
        match="Cannot save on-chain raw data",
    ):
        mempool_space.collect_mempool_space(
            collection_time=fixed_collection_time,
        )

    assert len(session.calls) == 4

    assert (
        captured["output_path"].name
        == "mempool_space_BTC.jsonl"
    )
