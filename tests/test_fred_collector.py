
"""Tests for the FRED economic data collector."""

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

import collectors.fred as fred

from collectors.fred import (
    FREDCollectorError,
    collect_fred_observations,
)


API_KEY = "A" * 32

COLLECTION_TIME = datetime(
    2026,
    11,
    19,
    10,
    30,
    tzinfo=timezone.utc,
)


def observation(
    date="2026-11-18",
    value="4.25",
    realtime_start="2026-11-18",
    realtime_end="2026-11-19",
):
    return {
        "date": date,
        "value": value,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
    }


class FakeResponse:
    """Minimal HTTP response test double."""

    def __init__(
        self,
        payload=None,
        status_error=None,
        json_error=None,
    ):
        self.payload = payload
        self.status_error = status_error
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error

        return self.payload


class FakeSession:
    """Mock requests.Session with queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        return False

    def get(self, url, **kwargs):
        self.calls.append(
            {
                "url": url,
                "kwargs": kwargs,
            }
        )

        if not self.responses:
            raise AssertionError(
                "Unexpected additional HTTP request."
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def api_payload(records, count=None):
    if count is None:
        count = len(records)

    return {
        "count": count,
        "observations": records,
    }


def install_session(monkeypatch, responses):
    session = FakeSession(responses)

    monkeypatch.setattr(
        fred.requests,
        "Session",
        lambda: session,
    )

    return session


def collect(
    output_dir,
    *,
    series_ids=None,
    **kwargs,
):
    """Call the collector with safe test defaults."""

    collection_time = kwargs.pop(
        "collection_time",
        COLLECTION_TIME,
    )

    return collect_fred_observations(
        series_ids=(
            ["DGS10"]
            if series_ids is None
            else series_ids
        ),
        output_dir=output_dir,
        api_key=API_KEY,
        collection_time=collection_time,
        **kwargs,
    )


# ==================================================
# Successful collection
# ==================================================

def test_collects_valid_observation(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    records = collect(tmp_path)

    assert len(records) == 1

    record = records[0]

    assert record["source"] == "fred"
    assert record["series_id"] == "DGS10"
    assert record["observation_date"] == "2026-11-18"
    assert record["value"] == 4.25
    assert record["raw_value"] == "4.25"
    assert record["value_status"] == "VALID"
    assert record["collected_at"] == "2026-11-19T10:30:00Z"


def test_collects_multiple_series(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation(value="4.25")]
                )
            ),
            FakeResponse(
                api_payload(
                    [observation(value="3.75")]
                )
            ),
        ],
    )

    records = collect(
        tmp_path,
        series_ids=["DGS10", "FEDFUNDS"],
    )

    assert len(records) == 2

    assert {
        record["series_id"]
        for record in records
    } == {
        "DGS10",
        "FEDFUNDS",
    }

    assert len(session.calls) == 2


def test_request_contains_required_parameters(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    collect(
        tmp_path,
        observation_start="2020-01-01",
        observation_end="2026-11-19",
    )

    call = session.calls[0]
    params = call["kwargs"]["params"]

    assert call["url"] == fred.API_URL
    assert params["api_key"] == API_KEY
    assert params["series_id"] == "DGS10"
    assert params["file_type"] == "json"
    assert params["sort_order"] == "asc"
    assert params["observation_start"] == "2020-01-01"
    assert params["observation_end"] == "2026-11-19"


def test_request_uses_configured_timeout(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    collect(
        tmp_path,
        timeout=12.5,
    )

    assert session.calls[0]["kwargs"]["timeout"] == 12.5


def test_request_uses_json_accept_header(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    collect(tmp_path)

    assert session.calls[0]["kwargs"]["headers"] == {
        "accept": "application/json",
    }


# ==================================================
# Pagination
# ==================================================

def test_fetches_multiple_pages(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation(date="2026-11-17")],
                    count=2,
                )
            ),
            FakeResponse(
                api_payload(
                    [observation(date="2026-11-18")],
                    count=2,
                )
            ),
        ],
    )

    records = collect(
        tmp_path,
        page_size=1,
    )

    assert len(records) == 2
    assert len(session.calls) == 2

    first_params = session.calls[0]["kwargs"]["params"]
    second_params = session.calls[1]["kwargs"]["params"]

    assert first_params["offset"] == 0
    assert second_params["offset"] == 1

    assert first_params["limit"] == 1
    assert second_params["limit"] == 1


def test_empty_fred_response_returns_empty_list(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload([], count=0)
            )
        ],
    )

    records = collect(tmp_path)

    assert records == []

    assert not (
        tmp_path / "fred_observations.jsonl"
    ).exists()


def test_pagination_stops_when_count_is_reached(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()],
                    count=1,
                )
            )
        ],
    )

    records = collect(
        tmp_path,
        page_size=1,
    )

    assert len(records) == 1
    assert len(session.calls) == 1


def test_changed_count_during_pagination_is_rejected(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation(date="2026-11-17")],
                    count=2,
                )
            ),
            FakeResponse(
                api_payload(
                    [observation(date="2026-11-18")],
                    count=3,
                )
            ),
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="count changed",
    ):
        collect(
            tmp_path,
            page_size=1,
        )


def test_empty_page_before_expected_count_is_rejected(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()],
                    count=2,
                )
            ),
            FakeResponse(
                api_payload(
                    [],
                    count=2,
                )
            ),
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="pagination stopped unexpectedly",
    ):
        collect(
            tmp_path,
            page_size=1,
        )


# ==================================================
# Missing and numeric values
# ==================================================

def test_missing_value_is_preserved_as_null(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [
                        observation(value=".")
                    ]
                )
            )
        ],
    )

    records = collect(tmp_path)

    assert records[0]["value"] is None
    assert records[0]["raw_value"] == "."
    assert records[0]["value_status"] == "MISSING"


@pytest.mark.parametrize(
    "invalid_value",
    [
        "invalid",
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_invalid_numeric_values_are_rejected(
    monkeypatch,
    tmp_path,
    invalid_value,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [
                        observation(value=invalid_value)
                    ]
                )
            )
        ],
    )

    with pytest.raises(FREDCollectorError):
        collect(tmp_path)


# ==================================================
# Observation validation
# ==================================================

@pytest.mark.parametrize(
    "invalid_date",
    [
        "",
        "2026-1-01",
        "2026/11/18",
        "not-a-date",
    ],
)
def test_invalid_observation_date_is_rejected(
    monkeypatch,
    tmp_path,
    invalid_date,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [
                        observation(date=invalid_date)
                    ]
                )
            )
        ],
    )

    with pytest.raises(FREDCollectorError):
        collect(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "realtime_start",
        "realtime_end",
    ],
)
def test_invalid_realtime_dates_are_rejected(
    monkeypatch,
    tmp_path,
    field,
):
    item = observation()
    item[field] = "invalid-date"

    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload([item])
            )
        ],
    )

    with pytest.raises(FREDCollectorError):
        collect(tmp_path)


def test_non_dictionary_observation_is_rejected(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(["invalid"])
            )
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="JSON object",
    ):
        collect(tmp_path)


def test_missing_observation_value_is_rejected(
    monkeypatch,
    tmp_path,
):
    item = observation()
    del item["value"]

    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload([item])
            )
        ],
    )

    with pytest.raises(FREDCollectorError):
        collect(tmp_path)


# ==================================================
# Duplicate prevention and revisions
# ==================================================

def test_duplicate_observations_are_not_written_twice(
    monkeypatch,
    tmp_path,
):
    response = FakeResponse(
        api_payload(
            [observation()]
        )
    )

    install_session(
        monkeypatch,
        [response, response],
    )

    first = collect(tmp_path)
    second = collect(tmp_path)

    assert len(first) == 1
    assert second == []

    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1


def test_revised_vintage_is_preserved(
    monkeypatch,
    tmp_path,
):
    original = observation(
        value="4.25",
        realtime_start="2026-11-18",
        realtime_end="2026-11-19",
    )

    revised = observation(
        value="4.30",
        realtime_start="2026-11-19",
        realtime_end="2026-11-20",
    )

    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload([original])
            ),
            FakeResponse(
                api_payload([revised])
            ),
        ],
    )

    first = collect(tmp_path)
    second = collect(tmp_path)

    assert len(first) == 1
    assert len(second) == 1

    assert first[0]["value"] == 4.25
    assert second[0]["value"] == 4.30

    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2


def test_existing_history_is_preserved(
    monkeypatch,
    tmp_path,
):
    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    original = {
        "series_id": "DGS10",
        "observation_date": "2026-11-17",
        "realtime_start": "2026-11-17",
        "realtime_end": "2026-11-18",
    }

    output_file.write_text(
        json.dumps(original) + "\n",
        encoding="utf-8",
    )

    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    records = collect(tmp_path)

    assert len(records) == 1

    lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2


def test_invalid_existing_jsonl_is_rejected(
    monkeypatch,
    tmp_path,
):
    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    output_file.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FREDCollectorError,
        match="Invalid JSONL",
    ):
        collect(tmp_path)


# ==================================================
# API response errors
# ==================================================

def test_http_error_is_translated(
    monkeypatch,
    tmp_path,
):
    error = requests.HTTPError(
        "HTTP request failed"
    )

    install_session(
        monkeypatch,
        [
            FakeResponse(
                status_error=error
            )
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="FRED request failed",
    ):
        collect(tmp_path)


def test_connection_error_is_translated(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            requests.ConnectionError(
                "Connection failed"
            )
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="FRED request failed",
    ):
        collect(tmp_path)


def test_invalid_json_response_is_rejected(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                json_error=ValueError(
                    "Invalid JSON"
                )
            )
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="invalid JSON",
    ):
        collect(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "invalid",
        {"observations": []},
        {"count": -1, "observations": []},
        {"count": "1", "observations": []},
        {"count": True, "observations": []},
        {
            "error_code": 400,
            "error_message": "Invalid request",
        },
    ],
)
def test_invalid_api_payload_is_rejected(
    monkeypatch,
    tmp_path,
    payload,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(payload)
        ],
    )

    with pytest.raises(FREDCollectorError):
        collect(tmp_path)


def test_more_records_than_page_size_is_rejected(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [
                        observation(date="2026-11-17"),
                        observation(date="2026-11-18"),
                    ],
                    count=2,
                )
            )
        ],
    )

    with pytest.raises(
        FREDCollectorError,
        match="more observations than requested",
    ):
        collect(
            tmp_path,
            page_size=1,
        )


# ==================================================
# Input validation
# ==================================================

def test_api_key_is_required(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        "FRED_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="FRED API key is required",
    ):
        collect_fred_observations(
            ["DGS10"],
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        "short",
        "A" * 31,
        "A" * 33,
        "!" * 32,
    ],
)
def test_invalid_api_key_is_rejected(
    tmp_path,
    api_key,
):
    with pytest.raises(
        ValueError,
        match="API key",
    ):
        collect_fred_observations(
            ["DGS10"],
            output_dir=tmp_path,
            api_key=api_key,
        )


@pytest.mark.parametrize(
    "series_ids",
    [
        [],
        "DGS10",
        [""],
        [None],
        ["invalid-id"],
        ["DGS 10"],
        [123],
    ],
)
def test_invalid_series_ids_are_rejected(
    tmp_path,
    series_ids,
):
    with pytest.raises(ValueError):
        collect_fred_observations(
            series_ids,
            output_dir=tmp_path,
            api_key=API_KEY,
        )


def test_duplicate_series_ids_are_normalized(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    records = collect(
        tmp_path,
        series_ids=[
            "DGS10",
            "DGS10",
            "dgs10",
        ],
    )

    assert len(records) == 1
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        float("inf"),
        float("nan"),
        True,
        "10",
    ],
)
def test_invalid_timeout_is_rejected(
    tmp_path,
    timeout,
):
    with pytest.raises(ValueError):
        collect(
            tmp_path,
            timeout=timeout,
        )


@pytest.mark.parametrize(
    "page_size",
    [
        0,
        -1,
        100001,
        True,
        1.5,
    ],
)
def test_invalid_page_size_is_rejected(
    tmp_path,
    page_size,
):
    with pytest.raises(ValueError):
        collect(
            tmp_path,
            page_size=page_size,
        )


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-02-01", "2026-01-01"),
        ("2026/01/01", None),
        (None, "2026-1-01"),
    ],
)
def test_invalid_date_range_is_rejected(
    tmp_path,
    start,
    end,
):
    with pytest.raises(ValueError):
        collect(
            tmp_path,
            observation_start=start,
            observation_end=end,
        )


def test_naive_collection_time_is_rejected(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="timezone-aware UTC",
    ):
        collect(
            tmp_path,
            collection_time=datetime(
                2026,
                11,
                19,
            ),
        )


def test_non_utc_collection_time_is_rejected(
    tmp_path,
):
    non_utc = datetime(
        2026,
        11,
        19,
        10,
        30,
        tzinfo=timezone(
            timedelta(hours=3)
        ),
    )

    with pytest.raises(
        ValueError,
        match="must use UTC",
    ):
        collect(
            tmp_path,
            collection_time=non_utc,
        )


# ==================================================
# Storage and security
# ==================================================

def test_output_file_is_append_only(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation(date="2026-11-17")]
                )
            ),
            FakeResponse(
                api_payload(
                    [observation(date="2026-11-18")]
                )
            ),
        ],
    )

    collect(tmp_path)
    collect(tmp_path)

    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    records = [
        json.loads(line)
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 2


def test_api_key_is_not_written_to_output(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    collect(tmp_path)

    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    content = output_file.read_text(
        encoding="utf-8"
    )

    assert API_KEY not in content
    assert "api_key" not in content


def test_api_key_is_not_exposed_in_request_error(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            requests.ConnectionError(
                f"Failed request with key {API_KEY}"
            )
        ],
    )

    with pytest.raises(FREDCollectorError) as error:
        collect(tmp_path)

    assert API_KEY not in str(error.value)


def test_output_is_valid_jsonl(
    monkeypatch,
    tmp_path,
):
    install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    collect(tmp_path)

    output_file = (
        tmp_path / "fred_observations.jsonl"
    )

    lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["source"] == "fred"
    assert record["series_id"] == "DGS10"
    assert record["value_status"] == "VALID"


def test_session_is_closed_after_success(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            FakeResponse(
                api_payload(
                    [observation()]
                )
            )
        ],
    )

    collect(tmp_path)

    assert session.closed is True


def test_session_is_closed_after_failure(
    monkeypatch,
    tmp_path,
):
    session = install_session(
        monkeypatch,
        [
            requests.ConnectionError(
                "Network failure"
            )
        ],
    )

    with pytest.raises(FREDCollectorError):
        collect(tmp_path)

    assert session.closed is True
