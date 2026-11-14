"""Tests for the integrated market analysis engine."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_analysis import engine


UTC = timezone.utc

ANALYSIS_TIME = datetime(
    2026, 11, 14, 12, 0, 0,
    tzinfo=UTC,
)

COMPONENTS = (
    "trend",
    "structure",
    "zones",
    "liquidity",
    "regime",
)


def build_candles(count=70):
    """Build valid, closed UTC candles."""

    rows = []

    start = datetime(
        2026, 11, 14, 0, 0,
        tzinfo=UTC,
    )

    for index in range(count):
        open_time = start + timedelta(
            minutes=5 * index
        )

        close_time = open_time + timedelta(
            minutes=5
        )

        price = 95000.0 + index * 10

        rows.append({
            "open_time": open_time,
            "close_time": close_time,
            "open": price,
            "high": price + 20,
            "low": price - 20,
            "close": price + 5,
            "volume": 100.0 + index,
            "source": "kraken",
        })

    return pd.DataFrame(rows)


def build_component_result(
    analysis_time,
    candles,
):
    """Build a valid component response."""

    return {
        "status": "SUCCESS",
        "analysis_timestamp": analysis_time.isoformat(),
        "source": ["kraken"],
        "last_candle_close_time": (
            candles["close_time"].iloc[-1].isoformat()
        ),
        "candles_analyzed": len(candles),
    }


def install_component_mocks(
    monkeypatch,
    fail_component=None,
):
    """Replace all analysis components with controlled mocks."""

    calls = {
        "components": [],
        "confluence": [],
    }

    def make_component(name):

        def fake_component(
            data,
            analysis_time=None,
            **kwargs,
        ):
            calls["components"].append({
                "name": name,
                "data": data,
                "analysis_time": analysis_time,
                "parameters": kwargs,
            })

            if name == fail_component:
                raise RuntimeError(
                    f"{name} failure"
                )

            return build_component_result(
                analysis_time,
                data,
            )

        return fake_component

    functions = {
        "trend": "analyze_trend",
        "structure": "analyze_price_structure",
        "zones": "analyze_supply_demand",
        "liquidity": "analyze_liquidity",
        "regime": "analyze_market_regime",
    }

    for component, function_name in functions.items():
        monkeypatch.setattr(
            engine,
            function_name,
            make_component(component),
        )

    def fake_confluence(
        trend_analysis,
        structure_analysis,
        zones_analysis,
        liquidity_analysis,
        regime_analysis,
        weights=None,
        bullish_threshold=20.0,
        bearish_threshold=20.0,
        analysis_time=None,
    ):
        calls["confluence"].append({
            "trend": trend_analysis,
            "structure": structure_analysis,
            "zones": zones_analysis,
            "liquidity": liquidity_analysis,
            "regime": regime_analysis,
            "analysis_time": analysis_time,
            "weights": weights,
            "bullish_threshold": bullish_threshold,
            "bearish_threshold": bearish_threshold,
        })

        if fail_component == "confluence":
            raise RuntimeError(
                "confluence failure"
            )

        return {
            "status": "SUCCESS",
            "analysis_timestamp": analysis_time.isoformat(),
            "confluence_score": 35.0,
            "direction": "BULLISH",
        }

    monkeypatch.setattr(
        engine,
        "analyze_market_confluence",
        fake_confluence,
    )

    return calls


# ---------------------------------------------------------
# Successful execution
# ---------------------------------------------------------

def test_all_components_are_executed(monkeypatch):
    calls = install_component_mocks(monkeypatch)

    result = engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
    )

    executed = [
        call["name"]
        for call in calls["components"]
    ]

    assert executed == list(COMPONENTS)
    assert len(calls["confluence"]) == 1
    assert result["status"] == "SUCCESS"


def test_all_components_receive_same_analysis_time(monkeypatch):
    calls = install_component_mocks(monkeypatch)

    engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
    )

    timestamps = [
        call["analysis_time"]
        for call in calls["components"]
    ]

    assert len(timestamps) == len(COMPONENTS)

    assert all(
        timestamp == ANALYSIS_TIME
        for timestamp in timestamps
    )

    assert (
        calls["confluence"][0]["analysis_time"]
        == ANALYSIS_TIME
    )


def test_all_components_receive_independent_dataframes(
    monkeypatch,
):
    calls = install_component_mocks(monkeypatch)

    engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
    )

    frames = [
        call["data"]
        for call in calls["components"]
    ]

    assert len(frames) == len(COMPONENTS)

    for index, frame in enumerate(frames):
        for other in frames[index + 1:]:
            assert frame is not other


def test_component_parameters_are_forwarded(monkeypatch):
    calls = install_component_mocks(monkeypatch)

    parameters = {
        "trend": {
            "fast_period": 10,
            "slow_period": 30,
        },
        "structure": {
            "swing_window": 4,
        },
    }

    engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
        component_parameters=parameters,
    )

    trend_call = next(
        call
        for call in calls["components"]
        if call["name"] == "trend"
    )

    structure_call = next(
        call
        for call in calls["components"]
        if call["name"] == "structure"
    )

    assert trend_call["parameters"] == {
        "fast_period": 10,
        "slow_period": 30,
    }

    assert structure_call["parameters"] == {
        "swing_window": 4,
    }


def test_output_contains_expected_schema(monkeypatch):
    install_component_mocks(monkeypatch)

    result = engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
    )

    expected_keys = {
        "analysis_timestamp",
        "status",
        "analyses",
        "confluence",
        "market_price",
        "source",
        "candles_analyzed",
        "last_candle_close_time",
        "parameters",
        "limitations",
    }

    assert set(result) == expected_keys
    assert result["status"] == "SUCCESS"

    assert set(result["analyses"]) == set(COMPONENTS)

    assert result["confluence"]["status"] == "SUCCESS"

    assert result["market_price"] == pytest.approx(
        95695.0
    )

    assert result["source"] == ["kraken"]
    assert result["candles_analyzed"] == 70
    assert isinstance(result["limitations"], list)


def test_analysis_timestamp_is_utc_isoformat(monkeypatch):
    install_component_mocks(monkeypatch)

    result = engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
    )

    assert (
        result["analysis_timestamp"]
        == ANALYSIS_TIME.isoformat()
    )


def test_input_dataframe_is_not_modified(monkeypatch):
    install_component_mocks(monkeypatch)

    data = build_candles()
    original = data.copy(deep=True)

    engine.analyze_market(
        data,
        analysis_time=ANALYSIS_TIME,
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def test_non_dataframe_input_is_rejected():
    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="DataFrame",
    ):
        engine.analyze_market(
            [],
            analysis_time=ANALYSIS_TIME,
        )


def test_empty_dataframe_is_rejected():
    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="empty",
    ):
        engine.analyze_market(
            pd.DataFrame(),
            analysis_time=ANALYSIS_TIME,
        )


def test_missing_required_column_is_rejected():
    data = build_candles().drop(
        columns=["volume"]
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="volume",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_naive_candle_timestamps_are_rejected():
    data = build_candles()

    data["open_time"] = (
        data["open_time"]
        .dt.tz_localize(None)
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="timezone-aware",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_non_utc_candle_timestamps_are_rejected():
    data = build_candles()

    data["open_time"] = (
        data["open_time"]
        .dt.tz_convert("Europe/Berlin")
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="UTC",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_future_candle_is_rejected():
    data = build_candles()

    data.loc[
        data.index[-1],
        "close_time",
    ] = ANALYSIS_TIME + timedelta(minutes=5)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Future or unclosed",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_unsorted_candles_are_rejected():
    data = build_candles()

    data = pd.concat(
        [
            data.iloc[:10],
            data.iloc[11:],
            data.iloc[10:11],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="sorted",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_duplicate_open_time_is_rejected():
    data = build_candles()

    data.loc[
        data.index[-1],
        "open_time",
    ] = data.loc[
        data.index[-2],
        "open_time",
    ]

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Duplicate open_time",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_invalid_ohlc_relationship_is_rejected():
    data = build_candles()

    data.loc[
        data.index[-1],
        "high",
    ] = 1.0

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Invalid OHLC",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_negative_volume_is_rejected():
    data = build_candles()

    data.loc[
        data.index[-1],
        "volume",
    ] = -1.0

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Volume cannot be negative",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_non_finite_price_is_rejected():
    data = build_candles()

    data.loc[
        data.index[-1],
        "close",
    ] = float("inf")

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="non-finite",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


def test_missing_source_is_rejected():
    data = build_candles()

    data.loc[
        data.index[-1],
        "source",
    ] = ""

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="valid source",
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )


# ---------------------------------------------------------
# Analysis time validation
# ---------------------------------------------------------

def test_naive_analysis_time_is_rejected():
    naive_time = datetime(
        2026, 11, 14, 12, 0
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="timezone-aware",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=naive_time,
        )


def test_non_utc_analysis_time_is_rejected():
    non_utc_time = datetime(
        2026, 11, 14, 12, 0,
        tzinfo=timezone(
            timedelta(hours=2)
        ),
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="UTC",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=non_utc_time,
        )


# ---------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------

def test_unknown_component_is_rejected():
    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Unknown component",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters={
                "unknown": {},
            },
        )


def test_unknown_component_parameter_is_rejected():
    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Unknown parameters",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters={
                "trend": {
                    "unknown_parameter": 10,
                },
            },
        )


def test_component_parameters_must_be_dictionary():
    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="must be a dictionary",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters={
                "trend": [],
            },
        )


def test_component_parameters_root_must_be_dictionary():
    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="component_parameters must be a dictionary",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters=[],
        )


# ---------------------------------------------------------
# Component and confluence failures
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "component",
    COMPONENTS,
)
def test_component_failure_is_wrapped(
    monkeypatch,
    component,
):
    install_component_mocks(
        monkeypatch,
        fail_component=component,
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == component
    assert component in str(exc_info.value)


def test_confluence_failure_is_wrapped(monkeypatch):
    install_component_mocks(
        monkeypatch,
        fail_component="confluence",
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Confluence analysis failed",
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == "confluence"


def test_component_timestamp_mismatch_is_rejected(
    monkeypatch,
):
    install_component_mocks(monkeypatch)

    original = engine.analyze_price_structure

    def mismatched_component(
        data,
        analysis_time=None,
        **kwargs,
    ):
        result = original(
            data,
            analysis_time=analysis_time,
            **kwargs,
        )

        result["analysis_timestamp"] = (
            "2026-11-14T11:00:00+00:00"
        )

        return result

    monkeypatch.setattr(
        engine,
        "analyze_price_structure",
        mismatched_component,
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="mismatched analysis timestamp",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_component_candle_count_mismatch_is_rejected(
    monkeypatch,
):
    install_component_mocks(monkeypatch)

    original = engine.analyze_price_structure

    def mismatched_component(
        data,
        analysis_time=None,
        **kwargs,
    ):
        result = original(
            data,
            analysis_time=analysis_time,
            **kwargs,
        )

        result["candles_analyzed"] -= 1

        return result

    monkeypatch.setattr(
        engine,
        "analyze_price_structure",
        mismatched_component,
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="same number of candles",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_invalid_confluence_result_is_rejected(
    monkeypatch,
):
    install_component_mocks(monkeypatch)

    monkeypatch.setattr(
        engine,
        "analyze_market_confluence",
        lambda **kwargs: [],
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Confluence must return a dictionary",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_unsuccessful_confluence_result_is_rejected(
    monkeypatch,
):
    install_component_mocks(monkeypatch)

    monkeypatch.setattr(
        engine,
        "analyze_market_confluence",
        lambda **kwargs: {
            "status": "FAILED",
            "analysis_timestamp": ANALYSIS_TIME.isoformat(),
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="did not return SUCCESS",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_confluence_timestamp_mismatch_is_rejected(
    monkeypatch,
):
    install_component_mocks(monkeypatch)

    monkeypatch.setattr(
        engine,
        "analyze_market_confluence",
        lambda **kwargs: {
            "status": "SUCCESS",
            "analysis_timestamp": (
                "2026-11-14T11:00:00+00:00"
            ),
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="mismatched analysis timestamp",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )