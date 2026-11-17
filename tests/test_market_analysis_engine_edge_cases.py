"""Edge-case tests for the integrated market analysis engine."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_analysis import engine


UTC = timezone.utc

ANALYSIS_TIME = datetime(
    2026, 11, 17, 12, 0,
    tzinfo=UTC,
)

COMPONENTS = (
    "trend",
    "structure",
    "zones",
    "liquidity",
    "regime",
)

COMPONENT_FUNCTIONS = {
    "trend": "analyze_trend",
    "structure": "analyze_price_structure",
    "zones": "analyze_supply_demand",
    "liquidity": "analyze_liquidity",
    "regime": "analyze_market_regime",
}


def build_candles(count=10):
    """Build valid closed UTC candles."""

    rows = []

    start = datetime(
        2026, 11, 17, 10, 0,
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


def valid_component_result(
    analysis_time,
    data,
):
    """Return a valid component result."""

    return {
        "status": "SUCCESS",
        "analysis_timestamp": analysis_time.isoformat(),
        "source": ["kraken"],
        "last_candle_close_time": (
            data["close_time"].iloc[-1].isoformat()
        ),
        "candles_analyzed": len(data),
    }


def install_mocks(
    monkeypatch,
    result_overrides=None,
    confluence_result=None,
):
    """Install deterministic component and confluence mocks."""

    result_overrides = result_overrides or {}

    for component, function_name in COMPONENT_FUNCTIONS.items():

        def make_component(name):

            def fake_component(
                data,
                analysis_time=None,
                **kwargs,
            ):
                result = valid_component_result(
                    analysis_time,
                    data,
                )

                override = result_overrides.get(name)

                if callable(override):
                    return override(
                        result,
                        analysis_time,
                        data,
                    )

                if isinstance(override, dict):
                    result.update(override)

                return result

            return fake_component

        monkeypatch.setattr(
            engine,
            function_name,
            make_component(component),
        )

    if confluence_result is None:

        def fake_confluence(
            analysis_time=None,
            **kwargs,
        ):
            return {
                "status": "SUCCESS",
                "analysis_timestamp": (
                    analysis_time.isoformat()
                ),
                "confluence_score": 25.0,
                "direction": "BULLISH",
            }

    else:
        fake_confluence = confluence_result

    monkeypatch.setattr(
        engine,
        "analyze_market_confluence",
        fake_confluence,
    )


def run_analysis(monkeypatch, data=None, **kwargs):
    """Run analysis with controlled component responses."""

    install_mocks(monkeypatch)

    return engine.analyze_market(
        build_candles() if data is None else data,
        analysis_time=ANALYSIS_TIME,
        **kwargs,
    )


# ---------------------------------------------------------
# Analysis timestamp edge cases
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_time",
    [
        "not-a-date",
        object(),
        float("inf"),
    ],
)
def test_invalid_analysis_time_is_rejected(
    monkeypatch,
    invalid_time,
):
    install_mocks(monkeypatch)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Invalid analysis_time",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=invalid_time,
        )


def test_missing_analysis_time_uses_utc_clock(
    monkeypatch,
):
    install_mocks(monkeypatch)

    result = engine.analyze_market(
        build_candles(),
    )

    timestamp = pd.Timestamp(
        result["analysis_timestamp"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


# ---------------------------------------------------------
# Candle timestamp edge cases
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "column",
    [
        "open_time",
        "close_time",
    ],
)
def test_invalid_candle_timestamp_is_rejected(
    monkeypatch,
    column,
):
    data = build_candles()

    data[column] = data[column].astype(object)
    data.loc[data.index[-1], column] = "invalid-date"

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match=f"Invalid timestamps in {column}",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


@pytest.mark.parametrize(
    "column",
    [
        "open_time",
        "close_time",
    ],
)
def test_missing_candle_timestamp_is_rejected(
    monkeypatch,
    column,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        column,
    ] = pd.NaT

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match=f"{column} contains missing timestamps",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


def test_candle_closing_before_opening_is_rejected(
    monkeypatch,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        "close_time",
    ] = (
        data.loc[
            data.index[-1],
            "open_time",
        ]
        - timedelta(minutes=1)
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="cannot close before it opens",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


def test_unsorted_close_times_are_rejected(
    monkeypatch,
):
    data = build_candles()

    previous_index = data.index[-2]
    last_index = data.index[-1]

    previous_close = data.loc[
        previous_index,
        "close_time",
    ]

    last_close = data.loc[
        last_index,
        "close_time",
    ]

    data.loc[
        previous_index,
        "close_time",
    ] = last_close

    data.loc[
        last_index,
        "close_time",
    ] = previous_close

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="close_time values must be sorted",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


def test_duplicate_close_time_is_rejected(
    monkeypatch,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        "close_time",
    ] = data.loc[
        data.index[-2],
        "close_time",
    ]

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Duplicate close_time",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


# ---------------------------------------------------------
# Numeric candle validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ],
)
def test_non_numeric_candle_value_is_rejected(
    monkeypatch,
    column,
):
    data = build_candles()

    data[column] = data[column].astype(object)
    data.loc[data.index[-1], column] = "invalid"

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match=f"{column} must contain numeric values",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ],
)
def test_missing_numeric_value_is_rejected(
    monkeypatch,
    column,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        column,
    ] = float("nan")

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="non-finite",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
    ],
)
def test_non_positive_ohlc_price_is_rejected(
    monkeypatch,
    column,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        column,
    ] = 0.0

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="greater than zero",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ],
)
def test_infinite_numeric_value_is_rejected(
    monkeypatch,
    column,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        column,
    ] = float("inf")

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="non-finite",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


@pytest.mark.parametrize(
    "source",
    [
        None,
        "",
        "   ",
        123,
        float("nan"),
    ],
)
def test_invalid_source_value_is_rejected(
    monkeypatch,
    source,
):
    data = build_candles()

    data.loc[
        data.index[-1],
        "source",
    ] = source

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="valid source",
    ):
        run_analysis(
            monkeypatch,
            data,
        )


# ---------------------------------------------------------
# Component parameter edge cases
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "parameters",
    [
        [],
        "trend",
        123,
        True,
    ],
)
def test_invalid_component_parameters_root_is_rejected(
    monkeypatch,
    parameters,
):
    install_mocks(monkeypatch)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="component_parameters must be a dictionary",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters=parameters,
        )


@pytest.mark.parametrize(
    "component",
    COMPONENTS,
)
def test_component_parameters_must_be_dictionary_for_each_group(
    monkeypatch,
    component,
):
    install_mocks(monkeypatch)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match=f"Parameters for '{component}' must be a dictionary",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters={
                component: [],
            },
        )


def test_multiple_unknown_component_groups_are_reported(
    monkeypatch,
):
    install_mocks(monkeypatch)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Unknown component parameter groups",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters={
                "unknown_b": {},
                "unknown_a": {},
            },
        )


def test_multiple_unknown_parameters_are_reported(
    monkeypatch,
):
    install_mocks(monkeypatch)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Unknown parameters",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
            component_parameters={
                "trend": {
                    "unknown_b": 1,
                    "unknown_a": 2,
                },
            },
        )


# ---------------------------------------------------------
# Component result validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "component",
    COMPONENTS,
)
def test_component_must_return_dictionary(
    monkeypatch,
    component,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            component: lambda *args: [],
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Component must return a dictionary",
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == component


@pytest.mark.parametrize(
    "component",
    COMPONENTS,
)
def test_component_must_return_success_status(
    monkeypatch,
    component,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            component: {
                "status": "FAILED",
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="did not return SUCCESS",
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == component


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "last_candle_close_time",
        "candles_analyzed",
    ],
)
@pytest.mark.parametrize(
    "component",
    COMPONENTS,
)
def test_component_missing_required_metadata_is_rejected(
    monkeypatch,
    component,
    field,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            component: {
                field: None,
            },
        },
    )

    original = engine.__dict__[
        COMPONENT_FUNCTIONS[component]
    ]

    def missing_metadata(
        data,
        analysis_time=None,
        **kwargs,
    ):
        result = original(
            data,
            analysis_time=analysis_time,
            **kwargs,
        )

        result.pop(field, None)
        return result

    monkeypatch.setattr(
        engine,
        COMPONENT_FUNCTIONS[component],
        missing_metadata,
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match=f"missing '{field}'",
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == component


@pytest.mark.parametrize(
    "invalid_sources",
    [
        [],
        "kraken",
        None,
        [""],
        ["kraken", None],
        [123],
    ],
)
def test_invalid_component_source_metadata_is_rejected(
    monkeypatch,
    invalid_sources,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "source": invalid_sources,
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="invalid source metadata",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
        True,
        1.5,
        "10",
        None,
    ],
)
def test_invalid_component_candle_count_is_rejected(
    monkeypatch,
    invalid_count,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "candles_analyzed": invalid_count,
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="invalid candle count",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


@pytest.mark.parametrize(
    "invalid_close_time",
    [
        "not-a-date",
        float("inf"),
    ],
)
def test_invalid_component_last_close_time_is_rejected(
    monkeypatch,
    invalid_close_time,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "last_candle_close_time": invalid_close_time,
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="invalid last candle close time",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_naive_component_last_close_time_is_rejected(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "last_candle_close_time": (
                    "2026-11-17T10:50:00"
                ),
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="must be timezone-aware",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_non_utc_component_last_close_time_is_rejected(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "last_candle_close_time": (
                    "2026-11-17T12:50:00+02:00"
                ),
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="must use UTC",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


# ---------------------------------------------------------
# Component consistency
# ---------------------------------------------------------

def test_component_last_close_time_mismatch_is_rejected(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "structure": {
                "last_candle_close_time": (
                    "2026-11-17T10:45:00+00:00"
                ),
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="same last candle",
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == "structure"


def test_component_candle_count_mismatch_is_rejected(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "liquidity": {
                "candles_analyzed": 9,
            },
        },
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="same number of candles",
    ) as exc_info:
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )

    assert exc_info.value.component == "liquidity"


# ---------------------------------------------------------
# Confluence result edge cases
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_non_dictionary_confluence_result_is_rejected(
    monkeypatch,
    invalid_result,
):
    install_mocks(
        monkeypatch,
        confluence_result=lambda **kwargs: invalid_result,
    )

    with pytest.raises(
        engine.MarketAnalysisEngineError,
        match="Confluence must return a dictionary",
    ):
        engine.analyze_market(
            build_candles(),
            analysis_time=ANALYSIS_TIME,
        )


def test_unsuccessful_confluence_status_is_rejected(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        confluence_result=lambda analysis_time=None, **kwargs: {
            "status": "FAILED",
            "analysis_timestamp": analysis_time.isoformat(),
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


def test_missing_confluence_status_is_rejected(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        confluence_result=lambda analysis_time=None, **kwargs: {
            "analysis_timestamp": analysis_time.isoformat(),
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
    install_mocks(
        monkeypatch,
        confluence_result=lambda **kwargs: {
            "status": "SUCCESS",
            "analysis_timestamp": (
                "2026-11-17T11:00:00+00:00"
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


def test_confluence_unexpected_exception_is_wrapped(
    monkeypatch,
):
    def failing_confluence(**kwargs):
        raise ValueError(
            "unexpected confluence failure"
        )

    install_mocks(
        monkeypatch,
        confluence_result=failing_confluence,
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


# ---------------------------------------------------------
# Output integrity
# ---------------------------------------------------------

def test_sources_are_combined_and_sorted(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "source": ["kraken", "coingecko"],
            },
            "structure": {
                "source": ["binance", "kraken"],
            },
        },
    )

    result = engine.analyze_market(
        build_candles(),
        analysis_time=ANALYSIS_TIME,
    )

    assert result["source"] == [
        "binance",
        "coingecko",
        "kraken",
    ]


def test_output_market_price_uses_last_close(
    monkeypatch,
):
    result = run_analysis(
        monkeypatch,
    )

    assert result["market_price"] == pytest.approx(
        95095.0
    )


def test_empty_component_parameters_are_normalized(
    monkeypatch,
):
    result = run_analysis(
        monkeypatch,
        component_parameters={},
    )

    assert set(
        result["parameters"]["components"]
    ) == set(COMPONENTS)

    assert all(
        value == {}
        for value in result["parameters"]["components"].values()
    )


def test_none_component_parameters_are_normalized(
    monkeypatch,
):
    result = run_analysis(
        monkeypatch,
        component_parameters=None,
    )

    assert set(
        result["parameters"]["components"]
    ) == set(COMPONENTS)


def test_input_dataframe_remains_unchanged_after_failure(
    monkeypatch,
):
    install_mocks(
        monkeypatch,
        result_overrides={
            "trend": {
                "status": "FAILED",
            },
        },
    )

    data = build_candles()
    original = data.copy(deep=True)

    with pytest.raises(
        engine.MarketAnalysisEngineError,
    ):
        engine.analyze_market(
            data,
            analysis_time=ANALYSIS_TIME,
        )

    pd.testing.assert_frame_equal(
        data,
        original,
    )