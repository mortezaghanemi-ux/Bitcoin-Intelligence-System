"""Tests for Bitcoin supply and demand zone analysis."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_analysis.zones import (
    ACTIVE,
    BROKEN,
    DEMAND,
    MITIGATED,
    SUPPLY,
    ZoneAnalysisError,
    analyze_supply_demand,
)


UTC = timezone.utc
START = datetime(2026, 11, 13, 0, 0, tzinfo=UTC)
ANALYSIS_TIME = datetime(2026, 11, 14, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------


def make_candles(
    base_type=None,
    final_status=None,
    candle_count=40,
):
    """Create deterministic OHLCV candles for zone tests."""

    rows = []

    for index in range(candle_count):
        price = 100.0

        rows.append(
            {
                "open_time": START + timedelta(minutes=5 * index),
                "close_time": START + timedelta(minutes=5 * (index + 1)),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 100.0,
                "source": "kraken",
            }
        )

    if base_type == SUPPLY:
        # Bullish base candle.
        rows[20].update(
            {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
            }
        )

        # Strong bearish displacement.
        rows[21].update(
            {
                "open": 101.0,
                "high": 102.0,
                "low": 93.0,
                "close": 94.0,
            }
        )

        # Subsequent candles remain below the supply zone.
        for index in range(22, candle_count):
            rows[index].update(
                {
                    "open": 90.0,
                    "high": 91.0,
                    "low": 89.0,
                    "close": 90.0,
                }
            )

        if final_status == MITIGATED:
            rows[30].update(
                {
                    "open": 90.0,
                    "high": 101.0,
                    "low": 89.0,
                    "close": 100.5,
                }
            )

        elif final_status == BROKEN:
            rows[30].update(
                {
                    "open": 101.0,
                    "high": 105.0,
                    "low": 100.0,
                    "close": 104.0,
                }
            )

    elif base_type == DEMAND:
        # Bearish base candle.
        rows[20].update(
            {
                "open": 101.0,
                "high": 102.0,
                "low": 99.0,
                "close": 100.0,
            }
        )

        # Strong bullish displacement.
        rows[21].update(
            {
                "open": 100.0,
                "high": 107.0,
                "low": 99.0,
                "close": 106.0,
            }
        )

        # Subsequent candles remain above the demand zone.
        for index in range(22, candle_count):
            rows[index].update(
                {
                    "open": 110.0,
                    "high": 111.0,
                    "low": 109.0,
                    "close": 110.0,
                }
            )

        if final_status == MITIGATED:
            rows[30].update(
                {
                    "open": 110.0,
                    "high": 111.0,
                    "low": 100.0,
                    "close": 100.5,
                }
            )

        elif final_status == BROKEN:
            rows[30].update(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 95.0,
                    "close": 96.0,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------


def analyze(data, **kwargs):
    """Run analysis with common test parameters."""

    parameters = {
        "atr_period": 14,
        "impulse_window": 5,
        "atr_multiplier": 1.5,
        "minimum_candles": 30,
        "analysis_time": ANALYSIS_TIME,
    }

    parameters.update(kwargs)

    return analyze_supply_demand(
        data=data,
        **parameters,
    )


def get_zone(result, zone_type):
    """Return the first zone of the requested type."""

    if zone_type == SUPPLY:
        zones = result["supply_zones"]
    else:
        zones = result["demand_zones"]

    assert zones, f"No {zone_type} zones were detected."

    return zones[0]


# ---------------------------------------------------------------------
# Basic analysis
# ---------------------------------------------------------------------


def test_analysis_returns_expected_structure():
    data = make_candles(base_type=SUPPLY)

    result = analyze(data)

    expected_keys = {
        "analysis_timestamp",
        "status",
        "source",
        "candles_analyzed",
        "last_candle_close_time",
        "supply_zones",
        "demand_zones",
        "zones",
        "summary",
        "parameters",
    }

    assert expected_keys.issubset(result.keys())


def test_analysis_uses_utc_timestamp():
    result = analyze(make_candles(base_type=SUPPLY))

    timestamp = result["analysis_timestamp"]

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_analysis_counts_input_candles():
    data = make_candles(base_type=SUPPLY)

    result = analyze(data)

    assert result["candles_analyzed"] == len(data)


def test_analysis_reports_source():
    result = analyze(make_candles(base_type=SUPPLY))

    assert result["source"] == "kraken"


def test_analysis_with_no_detectable_zones():
    data = make_candles()

    result = analyze(data)

    assert result["supply_zones"] == []
    assert result["demand_zones"] == []
    assert result["zones"] == []


def test_analysis_returns_summary_dictionary():
    result = analyze(make_candles(base_type=SUPPLY))

    assert isinstance(result["summary"], dict)


def test_analysis_returns_parameters_dictionary():
    result = analyze(make_candles(base_type=SUPPLY))

    assert isinstance(result["parameters"], dict)


# ---------------------------------------------------------------------
# Supply zone detection
# ---------------------------------------------------------------------


def test_detects_supply_zone():
    result = analyze(make_candles(base_type=SUPPLY))

    zone = get_zone(result, SUPPLY)

    assert zone["type"] == SUPPLY


def test_supply_zone_has_valid_price_boundaries():
    result = analyze(make_candles(base_type=SUPPLY))

    zone = get_zone(result, SUPPLY)

    assert zone["upper"] >= zone["lower"]
    assert zone["upper"] > 0
    assert zone["lower"] > 0


def test_supply_zone_contains_confirmation_information():
    result = analyze(make_candles(base_type=SUPPLY))

    zone = get_zone(result, SUPPLY)

    assert zone["base_time"] is not None
    assert zone["confirmation_time"] is not None
    assert zone["confirmation_time"] >= zone["base_time"]


def test_supply_zone_is_active_without_retest():
    result = analyze(
        make_candles(
            base_type=SUPPLY,
            final_status=ACTIVE,
        )
    )

    zone = get_zone(result, SUPPLY)

    assert zone["status"] == ACTIVE


def test_supply_zone_becomes_mitigated_after_retest():
    result = analyze(
        make_candles(
            base_type=SUPPLY,
            final_status=MITIGATED,
        )
    )

    zone = get_zone(result, SUPPLY)

    assert zone["status"] == MITIGATED


def test_supply_zone_becomes_broken_after_close_above_zone():
    result = analyze(
        make_candles(
            base_type=SUPPLY,
            final_status=BROKEN,
        )
    )

    zone = get_zone(result, SUPPLY)

    assert zone["status"] == BROKEN


# ---------------------------------------------------------------------
# Demand zone detection
# ---------------------------------------------------------------------


def test_detects_demand_zone():
    result = analyze(make_candles(base_type=DEMAND))

    zone = get_zone(result, DEMAND)

    assert zone["type"] == DEMAND


def test_demand_zone_has_valid_price_boundaries():
    result = analyze(make_candles(base_type=DEMAND))

    zone = get_zone(result, DEMAND)

    assert zone["upper"] >= zone["lower"]
    assert zone["upper"] > 0
    assert zone["lower"] > 0


def test_demand_zone_contains_confirmation_information():
    result = analyze(make_candles(base_type=DEMAND))

    zone = get_zone(result, DEMAND)

    assert zone["base_time"] is not None
    assert zone["confirmation_time"] is not None
    assert zone["confirmation_time"] >= zone["base_time"]


def test_demand_zone_is_active_without_retest():
    result = analyze(
        make_candles(
            base_type=DEMAND,
            final_status=ACTIVE,
        )
    )

    zone = get_zone(result, DEMAND)

    assert zone["status"] == ACTIVE


def test_demand_zone_becomes_mitigated_after_retest():
    result = analyze(
        make_candles(
            base_type=DEMAND,
            final_status=MITIGATED,
        )
    )

    zone = get_zone(result, DEMAND)

    assert zone["status"] == MITIGATED


def test_demand_zone_becomes_broken_after_close_below_zone():
    result = analyze(
        make_candles(
            base_type=DEMAND,
            final_status=BROKEN,
        )
    )

    zone = get_zone(result, DEMAND)

    assert zone["status"] == BROKEN


# ---------------------------------------------------------------------
# Summary and consistency
# ---------------------------------------------------------------------


def test_summary_contains_zone_counts():
    result = analyze(make_candles(base_type=SUPPLY))

    summary = result["summary"]

    expected_keys = {
        "total",
        "supply",
        "demand",
        "active",
        "mitigated",
        "broken",
    }

    assert expected_keys.issubset(summary.keys())


def test_summary_total_matches_detected_zones():
    result = analyze(make_candles(base_type=SUPPLY))

    assert result["summary"]["total"] == len(result["zones"])


def test_summary_supply_count_matches_supply_list():
    result = analyze(make_candles(base_type=SUPPLY))

    assert (
        result["summary"]["supply"]
        == len(result["supply_zones"])
    )


def test_summary_demand_count_matches_demand_list():
    result = analyze(make_candles(base_type=DEMAND))

    assert (
        result["summary"]["demand"]
        == len(result["demand_zones"])
    )


def test_summary_status_counts_match_zone_statuses():
    result = analyze(make_candles(base_type=SUPPLY))

    zones = result["zones"]
    summary = result["summary"]

    assert summary["active"] == sum(
        zone["status"] == ACTIVE for zone in zones
    )

    assert summary["mitigated"] == sum(
        zone["status"] == MITIGATED for zone in zones
    )

    assert summary["broken"] == sum(
        zone["status"] == BROKEN for zone in zones
    )


def test_zone_status_is_valid():
    result = analyze(make_candles(base_type=SUPPLY))

    allowed_statuses = {
        ACTIVE,
        MITIGATED,
        BROKEN,
    }

    for zone in result["zones"]:
        assert zone["status"] in allowed_statuses


def test_all_zones_are_present_in_combined_list():
    result = analyze(make_candles(base_type=SUPPLY))

    combined = (
        result["supply_zones"]
        + result["demand_zones"]
    )

    assert len(result["zones"]) == len(combined)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_rejects_non_dataframe_input():
    with pytest.raises(ZoneAnalysisError):
        analyze_supply_demand(
            data=[1, 2, 3],
            analysis_time=ANALYSIS_TIME,
        )


def test_rejects_empty_dataframe():
    with pytest.raises(ZoneAnalysisError):
        analyze_supply_demand(
            data=pd.DataFrame(),
            analysis_time=ANALYSIS_TIME,
        )


def test_rejects_missing_required_columns():
    data = make_candles(base_type=SUPPLY)

    data = data.drop(columns=["volume"])

    with pytest.raises(ZoneAnalysisError):
        analyze(data)


def test_rejects_insufficient_candles():
    data = make_candles(candle_count=10)

    with pytest.raises(ZoneAnalysisError):
        analyze(data)


def test_rejects_naive_analysis_timestamp():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze_supply_demand(
            data=data,
            analysis_time=datetime(2026, 11, 14, 0, 0),
        )


def test_rejects_unsorted_candles():
    data = make_candles(base_type=SUPPLY)

    data = data.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ZoneAnalysisError):
        analyze(data)


def test_rejects_duplicate_candles():
    data = make_candles(base_type=SUPPLY)

    data.loc[1, "open_time"] = data.loc[0, "open_time"]
    data.loc[1, "close_time"] = data.loc[0, "close_time"]

    with pytest.raises(ZoneAnalysisError):
        analyze(data)


def test_rejects_invalid_atr_period():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze(data, atr_period=0)


def test_rejects_invalid_impulse_window():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze(data, impulse_window=0)


def test_rejects_invalid_atr_multiplier():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze(data, atr_multiplier=0)


def test_rejects_invalid_minimum_candles():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze(data, minimum_candles=0)


def test_rejects_nan_atr_multiplier():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze(data, atr_multiplier=float("nan"))


def test_rejects_infinite_atr_multiplier():
    data = make_candles(base_type=SUPPLY)

    with pytest.raises(ZoneAnalysisError):
        analyze(data, atr_multiplier=float("inf"))


# ---------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------


def test_analysis_does_not_modify_input_dataframe():
    data = make_candles(base_type=SUPPLY)

    original = data.copy(deep=True)

    analyze(data)

    pd.testing.assert_frame_equal(data, original)


def test_analysis_is_deterministic():
    data = make_candles(base_type=SUPPLY)

    first = analyze(data)
    second = analyze(data)

    assert first == second


def test_analysis_does_not_accept_open_candles():
    data = make_candles(base_type=SUPPLY)

    data.loc[len(data) - 1, "close_time"] = (
        ANALYSIS_TIME + timedelta(minutes=5)
    )

    with pytest.raises(ZoneAnalysisError):
        analyze(data)


def test_analysis_rejects_non_utc_candle_timestamps():
    data = make_candles(base_type=SUPPLY)

    non_utc = timezone(timedelta(hours=3))

    data["open_time"] = data["open_time"].map(
        lambda value: value.astimezone(non_utc)
    )

    data["close_time"] = data["close_time"].map(
        lambda value: value.astimezone(non_utc)
    )

    with pytest.raises(ZoneAnalysisError):
        analyze(data)