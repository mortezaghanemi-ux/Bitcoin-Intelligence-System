"""Tests for trading signal position sizing and liquidation estimates."""

import pytest

from trader.signal_engine.engine import (
    TradingSignalEngineError,
    _calculate_position_risk,
)


def test_calculates_long_position_risk():
    result = _calculate_position_risk(
        direction="BULLISH",
        entry_price=100.0,
        stop_loss=95.0,
        account_equity=10_000.0,
        risk_fraction=0.005,
        leverage=2.0,
        maintenance_margin_rate=0.005,
    )

    position = result["position"]

    assert position["status"] == "CALCULATED"
    assert position["risk_amount"] == 50.0
    assert position["quantity_btc"] == 10.0
    assert position["notional_value"] == 1000.0
    assert position["required_margin"] == 500.0
    assert position["leverage"] == 2.0

    assert result["liquidation"]["status"] == "ESTIMATE_ONLY"
    assert result["liquidation"]["estimated_price"] == 50.5


def test_calculates_short_position_risk():
    result = _calculate_position_risk(
        direction="BEARISH",
        entry_price=100.0,
        stop_loss=105.0,
        account_equity=10_000.0,
        risk_fraction=0.005,
        leverage=2.0,
        maintenance_margin_rate=0.005,
    )

    position = result["position"]

    assert position["status"] == "CALCULATED"
    assert position["risk_amount"] == 50.0
    assert position["quantity_btc"] == 10.0
    assert position["notional_value"] == 1000.0
    assert position["required_margin"] == 500.0

    assert result["liquidation"]["estimated_price"] == 149.5


def test_rejects_position_exceeding_available_margin():
    with pytest.raises(
        TradingSignalEngineError,
        match="margin|equity|position",
    ):
        _calculate_position_risk(
            direction="BULLISH",
            entry_price=100.0,
            stop_loss=99.9,
            account_equity=10_000.0,
            risk_fraction=0.005,
            leverage=2.0,
            maintenance_margin_rate=0.005,
        )


def test_position_sizing_is_not_calculated_without_equity():
    result = _calculate_position_risk(
        direction="BULLISH",
        entry_price=100.0,
        stop_loss=95.0,
        account_equity=None,
        risk_fraction=0.005,
        leverage=2.0,
        maintenance_margin_rate=0.005,
    )

    position = result["position"]

    assert position["status"] == "NOT_CALCULATED"
    assert position["quantity_btc"] is None
    assert position["notional_value"] is None
    assert position["required_margin"] is None
    assert "reason" in position

    assert result["liquidation"]["estimated_price"] == 50.5


@pytest.mark.parametrize(
    "equity",
    [
        0,
        -100,
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        "10000",
    ],
)
def test_rejects_invalid_account_equity(equity):
    with pytest.raises(
        TradingSignalEngineError,
        match="account_equity",
    ):
        _calculate_position_risk(
            direction="BULLISH",
            entry_price=100.0,
            stop_loss=95.0,
            account_equity=equity,
            risk_fraction=0.005,
            leverage=2.0,
            maintenance_margin_rate=0.005,
        )


def test_liquidation_estimate_includes_maintenance_margin():
    long_result = _calculate_position_risk(
        direction="BULLISH",
        entry_price=100.0,
        stop_loss=95.0,
        account_equity=None,
        risk_fraction=0.005,
        leverage=2.0,
        maintenance_margin_rate=0.005,
    )

    short_result = _calculate_position_risk(
        direction="BEARISH",
        entry_price=100.0,
        stop_loss=105.0,
        account_equity=None,
        risk_fraction=0.005,
        leverage=2.0,
        maintenance_margin_rate=0.005,
    )

    assert long_result["liquidation"]["estimated_price"] == 50.5
    assert short_result["liquidation"]["estimated_price"] == 149.5


def test_liquidation_estimate_contains_warning():
    result = _calculate_position_risk(
        direction="BULLISH",
        entry_price=100.0,
        stop_loss=95.0,
        account_equity=None,
        risk_fraction=0.005,
        leverage=2.0,
        maintenance_margin_rate=0.005,
    )

    warning = result["liquidation"]["warning"]

    assert isinstance(warning, str)
    assert "Approximation only" in warning
    assert "exchange rules" in warning