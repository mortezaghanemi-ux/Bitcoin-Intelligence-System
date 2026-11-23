
"""Historical backtesting engine for the Bitcoin Intelligence System.

The engine uses the existing TradingSignalEngine directly.
It never connects to an exchange and never places real orders.

Execution rules:
- Signals are generated using candles closed at the decision timestamp.
- Entries are simulated at the next execution candle's open.
- Fees and adverse slippage are applied to both entry and exit.
- Stop loss takes priority when stop loss and take profit are both
  touched within the same candle.
- Open positions are closed at the final candle when the test ends.
"""

from __future__ import annotations

from datetime import timedelta
import math
from typing import Any

import pandas as pd

from trader.signal_engine.engine import (
    TradingSignalEngine,
    TradingSignalEngineError,
    SUPPORTED_TIMEFRAMES,
)


class BacktestingEngineError(RuntimeError):
    """Raised when historical backtesting cannot run safely."""


TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

REQUIRED_COLUMNS = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)

DIRECTION_SIGN = {
    "LONG": 1,
    "SHORT": -1,
}


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate and return a finite numeric value."""

    if isinstance(value, bool):
        raise BacktestingEngineError(
            f"{name} must be a numeric value."
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BacktestingEngineError(
            f"{name} must be a numeric value."
        ) from exc

    if not math.isfinite(number):
        raise BacktestingEngineError(
            f"{name} must be finite."
        )

    if minimum is not None and number < minimum:
        raise BacktestingEngineError(
            f"{name} must be at least {minimum}."
        )

    if maximum is not None and number > maximum:
        raise BacktestingEngineError(
            f"{name} must not exceed {maximum}."
        )

    return number


def _utc_series(
    values: pd.Series,
    *,
    timeframe: str,
    column: str,
) -> pd.Series:
    """Parse timestamps while requiring explicit UTC timezone data."""

    parsed = []

    for index, value in values.items():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BacktestingEngineError(
                f"{timeframe} has an invalid {column} at row {index}."
            ) from exc

        if pd.isna(timestamp):
            raise BacktestingEngineError(
                f"{timeframe} has a missing {column} at row {index}."
            )

        if timestamp.tzinfo is None:
            raise BacktestingEngineError(
                f"{timeframe} {column} must be timezone-aware UTC."
            )

        if timestamp.utcoffset() != timedelta(0):
            raise BacktestingEngineError(
                f"{timeframe} {column} must use UTC."
            )

        parsed.append(timestamp.tz_convert("UTC"))

    return pd.Series(
        pd.DatetimeIndex(parsed),
        index=values.index,
        name=column,
    )


def _prepare_timeframe(
    data: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """Validate and normalize one historical candle dataset."""

    if not isinstance(data, pd.DataFrame):
        raise BacktestingEngineError(
            f"{timeframe} data must be a pandas DataFrame."
        )

    if data.empty:
        raise BacktestingEngineError(
            f"{timeframe} historical data is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise BacktestingEngineError(
            f"{timeframe} is missing columns: "
            + ", ".join(missing)
        )

    result = data.loc[:, list(REQUIRED_COLUMNS)].copy(deep=True)

    result["open_time"] = _utc_series(
        result["open_time"],
        timeframe=timeframe,
        column="open_time",
    )

    result["close_time"] = _utc_series(
        result["close_time"],
        timeframe=timeframe,
        column="close_time",
    )

    interval = pd.Timedelta(
        minutes=TIMEFRAME_MINUTES[timeframe]
    )

    if (result["close_time"] <= result["open_time"]).any():
        raise BacktestingEngineError(
            f"{timeframe} contains invalid candle time ranges."
        )

    if not (
        result["close_time"] - result["open_time"]
    ).eq(interval).all():
        raise BacktestingEngineError(
            f"{timeframe} contains incorrect candle durations."
        )

    interval_ns = interval.value

    if result["open_time"].map(
        lambda value: value.value % interval_ns != 0
    ).any():
        raise BacktestingEngineError(
            f"{timeframe} contains misaligned candle timestamps."
        )

    result = result.sort_values(
        "open_time",
        kind="stable",
    ).reset_index(drop=True)

    if result["open_time"].duplicated().any():
        raise BacktestingEngineError(
            f"{timeframe} contains duplicate candle timestamps."
        )

    if result["close_time"].duplicated().any():
        raise BacktestingEngineError(
            f"{timeframe} contains duplicate candle close times."
        )

    if len(result) > 1:
        differences = result["open_time"].diff().iloc[1:]

        if not differences.eq(interval).all():
            raise BacktestingEngineError(
                f"{timeframe} contains missing, overlapping "
                "or out-of-order candle intervals."
            )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        original = result[column]

        if original.map(
            lambda value: isinstance(value, bool)
        ).any():
            raise BacktestingEngineError(
                f"{timeframe} {column} cannot contain booleans."
            )

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        if converted.isna().any():
            raise BacktestingEngineError(
                f"{timeframe} contains invalid {column} values."
            )

        converted = converted.astype(float)

        if not converted.map(math.isfinite).all():
            raise BacktestingEngineError(
                f"{timeframe} contains non-finite {column} values."
            )

        result[column] = converted

    if (
        result[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise BacktestingEngineError(
            f"{timeframe} prices must be greater than zero."
        )

    if (result["volume"] < 0).any():
        raise BacktestingEngineError(
            f"{timeframe} volume cannot be negative."
        )

    invalid_ohlc = (
        (result["high"] < result["low"])
        | (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():
        raise BacktestingEngineError(
            f"{timeframe} contains invalid OHLC relationships."
        )

    if not result["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    ).all():
        raise BacktestingEngineError(
            f"{timeframe} contains invalid source values."
        )

    return result


def _validate_datasets(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Validate all required trading timeframe datasets."""

    if not isinstance(data, dict):
        raise BacktestingEngineError(
            "data must be a dictionary of timeframe DataFrames."
        )

    if set(data) != set(SUPPORTED_TIMEFRAMES):
        missing = sorted(
            set(SUPPORTED_TIMEFRAMES) - set(data)
        )

        unknown = sorted(
            set(data) - set(SUPPORTED_TIMEFRAMES)
        )

        raise BacktestingEngineError(
            "Exactly 5m, 15m, 1h and 4h datasets are required. "
            f"Missing: {missing}; unsupported: {unknown}."
        )

    return {
        timeframe: _prepare_timeframe(
            data[timeframe],
            timeframe,
        )
        for timeframe in SUPPORTED_TIMEFRAMES
    }


class BacktestingEngine:
    """Run a historical simulation using the project's signal engine."""

    def __init__(
        self,
        *,
        initial_equity: float = 10000.0,
        leverage: float = 1.0,
        fee_rate: float = 0.0006,
        slippage_bps: float = 2.0,
        execution_timeframe: str = "5m",
        signal_engine: TradingSignalEngine | None = None,
        lookback_candles: int | None = 150,
        target_index: int = 0,
        decision_step: int = 1,
    ) -> None:
        if (
            isinstance(decision_step, bool)
            or not isinstance(decision_step, int)
            or decision_step < 1
        ):
            raise BacktestingEngineError(
                "decision_step must be an integer >= 1."
            )

        self.decision_step = decision_step

        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or target_index not in (0, 1, 2)
        ):
            raise BacktestingEngineError(
                "target_index must be 0, 1 or 2 (TP1, TP2 or TP3)."
            )

        self.target_index = target_index

        if lookback_candles is not None and (
            isinstance(lookback_candles, bool)
            or not isinstance(lookback_candles, int)
            or lookback_candles < 30
        ):
            raise BacktestingEngineError(
                "lookback_candles must be None or an integer >= 30."
            )

        self.lookback_candles = lookback_candles

        self.initial_equity = _finite_number(
            initial_equity,
            "initial_equity",
            minimum=0.00000001,
        )

        self.leverage = _finite_number(
            leverage,
            "leverage",
            minimum=1.0,
            maximum=2.0,
        )

        self.fee_rate = _finite_number(
            fee_rate,
            "fee_rate",
            minimum=0.0,
            maximum=0.05,
        )

        self.slippage_bps = _finite_number(
            slippage_bps,
            "slippage_bps",
            minimum=0.0,
            maximum=1000.0,
        )

        if execution_timeframe not in SUPPORTED_TIMEFRAMES:
            raise BacktestingEngineError(
                "Unsupported execution timeframe."
            )

        self.execution_timeframe = execution_timeframe

        if signal_engine is not None and not callable(
            getattr(signal_engine, "generate", None)
        ):
            raise BacktestingEngineError(
                "signal_engine must provide a callable generate method."
            )

        self.signal_engine = (
            signal_engine
            if signal_engine is not None
            else TradingSignalEngine(
                execution_timeframe=execution_timeframe,
            )
        )

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all state before a new backtest."""

        self.equity = self.initial_equity
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []
        self.open_position: dict[str, Any] | None = None
        self.total_fees = 0.0

    @property
    def slippage_fraction(self) -> float:
        """Return slippage in decimal form."""

        return self.slippage_bps / 10000.0

    def _build_snapshot(
        self,
        datasets: dict[str, pd.DataFrame],
        decision_time: pd.Timestamp,
    ) -> dict[str, pd.DataFrame] | None:
        """Expose only candles closed at or before decision_time."""

        snapshot = {}

        for timeframe in SUPPORTED_TIMEFRAMES:
            frame = datasets[timeframe]

            end = int(
                frame["close_time"].searchsorted(
                    decision_time,
                    side="right",
                )
            )

            start = (
                max(0, end - self.lookback_candles)
                if self.lookback_candles is not None
                else 0
            )

            available = frame.iloc[start:end].copy(deep=True)

            minimum_candles = int(
                getattr(
                    self.signal_engine,
                    "minimum_candles",
                    30,
                )
            )

            if len(available) < minimum_candles:
                return None

            snapshot[timeframe] = available.reset_index(drop=True)

        return snapshot

    def _extract_trade_levels(
        self,
        signal: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        """Extract entry reference, stop loss and first target."""

        if signal.get("status") != "REQUIRES_MANUAL_REVIEW":
            return None

        direction = signal.get("signal")

        if direction not in DIRECTION_SIGN:
            return None

        stop_loss = _finite_number(
            signal.get("stop_loss"),
            "signal stop_loss",
            minimum=0.00000001,
        )

        targets = signal.get("take_profit")

        if not isinstance(targets, list) or not targets:
            return None

        if len(targets) <= self.target_index:
            return None

        first_target = targets[self.target_index]

        if not isinstance(first_target, dict):
            return None

        target_price = _finite_number(
            first_target.get("price"),
            "first take-profit price",
            minimum=0.00000001,
        )

        entry_reference = _finite_number(
            signal.get("current_price"),
            "signal current_price",
            minimum=0.00000001,
        )

        if direction == "LONG":
            if not stop_loss < entry_reference < target_price:
                return None
        else:
            if not target_price < entry_reference < stop_loss:
                return None

        return entry_reference, stop_loss, target_price

    def _calculate_entry_price(
        self,
        direction: str,
        next_open: float,
    ) -> float:
        """Apply adverse slippage to the simulated entry."""

        if direction == "LONG":
            return next_open * (
                1.0 + self.slippage_fraction
            )

        return next_open * (
            1.0 - self.slippage_fraction
        )

    def _calculate_exit_price(
        self,
        direction: str,
        reference_price: float,
    ) -> float:
        """Apply adverse slippage to the simulated exit."""

        if direction == "LONG":
            return reference_price * (
                1.0 - self.slippage_fraction
            )

        return reference_price * (
            1.0 + self.slippage_fraction
        )

    def _open_trade(
        self,
        signal: dict[str, Any],
        next_candle: pd.Series,
        decision_time: pd.Timestamp,
    ) -> None:
        """Open a simulated position at the next candle's open."""

        levels = self._extract_trade_levels(signal)

        if levels is None:
            return

        _, stop_loss, target_price = levels

        direction = signal["signal"]
        next_open = float(next_candle["open"])

        entry_price = self._calculate_entry_price(
            direction,
            next_open,
        )

        if direction == "LONG":
            if not stop_loss < entry_price < target_price:
                return
        else:
            if not target_price < entry_price < stop_loss:
                return

        risk_distance = abs(entry_price - stop_loss)

        if risk_distance <= 0:
            return

        parameters = signal.get("parameters", {})

        if not isinstance(parameters, dict):
            parameters = {}

        risk_fraction = _finite_number(
            parameters.get("risk_fraction", 0.005),
            "signal risk_fraction",
            minimum=0.00000001,
            maximum=0.05,
        )

        risk_amount = self.equity * risk_fraction

        quantity_by_risk = risk_amount / risk_distance

        maximum_notional = self.equity * self.leverage

        quantity_by_leverage = maximum_notional / entry_price

        quantity = min(
            quantity_by_risk,
            quantity_by_leverage,
        )

        if not math.isfinite(quantity) or quantity <= 0:
            return

        notional = quantity * entry_price

        entry_fee = notional * self.fee_rate

        self.open_position = {
            "signal_id": signal.get("signal_id"),
            "direction": direction,
            "confidence": signal.get("confidence"),
            "quantity_btc": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": target_price,
            "entry_time": next_candle["open_time"].isoformat(),
            "decision_time": decision_time.isoformat(),
            "entry_fee": entry_fee,
            "entry_notional": notional,
            "leverage": self.leverage,
            "risk_amount": risk_amount,
        }

        self.total_fees += entry_fee

    def _check_exit(
        self,
        candle: pd.Series,
    ) -> tuple[float, str] | None:
        """Check stop loss and take profit against one candle.

        If both levels are touched in the same candle, stop loss
        is assumed to have been hit first.
        """

        position = self.open_position

        if position is None:
            return None

        direction = position["direction"]

        stop = float(position["stop_loss"])
        target = float(position["take_profit"])

        candle_open = float(candle["open"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])

        if direction == "LONG":
            if candle_open <= stop:
                return candle_open, "STOP_LOSS_GAP"

            if candle_open >= target:
                return candle_open, "TAKE_PROFIT_GAP"

            stop_hit = candle_low <= stop
            target_hit = candle_high >= target

        else:
            if candle_open >= stop:
                return candle_open, "STOP_LOSS_GAP"

            if candle_open <= target:
                return candle_open, "TAKE_PROFIT_GAP"

            stop_hit = candle_high >= stop
            target_hit = candle_low <= target

        if stop_hit:
            return stop, "STOP_LOSS"

        if target_hit:
            return target, "TAKE_PROFIT"

        return None

    def _close_trade(
        self,
        reference_exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ) -> None:
        """Close the active simulated position and record its result."""

        position = self.open_position

        if position is None:
            return

        direction = position["direction"]
        quantity = float(position["quantity_btc"])
        entry_price = float(position["entry_price"])

        exit_price = self._calculate_exit_price(
            direction,
            reference_exit_price,
        )

        direction_sign = DIRECTION_SIGN[direction]

        gross_pnl = (
            (exit_price - entry_price)
            * quantity
            * direction_sign
        )

        exit_notional = exit_price * quantity
        exit_fee = exit_notional * self.fee_rate

        entry_fee = float(position["entry_fee"])

        total_trade_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_trade_fees

        equity_before = self.equity

        self.equity += net_pnl
        self.total_fees += exit_fee

        if not math.isfinite(self.equity):
            raise BacktestingEngineError(
                "Backtest equity became non-finite."
            )

        trade = {
            "signal_id": position["signal_id"],
            "direction": direction,
            "confidence": position.get("confidence"),
            "entry_time": position["entry_time"],
            "decision_time": position["decision_time"],
            "exit_time": exit_time.isoformat(),
            "entry_price": round(entry_price, 8),
            "exit_price": round(exit_price, 8),
            "stop_loss": round(
                float(position["stop_loss"]),
                8,
            ),
            "take_profit": round(
                float(position["take_profit"]),
                8,
            ),
            "quantity_btc": round(quantity, 8),
            "leverage": position["leverage"],
            "gross_pnl": round(gross_pnl, 8),
            "entry_fee": round(entry_fee, 8),
            "exit_fee": round(exit_fee, 8),
            "total_fees": round(total_trade_fees, 8),
            "net_pnl": round(net_pnl, 8),
            "return_pct_on_equity": round(
                (net_pnl / equity_before) * 100.0
                if equity_before != 0
                else 0.0,
                6,
            ),
            "exit_reason": exit_reason,
            "equity_after_trade": round(self.equity, 8),
        }

        self.trades.append(trade)
        self.open_position = None

    def _record_equity(
        self,
        timestamp: pd.Timestamp,
    ) -> None:
        """Record the current realized equity observation.

        A second observation for the same timestamp (for example when
        a position is force-closed on the last candle) replaces the
        previous one so timestamps stay strictly increasing.
        """

        point = {
            "timestamp": timestamp.isoformat(),
            "equity": round(self.equity, 8),
        }

        if (
            self.equity_curve
            and self.equity_curve[-1]["timestamp"] == point["timestamp"]
        ):
            self.equity_curve[-1] = point
            return

        self.equity_curve.append(point)

    def _calculate_summary(self) -> dict[str, Any]:
        """Calculate basic backtest summary statistics."""

        wins = [
            trade
            for trade in self.trades
            if trade["net_pnl"] > 0
        ]

        losses = [
            trade
            for trade in self.trades
            if trade["net_pnl"] < 0
        ]

        final_equity = self.equity

        net_profit = final_equity - self.initial_equity

        net_return = (
            net_profit / self.initial_equity
        ) * 100.0

        peak = self.initial_equity
        maximum_drawdown = 0.0

        for point in self.equity_curve:
            current = float(point["equity"])
            peak = max(peak, current)

            if peak > 0:
                drawdown = (peak - current) / peak
                maximum_drawdown = max(
                    maximum_drawdown,
                    drawdown,
                )

        return {
            "initial_equity": round(self.initial_equity, 8),
            "final_equity": round(final_equity, 8),
            "net_profit": round(net_profit, 8),
            "net_return_pct": round(net_return, 6),
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "breakeven_trades": (
                len(self.trades) - len(wins) - len(losses)
            ),
            "win_rate_pct": round(
                (len(wins) / len(self.trades)) * 100.0
                if self.trades
                else 0.0,
                4,
            ),
            "total_fees": round(self.total_fees, 8),
            "maximum_drawdown_pct": round(
                maximum_drawdown * 100.0,
                6,
            ),
        }

    def run(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run a complete historical backtest.

        Args:
            data: Dictionary containing 5m, 15m, 1h and 4h
                historical candle DataFrames.

        Returns:
            A dictionary containing summary statistics,
            simulated trades and the realized equity curve.
        """

        self._reset_state()

        datasets = _validate_datasets(data)

        execution_data = datasets[
            self.execution_timeframe
        ]

        if len(execution_data) < 2:
            raise BacktestingEngineError(
                "At least two execution timeframe candles are required."
            )

        for index in range(len(execution_data) - 1):
            decision_candle = execution_data.iloc[index]
            next_candle = execution_data.iloc[index + 1]

            decision_time = decision_candle["close_time"]

            if self.open_position is not None:
                exit_result = self._check_exit(
                    next_candle
                )

                if exit_result is not None:
                    reference_price, reason = exit_result

                    self._close_trade(
                        reference_price,
                        next_candle["close_time"],
                        reason,
                    )

                self._record_equity(
                    next_candle["close_time"]
                )

                continue

            if index % self.decision_step != 0:
                self._record_equity(
                    next_candle["close_time"]
                )
                continue

            snapshot = self._build_snapshot(
                datasets,
                decision_time,
            )

            if snapshot is None:
                self._record_equity(
                    next_candle["close_time"]
                )
                continue

            try:
                signal = self.signal_engine.generate(
                    snapshot,
                    analysis_time=decision_time.to_pydatetime(),
                    account_equity=self.equity,
                    leverage=self.leverage,
                )

            except TradingSignalEngineError as exc:
                raise BacktestingEngineError(
                    "Signal generation failed during backtesting "
                    f"at {decision_time.isoformat()}: {exc}"
                ) from exc

            if not isinstance(signal, dict):
                raise BacktestingEngineError(
                    "Signal engine returned an invalid result."
                )

            if signal.get("signal") in DIRECTION_SIGN:
                self._open_trade(
                    signal,
                    next_candle,
                    decision_time,
                )

                if self.open_position is not None:
                    exit_result = self._check_exit(
                        next_candle
                    )

                    if exit_result is not None:
                        reference_price, reason = exit_result

                        self._close_trade(
                            reference_price,
                            next_candle["close_time"],
                            reason,
                        )

            self._record_equity(
                next_candle["close_time"]
            )

        if self.open_position is not None:
            final_candle = execution_data.iloc[-1]

            self._close_trade(
                float(final_candle["close"]),
                final_candle["close_time"],
                "END_OF_BACKTEST",
            )

            self._record_equity(
                final_candle["close_time"]
            )

        return {
            "status": "COMPLETED",
            "execution_timeframe": self.execution_timeframe,
            "execution": {
                "simulation_only": True,
                "exchange_connection": False,
                "order_execution": False,
                "fee_rate": self.fee_rate,
                "slippage_bps": self.slippage_bps,
                "leverage": self.leverage,
                "target": f"TP{self.target_index + 1}",
                "decision_step": self.decision_step,
            },
            "summary": self._calculate_summary(),
            "trades": list(self.trades),
            "equity_curve": list(self.equity_curve),
            "limitations": [
                "This is a candle-based simulation, not an exchange execution replay.",
                "Entry is simulated at the next candle open.",
                f"Only take-profit target TP{self.target_index + 1} is used.",
                "Stop loss is prioritized when both exit levels are touched in one candle.",
                "Funding payments and exchange-specific liquidation rules are not modeled.",
                "Trading fees and slippage are estimates supplied to the engine.",
                "No real orders are placed.",
            ],
        }


def run_backtest(
    data: dict[str, pd.DataFrame],
    *,
    initial_equity: float = 10000.0,
    leverage: float = 1.0,
    fee_rate: float = 0.0006,
    slippage_bps: float = 2.0,
    execution_timeframe: str = "5m",
    lookback_candles: int | None = 150,
    target_index: int = 0,
    decision_step: int = 1,
    signal_engine: Any = None,
) -> dict[str, Any]:
    """Convenience function for running one backtest."""

    engine = BacktestingEngine(
        signal_engine=signal_engine,
        lookback_candles=lookback_candles,
        target_index=target_index,
        decision_step=decision_step,
        initial_equity=initial_equity,
        leverage=leverage,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        execution_timeframe=execution_timeframe,
    )

    return engine.run(data)
