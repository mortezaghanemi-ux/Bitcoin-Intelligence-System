"""A deliberately simple trend-following signal: Donchian breakouts.

Rule (all parameters fixed in advance, see config/strategy_profiles.yaml):
  LONG  when a 4h candle closes above the highest high of the previous
        N candles, and the candle before it had not already done so.
  SHORT is the mirror image.
The stop sits `stop_atr` x ATR(14) from the close; targets are fixed
multiples of that risk. It uses only closed candles and has far fewer
tunable parts than the confluence engine, so it is harder to overfit.

This engine is for backtesting; it is not wired into the live trader.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

ATR_PERIOD = 14


class BreakoutEngineError(RuntimeError):
    """Raised for invalid breakout engine input."""


def _atr(candles: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high = candles["high"]
    low = candles["low"]
    previous = candles["close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)

    return float(true_range.rolling(period).mean().iloc[-1])


class DonchianBreakoutEngine:
    def __init__(
        self,
        *,
        timeframe: str = "4h",
        channel: int = 20,
        stop_atr: float = 2.0,
        targets_r: tuple[float, ...] = (1.5, 2.0, 3.0),
        fresh_window_minutes: float = 60.0,
        risk_fraction: float = 0.005,
    ) -> None:
        if channel < 5:
            raise BreakoutEngineError("channel must be at least 5.")

        if stop_atr <= 0 or any(r <= 0 for r in targets_r):
            raise BreakoutEngineError("stop_atr and targets must be > 0.")

        self.timeframe = timeframe
        self.channel = channel
        self.stop_atr = stop_atr
        self.targets_r = tuple(targets_r)
        self.fresh_window = timedelta(minutes=fresh_window_minutes)
        self.risk_fraction = risk_fraction
        self.minimum_candles = channel + ATR_PERIOD + 2

    def _no_trade(self, when: datetime, price: float, reason: str):
        return {
            "signal": "NO_TRADE",
            "status": "NO_TRADE",
            "analysis_timestamp": when.isoformat(),
            "current_price": price,
            "reasons": [reason],
        }

    def generate(
        self,
        snapshot: dict[str, pd.DataFrame],
        *,
        analysis_time: datetime,
        account_equity: float | None = None,
        leverage: float = 1.0,
    ) -> dict[str, Any]:
        candles = snapshot[self.timeframe]

        if len(candles) < self.minimum_candles:
            raise BreakoutEngineError("Not enough candles.")

        last = candles.iloc[-1]
        price = float(last["close"])

        # Only act right after the 4h candle closed.
        age = pd.Timestamp(analysis_time) - pd.Timestamp(last["close_time"])

        if age > self.fresh_window:
            return self._no_trade(
                analysis_time, price, "Breakout candle is not fresh."
            )

        def channel_at(position: int) -> tuple[float, float]:
            window = candles.iloc[position - self.channel: position]
            return float(window["high"].max()), float(window["low"].min())

        n = len(candles)
        upper, lower = channel_at(n - 1)
        previous_upper, previous_lower = channel_at(n - 2)
        previous_close = float(candles.iloc[-2]["close"])

        direction = None

        if price > upper and previous_close <= previous_upper:
            direction = "LONG"
        elif price < lower and previous_close >= previous_lower:
            direction = "SHORT"

        if direction is None:
            return self._no_trade(
                analysis_time, price, "No fresh channel breakout."
            )

        atr = _atr(candles)
        risk = self.stop_atr * atr

        if not risk > 0:
            return self._no_trade(analysis_time, price, "ATR unavailable.")

        sign = 1 if direction == "LONG" else -1
        stop = price - sign * risk
        targets = [
            {"level": index + 1, "price": price + sign * risk * r}
            for index, r in enumerate(self.targets_r)
        ]

        identity = json.dumps(
            [
                analysis_time.isoformat(), direction, self.channel,
                self.stop_atr, list(self.targets_r),
            ]
        )

        return {
            "signal_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
            "signal": direction,
            "status": "REQUIRES_MANUAL_REVIEW",
            "analysis_timestamp": analysis_time.isoformat(),
            "timeframe": self.timeframe,
            "current_price": price,
            "entry_range": {"lower": price, "upper": price},
            "stop_loss": stop,
            "take_profit": targets,
            "confidence": 70.0,
            "parameters": {"risk_fraction": self.risk_fraction},
            "reasons": [
                f"4h close broke the {self.channel}-candle "
                f"{'high' if direction == 'LONG' else 'low'}."
            ],
        }
