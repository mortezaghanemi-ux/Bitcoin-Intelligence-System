"""Bitcoin trading signal generation engine.

Generates LONG, SHORT and NO_TRADE signals for manual review.
No exchange connection, API keys or order execution are supported.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any

import pandas as pd

from trader.order_blocks.detector import OrderBlockDetector
from trader.patterns.detector import CandlestickPatternDetector
from trader.structure.analyzer import MarketStructureAnalyzer
from trader.timeframe.analyzer import MultiTimeframeAnalyzer


SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h", "4h")

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

DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL")


class TradingSignalEngineError(RuntimeError):
    """Raised when trading signal generation fails."""


def _normalize_utc(value: datetime | None) -> pd.Timestamp:
    """Validate and normalize an analysis timestamp to UTC."""

    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TradingSignalEngineError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise TradingSignalEngineError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise TradingSignalEngineError(
            "analysis_time must be timezone-aware UTC."
        )

    if timestamp.utcoffset() != timedelta(0):
        raise TradingSignalEngineError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_parameters(
    minimum_candles: int,
    minimum_confidence: float,
    minimum_risk_reward: float,
    maximum_leverage: float,
    swing_window: int,
    risk_fraction: float,
    maintenance_margin_rate: float,
) -> None:
    """Validate engine configuration."""

    if (
        isinstance(minimum_candles, bool)
        or not isinstance(minimum_candles, int)
        or minimum_candles < 5
    ):
        raise TradingSignalEngineError(
            "minimum_candles must be an integer of at least 5."
        )

    for name, value in (
        ("minimum_confidence", minimum_confidence),
        ("minimum_risk_reward", minimum_risk_reward),
        ("maximum_leverage", maximum_leverage),
        ("risk_fraction", risk_fraction),
        ("maintenance_margin_rate", maintenance_margin_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise TradingSignalEngineError(
                f"{name} must be a finite number."
            )

    if not 0 <= minimum_confidence <= 100:
        raise TradingSignalEngineError(
            "minimum_confidence must be between 0 and 100."
        )

    if minimum_risk_reward <= 0:
        raise TradingSignalEngineError(
            "minimum_risk_reward must be greater than zero."
        )

    if not 1 <= maximum_leverage <= 2:
        raise TradingSignalEngineError(
            "maximum_leverage must be between 1 and 2."
        )

    if not 0 < risk_fraction <= 0.05:
        raise TradingSignalEngineError(
            "risk_fraction must be greater than zero and at most 0.05."
        )

    if not 0 <= maintenance_margin_rate < 0.1:
        raise TradingSignalEngineError(
            "maintenance_margin_rate must be between 0 and 0.1."
        )

    if (
        isinstance(swing_window, bool)
        or not isinstance(swing_window, int)
        or swing_window <= 0
    ):
        raise TradingSignalEngineError(
            "swing_window must be a positive integer."
        )

    if minimum_candles < (2 * swing_window + 1):
        raise TradingSignalEngineError(
            "minimum_candles must be at least 2 * swing_window + 1."
        )


def _validate_input_data(
    data: dict[str, pd.DataFrame],
    analysis_time: pd.Timestamp,
    minimum_candles: int,
) -> dict[str, pd.DataFrame]:
    """Validate the supplied timeframe collection."""

    if not isinstance(data, dict):
        raise TradingSignalEngineError(
            "data must be a dictionary of timeframe DataFrames."
        )

    if set(data) != set(SUPPORTED_TIMEFRAMES):
        missing = sorted(set(SUPPORTED_TIMEFRAMES) - set(data))
        unknown = sorted(set(data) - set(SUPPORTED_TIMEFRAMES))

        details = []

        if missing:
            details.append("missing: " + ", ".join(missing))

        if unknown:
            details.append("unsupported: " + ", ".join(map(str, unknown)))

        raise TradingSignalEngineError(
            "Exactly 5m, 15m, 1h and 4h data are required. "
            + "; ".join(details)
        )

    validated: dict[str, pd.DataFrame] = {}

    for timeframe in SUPPORTED_TIMEFRAMES:
        candles = data[timeframe]

        if not isinstance(candles, pd.DataFrame):
            raise TradingSignalEngineError(
                f"{timeframe} data must be a pandas DataFrame."
            )

        if candles.empty:
            raise TradingSignalEngineError(
                f"{timeframe} candle data is empty."
            )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in candles.columns
        ]

        if missing_columns:
            raise TradingSignalEngineError(
                f"{timeframe} is missing columns: "
                + ", ".join(missing_columns)
            )

        if len(candles) < minimum_candles:
            raise TradingSignalEngineError(
                f"{timeframe} requires at least "
                f"{minimum_candles} candles."
            )

        copy = candles.loc[:, REQUIRED_COLUMNS].copy(deep=True)

        for column in ("open_time", "close_time"):
            try:
                timestamps = pd.to_datetime(
                    copy[column],
                    errors="raise",
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise TradingSignalEngineError(
                    f"{timeframe} has invalid {column} values."
                ) from exc

            if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
                raise TradingSignalEngineError(
                    f"{timeframe} {column} must be timezone-aware."
                )

            if str(timestamps.dt.tz) != "UTC":
                raise TradingSignalEngineError(
                    f"{timeframe} {column} must use UTC."
                )

            if timestamps.isna().any():
                raise TradingSignalEngineError(
                    f"{timeframe} {column} contains missing values."
                )

            copy[column] = timestamps.dt.tz_convert("UTC")

        interval_seconds = {
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
        }[timeframe]

        durations = copy["close_time"] - copy["open_time"]

        if not durations.eq(
            pd.Timedelta(seconds=interval_seconds)
        ).all():
            raise TradingSignalEngineError(
                f"{timeframe} contains incorrect candle durations."
            )

        if (copy["close_time"] <= copy["open_time"]).any():
            raise TradingSignalEngineError(
                f"{timeframe} contains invalid candle timestamps."
            )

        if (copy["close_time"] > analysis_time).any():
            raise TradingSignalEngineError(
                f"{timeframe} contains future or unclosed candles."
            )

        if not copy["open_time"].is_monotonic_increasing:
            raise TradingSignalEngineError(
                f"{timeframe} candles must be sorted."
            )

        if copy["open_time"].duplicated().any():
            raise TradingSignalEngineError(
                f"{timeframe} contains duplicate candles."
            )

        if copy["close_time"].duplicated().any():
            raise TradingSignalEngineError(
                f"{timeframe} contains duplicate close times."
            )

        if not copy["source"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise TradingSignalEngineError(
                f"{timeframe} contains invalid source values."
            )

        for column in ("open", "high", "low", "close", "volume"):
            original = copy[column]

            if original.map(
                lambda value: isinstance(value, bool)
            ).any():
                raise TradingSignalEngineError(
                    f"{timeframe} {column} cannot contain booleans."
                )

            try:
                numeric = pd.to_numeric(
                    original,
                    errors="raise",
                ).astype(float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TradingSignalEngineError(
                    f"{timeframe} {column} must be numeric."
                ) from exc

            if not numeric.map(math.isfinite).all():
                raise TradingSignalEngineError(
                    f"{timeframe} {column} contains non-finite values."
                )

            copy[column] = numeric

        if (
            copy[["open", "high", "low", "close"]] <= 0
        ).any().any():
            raise TradingSignalEngineError(
                f"{timeframe} prices must be greater than zero."
            )

        if (copy["volume"] < 0).any():
            raise TradingSignalEngineError(
                f"{timeframe} volume cannot be negative."
            )

        invalid_ohlc = (
            (copy["high"] < copy["low"])
            | (copy["high"] < copy["open"])
            | (copy["high"] < copy["close"])
            | (copy["low"] > copy["open"])
            | (copy["low"] > copy["close"])
        )

        if invalid_ohlc.any():
            raise TradingSignalEngineError(
                f"{timeframe} contains invalid OHLC relationships."
            )

        validated[timeframe] = copy.reset_index(drop=True)

    return validated


def _direction(value: Any) -> str:
    """Normalize a direction value."""

    if not isinstance(value, str):
        return "NEUTRAL"

    normalized = value.strip().upper()

    if normalized in DIRECTIONS:
        return normalized

    if normalized == "LONG":
        return "BULLISH"

    if normalized == "SHORT":
        return "BEARISH"

    return "NEUTRAL"


def _safe_float(value: Any) -> float | None:
    """Convert a value to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _calculate_atr(candles: pd.DataFrame, period: int = 14) -> float:
    """Calculate a simple average true range."""

    previous_close = candles["close"].shift(1)

    true_range = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - previous_close).abs(),
            (candles["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.tail(min(period, len(true_range))).mean()

    result = float(atr)

    if not math.isfinite(result) or result <= 0:
        raise TradingSignalEngineError(
            "ATR calculation produced an invalid value."
        )

    return result


def _build_signal_identity(
    data: dict[str, pd.DataFrame],
) -> str:
    """Build a stable identity from the latest candle of every timeframe."""

    identity = {}

    for timeframe in SUPPORTED_TIMEFRAMES:
        candles = data[timeframe]
        latest = candles.iloc[-1]

        identity[timeframe] = {
            "open_time": latest["open_time"].isoformat(),
            "close_time": latest["close_time"].isoformat(),
            "source": str(latest["source"]),
            "close": round(float(latest["close"]), 10),
        }

    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _extract_structure_direction(
    structure: dict[str, Any],
) -> str:
    """Read the direction from a structure analysis."""

    trend = structure.get("trend", {})

    if isinstance(trend, dict):
        return _direction(trend.get("direction"))

    return _direction(trend)


def _calculate_confidence(
    overall_direction: str,
    alignment_ratio: float,
    structure_direction: str,
    order_block_direction: str,
    pattern_direction: str,
    volume_direction: str,
    conflict_count: int,
) -> tuple[float, list[str]]:
    """Calculate a transparent rule-based confidence score."""

    reasons: list[str] = []

    score = max(0.0, min(40.0, alignment_ratio * 40.0))

    if overall_direction != "NEUTRAL":
        reasons.append(
            f"Multi-timeframe direction: {overall_direction}."
        )

    if structure_direction == overall_direction:
        score += 20.0
        reasons.append("Market structure confirms the directional bias.")
    elif structure_direction != "NEUTRAL":
        reasons.append("Market structure conflicts with the directional bias.")

    if order_block_direction == overall_direction:
        score += 15.0
        reasons.append("An active order block supports the directional bias.")

    if pattern_direction == overall_direction:
        score += 10.0
        reasons.append("Candlestick patterns support the directional bias.")

    if volume_direction == overall_direction:
        score += 10.0
        reasons.append("Volume confirmation supports the directional bias.")

    if conflict_count == 0:
        score += 5.0
        reasons.append("No directional conflict was detected between timeframes.")
    else:
        reasons.append(
            f"{conflict_count} timeframe conflicts were detected."
        )
        score -= min(20.0, conflict_count * 5.0)

    return round(max(0.0, min(100.0, score)), 2), reasons


def _select_order_block_direction(
    order_block_results: dict[str, dict[str, Any]],
    current_price: float,
) -> str:
    """Find the nearest active order block relative to current price."""

    candidates = []

    for timeframe, result in order_block_results.items():
        for zone in result.get("order_blocks", []):
            if zone.get("status") != "ACTIVE":
                continue

            direction = _direction(zone.get("direction"))

            if direction == "NEUTRAL":
                continue

            lower = _safe_float(zone.get("lower"))
            upper = _safe_float(zone.get("upper"))

            if lower is None or upper is None or lower > upper:
                continue

            distance = (
                0.0
                if lower <= current_price <= upper
                else min(
                    abs(current_price - lower),
                    abs(current_price - upper),
                )
            )

            candidates.append(
                (distance, timeframe, direction)
            )

    if not candidates:
        return "NEUTRAL"

    candidates.sort(key=lambda item: item[0])

    nearest_distance = candidates[0][0]

    nearest = [
        item
        for item in candidates
        if math.isclose(
            item[0],
            nearest_distance,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ]

    directions = {item[2] for item in nearest}

    if len(directions) != 1:
        return "NEUTRAL"

    return nearest[0][2]


def _select_pattern_direction(
    pattern_results: dict[str, pd.DataFrame],
) -> str:
    """Determine the dominant direction of newly detected patterns."""

    bullish = 0
    bearish = 0

    for patterns in pattern_results.values():
        if patterns.empty:
            continue

        for value in patterns["direction"].tolist():
            direction = _direction(value)

            if direction == "BULLISH":
                bullish += 1
            elif direction == "BEARISH":
                bearish += 1

    if bullish > bearish:
        return "BULLISH"

    if bearish > bullish:
        return "BEARISH"

    return "NEUTRAL"


def _build_risk_levels(
    direction: str,
    candles: pd.DataFrame,
    atr: float,
    minimum_risk_reward: float,
) -> dict[str, Any]:
    """Build entry, stop-loss and take-profit reference levels."""

    current_price = float(candles["close"].iloc[-1])

    recent = candles.tail(min(14, len(candles)))

    buffer = atr * 0.25
    entry_width = atr * 0.10

    if direction == "BULLISH":
        entry_lower = current_price - entry_width
        entry_upper = current_price

        structural_stop = float(recent["low"].min()) - buffer
        volatility_stop = entry_lower - (atr * 1.5)

        stop_loss = min(structural_stop, volatility_stop)

        risk = entry_upper - stop_loss

        if risk <= 0:
            raise TradingSignalEngineError(
                "Calculated LONG stop loss is invalid."
            )

        targets = [
            entry_upper + (risk * 1.0),
            entry_upper + (risk * max(1.5, minimum_risk_reward)),
            entry_upper + (risk * 2.0),
        ]

    elif direction == "BEARISH":
        entry_lower = current_price
        entry_upper = current_price + entry_width

        structural_stop = float(recent["high"].max()) + buffer
        volatility_stop = entry_upper + (atr * 1.5)

        stop_loss = max(structural_stop, volatility_stop)

        risk = stop_loss - entry_lower

        if risk <= 0:
            raise TradingSignalEngineError(
                "Calculated SHORT stop loss is invalid."
            )

        targets = [
            entry_lower - (risk * 1.0),
            entry_lower - (risk * max(1.5, minimum_risk_reward)),
            entry_lower - (risk * 2.0),
        ]

    else:
        raise TradingSignalEngineError(
            "Risk levels require a directional signal."
        )

    reward = (
        abs(targets[-1] - (
            entry_upper if direction == "BULLISH" else entry_lower
        ))
    )

    risk_reward = reward / risk

    if (
        not all(
            math.isfinite(value)
            for value in (
                entry_lower,
                entry_upper,
                stop_loss,
                risk,
                risk_reward,
                *targets,
            )
        )
        or entry_lower <= 0
        or entry_upper <= 0
        or stop_loss <= 0
        or risk_reward < minimum_risk_reward
    ):
        raise TradingSignalEngineError(
            "Calculated risk levels do not satisfy risk/reward requirements."
        )

    return {
        "entry_range": {
            "lower": round(entry_lower, 8),
            "upper": round(entry_upper, 8),
        },
        "stop_loss": round(stop_loss, 8),
        "take_profit": [
            {
                "level": index + 1,
                "price": round(target, 8),
                "risk_reward": round(
                    abs(target - (
                        entry_upper
                        if direction == "BULLISH"
                        else entry_lower
                    )) / risk,
                    4,
                ),
            }
            for index, target in enumerate(targets)
        ],
        "risk_per_unit": round(risk, 8),
        "risk_reward": round(risk_reward, 4),
    }


def _calculate_position_risk(
    direction: str,
    entry_price: float,
    stop_loss: float,
    account_equity: float | None,
    risk_fraction: float,
    leverage: float,
    maintenance_margin_rate: float,
) -> dict[str, Any]:
    """Calculate estimated position sizing and liquidation reference."""

    if account_equity is not None:
        if (
            isinstance(account_equity, bool)
            or not isinstance(account_equity, (int, float))
            or not math.isfinite(account_equity)
            or account_equity <= 0
        ):
            raise TradingSignalEngineError(
                "account_equity must be a finite positive number."
            )

        risk_amount = float(account_equity) * risk_fraction
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            raise TradingSignalEngineError(
                "Cannot calculate position size with zero risk."
            )

        quantity = risk_amount / risk_per_unit
        notional_value = quantity * entry_price
        required_margin = notional_value / leverage

        position = {
            "status": "CALCULATED",
            "account_equity": round(float(account_equity), 8),
            "risk_fraction": risk_fraction,
            "risk_amount": round(risk_amount, 8),
            "quantity_btc": round(quantity, 8),
            "notional_value": round(notional_value, 8),
            "required_margin": round(required_margin, 8),
            "leverage": leverage,
        }

    else:
        position = {
            "status": "NOT_CALCULATED",
            "reason": "Account equity was not supplied.",
            "quantity_btc": None,
            "notional_value": None,
            "required_margin": None,
            "leverage": leverage,
        }

    if direction == "BULLISH":
        liquidation_price = entry_price * (
            1.0 - (1.0 / leverage) + maintenance_margin_rate
        )
    else:
        liquidation_price = entry_price * (
            1.0 + (1.0 / leverage) - maintenance_margin_rate
        )

    if liquidation_price <= 0 or not math.isfinite(liquidation_price):
        liquidation_price = None

    return {
        "position": position,
        "liquidation": {
            "status": "ESTIMATE_ONLY",
            "estimated_price": (
                round(liquidation_price, 8)
                if liquidation_price is not None
                else None
            ),
            "leverage": leverage,
            "maintenance_margin_rate": maintenance_margin_rate,
            "warning": (
                "Approximation only. Actual liquidation depends on "
                "exchange rules, fees, funding, margin mode and maintenance tiers."
            ),
        },
    }


class TradingSignalEngine:
    """
    Generate trading signals for manual review.

    Features:
    - LONG, SHORT and NO_TRADE outputs.
    - Multi-timeframe confirmation.
    - Market structure analysis.
    - Order block and FVG context.
    - Candlestick pattern confirmation.
    - Confidence scoring and conflict rejection.
    - Entry, stop-loss and take-profit levels.
    - Risk/reward and optional position sizing.
    - Estimated liquidation reference.
    - UTC timestamps and closed candles only.
    - In-memory signal and rejection history.

    This module never connects to an exchange or executes trades.
    """

    def __init__(
        self,
        minimum_candles: int = 30,
        minimum_confidence: float = 60.0,
        minimum_risk_reward: float = 1.5,
        maximum_leverage: float = 2.0,
        swing_window: int = 3,
        risk_fraction: float = 0.005,
        maintenance_margin_rate: float = 0.005,
        execution_timeframe: str = "5m",
    ) -> None:

        _validate_parameters(
            minimum_candles,
            minimum_confidence,
            minimum_risk_reward,
            maximum_leverage,
            swing_window,
            risk_fraction,
            maintenance_margin_rate,
        )

        if execution_timeframe not in SUPPORTED_TIMEFRAMES:
            raise TradingSignalEngineError(
                "Unsupported execution_timeframe."
            )

        self.minimum_candles = minimum_candles
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_risk_reward = float(minimum_risk_reward)
        self.maximum_leverage = float(maximum_leverage)
        self.swing_window = swing_window
        self.risk_fraction = float(risk_fraction)
        self.maintenance_margin_rate = float(
            maintenance_margin_rate
        )
        self.execution_timeframe = execution_timeframe

        self._history: list[dict[str, Any]] = []
        self._known_ids: set[str] = set()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a defensive copy of all generated and rejected signals."""

        return json.loads(
            json.dumps(self._history)
        )

    def clear_history(self) -> None:
        """Clear in-memory signal history."""

        self._history.clear()
        self._known_ids.clear()

    def _record(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Preserve a signal once without modifying historical records."""

        signal_id = result["signal_id"]

        if signal_id not in self._known_ids:
            self._history.append(
                json.loads(json.dumps(result))
            )
            self._known_ids.add(signal_id)

            return json.loads(json.dumps(result))

        for previous in self._history:
            if previous["signal_id"] == signal_id:
                duplicate = json.loads(json.dumps(previous))
                duplicate["duplicate"] = True
                return duplicate

        return json.loads(json.dumps(result))

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        *,
        analysis_time: datetime | None = None,
        account_equity: float | None = None,
        leverage: float = 1.0,
    ) -> dict[str, Any]:
        """Generate one signal from four validated timeframe datasets."""

        timestamp = _normalize_utc(analysis_time)

        if (
            isinstance(leverage, bool)
            or not isinstance(leverage, (int, float))
            or not math.isfinite(leverage)
            or not 1 <= leverage <= self.maximum_leverage
        ):
            raise TradingSignalEngineError(
                f"leverage must be between 1 and {self.maximum_leverage}."
            )

        candles_by_timeframe = _validate_input_data(
            data,
            timestamp,
            self.minimum_candles,
        )

        identity = _build_signal_identity(
            candles_by_timeframe
        )

        signal_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        if signal_id in self._known_ids:
            return self._record({
                "signal_id": signal_id,
            })

        multi_timeframe = MultiTimeframeAnalyzer(
            minimum_candles=self.minimum_candles,
            required_timeframes=SUPPORTED_TIMEFRAMES,
        ).analyze(
            candles_by_timeframe,
            analysis_time=timestamp.to_pydatetime(),
        )

        structure_results = {}
        order_block_results = {}
        pattern_results = {}

        for timeframe in SUPPORTED_TIMEFRAMES:
            candles = candles_by_timeframe[timeframe]

            structure_results[timeframe] = (
                MarketStructureAnalyzer(
                    timeframe=timeframe,
                    swing_window=self.swing_window,
                    minimum_candles=self.minimum_candles,
                ).analyze(
                    candles,
                    analysis_time=timestamp.to_pydatetime(),
                )
            )

            order_block_results[timeframe] = (
                OrderBlockDetector(
                    timeframe=timeframe,
                    minimum_candles=self.minimum_candles,
                ).detect(
                    candles,
                    analysis_time=timestamp.to_pydatetime(),
                )
            )

            pattern_results[timeframe] = (
                CandlestickPatternDetector(
                    timeframe=timeframe
                ).detect(
                    candles,
                    detection_time=timestamp.to_pydatetime(),
                )
            )

        execution_candles = candles_by_timeframe[
            self.execution_timeframe
        ]

        latest_price = float(
            execution_candles["close"].iloc[-1]
        )

        overall_direction = _direction(
            multi_timeframe["overall_direction"]
        )

        alignment = multi_timeframe["alignment"]

        alignment_ratio = float(
            alignment.get("alignment_ratio", 0.0)
        )

        conflict_count = int(
            alignment.get("conflict_count", 0)
        )

        structure_direction = _extract_structure_direction(
            structure_results[self.execution_timeframe]
        )

        order_block_direction = _select_order_block_direction(
            order_block_results,
            latest_price,
        )

        pattern_direction = _select_pattern_direction(
            pattern_results
        )

        volume_direction = _direction(
            multi_timeframe["timeframes"][
                self.execution_timeframe
            ]["volume"]["direction"]
        )

        confidence, reasons = _calculate_confidence(
            overall_direction,
            alignment_ratio,
            structure_direction,
            order_block_direction,
            pattern_direction,
            volume_direction,
            conflict_count,
        )

        rejection_reasons = []

        if overall_direction == "NEUTRAL":
            rejection_reasons.append(
                "Multi-timeframe direction is neutral."
            )

        if conflict_count:
            rejection_reasons.append(
                "Conflicting timeframe directions were detected."
            )

        if confidence < self.minimum_confidence:
            rejection_reasons.append(
                "Confidence is below the configured minimum."
            )

        if (
            structure_direction not in ("NEUTRAL", overall_direction)
            and overall_direction != "NEUTRAL"
        ):
            rejection_reasons.append(
                "Market structure conflicts with the multi-timeframe direction."
            )

        if rejection_reasons:
            signal_type = "NO_TRADE"
            risk_levels = None
            risk_calculation = None

        else:
            signal_type = (
                "LONG"
                if overall_direction == "BULLISH"
                else "SHORT"
            )

            atr = _calculate_atr(execution_candles)

            try:
                risk_levels = _build_risk_levels(
                    overall_direction,
                    execution_candles,
                    atr,
                    self.minimum_risk_reward,
                )

                risk_calculation = _calculate_position_risk(
                    overall_direction,
                    latest_price,
                    risk_levels["stop_loss"],
                    account_equity,
                    self.risk_fraction,
                    float(leverage),
                    self.maintenance_margin_rate,
                )

            except TradingSignalEngineError as exc:
                rejection_reasons.append(str(exc))
                signal_type = "NO_TRADE"
                risk_levels = None
                risk_calculation = None

        if signal_type == "NO_TRADE":
            status = "REJECTED"
            reasons.extend(rejection_reasons)
        else:
            status = "REQUIRES_MANUAL_REVIEW"

        all_sources = sorted({
            str(source)
            for timeframe in SUPPORTED_TIMEFRAMES
            for source in candles_by_timeframe[timeframe]["source"].unique()
        })

        result = {
            "signal_id": signal_id,
            "analysis_timestamp": timestamp.isoformat(),
            "status": status,
            "signal": signal_type,
            "direction": overall_direction,
            "timeframe": self.execution_timeframe,
            "signal_timestamp": (
                execution_candles["close_time"].iloc[-1].isoformat()
            ),
            "current_price": round(latest_price, 8),
            "confidence": confidence,
            "confidence_threshold": self.minimum_confidence,
            "alignment": {
                "direction": overall_direction,
                "ratio": alignment_ratio,
                "status": alignment.get("status"),
                "conflict_count": conflict_count,
                "timeframe_directions": alignment.get("directions", {}),
            },
            "supporting_analysis": {
                "market_structure_direction": structure_direction,
                "order_block_direction": order_block_direction,
                "candlestick_pattern_direction": pattern_direction,
                "volume_direction": volume_direction,
                "structure_by_timeframe": {
                    timeframe: _extract_structure_direction(value)
                    for timeframe, value in structure_results.items()
                },
                "active_order_blocks": {
                    timeframe: value["summary"]["active_order_blocks"]
                    for timeframe, value in order_block_results.items()
                },
                "detected_patterns": {
                    timeframe: len(value)
                    for timeframe, value in pattern_results.items()
                },
            },
            "entry_range": (
                risk_levels["entry_range"]
                if risk_levels is not None
                else None
            ),
            "stop_loss": (
                risk_levels["stop_loss"]
                if risk_levels is not None
                else None
            ),
            "take_profit": (
                risk_levels["take_profit"]
                if risk_levels is not None
                else []
            ),
            "risk_reward": (
                risk_levels["risk_reward"]
                if risk_levels is not None
                else None
            ),
            "risk_per_unit": (
                risk_levels["risk_per_unit"]
                if risk_levels is not None
                else None
            ),
            "risk_calculation": risk_calculation,
            "reasons": reasons,
            "rejection_reasons": rejection_reasons,
            "data_quality": {
                "status": "VALIDATED",
                "validated_timeframes": list(SUPPORTED_TIMEFRAMES),
                "closed_candles_only": True,
                "future_data_used": False,
                "sources": all_sources,
                "last_candle_close_times": {
                    timeframe: (
                        candles_by_timeframe[timeframe]
                        ["close_time"].iloc[-1].isoformat()
                    )
                    for timeframe in SUPPORTED_TIMEFRAMES
                },
            },
            "parameters": {
                "minimum_candles": self.minimum_candles,
                "minimum_confidence": self.minimum_confidence,
                "minimum_risk_reward": self.minimum_risk_reward,
                "maximum_leverage": self.maximum_leverage,
                "applied_leverage": float(leverage),
                "risk_fraction": self.risk_fraction,
                "execution_timeframe": self.execution_timeframe,
            },
            "execution": {
                "manual_review_only": True,
                "auto_trading": False,
                "exchange_connection": False,
                "order_execution": False,
            },
            "limitations": [
                "Confidence is a rule-based score, not a calibrated probability.",
                "Position sizing is calculated only when account equity is supplied.",
                "Liquidation price is an estimate and is not exchange-specific.",
                "Entry, stop-loss and take-profit levels are technical references.",
                "Signals require independent manual review before any decision.",
                "This module does not connect to exchanges.",
                "This module does not execute trades.",
            ],
        }

        return self._record(result)


def generate_trading_signal(
    data: dict[str, pd.DataFrame],
    *,
    analysis_time: datetime | None = None,
    account_equity: float | None = None,
    leverage: float = 1.0,
    minimum_candles: int = 30,
    minimum_confidence: float = 60.0,
    minimum_risk_reward: float = 1.5,
    maximum_leverage: float = 2.0,
    swing_window: int = 3,
    risk_fraction: float = 0.005,
    maintenance_margin_rate: float = 0.005,
    execution_timeframe: str = "5m",
) -> dict[str, Any]:
    """Generate a one-time trading signal."""

    engine = TradingSignalEngine(
        minimum_candles=minimum_candles,
        minimum_confidence=minimum_confidence,
        minimum_risk_reward=minimum_risk_reward,
        maximum_leverage=maximum_leverage,
        swing_window=swing_window,
        risk_fraction=risk_fraction,
        maintenance_margin_rate=maintenance_margin_rate,
        execution_timeframe=execution_timeframe,
    )

    return engine.generate(
        data,
        analysis_time=analysis_time,
        account_equity=account_equity,
        leverage=leverage,
    )


if __name__ == "__main__":
    print("Bitcoin trading signal engine loaded.")