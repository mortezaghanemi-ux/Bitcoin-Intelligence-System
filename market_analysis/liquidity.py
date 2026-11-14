"""Bitcoin market liquidity analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from validation.candles import validate_candles


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

BUY_SIDE = "BUY_SIDE"
SELL_SIDE = "SELL_SIDE"

ACTIVE = "ACTIVE"
SWEPT = "SWEPT"


class LiquidityAnalysisError(RuntimeError):
    """Raised when liquidity analysis cannot be completed."""


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Normalize analysis timestamp to UTC."""

    if analysis_time is None:
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        timestamp = pd.Timestamp(analysis_time)

    except (TypeError, ValueError, OverflowError) as exc:
        raise LiquidityAnalysisError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise LiquidityAnalysisError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise LiquidityAnalysisError(
            "analysis_time must be timezone-aware."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise LiquidityAnalysisError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_parameters(
    swing_window: int,
    minimum_touches: int,
    minimum_candles: int,
    level_tolerance: float,
) -> None:
    """Validate liquidity analysis parameters."""

    integer_parameters = (
        ("swing_window", swing_window),
        ("minimum_touches", minimum_touches),
        ("minimum_candles", minimum_candles),
    )

    for name, value in integer_parameters:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise LiquidityAnalysisError(
                f"{name} must be a positive integer."
            )

    if minimum_touches < 2:
        raise LiquidityAnalysisError(
            "minimum_touches must be at least 2."
        )

    if minimum_candles < (2 * swing_window + 1):
        raise LiquidityAnalysisError(
            "minimum_candles must be at least "
            "2 * swing_window + 1."
        )

    if not isinstance(
        level_tolerance,
        (int, float),
    ) or isinstance(level_tolerance, bool):
        raise LiquidityAnalysisError(
            "level_tolerance must be numeric."
        )

    if (
        not np.isfinite(level_tolerance)
        or level_tolerance <= 0
        or level_tolerance >= 0.05
    ):
        raise LiquidityAnalysisError(
            "level_tolerance must be greater than 0 "
            "and smaller than 0.05."
        )


def _validate_candles(
    data: pd.DataFrame,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate OHLCV candles using the central validator."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise LiquidityAnalysisError(
            "Cannot analyze empty candle data."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise LiquidityAnalysisError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    try:
        accepted, rejected, report = validate_candles(
            data=data,
            validation_time=analysis_time.to_pydatetime(),
            require_closed=True,
        )

    except (TypeError, ValueError) as exc:
        raise LiquidityAnalysisError(
            f"Candle validation failed: {exc}"
        ) from exc

    if not rejected.empty:
        raise LiquidityAnalysisError(
            "Invalid candles detected. "
            f"Rejected: {len(rejected)}. "
            f"Validation errors: {report['errors']}."
        )

    if accepted.empty:
        raise LiquidityAnalysisError(
            "No valid candles available."
        )

    result = accepted.copy(deep=True)

    if not result["open_time"].is_monotonic_increasing:
        raise LiquidityAnalysisError(
            "Candles must be sorted by open_time."
        )

    if not result["close_time"].is_monotonic_increasing:
        raise LiquidityAnalysisError(
            "Candles must be sorted by close_time."
        )

    if result["open_time"].duplicated().any():
        raise LiquidityAnalysisError(
            "Duplicate candle open_time values detected."
        )

    if result["close_time"].duplicated().any():
        raise LiquidityAnalysisError(
            "Duplicate candle close_time values detected."
        )

    for column in ("open_time", "close_time"):
        if not isinstance(
            result[column].dtype,
            pd.DatetimeTZDtype,
        ):
            raise LiquidityAnalysisError(
                f"{column} must be timezone-aware."
            )

        if str(result[column].dt.tz) != "UTC":
            raise LiquidityAnalysisError(
                f"{column} must use UTC."
            )

    valid_sources = result["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise LiquidityAnalysisError(
            "Every candle must have a valid source."
        )

    return result.reset_index(drop=True)


def _find_confirmed_swings(
    data: pd.DataFrame,
    swing_window: int,
) -> list[dict[str, Any]]:
    """
    Detect confirmed swing highs and lows.

    A swing is only available after all right-side
    confirmation candles have closed.
    """

    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)

    swings: list[dict[str, Any]] = []

    for index in range(
        swing_window,
        len(data) - swing_window,
    ):
        left = index - swing_window
        right = index + swing_window + 1

        other_highs = np.concatenate(
            (
                highs[left:index],
                highs[index + 1:right],
            )
        )

        other_lows = np.concatenate(
            (
                lows[left:index],
                lows[index + 1:right],
            )
        )

        if highs[index] > other_highs.max():
            swings.append(
                {
                    "type": "HIGH",
                    "index": index,
                    "price": float(highs[index]),
                    "time": data.iloc[index][
                        "close_time"
                    ].isoformat(),
                    "confirmation_index": (
                        index + swing_window
                    ),
                    "confirmation_time": data.iloc[
                        index + swing_window
                    ]["close_time"].isoformat(),
                }
            )

        if lows[index] < other_lows.min():
            swings.append(
                {
                    "type": "LOW",
                    "index": index,
                    "price": float(lows[index]),
                    "time": data.iloc[index][
                        "close_time"
                    ].isoformat(),
                    "confirmation_index": (
                        index + swing_window
                    ),
                    "confirmation_time": data.iloc[
                        index + swing_window
                    ]["close_time"].isoformat(),
                }
            )

    swings.sort(
        key=lambda item: (
            item["confirmation_index"],
            item["index"],
            item["type"],
        )
    )

    return swings


def _group_liquidity_levels(
    swings: list[dict[str, Any]],
    minimum_touches: int,
    level_tolerance: float,
) -> list[dict[str, Any]]:
    """
    Group nearby confirmed swing levels.

    Only levels of the same type are grouped.
    Pool prices are calculated from their confirmed
    member swings.
    """

    groups: list[dict[str, Any]] = []

    for swing in swings:
        matching_group = None

        for group in groups:
            if group["swing_type"] != swing["type"]:
                continue

            reference_price = group["reference_price"]

            difference = abs(
                swing["price"] - reference_price
            ) / reference_price

            if difference <= level_tolerance:
                matching_group = group
                break

        if matching_group is None:
            groups.append(
                {
                    "swing_type": swing["type"],
                    "reference_price": swing["price"],
                    "members": [swing],
                }
            )

        else:
            matching_group["members"].append(swing)

            prices = [
                item["price"]
                for item in matching_group["members"]
            ]

            matching_group["reference_price"] = float(
                np.mean(prices)
            )

    pools: list[dict[str, Any]] = []

    for group in groups:
        members = group["members"]

        if len(members) < minimum_touches:
            continue

        prices = [
            item["price"]
            for item in members
        ]

        confirmation_index = max(
            item["confirmation_index"]
            for item in members
        )

        confirmation_time = max(
            members,
            key=lambda item: item["confirmation_index"],
        )["confirmation_time"]

        if group["swing_type"] == "HIGH":
            pool_type = BUY_SIDE
        else:
            pool_type = SELL_SIDE

        level = float(np.mean(prices))

        pools.append(
            {
                "type": pool_type,
                "level": level,
                "upper": float(max(prices)),
                "lower": float(min(prices)),
                "touch_count": len(members),
                "member_prices": prices,
                "member_times": [
                    item["time"]
                    for item in members
                ],
                "first_touch_time": members[0]["time"],
                "last_touch_time": members[-1]["time"],
                "confirmation_index": confirmation_index,
                "confirmation_time": confirmation_time,
            }
        )

    pools.sort(
        key=lambda item: (
            item["confirmation_index"],
            item["type"],
            item["level"],
        )
    )

    return pools


def _detect_liquidity_sweeps(
    data: pd.DataFrame,
    pools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Detect liquidity sweeps after pool confirmation.

    BUY_SIDE:
        Price trades above the level and closes below it.

    SELL_SIDE:
        Price trades below the level and closes above it.

    Candles before pool confirmation are never evaluated.
    """

    evaluated_pools: list[dict[str, Any]] = []
    all_sweeps: list[dict[str, Any]] = []

    for pool_number, original_pool in enumerate(
        pools,
        start=1,
    ):
        pool = dict(original_pool)

        sweeps: list[dict[str, Any]] = []

        level = pool["level"]
        confirmation_index = pool[
            "confirmation_index"
        ]

        for index in range(
            confirmation_index + 1,
            len(data),
        ):
            candle = data.iloc[index]

            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])

            swept = False

            if pool["type"] == BUY_SIDE:
                swept = (
                    high > level
                    and close < level
                )

            elif pool["type"] == SELL_SIDE:
                swept = (
                    low < level
                    and close > level
                )

            if not swept:
                continue

            sweep = {
                "pool_id": pool_number,
                "type": pool["type"],
                "level": level,
                "candle_index": index,
                "time": candle["close_time"].isoformat(),
                "high": high,
                "low": low,
                "close": close,
                "wick_distance": (
                    high - level
                    if pool["type"] == BUY_SIDE
                    else level - low
                ),
                "source": str(candle["source"]),
            }

            sweeps.append(sweep)
            all_sweeps.append(sweep)

        pool["sweeps"] = sweeps
        pool["sweep_count"] = len(sweeps)

        if sweeps:
            pool["status"] = SWEPT
            pool["first_sweep_time"] = sweeps[0]["time"]
            pool["last_sweep_time"] = sweeps[-1]["time"]
        else:
            pool["status"] = ACTIVE
            pool["first_sweep_time"] = None
            pool["last_sweep_time"] = None

        evaluated_pools.append(pool)

    return evaluated_pools, all_sweeps


def analyze_liquidity(
    data: pd.DataFrame,
    swing_window: int = 3,
    minimum_touches: int = 2,
    level_tolerance: float = 0.001,
    minimum_candles: int = 30,
    analysis_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Analyze Bitcoin market liquidity levels.

    Features:
    - Confirmed swing highs and lows.
    - Buy-side liquidity pools.
    - Sell-side liquidity pools.
    - Equal-high and equal-low level grouping.
    - Liquidity sweep detection.
    - Active and swept pool states.
    - UTC timestamps and source tracking.

    Only validated closed candles are accepted.
    Swing points require right-side confirmation.
    Liquidity sweeps are evaluated only after pool
    confirmation, preventing look-ahead bias.

    This module does not generate trading signals.
    It does not execute trades.
    It does not modify input data.
    """

    _validate_parameters(
        swing_window=swing_window,
        minimum_touches=minimum_touches,
        minimum_candles=minimum_candles,
        level_tolerance=level_tolerance,
    )

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    candles = _validate_candles(
        data,
        timestamp,
    )

    if len(candles) < minimum_candles:
        raise LiquidityAnalysisError(
            f"At least {minimum_candles} candles are required. "
            f"Received: {len(candles)}."
        )

    swings = _find_confirmed_swings(
        candles,
        swing_window,
    )

    raw_pools = _group_liquidity_levels(
        swings=swings,
        minimum_touches=minimum_touches,
        level_tolerance=float(level_tolerance),
    )

    pools, sweeps = _detect_liquidity_sweeps(
        candles,
        raw_pools,
    )

    buy_side_pools = [
        pool
        for pool in pools
        if pool["type"] == BUY_SIDE
    ]

    sell_side_pools = [
        pool
        for pool in pools
        if pool["type"] == SELL_SIDE
    ]

    sources = sorted(
        candles["source"].astype(str).unique().tolist()
    )

    buy_side_sweeps = [
        sweep
        for sweep in sweeps
        if sweep["type"] == BUY_SIDE
    ]

    sell_side_sweeps = [
        sweep
        for sweep in sweeps
        if sweep["type"] == SELL_SIDE
    ]

    return {
        "analysis_timestamp": timestamp.isoformat(),
        "status": "SUCCESS",
        "source": sources,
        "candles_analyzed": len(candles),
        "last_candle_close_time": (
            candles["close_time"].iloc[-1].isoformat()
        ),
        "swing_highs": [
            swing
            for swing in swings
            if swing["type"] == "HIGH"
        ],
        "swing_lows": [
            swing
            for swing in swings
            if swing["type"] == "LOW"
        ],
        "swings": swings,
        "buy_side_liquidity": buy_side_pools,
        "sell_side_liquidity": sell_side_pools,
        "pools": pools,
        "sweeps": sweeps,
        "summary": {
            "swing_high_count": sum(
                swing["type"] == "HIGH"
                for swing in swings
            ),
            "swing_low_count": sum(
                swing["type"] == "LOW"
                for swing in swings
            ),
            "total_pools": len(pools),
            "buy_side_pool_count": len(
                buy_side_pools
            ),
            "sell_side_pool_count": len(
                sell_side_pools
            ),
            "active_pool_count": sum(
                pool["status"] == ACTIVE
                for pool in pools
            ),
            "swept_pool_count": sum(
                pool["status"] == SWEPT
                for pool in pools
            ),
            "total_sweeps": len(sweeps),
            "buy_side_sweep_count": len(
                buy_side_sweeps
            ),
            "sell_side_sweep_count": len(
                sell_side_sweeps
            ),
        },
        "parameters": {
            "swing_window": swing_window,
            "minimum_touches": minimum_touches,
            "level_tolerance": float(
                level_tolerance
            ),
            "minimum_candles": minimum_candles,
        },
    }


if __name__ == "__main__":
    print(
        "Bitcoin market liquidity analysis module loaded."
    )