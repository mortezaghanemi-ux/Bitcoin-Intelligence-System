"""Strategy profiles: switchable rules layered on the signal engine.

A profile changes which signals are allowed and how stops are placed.
Every layer is optional and off by default, so an empty profile behaves
exactly like the original engine. Profiles exist so each layer can be
measured separately with the backtest before it is trusted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

TIMEFRAMES = ("5m", "15m", "1h", "4h")
REGIMES = (
    "TRENDING",
    "RANGING",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    "TRANSITIONAL",
)

DEFAULT_PROFILE_FILE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "strategy_profiles.yaml"
)


class StrategyProfileError(ValueError):
    """Raised for an invalid strategy profile."""


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    description: str = ""
    execution_timeframe: str = "5m"
    # Timeframe used to simulate fills in a backtest.
    fill_timeframe: str = "5m"
    minimum_confidence: float = 60.0
    minimum_risk_reward: float = 1.5
    stop_atr_multiplier: float = 1.5

    # Layer 2: market regime.
    allowed_regimes: tuple[str, ...] = ()
    regime_timeframe: str = "4h"
    require_regime_direction: bool = False
    minimum_adx: float | None = None

    # Layer 3: entry location.
    higher_timeframe_bias: str | None = None
    maximum_pullback_atr: float | None = None

    # Which signal generator: the confluence engine or the simple
    # Donchian breakout (trader/breakout_engine.py).
    engine: str = "confluence"
    breakout_channel: int = 20
    breakout_stop_atr: float = 2.0
    breakout_targets_r: tuple[float, ...] = (1.5, 2.0, 3.0)

    # Layer 5: scheduled-event blackout, as (start, end) ISO times.
    blackout_windows: tuple[tuple[str, str], ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).replace("_", "").isalnum():
            raise StrategyProfileError(
                "Profile name must be letters, digits and underscores."
            )

        for label, value in (
            ("execution_timeframe", self.execution_timeframe),
            ("fill_timeframe", self.fill_timeframe),
            ("regime_timeframe", self.regime_timeframe),
        ):
            if value not in TIMEFRAMES:
                raise StrategyProfileError(f"Invalid {label}: {value}")

        if (
            self.higher_timeframe_bias is not None
            and self.higher_timeframe_bias not in TIMEFRAMES
        ):
            raise StrategyProfileError("Invalid higher_timeframe_bias.")

        for regime in self.allowed_regimes:
            if regime not in REGIMES:
                raise StrategyProfileError(f"Unknown regime: {regime}")

        if self.engine not in ("confluence", "donchian"):
            raise StrategyProfileError(f"Unknown engine: {self.engine}")

        if self.breakout_channel < 5 or self.breakout_stop_atr <= 0:
            raise StrategyProfileError("Invalid breakout settings.")

        if not 0 <= self.minimum_confidence <= 100:
            raise StrategyProfileError(
                "minimum_confidence must be between 0 and 100."
            )

        if self.minimum_risk_reward <= 0 or self.stop_atr_multiplier <= 0:
            raise StrategyProfileError(
                "Risk/reward and stop multiplier must be positive."
            )

        if (
            self.maximum_pullback_atr is not None
            and self.maximum_pullback_atr <= 0
        ):
            raise StrategyProfileError(
                "maximum_pullback_atr must be positive."
            )

        for window in self.blackout_windows:
            start, end = (_parse_time(item) for item in window)

            if end <= start:
                raise StrategyProfileError(
                    "Blackout window must end after it starts."
                )

    def fingerprint(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.fingerprint())


def _parse_time(value: str) -> datetime:
    parsed = pd.Timestamp(value)

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")

    return parsed.tz_convert("UTC").to_pydatetime()


def profile_from_mapping(
    name: str,
    mapping: dict[str, Any],
) -> StrategyProfile:
    allowed = {
        "description", "execution_timeframe", "fill_timeframe",
        "minimum_confidence", "minimum_risk_reward",
        "stop_atr_multiplier", "allowed_regimes", "regime_timeframe",
        "require_regime_direction", "minimum_adx",
        "higher_timeframe_bias", "maximum_pullback_atr",
        "blackout_windows", "engine", "breakout_channel",
        "breakout_stop_atr", "breakout_targets_r",
    }
    unknown = set(mapping) - allowed

    if unknown:
        raise StrategyProfileError(
            f"Unknown profile keys in {name}: {sorted(unknown)}"
        )

    values = dict(mapping)
    values["allowed_regimes"] = tuple(values.get("allowed_regimes", ()))
    values["breakout_targets_r"] = tuple(
        float(x) for x in values.get("breakout_targets_r", (1.5, 2.0, 3.0))
    )
    values["blackout_windows"] = tuple(
        tuple(item) for item in values.get("blackout_windows", ())
    )

    return StrategyProfile(name=name, **values)


def load_profiles(
    path: str | Path | None = None,
) -> tuple[dict[str, StrategyProfile], str]:
    """Return (profiles by name, active profile name)."""

    import yaml

    file = Path(path or DEFAULT_PROFILE_FILE)

    if not file.is_file():
        return {"baseline": StrategyProfile(name="baseline")}, "baseline"

    document = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    raw = document.get("profiles") or {}

    profiles = {
        str(name): profile_from_mapping(str(name), mapping or {})
        for name, mapping in raw.items()
    }

    if "baseline" not in profiles:
        profiles["baseline"] = StrategyProfile(name="baseline")

    active = str(document.get("active", "baseline"))

    if active not in profiles:
        raise StrategyProfileError(f"Active profile {active} is not defined.")

    return profiles, active


def build_engine(profile: StrategyProfile | None):
    """Create the signal generator a profile asks for."""

    from trader.signal_engine.engine import TradingSignalEngine

    if profile is None:
        return None

    if profile.engine == "donchian":
        from trader.breakout_engine import DonchianBreakoutEngine

        return DonchianBreakoutEngine(
            timeframe=profile.execution_timeframe,
            channel=profile.breakout_channel,
            stop_atr=profile.breakout_stop_atr,
            targets_r=profile.breakout_targets_r,
        )

    return TradingSignalEngine(profile=profile)


def get_profile(
    name: str | None = None,
    path: str | Path | None = None,
) -> StrategyProfile:
    profiles, active = load_profiles(path)
    chosen = name or active

    if chosen not in profiles:
        raise StrategyProfileError(
            f"Unknown profile {chosen}. Available: {sorted(profiles)}"
        )

    return profiles[chosen]


def _ema_distance_in_atr(candles: pd.DataFrame, atr: float) -> float:
    ema = candles["close"].ewm(span=20, adjust=False).mean().iloc[-1]

    return abs(float(candles["close"].iloc[-1]) - float(ema)) / atr


def evaluate_filters(
    profile: StrategyProfile,
    *,
    analysis_time: datetime,
    candles_by_timeframe: dict[str, pd.DataFrame],
    overall_direction: str,
    timeframe_directions: dict[str, str],
    execution_candles: pd.DataFrame,
    atr: float,
) -> tuple[list[str], dict[str, Any]]:
    """Return (rejection reasons, diagnostics) for a candidate signal."""

    reasons: list[str] = []
    details: dict[str, Any] = {"profile": profile.name}

    if overall_direction not in ("BULLISH", "BEARISH"):
        return reasons, details

    for start, end in profile.blackout_windows:
        if _parse_time(start) <= analysis_time <= _parse_time(end):
            reasons.append(
                f"Inside a scheduled-event blackout ({start} to {end})."
            )

    if profile.higher_timeframe_bias is not None:
        bias = timeframe_directions.get(profile.higher_timeframe_bias)
        details["higher_timeframe_bias"] = bias

        if bias != overall_direction:
            reasons.append(
                f"{profile.higher_timeframe_bias} direction ({bias}) "
                "does not match the signal direction."
            )

    needs_regime = bool(
        profile.allowed_regimes
        or profile.require_regime_direction
        or profile.minimum_adx is not None
    )

    if needs_regime:
        from market_analysis.regime import analyze_market_regime

        try:
            regime = analyze_market_regime(
                candles_by_timeframe[profile.regime_timeframe],
                analysis_time=analysis_time,
            )
        except Exception as error:  # fail closed: no regime, no signal
            reasons.append(f"Regime analysis unavailable: {error}")
            regime = None

        if regime is not None:
            details["regime"] = regime["regime"]
            details["regime_direction"] = regime["direction"]
            details["adx"] = regime["trend"]["adx"]

            if (
                profile.allowed_regimes
                and regime["regime"] not in profile.allowed_regimes
            ):
                reasons.append(
                    f"Market regime {regime['regime']} is not allowed."
                )

            if (
                profile.require_regime_direction
                and regime["direction"] != overall_direction
            ):
                reasons.append(
                    "Trend (DI) direction does not match the signal."
                )

            if (
                profile.minimum_adx is not None
                and regime["trend"]["adx"] < profile.minimum_adx
            ):
                reasons.append(
                    f"ADX {regime['trend']['adx']:.1f} is below "
                    f"{profile.minimum_adx}."
                )

    if profile.maximum_pullback_atr is not None:
        distance = _ema_distance_in_atr(execution_candles, atr)
        details["ema20_distance_atr"] = round(distance, 3)

        if distance > profile.maximum_pullback_atr:
            reasons.append(
                f"Price is {distance:.2f} ATR from EMA20; "
                "waiting for a pullback."
            )

    return reasons, details
