
"""Bitcoin integrated market analysis engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from market_analysis.trend import analyze_trend
from market_analysis.structure import analyze_price_structure
from market_analysis.zones import analyze_supply_demand
from market_analysis.liquidity import analyze_liquidity
from market_analysis.regime import analyze_market_regime
from market_analysis.confluence import analyze_market_confluence


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

COMPONENTS = (
    "trend",
    "structure",
    "zones",
    "liquidity",
    "regime",
)

DEFAULT_COMPONENT_PARAMETERS = {
    "trend": {},
    "structure": {},
    "zones": {},
    "liquidity": {},
    "regime": {},
}

ALLOWED_COMPONENT_PARAMETERS = {
    "trend": {
        "fast_period",
        "slow_period",
        "adx_period",
        "minimum_candles",
    },
    "structure": {
        "swing_window",
        "minimum_candles",
    },
    "zones": {
        "atr_period",
        "impulse_window",
        "atr_multiplier",
        "minimum_candles",
    },
    "liquidity": {
        "swing_window",
        "minimum_touches",
        "level_tolerance",
        "minimum_candles",
    },
    "regime": {
        "adx_period",
        "atr_period",
        "volatility_lookback",
        "bollinger_period",
        "minimum_candles",
        "trending_threshold",
        "ranging_threshold",
        "high_volatility_ratio",
        "low_volatility_ratio",
    },
}


class MarketAnalysisEngineError(RuntimeError):
    """Raised when integrated market analysis fails."""

    def __init__(
        self,
        message: str,
        component: str = "engine",
    ) -> None:
        self.component = component
        super().__init__(
            f"[{component}] {message}"
        )


def _normalize_analysis_time(
    analysis_time: datetime | None,
) -> pd.Timestamp:
    """Validate and normalize the common UTC analysis time."""

    if analysis_time is None:
        return pd.Timestamp(
            datetime.now(timezone.utc)
        )

    try:
        timestamp = pd.Timestamp(
            analysis_time
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketAnalysisEngineError(
            "Invalid analysis_time."
        ) from exc

    if pd.isna(timestamp):
        raise MarketAnalysisEngineError(
            "analysis_time cannot be missing."
        )

    if timestamp.tzinfo is None:
        raise MarketAnalysisEngineError(
            "analysis_time must be timezone-aware UTC."
        )

    if timestamp.utcoffset().total_seconds() != 0:
        raise MarketAnalysisEngineError(
            "analysis_time must use UTC."
        )

    return timestamp.tz_convert("UTC")


def _validate_input(
    data: pd.DataFrame,
    analysis_time: pd.Timestamp,
) -> pd.DataFrame:
    """Validate and defensively copy input candles."""

    if not isinstance(data, pd.DataFrame):
        raise MarketAnalysisEngineError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise MarketAnalysisEngineError(
            "Cannot analyze an empty DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise MarketAnalysisEngineError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    candles = data.copy(deep=True)

    for column in ("open_time", "close_time"):
        try:
            timestamps = pd.to_datetime(
                candles[column],
                errors="raise",
            )

        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketAnalysisEngineError(
                f"Invalid timestamps in {column}."
            ) from exc

        if not isinstance(
            timestamps.dtype,
            pd.DatetimeTZDtype,
        ):
            raise MarketAnalysisEngineError(
                f"{column} must contain timezone-aware timestamps."
            )

        if str(timestamps.dt.tz) != "UTC":
            raise MarketAnalysisEngineError(
                f"{column} timestamps must use UTC."
            )

        if timestamps.isna().any():
            raise MarketAnalysisEngineError(
                f"{column} contains missing timestamps."
            )

        candles[column] = timestamps

    if (
        candles["open_time"]
        > candles["close_time"]
    ).any():
        raise MarketAnalysisEngineError(
            "A candle cannot close before it opens."
        )

    if (
        candles["close_time"]
        > analysis_time
    ).any():
        raise MarketAnalysisEngineError(
            "Future or unclosed candles are not allowed."
        )

    if not candles["open_time"].is_monotonic_increasing:
        raise MarketAnalysisEngineError(
            "Candles must be sorted by open_time."
        )

    if not candles["close_time"].is_monotonic_increasing:
        raise MarketAnalysisEngineError(
            "Candle close_time values must be sorted."
        )

    if candles["open_time"].duplicated().any():
        raise MarketAnalysisEngineError(
            "Duplicate open_time values detected."
        )

    if candles["close_time"].duplicated().any():
        raise MarketAnalysisEngineError(
            "Duplicate close_time values detected."
        )

    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    for column in numeric_columns:
        try:
            values = pd.to_numeric(
                candles[column],
                errors="raise",
            )

        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketAnalysisEngineError(
                f"{column} must contain numeric values."
            ) from exc

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise MarketAnalysisEngineError(
                f"{column} contains non-finite values."
            )

        candles[column] = values.astype(float)

    if (
        candles[list(numeric_columns)]
        .isna()
        .any()
        .any()
    ):
        raise MarketAnalysisEngineError(
            "Numeric candle columns cannot contain missing values."
        )

    if (
        candles[["open", "high", "low", "close"]]
        <= 0
    ).any().any():
        raise MarketAnalysisEngineError(
            "OHLC prices must be greater than zero."
        )

    if (candles["volume"] < 0).any():
        raise MarketAnalysisEngineError(
            "Volume cannot be negative."
        )

    invalid_ohlc = (
        (candles["high"] < candles["low"])
        | (candles["high"] < candles["open"])
        | (candles["high"] < candles["close"])
        | (candles["low"] > candles["open"])
        | (candles["low"] > candles["close"])
    )

    if invalid_ohlc.any():
        raise MarketAnalysisEngineError(
            "Invalid OHLC candle relationships detected."
        )

    valid_sources = candles["source"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )

    if not valid_sources.all():
        raise MarketAnalysisEngineError(
            "Every candle must have a valid source."
        )

    return candles.reset_index(drop=True)


def _validate_component_parameters(
    component_parameters: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Validate component parameter names before execution."""

    if component_parameters is None:
        return {
            name: {}
            for name in COMPONENTS
        }

    if not isinstance(component_parameters, dict):
        raise MarketAnalysisEngineError(
            "component_parameters must be a dictionary."
        )

    unknown_components = (
        set(component_parameters)
        - set(COMPONENTS)
    )

    if unknown_components:
        raise MarketAnalysisEngineError(
            "Unknown component parameter groups: "
            + ", ".join(sorted(unknown_components))
        )

    normalized = {
        name: {}
        for name in COMPONENTS
    }

    for component, parameters in component_parameters.items():

        if not isinstance(parameters, dict):
            raise MarketAnalysisEngineError(
                f"Parameters for '{component}' must be a dictionary."
            )

        unknown_parameters = (
            set(parameters)
            - ALLOWED_COMPONENT_PARAMETERS[component]
        )

        if unknown_parameters:
            raise MarketAnalysisEngineError(
                "Unknown parameters for "
                f"'{component}': "
                + ", ".join(sorted(unknown_parameters))
            )

        normalized[component] = parameters.copy()

    return normalized


def _validate_component_result(
    component: str,
    result: Any,
    analysis_timestamp: str,
) -> dict[str, Any]:
    """Validate the common metadata returned by a component."""

    if not isinstance(result, dict):
        raise MarketAnalysisEngineError(
            "Component must return a dictionary.",
            component,
        )

    if result.get("status") != "SUCCESS":
        raise MarketAnalysisEngineError(
            "Component did not return SUCCESS status.",
            component,
        )

    if result.get("analysis_timestamp") != analysis_timestamp:
        raise MarketAnalysisEngineError(
            "Component returned a mismatched analysis timestamp.",
            component,
        )

    required_metadata = (
        "source",
        "last_candle_close_time",
        "candles_analyzed",
    )

    for field in required_metadata:
        if field not in result:
            raise MarketAnalysisEngineError(
                f"Component result is missing '{field}'.",
                component,
            )

    sources = result["source"]

    if (
        not isinstance(sources, list)
        or not sources
        or not all(
            isinstance(source, str) and source.strip()
            for source in sources
        )
    ):
        raise MarketAnalysisEngineError(
            "Component returned invalid source metadata.",
            component,
        )

    candle_count = result["candles_analyzed"]

    if (
        isinstance(candle_count, bool)
        or not isinstance(candle_count, int)
        or candle_count <= 0
    ):
        raise MarketAnalysisEngineError(
            "Component returned an invalid candle count.",
            component,
        )

    try:
        close_time = pd.Timestamp(
            result["last_candle_close_time"]
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketAnalysisEngineError(
            "Component returned an invalid last candle close time.",
            component,
        ) from exc

    if close_time.tzinfo is None:
        raise MarketAnalysisEngineError(
            "Last candle close time must be timezone-aware.",
            component,
        )

    if close_time.utcoffset().total_seconds() != 0:
        raise MarketAnalysisEngineError(
            "Last candle close time must use UTC.",
            component,
        )

    return result


def analyze_market(
    data: pd.DataFrame,
    analysis_time: datetime | None = None,
    component_parameters: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    bullish_threshold: float = 20.0,
    bearish_threshold: float = 20.0,
) -> dict[str, Any]:
    """
    Run all market analysis components using one validated dataset.

    Components:
    - Trend analysis.
    - Price structure.
    - Supply and demand zones.
    - Liquidity analysis.
    - Market regime.
    - Confluence analysis.

    The engine:
    - Uses one shared UTC analysis timestamp.
    - Rejects invalid and unclosed candles.
    - Uses defensive DataFrame copies.
    - Validates component parameter names.
    - Requires consistent component metadata.
    - Does not persist data.
    - Does not generate trading signals.
    - Does not execute trades.
    """

    timestamp = _normalize_analysis_time(
        analysis_time
    )

    timestamp_string = timestamp.isoformat()

    candles = _validate_input(
        data,
        timestamp,
    )

    parameters = _validate_component_parameters(
        component_parameters
    )

    component_functions = {
        "trend": analyze_trend,
        "structure": analyze_price_structure,
        "zones": analyze_supply_demand,
        "liquidity": analyze_liquidity,
        "regime": analyze_market_regime,
    }

    analyses = {}

    for component in COMPONENTS:
        try:
            component_data = candles.copy(
                deep=True
            )

            result = component_functions[component](
                data=component_data,
                analysis_time=timestamp.to_pydatetime(),
                **parameters[component],
            )

            analyses[component] = (
                _validate_component_result(
                    component=component,
                    result=result,
                    analysis_timestamp=timestamp_string,
                )
            )

        except MarketAnalysisEngineError:
            raise

        except Exception as exc:
            raise MarketAnalysisEngineError(
                f"Analysis failed: {exc}",
                component,
            ) from exc

    reference_close_time = (
        analyses["trend"]["last_candle_close_time"]
    )

    reference_candle_count = (
        analyses["trend"]["candles_analyzed"]
    )

    for component in COMPONENTS[1:]:
        result = analyses[component]

        if (
            result["last_candle_close_time"]
            != reference_close_time
        ):
            raise MarketAnalysisEngineError(
                "Components did not analyze the same last candle.",
                component,
            )

        if (
            result["candles_analyzed"]
            != reference_candle_count
        ):
            raise MarketAnalysisEngineError(
                "Components did not analyze the same number of candles.",
                component,
            )

    try:
        confluence = analyze_market_confluence(
            trend_analysis=analyses["trend"],
            structure_analysis=analyses["structure"],
            zones_analysis=analyses["zones"],
            liquidity_analysis=analyses["liquidity"],
            regime_analysis=analyses["regime"],
            weights=weights,
            bullish_threshold=bullish_threshold,
            bearish_threshold=bearish_threshold,
            analysis_time=timestamp.to_pydatetime(),
        )

    except Exception as exc:
        raise MarketAnalysisEngineError(
            f"Confluence analysis failed: {exc}",
            "confluence",
        ) from exc

    if not isinstance(confluence, dict):
        raise MarketAnalysisEngineError(
            "Confluence must return a dictionary.",
            "confluence",
        )

    if confluence.get("status") != "SUCCESS":
        raise MarketAnalysisEngineError(
            "Confluence analysis did not return SUCCESS status.",
            "confluence",
        )

    if (
        confluence.get("analysis_timestamp")
        != timestamp_string
    ):
        raise MarketAnalysisEngineError(
            "Confluence returned a mismatched analysis timestamp.",
            "confluence",
        )

    sources = sorted(
        {
            source
            for result in analyses.values()
            for source in result["source"]
        }
    )

    return {
        "analysis_timestamp": timestamp_string,
        "status": "SUCCESS",
        "analyses": analyses,
        "confluence": confluence,
        "market_price": float(
            candles["close"].iloc[-1]
        ),
        "source": sources,
        "candles_analyzed": len(candles),
        "last_candle_close_time": reference_close_time,
        "parameters": {
            "components": parameters,
            "weights": (
                weights.copy()
                if weights is not None
                else None
            ),
            "bullish_threshold": bullish_threshold,
            "bearish_threshold": bearish_threshold,
        },
        "limitations": [
            "Confluence scores are heuristic, not probabilities.",
            "All components use the same validated candle dataset.",
            "Only closed candles are allowed.",
            "This engine does not generate trading signals.",
            "This engine does not execute trades.",
            "This engine does not persist analysis results.",
        ],
    }


if __name__ == "__main__":
    print(
        "Bitcoin integrated market analysis engine loaded."
    )