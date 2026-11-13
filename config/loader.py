"""Central configuration loader and validator."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = (
    "settings.yaml",
    "data_sources.yaml",
    "data_quality.yaml",
    "timeframes.yaml",
    "indicators.yaml",
    "forecast.yaml",
    "trader.yaml",
    "schedules.yaml",
)

SECRET_PATTERN = re.compile(
    r"(api[_-]?key|secret|password|access[_-]?token|private[_-]?key)",
    re.IGNORECASE,
)

SECRET_ENVIRONMENT_KEYS = {
    "protect_api_keys",
    "never_store_api_keys_in_files",
    "exchange_api_keys_required",
}


class ConfigurationError(Exception):
    """Raised when project configuration is invalid."""


def _require_mapping(value: Any, name: str) -> dict:
    """Require a dictionary-like YAML mapping."""

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"{name} must be a YAML mapping."
        )

    return value


def _require_bool(value: Any, name: str) -> None:
    """Require an actual boolean value."""

    if not isinstance(value, bool):
        raise ConfigurationError(
            f"{name} must be a boolean."
        )


def _require_positive_number(value: Any, name: str) -> None:
    """Require a positive finite numeric value."""

    import math

    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise ConfigurationError(
            f"{name} must be numeric."
        )

    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(
            f"{name} must be greater than zero."
        )


def _contains_secret_key(
    value: Any,
    location: str = "",
) -> list[str]:
    """Detect possible hard-coded credentials."""

    findings = []

    if isinstance(value, dict):

        for key, child in value.items():

            key_text = str(key)

            child_location = (
                f"{location}.{key_text}"
                if location
                else key_text
            )

            is_secret = (
                SECRET_PATTERN.search(key_text)
                and key_text.lower()
                not in SECRET_ENVIRONMENT_KEYS
            )

            if is_secret:

                is_environment_reference = (
                    isinstance(child, str)
                    and bool(
                        re.fullmatch(
                            r"[A-Z][A-Z0-9_]*",
                            child,
                        )
                    )
                )

                if not is_environment_reference:
                    findings.append(child_location)

            findings.extend(
                _contains_secret_key(
                    child,
                    child_location,
                )
            )

    elif isinstance(value, list):

        for index, child in enumerate(value):

            findings.extend(
                _contains_secret_key(
                    child,
                    f"{location}[{index}]",
                )
            )

    return findings


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and perform basic security checks on a YAML file."""

    try:

        with path.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file)

    except yaml.YAMLError as exc:

        raise ConfigurationError(
            f"Invalid YAML in {path.name}: {exc}"
        ) from exc

    except OSError as exc:

        raise ConfigurationError(
            f"Cannot read {path.name}: {exc}"
        ) from exc

    if not isinstance(content, dict):

        raise ConfigurationError(
            f"{path.name} must contain a YAML mapping."
        )

    secret_paths = _contains_secret_key(content)

    if secret_paths:

        raise ConfigurationError(
            f"Possible hard-coded credentials in "
            f"{path.name}: "
            + ", ".join(secret_paths)
        )

    return content


def load_config(
    config_dir: str | Path = CONFIG_DIR,
) -> dict[str, dict[str, Any]]:
    """Load all required project configuration files."""

    directory = Path(config_dir).expanduser().resolve()

    if not directory.is_dir():

        raise ConfigurationError(
            f"Configuration directory not found: {directory}"
        )

    missing = [
        name
        for name in REQUIRED_FILES
        if not (directory / name).is_file()
    ]

    if missing:

        raise ConfigurationError(
            "Missing configuration files: "
            + ", ".join(missing)
        )

    loaded = {}

    for filename in REQUIRED_FILES:

        name = filename.removesuffix(".yaml")

        loaded[name] = _load_yaml(
            directory / filename
        )

    _validate_configuration(loaded)

    return loaded


def _validate_configuration(
    config: dict[str, dict[str, Any]],
) -> None:
    """Validate configuration structure and safety rules."""

    # --------------------------------------------------
    # General settings
    # --------------------------------------------------

    settings = _require_mapping(
        config["settings"].get("system"),
        "settings.system",
    )

    if settings.get("timezone") != "UTC":

        raise ConfigurationError(
            "system.timezone must be UTC."
        )

    logging_config = _require_mapping(
        settings.get("logging"),
        "settings.system.logging",
    )

    _require_bool(
        logging_config.get("enabled"),
        "logging.enabled",
    )

    allowed_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if logging_config.get("level") not in allowed_log_levels:

        raise ConfigurationError(
            "Invalid logging level."
        )

    security = _require_mapping(
        settings.get("security"),
        "settings.system.security",
    )

    _require_bool(
        security.get("require_validated_data"),
        "security.require_validated_data",
    )

    _require_bool(
        security.get("protect_api_keys"),
        "security.protect_api_keys",
    )

    # --------------------------------------------------
    # Data source policies
    # --------------------------------------------------

    sources_config = config["data_sources"]

    data_policy = _require_mapping(
        sources_config.get("data_policy"),
        "data_sources.data_policy",
    )

    if data_policy.get("timezone") != "UTC":

        raise ConfigurationError(
            "data_policy.timezone must be UTC."
        )

    for key in (
        "preserve_raw_data",
        "never_store_api_keys_in_files",
        "validate_before_processing",
        "prevent_duplicate_records",
        "reject_stale_data",
    ):

        if key in data_policy:

            _require_bool(
                data_policy[key],
                f"data_policy.{key}",
            )

    if data_policy.get("preserve_raw_data") is not True:

        raise ConfigurationError(
            "Raw data preservation must remain enabled."
        )

    if data_policy.get(
        "never_store_api_keys_in_files"
    ) is not True:

        raise ConfigurationError(
            "API keys must never be stored in project files."
        )

    # --------------------------------------------------
    # Data quality
    # --------------------------------------------------

    quality = config["data_quality"]

    freshness = _require_mapping(
        quality.get("freshness"),
        "data_quality.freshness",
    )

    if freshness.get("reject_stale_data") is not True:

        raise ConfigurationError(
            "Stale data rejection must remain enabled."
        )

    max_age = _require_mapping(
        freshness.get("max_age_minutes"),
        "data_quality.freshness.max_age_minutes",
    )

    for source, minutes in max_age.items():

        _require_positive_number(
            minutes,
            f"max_age_minutes.{source}",
        )

    # --------------------------------------------------
    # Timeframes
    # --------------------------------------------------

    timeframe_config = config["timeframes"]

    if timeframe_config.get("timezone") != "UTC":

        raise ConfigurationError(
            "timeframes.timezone must be UTC."
        )

    timeframe_trader = _require_mapping(
        timeframe_config.get("trader"),
        "timeframes.trader",
    )

    timeframe_items = timeframe_trader.get("timeframes")

    if not isinstance(timeframe_items, list) or not timeframe_items:

        raise ConfigurationError(
            "timeframes.trader.timeframes must be a non-empty list."
        )

    timeframe_names = []

    for index, item in enumerate(timeframe_items):

        item = _require_mapping(
            item,
            f"timeframes.trader.timeframes[{index}]",
        )

        name = item.get("name")

        if not isinstance(name, str) or not name.strip():

            raise ConfigurationError(
                "Every timeframe must have a valid name."
            )

        if name in timeframe_names:

            raise ConfigurationError(
                f"Duplicate timeframe: {name}"
            )

        timeframe_names.append(name)

        _require_positive_number(
            item.get("minutes"),
            f"timeframes.{name}.minutes",
        )

        _require_positive_number(
            item.get("seconds"),
            f"timeframes.{name}.seconds",
        )

    # --------------------------------------------------
    # Indicators
    # --------------------------------------------------

    indicators = _require_mapping(
        config["indicators"].get("indicators"),
        "indicators.indicators",
    )

    _require_bool(
        indicators.get("enabled"),
        "indicators.enabled",
    )

    indicator_rules = _require_mapping(
        indicators.get("rules"),
        "indicators.rules",
    )

    for key in (
        "use_closed_candles_only",
        "calculate_for_all_timeframes",
        "preserve_indicator_history",
        "prevent_lookahead_bias",
        "use_validated_data_only",
    ):

        _require_bool(
            indicator_rules.get(key),
            f"indicators.rules.{key}",
        )

    # --------------------------------------------------
    # Forecast
    # --------------------------------------------------

    forecast = _require_mapping(
        config["forecast"].get("forecast"),
        "forecast.forecast",
    )

    if forecast.get("horizon_days") != 7:

        raise ConfigurationError(
            "Forecast horizon must be 7 days."
        )

    forecast_rules = _require_mapping(
        forecast.get("rules"),
        "forecast.rules",
    )

    for key in (
        "probabilities_must_total_100",
        "use_validated_data_only",
        "prevent_lookahead_bias",
        "use_closed_candles_only",
        "preserve_historical_forecasts",
    ):

        _require_bool(
            forecast_rules.get(key),
            f"forecast.rules.{key}",
        )

    # --------------------------------------------------
    # Trader safety
    # --------------------------------------------------

    trader = _require_mapping(
        config["trader"].get("trader"),
        "trader.trader",
    )

    execution = _require_mapping(
        trader.get("execution"),
        "trader.execution",
    )

    if execution.get("auto_trading") is not False:

        raise ConfigurationError(
            "Automatic trading must remain disabled."
        )

    if execution.get("exchange_connection") is not False:

        raise ConfigurationError(
            "Exchange connections must remain disabled."
        )

    if execution.get("manual_review_only") is not True:

        raise ConfigurationError(
            "Signals must require manual review."
        )

    if execution.get(
        "exchange_api_keys_required"
    ) is not False:

        raise ConfigurationError(
            "Exchange API keys must not be required."
        )

    trader_timeframes = trader.get("timeframes")

    if (
        not isinstance(trader_timeframes, list)
        or not trader_timeframes
    ):

        raise ConfigurationError(
            "Trader timeframes must be a non-empty list."
        )

    if len(trader_timeframes) != len(set(trader_timeframes)):

        raise ConfigurationError(
            "Duplicate trader timeframes detected."
        )

    if set(trader_timeframes) != set(timeframe_names):

        raise ConfigurationError(
            "Trader timeframes do not match timeframe configuration."
        )

    signals = _require_mapping(
        trader.get("signals"),
        "trader.signals",
    )

    allowed_signals = signals.get("allowed")

    if not isinstance(allowed_signals, list):

        raise ConfigurationError(
            "Trader allowed signals must be a list."
        )

    required_signals = {
        "LONG",
        "SHORT",
        "NO_TRADE",
    }

    if not required_signals.issubset(set(allowed_signals)):

        raise ConfigurationError(
            "LONG, SHORT and NO_TRADE must be enabled."
        )

    position = _require_mapping(
        trader.get("position"),
        "trader.position",
    )

    for key in (
        "require_entry_range",
        "require_stop_loss",
        "require_take_profit",
        "multiple_take_profit_targets",
        "require_risk_reward",
    ):

        _require_bool(
            position.get(key),
            f"trader.position.{key}",
        )

    if position.get("require_stop_loss") is not True:

        raise ConfigurationError(
            "Every trading signal must require a stop loss."
        )

    _require_positive_number(
        position.get("minimum_risk_reward"),
        "trader.position.minimum_risk_reward",
    )

    risk = _require_mapping(
        trader.get("risk"),
        "trader.risk",
    )

    _require_positive_number(
        risk.get("maximum_leverage"),
        "trader.risk.maximum_leverage",
    )

    # --------------------------------------------------
    # Scheduling
    # --------------------------------------------------

    schedules = _require_mapping(
        config["schedules"].get("schedules"),
        "schedules.schedules",
    )

    if schedules.get("timezone") != "UTC":

        raise ConfigurationError(
            "schedules.timezone must be UTC."
        )

    schedule_execution = _require_mapping(
        schedules.get("execution"),
        "schedules.execution",
    )

    for key in (
        "prevent_overlapping_runs",
        "sequential_dependencies",
        "stop_on_critical_error",
    ):

        _require_bool(
            schedule_execution.get(key),
            f"schedules.execution.{key}",
        )

    retries = schedule_execution.get("maximum_retries")

    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries < 0
    ):

        raise ConfigurationError(
            "maximum_retries must be a non-negative integer."
        )


def main() -> int:
    """Validate configuration from the command line."""

    try:

        configs = load_config()

    except ConfigurationError as exc:

        print(
            f"CONFIGURATION ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    print("CONFIGURATION OK")
    print(f"Project: {PROJECT_ROOT.name}")
    print(f"Loaded files: {len(configs)}")

    for name in configs:

        print(f"  - {name}.yaml")

    print("Safety checks: PASSED")

    return 0


if __name__ == "__main__":

    raise SystemExit(main())