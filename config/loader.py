"""Central configuration loader."""

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


class ConfigurationError(Exception):
    pass


def _contains_secret_key(value: Any, location: str = "") -> list[str]:

    findings = []

    if isinstance(value, dict):

        for key, child in value.items():

            key_text = str(key)

            child_location = (
                f"{location}.{key_text}"
                if location else key_text
            )

            is_secret = (
                SECRET_PATTERN.search(key_text)
                and key_text.lower() not in {
                    "protect_api_keys",
                    "never_store_api_keys_in_files",
                    "exchange_api_keys_required",
                }
            )

            if is_secret:

                if not (
                    isinstance(child, str)
                    and child.isupper()
                    and re.fullmatch(r"[A-Z][A-Z0-9_]*", child)
                ):

                    findings.append(child_location)

            findings.extend(
                _contains_secret_key(child, child_location)
            )

    elif isinstance(value, list):

        for index, child in enumerate(value):

            findings.extend(
                _contains_secret_key(
                    child,
                    f"{location}[{index}]"
                )
            )

    return findings


def _load_yaml(path: Path) -> dict[str, Any]:

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
            f"Possible hard-coded credentials in {path.name}: "
            + ", ".join(secret_paths)
        )

    return content


def load_config(
    config_dir: str | Path = CONFIG_DIR
) -> dict[str, dict[str, Any]]:

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


def _validate_configuration(config):

    settings = config["settings"].get("system", {})

    if settings.get("timezone") != "UTC":

        raise ConfigurationError(
            "system.timezone must be UTC."
        )

    data_policy = config["data_sources"].get(
        "data_policy", {}
    )

    if data_policy.get("timezone") != "UTC":

        raise ConfigurationError(
            "data_policy.timezone must be UTC."
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

    trader = config["trader"].get("trader", {})

    execution = trader.get("execution", {})

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

    timeframes = trader.get("timeframes")

    if not isinstance(timeframes, list) or not timeframes:

        raise ConfigurationError(
            "Trader timeframes must be a non-empty list."
        )

    if len(timeframes) != len(set(timeframes)):

        raise ConfigurationError(
            "Duplicate trader timeframes detected."
        )

    allowed = trader.get("signals", {}).get(
        "allowed", []
    )

    required = {
        "LONG",
        "SHORT",
        "NO_TRADE"
    }

    if not required.issubset(set(allowed)):

        raise ConfigurationError(
            "LONG, SHORT and NO_TRADE must be enabled."
        )

    forecast = config["forecast"].get("forecast", {})

    if forecast.get("horizon_days") != 7:

        raise ConfigurationError(
            "Forecast horizon must be 7 days."
        )

    freshness = config["data_quality"].get(
        "freshness", {}
    )

    if freshness.get("reject_stale_data") is not True:

        raise ConfigurationError(
            "Stale data rejection must remain enabled."
        )


def main():

    try:

        configs = load_config()

    except ConfigurationError as exc:

        print(
            f"CONFIGURATION ERROR: {exc}",
            file=sys.stderr
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