"""Load and validate FRED macroeconomic series configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import re

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "macro_series.yaml"
)

SERIES_ID_PATTERN = re.compile(
    r"^[A-Z0-9_]+$"
)


class MacroConfigurationError(Exception):
    """Raised when macroeconomic configuration is invalid."""


def _require_mapping(
    value: Any,
    name: str,
) -> dict:
    if not isinstance(value, dict):
        raise MacroConfigurationError(
            f"{name} must be a YAML mapping."
        )

    return value


def _require_bool(
    value: Any,
    name: str,
) -> None:
    if not isinstance(value, bool):
        raise MacroConfigurationError(
            f"{name} must be a boolean."
        )


def _validate_date(
    value: Any,
    name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise MacroConfigurationError(
            f"{name} must be YYYY-MM-DD or null."
        )

    try:
        parsed = date.fromisoformat(value)

    except ValueError as exc:
        raise MacroConfigurationError(
            f"{name} is not a valid date."
        ) from exc

    if parsed.isoformat() != value:
        raise MacroConfigurationError(
            f"{name} must use YYYY-MM-DD format."
        )

    return value


def _validate_series(
    series: Any,
) -> list[dict[str, Any]]:
    if not isinstance(series, list) or not series:
        raise MacroConfigurationError(
            "macro_series.series must be a non-empty list."
        )

    normalized = []
    seen_ids = set()

    for index, item in enumerate(series):

        location = (
            f"macro_series.series[{index}]"
        )

        item = _require_mapping(
            item,
            location,
        )

        series_id = item.get("id")

        if (
            not isinstance(series_id, str)
            or not SERIES_ID_PATTERN.fullmatch(series_id)
        ):
            raise MacroConfigurationError(
                f"{location}.id is invalid."
            )

        if series_id in seen_ids:
            raise MacroConfigurationError(
                f"Duplicate FRED series ID: {series_id}."
            )

        seen_ids.add(series_id)

        name = item.get("name")

        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise MacroConfigurationError(
                f"{location}.name must be a non-empty string."
            )

        description = item.get(
            "description",
            "",
        )

        if not isinstance(description, str):
            raise MacroConfigurationError(
                f"{location}.description must be a string."
            )

        enabled = item.get("enabled")

        _require_bool(
            enabled,
            f"{location}.enabled",
        )

        normalized.append(
            {
                "id": series_id,
                "name": name.strip(),
                "description": description.strip(),
                "enabled": enabled,
            }
        )

    if not any(
        item["enabled"]
        for item in normalized
    ):
        raise MacroConfigurationError(
            "At least one FRED series must be enabled."
        )

    return normalized


def _validate_collection(
    collection: Any,
) -> dict[str, Any]:
    collection = _require_mapping(
        collection,
        "macro_series.collection",
    )

    start = _validate_date(
        collection.get("observation_start"),
        "collection.observation_start",
    )

    end = _validate_date(
        collection.get("observation_end"),
        "collection.observation_end",
    )

    if (
        start is not None
        and end is not None
        and start > end
    ):
        raise MacroConfigurationError(
            "observation_start cannot be after observation_end."
        )

    boolean_fields = (
        "preserve_revisions",
        "preserve_missing_values",
        "prevent_duplicates",
        "append_only",
    )

    normalized = {
        "observation_start": start,
        "observation_end": end,
    }

    for field in boolean_fields:

        value = collection.get(field)

        _require_bool(
            value,
            f"collection.{field}",
        )

        normalized[field] = value

    if normalized["append_only"] is not True:
        raise MacroConfigurationError(
            "Macro raw data must remain append-only."
        )

    if normalized["prevent_duplicates"] is not True:
        raise MacroConfigurationError(
            "Macro duplicate prevention must remain enabled."
        )

    if normalized["preserve_missing_values"] is not True:
        raise MacroConfigurationError(
            "Missing macroeconomic values must be preserved."
        )

    if normalized["preserve_revisions"] is not True:
        raise MacroConfigurationError(
            "FRED historical revisions must be preserved."
        )

    return normalized


def _validate_rules(
    validation: Any,
) -> dict[str, Any]:
    validation = _require_mapping(
        validation,
        "macro_series.validation",
    )

    boolean_fields = (
        "require_series_id",
        "require_observation_date",
        "require_utc_collection_time",
        "reject_non_finite_values",
    )

    normalized = {}

    for field in boolean_fields:

        value = validation.get(field)

        _require_bool(
            value,
            f"validation.{field}",
        )

        normalized[field] = value

    required_rules = (
        "require_series_id",
        "require_observation_date",
        "require_utc_collection_time",
        "reject_non_finite_values",
    )

    for field in required_rules:

        if normalized[field] is not True:
            raise MacroConfigurationError(
                f"Safety rule {field} must remain enabled."
            )

    missing_policy = validation.get(
        "missing_value_policy"
    )

    if missing_policy != "preserve_as_null":
        raise MacroConfigurationError(
            "Missing value policy must be preserve_as_null."
        )

    normalized["missing_value_policy"] = missing_policy

    return normalized


def load_macro_series_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load and validate macroeconomic series configuration.

    Only enabled FRED series are returned in the series list.

    No API key is read or stored by this module.
    """

    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise MacroConfigurationError(
            "Invalid YAML in macro series configuration."
        ) from exc

    except OSError as exc:
        raise MacroConfigurationError(
            "Cannot read macro series configuration."
        ) from exc

    root = _require_mapping(
        raw_config,
        "configuration",
    )

    macro = _require_mapping(
        root.get("macro_series"),
        "macro_series",
    )

    enabled = macro.get("enabled")

    _require_bool(
        enabled,
        "macro_series.enabled",
    )

    if enabled is not True:
        raise MacroConfigurationError(
            "Macroeconomic data collection is disabled."
        )

    if macro.get("provider") != "fred":
        raise MacroConfigurationError(
            "Unsupported macroeconomic provider."
        )

    if macro.get("timezone") != "UTC":
        raise MacroConfigurationError(
            "Macro collection timezone must be UTC."
        )

    series = _validate_series(
        macro.get("series")
    )

    collection = _validate_collection(
        macro.get("collection")
    )

    validation = _validate_rules(
        macro.get("validation")
    )

    enabled_series = [
        item
        for item in series
        if item["enabled"]
    ]

    return {
        "enabled": True,
        "provider": "fred",
        "timezone": "UTC",
        "series": enabled_series,
        "collection": collection,
        "validation": validation,
    }


if __name__ == "__main__":
    try:
        config = load_macro_series_config()

    except MacroConfigurationError as exc:
        raise SystemExit(
            f"MACRO CONFIGURATION ERROR: {exc}"
        ) from exc

    print("MACRO CONFIGURATION OK")
    print(
        "Enabled FRED series:",
        len(config["series"]),
    )

    for item in config["series"]:
        print(
            f"  - {item['id']}: {item['name']}"
        )