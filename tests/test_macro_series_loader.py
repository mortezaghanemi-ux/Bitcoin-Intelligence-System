"""Tests for the FRED macro series configuration loader."""

import shutil
from pathlib import Path

import pytest
import yaml

from config.macro_series_loader import (
    DEFAULT_CONFIG_PATH,
    MacroConfigurationError,
    load_macro_series_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config_file(tmp_path):
    """Create an isolated copy of the macro configuration."""

    destination = tmp_path / "macro_series.yaml"

    shutil.copyfile(
        DEFAULT_CONFIG_PATH,
        destination,
    )

    return destination


def read_config(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def write_config(path, content):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            content,
            file,
            sort_keys=False,
        )


def modify_config(path, callback):
    config = read_config(path)
    callback(config)
    write_config(path, config)


# --------------------------------------------------
# Basic loading
# --------------------------------------------------

def test_loads_default_macro_configuration():
    config = load_macro_series_config()

    assert config["enabled"] is True
    assert config["provider"] == "fred"
    assert config["timezone"] == "UTC"


def test_default_configuration_contains_expected_series():
    config = load_macro_series_config()

    series_ids = {
        item["id"]
        for item in config["series"]
    }

    assert series_ids == {
        "DGS10",
        "FEDFUNDS",
        "CPIAUCSL",
        "UNRATE",
        "WALCL",
        "DTWEXBGS",
    }


def test_only_enabled_series_are_returned(config_file):
    def update(config):
        config["macro_series"]["series"][0]["enabled"] = False

    modify_config(
        config_file,
        update,
    )

    config = load_macro_series_config(config_file)

    series_ids = {
        item["id"]
        for item in config["series"]
    }

    assert "DGS10" not in series_ids
    assert len(series_ids) == 5


def test_series_names_are_normalized(config_file):
    def update(config):
        config["macro_series"]["series"][0]["name"] = (
            "  treasury_yield  "
        )

    modify_config(
        config_file,
        update,
    )

    config = load_macro_series_config(config_file)

    assert (
        config["series"][0]["name"]
        == "treasury_yield"
    )


def test_collection_settings_are_returned():
    config = load_macro_series_config()

    assert config["collection"] == {
        "observation_start": None,
        "observation_end": None,
        "preserve_revisions": True,
        "preserve_missing_values": True,
        "prevent_duplicates": True,
        "append_only": True,
    }


# --------------------------------------------------
# File and YAML errors
# --------------------------------------------------

def test_missing_configuration_file(tmp_path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(
        MacroConfigurationError,
        match="Cannot read",
    ):
        load_macro_series_config(missing)


def test_invalid_yaml_file(tmp_path):
    path = tmp_path / "invalid.yaml"

    path.write_text(
        "macro_series: [invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        MacroConfigurationError,
        match="Invalid YAML",
    ):
        load_macro_series_config(path)


def test_empty_yaml_file(tmp_path):
    path = tmp_path / "empty.yaml"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        MacroConfigurationError,
        match="configuration must be a YAML mapping",
    ):
        load_macro_series_config(path)


def test_non_mapping_root_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"

    path.write_text(
        "- item_one\n- item_two\n",
        encoding="utf-8",
    )

    with pytest.raises(
        MacroConfigurationError,
    ):
        load_macro_series_config(path)


def test_missing_macro_section_is_rejected(tmp_path):
    path = tmp_path / "missing_section.yaml"

    path.write_text(
        "other_section: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        MacroConfigurationError,
        match="macro_series must be a YAML mapping",
    ):
        load_macro_series_config(path)


# --------------------------------------------------
# Provider and system settings
# --------------------------------------------------

def test_disabled_macro_collection_is_rejected(config_file):
    def update(config):
        config["macro_series"]["enabled"] = False

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="disabled",
    ):
        load_macro_series_config(config_file)


def test_non_boolean_enabled_flag_is_rejected(config_file):
    def update(config):
        config["macro_series"]["enabled"] = "true"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="must be a boolean",
    ):
        load_macro_series_config(config_file)


def test_unsupported_provider_is_rejected(config_file):
    def update(config):
        config["macro_series"]["provider"] = "other"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="Unsupported macroeconomic provider",
    ):
        load_macro_series_config(config_file)


def test_non_utc_timezone_is_rejected(config_file):
    def update(config):
        config["macro_series"]["timezone"] = "LOCAL"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="timezone must be UTC",
    ):
        load_macro_series_config(config_file)


# --------------------------------------------------
# Series validation
# --------------------------------------------------

@pytest.mark.parametrize(
    "invalid_id",
    [
        "",
        "lowercase",
        "INVALID-ID",
        "INVALID ID",
        "ABC/DEF",
        None,
        123,
    ],
)
def test_invalid_series_id_is_rejected(
    config_file,
    invalid_id,
):
    def update(config):
        config["macro_series"]["series"][0]["id"] = (
            invalid_id
        )

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="id is invalid",
    ):
        load_macro_series_config(config_file)


def test_duplicate_series_ids_are_rejected(config_file):
    def update(config):
        config["macro_series"]["series"][1]["id"] = (
            config["macro_series"]["series"][0]["id"]
        )

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="Duplicate FRED series ID",
    ):
        load_macro_series_config(config_file)


def test_empty_series_list_is_rejected(config_file):
    def update(config):
        config["macro_series"]["series"] = []

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="non-empty list",
    ):
        load_macro_series_config(config_file)


def test_non_list_series_is_rejected(config_file):
    def update(config):
        config["macro_series"]["series"] = {}

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
    ):
        load_macro_series_config(config_file)


def test_non_mapping_series_item_is_rejected(config_file):
    def update(config):
        config["macro_series"]["series"][0] = "DGS10"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="must be a YAML mapping",
    ):
        load_macro_series_config(config_file)


def test_empty_series_name_is_rejected(config_file):
    def update(config):
        config["macro_series"]["series"][0]["name"] = " "

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="name must be a non-empty string",
    ):
        load_macro_series_config(config_file)


def test_non_string_description_is_rejected(config_file):
    def update(config):
        config["macro_series"]["series"][0]["description"] = 123

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="description must be a string",
    ):
        load_macro_series_config(config_file)


def test_non_boolean_series_enabled_is_rejected(config_file):
    def update(config):
        config["macro_series"]["series"][0]["enabled"] = "yes"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="must be a boolean",
    ):
        load_macro_series_config(config_file)


def test_all_disabled_series_are_rejected(config_file):
    def update(config):
        for item in config["macro_series"]["series"]:
            item["enabled"] = False

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="At least one FRED series",
    ):
        load_macro_series_config(config_file)


# --------------------------------------------------
# Collection date validation
# --------------------------------------------------

@pytest.mark.parametrize(
    "invalid_date",
    [
        "",
        "2026-1-01",
        "2026/01/01",
        "not-a-date",
        20260101,
    ],
)
def test_invalid_observation_start_is_rejected(
    config_file,
    invalid_date,
):
    def update(config):
        config["macro_series"]["collection"][
            "observation_start"
        ] = invalid_date

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
    ):
        load_macro_series_config(config_file)


def test_start_date_after_end_date_is_rejected(config_file):
    def update(config):
        collection = config["macro_series"]["collection"]
        collection["observation_start"] = "2026-02-01"
        collection["observation_end"] = "2026-01-01"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="cannot be after",
    ):
        load_macro_series_config(config_file)


def test_equal_observation_dates_are_allowed(config_file):
    def update(config):
        collection = config["macro_series"]["collection"]
        collection["observation_start"] = "2026-01-01"
        collection["observation_end"] = "2026-01-01"

    modify_config(
        config_file,
        update,
    )

    config = load_macro_series_config(config_file)

    assert (
        config["collection"]["observation_start"]
        == "2026-01-01"
    )


# --------------------------------------------------
# Collection safety rules
# --------------------------------------------------

@pytest.mark.parametrize(
    "field",
    [
        "preserve_revisions",
        "preserve_missing_values",
        "prevent_duplicates",
        "append_only",
    ],
)
def test_collection_safety_flags_must_be_boolean(
    config_file,
    field,
):
    def update(config):
        config["macro_series"]["collection"][field] = "true"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="must be a boolean",
    ):
        load_macro_series_config(config_file)


@pytest.mark.parametrize(
    "field",
    [
        "preserve_revisions",
        "preserve_missing_values",
        "prevent_duplicates",
        "append_only",
    ],
)
def test_collection_safety_rules_cannot_be_disabled(
    config_file,
    field,
):
    def update(config):
        config["macro_series"]["collection"][field] = False

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
    ):
        load_macro_series_config(config_file)


# --------------------------------------------------
# Validation rules
# --------------------------------------------------

@pytest.mark.parametrize(
    "field",
    [
        "require_series_id",
        "require_observation_date",
        "require_utc_collection_time",
        "reject_non_finite_values",
    ],
)
def test_validation_flags_must_be_boolean(
    config_file,
    field,
):
    def update(config):
        config["macro_series"]["validation"][field] = "true"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="must be a boolean",
    ):
        load_macro_series_config(config_file)


@pytest.mark.parametrize(
    "field",
    [
        "require_series_id",
        "require_observation_date",
        "require_utc_collection_time",
        "reject_non_finite_values",
    ],
)
def test_required_validation_rules_cannot_be_disabled(
    config_file,
    field,
):
    def update(config):
        config["macro_series"]["validation"][field] = False

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="must remain enabled",
    ):
        load_macro_series_config(config_file)


def test_invalid_missing_value_policy_is_rejected(config_file):
    def update(config):
        config["macro_series"]["validation"][
            "missing_value_policy"
        ] = "replace_with_zero"

    modify_config(
        config_file,
        update,
    )

    with pytest.raises(
        MacroConfigurationError,
        match="preserve_as_null",
    ):
        load_macro_series_config(config_file)


# --------------------------------------------------
# File integrity
# --------------------------------------------------

def test_loading_does_not_modify_configuration(config_file):
    before = config_file.read_bytes()

    load_macro_series_config(config_file)

    after = config_file.read_bytes()

    assert before == after


def test_loader_does_not_return_api_credentials():
    config = load_macro_series_config()

    assert "api_key" not in config
    assert "api_secret" not in config
    assert "secret" not in config