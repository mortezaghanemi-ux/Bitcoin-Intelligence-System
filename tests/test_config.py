"""Automated tests for the central configuration loader."""

import shutil
from pathlib import Path

import pytest
import yaml

from config.loader import (
    ConfigurationError,
    REQUIRED_FILES,
    _load_yaml,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CONFIG_DIR = PROJECT_ROOT / "config"


def copy_config(tmp_path):
    target = tmp_path / "config"

    shutil.copytree(
        SOURCE_CONFIG_DIR,
        target,
    )

    return target


def read_config(config_dir, filename):
    path = config_dir / filename

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def write_config(config_dir, filename, content):
    path = config_dir / filename

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            content,
            file,
            sort_keys=False,
        )


# --------------------------------------------------
# Basic configuration loading
# --------------------------------------------------

def test_loads_all_required_configuration_files():
    config = load_config()

    assert len(config) == len(REQUIRED_FILES)

    for filename in REQUIRED_FILES:
        name = filename.removesuffix(".yaml")
        assert name in config


def test_configuration_files_are_present():
    for filename in REQUIRED_FILES:
        assert (SOURCE_CONFIG_DIR / filename).is_file()


def test_loaded_configuration_uses_utc():
    config = load_config()

    assert config["settings"]["system"]["timezone"] == "UTC"
    assert config["data_sources"]["data_policy"]["timezone"] == "UTC"


def test_missing_configuration_directory(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(ConfigurationError):
        load_config(missing)


def test_missing_configuration_file(tmp_path):
    config_dir = copy_config(tmp_path)

    (config_dir / "forecast.yaml").unlink()

    with pytest.raises(
        ConfigurationError,
        match="Missing configuration files",
    ):
        load_config(config_dir)


# --------------------------------------------------
# YAML file validation
# --------------------------------------------------

def test_rejects_invalid_yaml(tmp_path):
    path = tmp_path / "invalid.yaml"

    path.write_text(
        "system: [invalid",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        _load_yaml(path)


def test_rejects_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        _load_yaml(path)


def test_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "list.yaml"

    path.write_text(
        "- item_one\n- item_two\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        _load_yaml(path)


def test_rejects_hard_coded_api_key(tmp_path):
    path = tmp_path / "secret.yaml"

    path.write_text(
        "api_key: my_secret_value\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="hard-coded credentials",
    ):
        _load_yaml(path)


def test_accepts_environment_variable_reference(tmp_path):
    path = tmp_path / "environment.yaml"

    path.write_text(
        "api_key: MY_API_KEY\n",
        encoding="utf-8",
    )

    result = _load_yaml(path)

    assert result["api_key"] == "MY_API_KEY"


# --------------------------------------------------
# Data safety rules
# --------------------------------------------------

def test_rejects_non_utc_data_policy(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "data_sources.yaml",
    )

    config["data_policy"]["timezone"] = "LOCAL"

    write_config(
        config_dir,
        "data_sources.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="data_policy.timezone",
    ):
        load_config(config_dir)


def test_raw_data_preservation_must_remain_enabled(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "data_sources.yaml",
    )

    config["data_policy"]["preserve_raw_data"] = False

    write_config(
        config_dir,
        "data_sources.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Raw data preservation",
    ):
        load_config(config_dir)


# --------------------------------------------------
# Trader safety rules
# --------------------------------------------------

def test_automatic_trading_must_remain_disabled(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "trader.yaml",
    )

    config["trader"]["execution"]["auto_trading"] = True

    write_config(
        config_dir,
        "trader.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Automatic trading",
    ):
        load_config(config_dir)


def test_exchange_connection_must_remain_disabled(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "trader.yaml",
    )

    config["trader"]["execution"]["exchange_connection"] = True

    write_config(
        config_dir,
        "trader.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Exchange connections",
    ):
        load_config(config_dir)


def test_manual_review_must_remain_enabled(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "trader.yaml",
    )

    config["trader"]["execution"]["manual_review_only"] = False

    write_config(
        config_dir,
        "trader.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="manual review",
    ):
        load_config(config_dir)


def test_duplicate_trader_timeframes_are_rejected(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "trader.yaml",
    )

    config["trader"]["timeframes"] = [
        "5m",
        "15m",
        "1h",
        "5m",
    ]

    write_config(
        config_dir,
        "trader.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Duplicate trader timeframes",
    ):
        load_config(config_dir)


def test_required_signal_types_must_exist(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "trader.yaml",
    )

    config["trader"]["signals"]["allowed"] = [
        "LONG",
        "SHORT",
    ]

    write_config(
        config_dir,
        "trader.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="LONG, SHORT and NO_TRADE",
    ):
        load_config(config_dir)


# --------------------------------------------------
# Forecast and data quality rules
# --------------------------------------------------

def test_forecast_horizon_must_be_seven_days(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "forecast.yaml",
    )

    config["forecast"]["horizon_days"] = 30

    write_config(
        config_dir,
        "forecast.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Forecast horizon",
    ):
        load_config(config_dir)


def test_stale_data_rejection_must_remain_enabled(tmp_path):
    config_dir = copy_config(tmp_path)

    config = read_config(
        config_dir,
        "data_quality.yaml",
    )

    config["freshness"]["reject_stale_data"] = False

    write_config(
        config_dir,
        "data_quality.yaml",
        config,
    )

    with pytest.raises(
        ConfigurationError,
        match="Stale data rejection",
    ):
        load_config(config_dir)


# --------------------------------------------------
# Configuration integrity
# --------------------------------------------------

def test_loading_does_not_modify_configuration_files(tmp_path):
    config_dir = copy_config(tmp_path)

    before = {
        filename: (config_dir / filename).read_bytes()
        for filename in REQUIRED_FILES
    }

    load_config(config_dir)

    after = {
        filename: (config_dir / filename).read_bytes()
        for filename in REQUIRED_FILES
    }

    assert before == after


def test_loaded_configuration_contains_all_expected_sections():
    config = load_config()

    expected = {
        "settings",
        "data_sources",
        "data_quality",
        "timeframes",
        "indicators",
        "forecast",
        "trader",
        "schedules",
    }

    assert set(config.keys()) == expected