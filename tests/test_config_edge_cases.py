"""Edge-case tests for the central configuration loader."""

from pathlib import Path

import pytest

import config.loader as loader
from config.loader import (
    ConfigurationError,
    _contains_secret_key,
    _load_yaml,
    _require_bool,
    _require_mapping,
    _require_positive_number,
    load_config,
)


def test_require_mapping_rejects_non_mapping_values():
    for value in (None, [], "text", 123):
        with pytest.raises(ConfigurationError):
            _require_mapping(value, "test")


def test_require_mapping_accepts_dictionary():
    value = {"key": "value"}

    assert _require_mapping(value, "test") is value


def test_require_bool_rejects_non_boolean_values():
    for value in (None, 0, 1, "true", "false", []):
        with pytest.raises(ConfigurationError):
            _require_bool(value, "test")


def test_require_bool_accepts_boolean_values():
    assert _require_bool(True, "test") is None
    assert _require_bool(False, "test") is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "10",
        [],
        {},
    ],
)
def test_positive_number_rejects_non_numeric_values(value):
    with pytest.raises(ConfigurationError):
        _require_positive_number(value, "test")


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        0.0,
        -2.5,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_positive_number_rejects_invalid_numeric_values(value):
    with pytest.raises(ConfigurationError):
        _require_positive_number(value, "test")


@pytest.mark.parametrize("value", [1, 0.5, 100, 2.75])
def test_positive_number_accepts_valid_values(value):
    assert _require_positive_number(value, "test") is None


def test_secret_detection_finds_nested_credentials():
    data = {
        "settings": {
            "services": [
                {
                    "api_key": "hardcoded-value",
                }
            ]
        }
    }

    findings = _contains_secret_key(data)

    assert findings
    assert "settings.services[0].api_key" in findings


def test_secret_detection_finds_password():
    findings = _contains_secret_key(
        {"database": {"password": "my-password"}}
    )

    assert "database.password" in findings


def test_secret_detection_finds_private_key():
    findings = _contains_secret_key(
        {"private_key": "hardcoded"}
    )

    assert findings == ["private_key"]


def test_secret_detection_accepts_environment_reference():
    findings = _contains_secret_key(
        {"api_key": "MY_SERVICE_API_KEY"}
    )

    assert findings == []


def test_secret_detection_accepts_allowed_security_keys():
    data = {
        "protect_api_keys": True,
        "never_store_api_keys_in_files": True,
        "exchange_api_keys_required": False,
    }

    assert _contains_secret_key(data) == []


def test_secret_detection_handles_nested_lists():
    data = {
        "items": [
            {"token": "hardcoded-token"},
            {"name": "safe"},
        ]
    }

    findings = _contains_secret_key(data)

    assert "items[0].token" in findings


def test_secret_detection_returns_empty_for_safe_data():
    data = {
        "name": "Bitcoin",
        "enabled": True,
        "items": ["one", "two"],
    }

    assert _contains_secret_key(data) == []


def test_load_yaml_rejects_unreadable_file(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(
        ConfigurationError,
        match="Cannot read",
    ):
        _load_yaml(path)


def test_load_yaml_rejects_yaml_scalar(tmp_path):
    path = tmp_path / "scalar.yaml"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="must contain a YAML mapping",
    ):
        _load_yaml(path)


def test_load_yaml_rejects_hardcoded_nested_secret(tmp_path):
    path = tmp_path / "nested_secret.yaml"

    path.write_text(
        "services:\n"
        "  - name: example\n"
        "    access_token: hardcoded-value\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="hard-coded credentials",
    ):
        _load_yaml(path)


def test_load_config_rejects_file_instead_of_directory(tmp_path):
    path = tmp_path / "not_a_directory.yaml"
    path.write_text("test: true", encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="Configuration directory not found",
    ):
        load_config(path)


def test_load_config_reports_all_missing_files(tmp_path):
    config_dir = tmp_path / "empty_config"
    config_dir.mkdir()

    with pytest.raises(
        ConfigurationError,
        match="Missing configuration files",
    ) as error:
        load_config(config_dir)

    message = str(error.value)

    assert "settings.yaml" in message
    assert "trader.yaml" in message
    assert "schedules.yaml" in message


def test_load_config_accepts_path_object():
    config_dir = Path(loader.CONFIG_DIR)

    config = load_config(config_dir)

    assert "settings" in config
    assert "trader" in config


def test_main_returns_success_and_prints_status(monkeypatch, capsys):
    monkeypatch.setattr(
        loader,
        "load_config",
        lambda: {
            "settings": {},
            "trader": {},
        },
    )

    result = loader.main()
    captured = capsys.readouterr()

    assert result == 0
    assert "CONFIGURATION OK" in captured.out
    assert "Loaded files: 2" in captured.out
    assert "settings.yaml" in captured.out
    assert "trader.yaml" in captured.out
    assert "Safety checks: PASSED" in captured.out


def test_main_returns_error_when_configuration_is_invalid(
    monkeypatch,
    capsys,
):
    def raise_configuration_error():
        raise ConfigurationError("test configuration error")

    monkeypatch.setattr(
        loader,
        "load_config",
        raise_configuration_error,
    )

    result = loader.main()
    captured = capsys.readouterr()

    assert result == 1
    assert "CONFIGURATION ERROR" in captured.err
    assert "test configuration error" in captured.err


def test_validation_rejects_invalid_logging_level():
    config = load_config()
    config["settings"]["system"]["logging"]["level"] = "TRACE"

    with pytest.raises(
        ConfigurationError,
        match="Invalid logging level",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_non_utc_system_timezone():
    config = load_config()
    config["settings"]["system"]["timezone"] = "LOCAL"

    with pytest.raises(
        ConfigurationError,
        match="system.timezone",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_maximum_data_age():
    config = load_config()
    max_age = config["data_quality"]["freshness"]["max_age_minutes"]
    max_age["market"] = 0

    with pytest.raises(
        ConfigurationError,
        match="greater than zero",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_empty_timeframes():
    config = load_config()
    config["timeframes"]["trader"]["timeframes"] = []

    with pytest.raises(
        ConfigurationError,
        match="non-empty list",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_duplicate_timeframe_names():
    config = load_config()
    items = config["timeframes"]["trader"]["timeframes"]
    items.append(dict(items[0]))

    with pytest.raises(
        ConfigurationError,
        match="Duplicate timeframe",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_timeframe_name():
    config = load_config()
    config["timeframes"]["trader"]["timeframes"][0]["name"] = ""

    with pytest.raises(
        ConfigurationError,
        match="valid name",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_timeframe_item():
    config = load_config()
    config["timeframes"]["trader"]["timeframes"][0] = "5m"

    with pytest.raises(ConfigurationError):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_timeframe_minutes():
    config = load_config()
    config["timeframes"]["trader"]["timeframes"][0]["minutes"] = 0

    with pytest.raises(ConfigurationError):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_indicator_rules():
    config = load_config()
    config["indicators"]["indicators"]["rules"][
        "prevent_lookahead_bias"
    ] = "yes"

    with pytest.raises(ConfigurationError):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_forecast_rules():
    config = load_config()
    config["forecast"]["forecast"]["rules"][
        "use_closed_candles_only"
    ] = 1

    with pytest.raises(ConfigurationError):
        loader._validate_configuration(config)


def test_validation_rejects_exchange_api_keys_requirement():
    config = load_config()
    config["trader"]["trader"]["execution"][
        "exchange_api_keys_required"
    ] = True

    with pytest.raises(
        ConfigurationError,
        match="API keys must not be required",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_trader_timeframe_mismatch():
    config = load_config()
    config["trader"]["trader"]["timeframes"] = ["5m"]

    with pytest.raises(
        ConfigurationError,
        match="do not match",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_non_list_allowed_signals():
    config = load_config()
    config["trader"]["trader"]["signals"]["allowed"] = "LONG"

    with pytest.raises(
        ConfigurationError,
        match="must be a list",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_missing_stop_loss_requirement():
    config = load_config()
    config["trader"]["trader"]["position"]["require_stop_loss"] = False

    with pytest.raises(
        ConfigurationError,
        match="must require a stop loss",
    ):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_risk_reward():
    config = load_config()
    config["trader"]["trader"]["position"][
        "minimum_risk_reward"
    ] = 0

    with pytest.raises(ConfigurationError):
        loader._validate_configuration(config)


def test_validation_rejects_invalid_maximum_leverage():
    config = load_config()
    config["trader"]["trader"]["risk"]["maximum_leverage"] = -2

    with pytest.raises(ConfigurationError):
        loader._validate_configuration(config)


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        1.5,
        "3",
        None,
    ],
)
def test_validation_rejects_invalid_maximum_retries(value):
    config = load_config()
    config["schedules"]["schedules"]["execution"][
        "maximum_retries"
    ] = value

    with pytest.raises(
        ConfigurationError,
        match="non-negative integer",
    ):
        loader._validate_configuration(config)


def test_validation_accepts_zero_maximum_retries():
    config = load_config()
    config["schedules"]["schedules"]["execution"][
        "maximum_retries"
    ] = 0

    loader._validate_configuration(config)


def test_validation_rejects_non_utc_schedule_timezone():
    config = load_config()
    config["schedules"]["schedules"]["timezone"] = "LOCAL"

    with pytest.raises(
        ConfigurationError,
        match="schedules.timezone",
    ):
        loader._validate_configuration(config)