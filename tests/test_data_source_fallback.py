"""Tests for data source fallback configuration and current collector behavior."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kraken_is_primary_market_source():
    """Kraken must remain the primary market data source."""
    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    market_config = config["market"]

    assert market_config["primary"] == "kraken"


def test_coingecko_is_configured_as_fallback():
    """CoinGecko fallback configuration must be preserved."""
    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    market_config = config["market"]

    assert market_config.get("fallback") == "coingecko"


def test_coingecko_collector_does_not_claim_to_provide_ohlcv():
    """The current CoinGecko collector must not be treated as an OHLCV provider."""
    collector_path = PROJECT_ROOT / "collectors" / "coingecko.py"

    assert collector_path.exists()

    source = collector_path.read_text(encoding="utf-8")

    assert "market_chart" in source
    assert "ohlc" not in source.lower()


def test_kraken_collector_exists():
    """The primary Kraken OHLCV collector must remain available."""
    collector_path = PROJECT_ROOT / "collectors" / "kraken_ohlcv.py"

    assert collector_path.exists()