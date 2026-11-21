
"""Tests for market data source configuration and CoinGecko data behavior."""

from datetime import datetime, timezone
from pathlib import Path

import yaml

from collectors.coingecko import _parse_series


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCES_PATH = PROJECT_ROOT / "config" / "data_sources.yaml"


def load_data_sources_config():
    """Load and return the project's data source configuration."""
    with DATA_SOURCES_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert isinstance(config, dict)
    assert "sources" in config
    assert isinstance(config["sources"], dict)

    return config["sources"]


def test_kraken_is_primary_market_source():
    """Kraken must remain the primary market data source."""
    sources = load_data_sources_config()

    market_config = sources["market"]

    assert market_config["enabled"] is True
    assert market_config["primary"] == "kraken"


def test_coingecko_is_configured_as_market_fallback():
    """CoinGecko must be listed in the market fallback configuration."""
    sources = load_data_sources_config()

    market_config = sources["market"]
    fallback = market_config.get("fallback", [])

    assert isinstance(fallback, list)
    assert "coingecko" in fallback


def test_coingecko_provider_is_enabled():
    """CoinGecko provider must be enabled in the configuration."""
    sources = load_data_sources_config()

    providers = sources.get("providers", {})
    coingecko_config = providers.get("coingecko", {})

    assert coingecko_config.get("enabled") is True


def test_coingecko_parser_returns_metrics_not_ohlcv_candles():
    """CoinGecko market-chart observations must not be treated as OHLCV."""
    collection_time = datetime(
        2026, 1, 1, 1, 0, tzinfo=timezone.utc
    )

    payload = {
        "prices": [[1767225600000, 95000.0]],
        "market_caps": [[1767225600000, 1_900_000_000_000.0]],
        "total_volumes": [[1767225600000, 25_000_000_000.0]],
    }

    records = _parse_series(
        payload,
        "prices",
        collection_time,
    )

    assert len(records) == 1

    record = records[0]

    assert record["source"] == "coingecko"
    assert record["asset"] == "BTC"
    assert record["metric"] == "prices"
    assert record["value"] == 95000.0
    assert record["schema_version"] == 1

    forbidden_candle_fields = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_time",
        "close_time",
    }

    assert forbidden_candle_fields.isdisjoint(record.keys())


def test_kraken_ohlcv_collector_exists():
    """The primary Kraken OHLCV collector must remain available."""
    collector_path = (
        PROJECT_ROOT / "collectors" / "kraken_ohlcv.py"
    )

    assert collector_path.is_file()
