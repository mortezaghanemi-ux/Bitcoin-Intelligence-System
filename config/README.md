# Configuration

This directory contains the central configuration files
for the Bitcoin Intelligence System.

## Configuration Files

- settings.yaml: Global system settings, security and execution rules.
- data_sources.yaml: Market data providers and source policies.
- data_quality.yaml: Data validation and freshness rules.
- timeframes.yaml: Official timeframe definitions.
- indicators.yaml: Technical indicator settings.
- forecast.yaml: Bitcoin 7-day forecast settings.
- trader.yaml: Manual trading signal engine settings.
- schedules.yaml: Collection and processing schedules.

## Core Rules

- All timestamps must use UTC.
- API keys must never be stored in configuration files.
- Only validated data can enter analysis engines.
- Historical records must not be overwritten.
- Forecast and Trader engines must remain independent.
- Automatic trading is permanently disabled unless explicitly redesigned.
- Configuration changes must be version-controlled.

## Configuration Ownership

Each configuration file has one specific responsibility.

Duplicate settings should be avoided.
When a setting is shared, its official source must be clearly defined.