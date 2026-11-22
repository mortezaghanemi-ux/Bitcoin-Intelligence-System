# Raw Data

This directory stores original, unmodified data
collected from external providers.

## Data Categories

- Bitcoin market prices
- OHLCV candles
- Derivatives data
- ETF flows
- On-chain data
- Macroeconomic data

## Data Collection Rules

- Preserve original source data.
- Never overwrite historical records.
- Record collection timestamps in UTC.
- Record provider and source identifiers.
- Preserve original timestamps when available.
- Do not store API keys or secrets.
- Record collection failures.
- Prevent duplicate records.

## Data Integrity

- Raw data must remain unchanged after collection.
- Validation must happen before processing.
- Invalid records must be preserved for investigation.
- Missing values must not be replaced with zero.
- Every dataset must have a defined schema.

## Ownership

The collectors are the only authorized writers
to this directory.

Processing engines must never modify raw data.