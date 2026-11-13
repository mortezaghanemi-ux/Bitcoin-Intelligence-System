# Processed Data

This directory stores cleaned, standardized and validated
market data ready for analysis.

## Data Categories

- Validated Bitcoin prices
- Processed OHLCV candles
- Technical indicators
- Market structure data
- Derivatives analysis
- ETF flow analysis
- On-chain metrics
- Macroeconomic indicators

## Processing Rules

- Only validated data is allowed.
- Preserve historical records.
- Use UTC timestamps.
- Prevent duplicate records.
- Never overwrite original raw data.
- Record data processing timestamps.
- Preserve source and data provenance.
- Apply consistent data schemas.
- Reject invalid or stale input data.

## Data Integrity

- Every dataset must have a defined schema.
- Missing values must not be treated as zero.
- Processing must be reproducible.
- Reprocessing must not create duplicate records.
- Historical corrections must be traceable.
- Processing errors must be recorded.

## Ownership

The validation and processing layer is the
authorized writer to this directory.

Raw collectors must never modify processed data.
Forecast and Trader engines must only read
approved datasets.