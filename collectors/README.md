# Data Collectors

This directory contains independent data collectors
responsible for retrieving external market data.

## Responsibilities

- Collect Bitcoin market data.
- Collect OHLCV candles.
- Collect derivatives data.
- Collect ETF flow data.
- Collect macroeconomic data.
- Collect on-chain data.

## Collection Rules

- All timestamps must use UTC.
- Raw data must be preserved.
- Data sources must be validated.
- Collectors must not generate trading signals.
- Collectors must not generate forecasts.
- Each collector must have a clearly defined responsibility.
- Each dataset must have one designated writer.
- Collection timestamps must be recorded.
- Original provider timestamps must be preserved.
- Duplicate records must be prevented.
- Collection failures must be recorded.
- API keys must never be stored in source code.
- Missing data must not be treated as zero.

## Data Integrity

- Preserve original provider responses whenever practical.
- Do not overwrite historical raw records.
- Validate response structure before storage.
- Handle provider errors and rate limits safely.
- Do not silently replace invalid data.
- Record data source and collection status.

## Architecture

- Collectors write only to the raw data layer.
- Validation and processing are handled separately.
- Forecast and Trader engines must not collect external data directly.
- Collectors must operate independently.
- One collector failure must not corrupt unrelated datasets.

## Security

- Use environment variables for API credentials.
- Never expose API keys in logs.
- Never commit secrets to GitHub.

## Execution

- Follow centralized scheduling configuration.
- Prevent overlapping collection runs.
- Support safe retries after temporary failures.
- Record execution timestamps and status.