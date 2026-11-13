# Data Validation

This module validates all collected and processed data
before it becomes available to analysis engines.

## Validation Checks

- Missing values
- Invalid timestamps
- Non-UTC timestamps
- Duplicate records
- Stale data
- Invalid prices and volumes
- OHLC consistency
- Invalid candle data
- Unclosed candles
- Invalid indicator values
- Invalid derivatives data
- Invalid ETF data
- Invalid on-chain metrics
- Invalid macroeconomic data
- Source and schema consistency

## Validation Rules

- All timestamps must use UTC.
- Invalid data must be rejected.
- Validation results must be recorded.
- Dependent analysis must stop when required data is invalid.
- Missing data must never be treated as zero.
- Historical raw data must never be modified.
- Duplicate records must be prevented.
- Only validated data may enter analysis engines.
- Look-ahead bias must be prevented.
- Closed candles must be used for analysis and signals.

## Data Quality

- Check data freshness against source-specific limits.
- Verify numeric values and acceptable ranges.
- Verify OHLC relationships.
- Verify dataset schemas.
- Preserve validation history.
- Record validation errors and warnings.
- Track the quality status of each data source.

## Failure Handling

- Reject invalid records.
- Record validation failures.
- Stop dependent workflows when critical data is invalid.
- Never silently replace invalid values.
- Allow safe revalidation after data correction.

## Architecture

- Validation must run after data collection.
- Processing must use validated raw data only.
- Forecast and Trader engines must use approved datasets.
- Validation must remain independent of Forecast and Trader engines.
- Every dataset must have a defined validation policy.

## Validation Reports

Each validation report should include:

- Dataset name
- Data source
- Validation timestamp
- Records checked
- Valid records
- Rejected records
- Warnings
- Errors
- Overall validation status