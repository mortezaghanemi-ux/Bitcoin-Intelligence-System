# Data Architecture

This directory is the central data storage layer
of the Bitcoin Intelligence System.

## Directory Structure

- raw/: Original data collected from external providers.
- processed/: Validated and standardized market data.
- forecast/: Forecast outputs and evaluation history.
- trader/: Trading signals, position analysis and signal history.

## Data Lifecycle

1. Collect data from external providers.
2. Preserve original raw records.
3. Validate timestamps, values and data integrity.
4. Reject invalid or stale records.
5. Normalize and store validated data.
6. Provide approved datasets to analysis engines.
7. Preserve all generated outputs and history.

## Data Ownership

- raw/: Data collection layer.
- processed/: Validation and processing layer.
- forecast/: Forecast engine.
- trader/: Trader engine.

Every dataset must have one designated writer.

## Data Integrity Rules

- All timestamps must use UTC.
- Raw data must never be overwritten.
- Historical records must be preserved.
- Duplicate records must be prevented.
- Invalid data must never reach analysis engines.
- Forecast and Trader data must remain separate.
- API keys and secrets must never be stored here.
- Every dataset must have a defined schema.
- Data provenance must be traceable to its source.

## Reliability

- Failed collection must be recorded.
- Validation errors must be preserved.
- Missing data must not be treated as zero.
- Reprocessing must not create duplicate records.
- Historical corrections must be traceable.