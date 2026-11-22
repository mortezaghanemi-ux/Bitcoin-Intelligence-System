# Forecast Data

This directory stores Bitcoin forecast outputs
and historical forecast evaluation records.

## Data Categories

- Daily forecast results
- 7-day probability forecasts
- Forecast history
- Actual market outcomes
- Forecast accuracy metrics
- Forecast confidence scores
- Forecast evaluation reports

## Forecast Rules

- Preserve every historical forecast.
- Never modify previously published forecasts.
- Use UTC timestamps.
- Record forecast generation time.
- Store actual outcomes separately.
- Support historical performance evaluation.
- Prevent duplicate forecast records.
- Ensure probabilities total 100%.
- Use validated data only.
- Prevent look-ahead bias.

## Evaluation

- Compare forecasts with actual market outcomes.
- Evaluate forecasts after the forecast horizon completes.
- Preserve all evaluation results.
- Track directional accuracy.
- Calculate Brier Score and Log Loss.
- Never rewrite historical forecast results.

## Data Integrity

- Every dataset must have a defined schema.
- Missing values must not be treated as zero.
- Historical corrections must be traceable.
- Evaluation records must reference their original forecasts.
- All timestamps must use UTC.

## Ownership

The Forecast engine is the authorized writer
to this directory.

The Trader engine must not modify forecast data.
Actual market outcomes must remain separate
from published forecast records.