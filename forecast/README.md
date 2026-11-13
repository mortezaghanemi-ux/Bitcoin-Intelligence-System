# Bitcoin Forecast Engine

This module generates probabilistic Bitcoin market
forecasts for a 7-day prediction horizon.

## Main Outputs

- 7-day bullish probability
- 7-day bearish probability
- 7-day neutral probability
- Market condition
- Forecast confidence
- Supporting factors
- Risk factors
- Forecast generation timestamp
- Historical forecast records

## Analysis Inputs

- Validated Bitcoin market data
- OHLCV candles
- Technical indicators
- Market structure analysis
- Derivatives data
- ETF flows
- On-chain metrics
- Macroeconomic indicators
- Global market conditions

## Forecast Rules

- Use validated data only.
- Probabilities must total 100%.
- Never use future information.
- Store every forecast with a UTC timestamp.
- Use closed candles only.
- Prevent look-ahead bias.
- Preserve every historical forecast.
- Never modify previously published forecasts.
- Record the data sources used for each forecast.
- Record the model version used for each forecast.
- Prevent duplicate forecast records.
- Missing data must not be treated as zero.

## Probability Model

The engine must estimate three mutually exclusive
market scenarios:

- Bullish
- Bearish
- Neutral

All probabilities must be normalized
and their total must equal 100%.

Forecast probabilities represent estimates,
not guaranteed market outcomes.

## Forecast Evaluation

Evaluate forecasts against actual market results.

Evaluation metrics:

- Directional accuracy
- Brier Score
- Log Loss
- Probability calibration
- Forecast confidence performance

Evaluation must occur after the forecast horizon
has completed.

## Historical Records

- Preserve all generated forecasts.
- Store actual market outcomes separately.
- Preserve historical evaluation results.
- Never overwrite published forecasts.
- Keep forecast history available for backtesting.
- Make historical results traceable.

## Architecture

- Read only validated datasets.
- Use centralized configuration from config/forecast.yaml.
- Store outputs in the forecast data layer.
- Remain independent from the Trader engine.
- Do not generate trading execution orders.
- Do not connect to exchange trading accounts.

## Reliability

- Record forecast generation failures.
- Record missing or unavailable input sources.
- Do not generate forecasts from invalid critical data.
- Preserve model and configuration versions.
- Make forecast generation reproducible.