# Bitcoin Forecast Engine

This module generates probabilistic Bitcoin price forecasts.

Main outputs:
- 7-day bullish probability
- 7-day bearish probability
- 7-day neutral probability
- Market condition
- Forecast confidence
- Supporting factors
- Historical forecast records

Rules:
- Use validated data only.
- Probabilities must total 100%.
- Never use future information.
- Store every forecast with a UTC timestamp.
- Evaluate forecasts against actual market results.
- Keep historical forecasts immutable.