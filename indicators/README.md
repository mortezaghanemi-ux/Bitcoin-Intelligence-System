# Technical Indicators

This module calculates technical indicators
from validated and standardized market data.

## Indicator Categories

### Trend Indicators
- EMA
- SMA
- ADX
- Ichimoku Cloud
- Parabolic SAR

### Momentum Indicators
- RSI
- MACD
- Stochastic
- CCI
- Williams %R
- ROC

### Volatility Indicators
- ATR
- Bollinger Bands

### Volume Indicators
- Volume SMA
- OBV
- MFI
- VWAP

## Calculation Rules

- Use validated input data only.
- Never use future candle information.
- Use closed candles for calculations.
- Preserve calculation timestamps in UTC.
- Keep calculations reproducible.
- Use centralized indicator configuration.
- Prevent duplicate calculation records.
- Preserve historical indicator values.
- Handle insufficient historical data safely.
- Never treat missing values as zero.

## Multi-Timeframe Support

Indicators must support:

- 5m
- 15m
- 1h
- 4h
- 1d

Each timeframe must be calculated independently
using correctly aligned candle data.

## Data Integrity

- Validate input candle sequences.
- Verify indicator value ranges.
- Prevent look-ahead bias.
- Record calculation errors.
- Preserve source candle references.
- Never modify original market data.

## Architecture

- Read only approved processed market data.
- Use centralized settings from config/indicators.yaml.
- Store calculated outputs in the processed data layer.
- Do not generate forecasts or trading signals.
- Keep indicator calculations independent
  from Forecast and Trader engines.

## Special Requirements

- Ichimoku calculations must prevent future data leakage.
- VWAP calculations must use correctly defined
  volume and price inputs.
- Indicator warm-up periods must be respected.
- Historical recalculations must be traceable.