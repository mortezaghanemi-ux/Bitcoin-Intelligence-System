# Trader Data

This directory stores Bitcoin trading analysis data
and historical signal records.

## Data Categories

- OHLCV candles
- Technical analysis
- Candlestick patterns
- Market structure
- Support and resistance
- Order Blocks
- Fair Value Gaps
- Liquidity and liquidity sweeps
- Fibonacci analysis
- Multi-timeframe analysis
- Derivatives analysis
- Trading signals
- Rejected signals
- Signal history
- Position and risk calculations

## Supported Timeframes

- 5m
- 15m
- 1h
- 4h

## Signal Types

- LONG
- SHORT
- NO_TRADE

## Data Rules

- Use validated market data only.
- Preserve historical records.
- Use UTC timestamps.
- Prevent duplicate records.
- Never use future candle information.
- Use closed candles for signal generation.
- Prevent look-ahead bias.
- Store rejected signals internally.
- Preserve signal generation timestamps.
- Record reasons behind every signal.
- Preserve original entry, stop-loss and take-profit levels.
- Record confidence and risk-to-reward calculations.

## Risk Management

- Every actionable signal must include a stop-loss.
- Every actionable signal must include take-profit targets.
- Position size and risk calculations must be recorded.
- Maximum leverage must respect system configuration.
- Liquidation price calculations must be preserved.

## Execution Restrictions

- Manual review and approval only.
- No automatic trade execution.
- No exchange account connection.
- No exchange API keys.
- No order placement.

## Data Integrity

- Every dataset must have a defined schema.
- Missing values must not be treated as zero.
- Historical corrections must be traceable.
- Reprocessing must not create duplicate records.
- Rejected signals must remain available for internal analysis.
- Rejected signals must not appear in the normal dashboard.

## Ownership

The Trader engine is the authorized writer
to this directory.

The Forecast engine must not modify Trader data.