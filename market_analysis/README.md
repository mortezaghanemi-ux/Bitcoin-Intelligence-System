# Market Analysis

This module analyzes Bitcoin market conditions
and identifies market structure and price behavior.

## Analysis Components

### Trend Analysis
- Market trend detection
- Trend strength
- Trend continuation and reversal
- Moving average structure

### Price Structure
- Support and resistance
- Swing highs and lows
- Higher highs and higher lows
- Lower highs and lower lows
- Break of Structure (BOS)
- Change of Character (CHoCH)

### Supply and Demand
- Supply zones
- Demand zones
- Order Block references
- Fair Value Gap references

### Liquidity Analysis
- Liquidity levels
- Equal highs and lows
- Liquidity sweeps
- Stop-hunt detection

### Fibonacci Analysis
- Fibonacci retracement
- Fibonacci extension
- Key retracement levels

### Market Conditions
- Volatility analysis
- Market regime detection
- Trending and ranging conditions
- Momentum analysis
- Volume confirmation

## Multi-Timeframe Analysis

Support the following timeframes:

- 5m
- 15m
- 1h
- 4h
- 1d

Identify alignment and conflicts between timeframes.

## Analysis Rules

- Use validated market data only.
- Never use future information.
- Use closed candles for confirmed analysis.
- Support multi-timeframe analysis.
- Record analysis timestamps in UTC.
- Preserve historical analysis results.
- Prevent duplicate analysis records.
- Prevent look-ahead bias.
- Record the source candles used in each analysis.
- Distinguish confirmed structures from developing structures.

## Data Integrity

- Every analysis output must have a defined schema.
- Missing values must not be treated as zero.
- Historical results must remain traceable.
- Recalculations must not create duplicate records.
- Analysis errors must be recorded.

## Architecture

- Read validated data from the processed data layer.
- Use centralized configuration where applicable.
- Store analysis outputs in the processed data layer.
- Do not generate final trading signals.
- Do not execute trades.
- Provide analysis outputs independently to Forecast and Trader engines.