# Trader Data

This directory stores Bitcoin trading analysis data.

Data includes:
- OHLCV candles
- Technical analysis
- Candlestick patterns
- Market structure
- Order Blocks
- Fair Value Gaps
- Multi-timeframe analysis
- Trading signals
- Rejected signals
- Signal history

Supported timeframes:
- 5m
- 15m
- 1h
- 4h

Rules:
- Use validated market data only.
- Preserve historical records.
- Use UTC timestamps.
- Prevent duplicate records.
- Never use future candle information.
- Store rejected signals internally.
- No automatic trade execution.