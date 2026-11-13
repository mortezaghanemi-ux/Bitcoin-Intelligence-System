# Trader Candle Data

This module manages Bitcoin OHLCV candle data
for trading analysis.

Supported timeframes:
- 5m
- 15m
- 1h
- 4h

Required candle fields:
- Open time
- Close time
- Open price
- High price
- Low price
- Close price
- Volume
- Source

Rules:
- Use UTC timestamps.
- Store only valid candle records.
- Detect missing candles.
- Prevent duplicate records.
- Verify OHLC consistency.
- Analyze closed candles only.
- Preserve historical candle data.
- Never modify original historical records.