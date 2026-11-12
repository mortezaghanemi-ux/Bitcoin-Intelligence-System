# Bitcoin Trader Engine

This module identifies potential Bitcoin trading opportunities
for manual review only.

Supported timeframes:
- 5m
- 15m
- 1h
- 4h

Analysis components:
- Candlestick patterns
- Technical indicators
- Market structure
- Support and resistance
- Order Blocks
- Fair Value Gaps
- Liquidity analysis
- Multi-timeframe confirmation
- Derivatives data

Signal outputs:
- LONG
- SHORT
- NO TRADE

Each candidate signal must include:
- Signal timestamp
- Timeframe
- Entry range
- Stop loss
- Take profit targets
- Risk/reward ratio
- Confidence score
- Supporting reasons

Rules:
- No automatic trading.
- No exchange account connection.
- No order execution.
- Reject stale or invalid data.
- Prevent look-ahead bias.
- Store rejected signals internally.
- Preserve signal history.
- Use UTC timestamps.