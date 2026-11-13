# Order Block Analysis

This module identifies institutional order blocks
and Fair Value Gaps in Bitcoin markets.

Analysis components:
- Bullish Order Blocks
- Bearish Order Blocks
- Order Block Validation
- Order Block Mitigation
- Fair Value Gaps (FVG)
- Bullish FVG
- Bearish FVG
- FVG Mitigation
- Consequent Encroachment
- Supply and Demand Zones

Rules:
- Use closed candles only.
- Never use future information.
- Support multiple timeframes.
- Validate zones before generating signals.
- Record detection timestamps in UTC.
- Preserve historical analysis.
- Use validated market data only.