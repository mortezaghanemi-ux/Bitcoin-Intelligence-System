# Trading Signal Engine

This module generates potential Bitcoin trading
signals for manual review.

Signal types:
- LONG
- SHORT
- NO TRADE

Signal requirements:
- Entry price range
- Stop loss
- Take profit targets
- Risk/reward ratio
- Confidence score
- Signal timeframe
- Signal timestamp
- Supporting analysis
- Data quality status

Validation rules:
- Require valid and fresh market data.
- Require closed candle confirmation.
- Validate multi-timeframe alignment.
- Validate market structure.
- Validate risk/reward conditions.
- Reject conflicting or incomplete signals.
- Never use future information.

Safety rules:
- No automatic trading.
- No exchange account connection.
- No order execution.
- No API keys for trading accounts.

History:
- Preserve all generated signals.
- Store rejected signals internally.
- Never modify historical signals.
- Use UTC timestamps.