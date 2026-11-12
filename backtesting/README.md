# Backtesting Engine

This module evaluates historical forecast and trading performance.

Forecast evaluation:
- Directional accuracy
- Probability calibration
- Brier score
- Log loss
- Forecast performance by horizon

Trading evaluation:
- Win rate
- Profit factor
- Maximum drawdown
- Risk/reward performance
- Expectancy
- Performance by timeframe

Rules:
- Prevent look-ahead bias.
- Use only information available at the historical decision time.
- Include trading fees and slippage.
- Preserve all backtest results.
- Never modify historical signals.
- Separate training and testing periods.