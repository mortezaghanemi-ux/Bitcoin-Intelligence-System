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
## Measuring past performance (win rate)

Kraken's public OHLC API returns only the latest ~720 candles (about 2.5
days of 5m data). That is far too little for a trustworthy win rate, so
import Kraken's downloadable history first.

1. Download the **5-minute** OHLCVT CSV for your pair (for example
   `XBT_5.csv`) from Kraken's "Downloadable historical OHLCVT data".
2. Convert it (also derives 15m, 1h and 4h candles):

       python main.py import-history path/to/XBT_5.csv --pair XBTUSD

   Files go to `data/history/ohlcv/`. `data/raw/` is never touched.
   Gaps up to 1 hour are filled with flat zero-volume candles; a longer
   gap cuts the data and only the newest continuous part is kept.
3. Run the backtest (replays the real signal engine, no look-ahead):

       python main.py backtest --target 1 --step 12 --last-days 180

   - `--target 1|2|3` picks which take-profit counts as a win.
   - `--step 12` checks for a signal once per hour. Use `--step 1` for
     every 5 minutes (about 12x slower).
   - Each signal analysis takes roughly 0.5 to 1.5 seconds depending on
     the machine, so 6 months at step 12 takes about 40 to 90 minutes.
     Run it separately from the scheduler.

The report is saved to `data/backtests/performance_latest.json`, and a
copy is kept in `data/backtests/history/`. It contains the overall win
rate, the out-of-sample part (last 30% of the period), LONG vs SHORT,
and confidence buckets, each with a 95% interval and a sample-size flag.

`python main.py trader` and the dashboard then show the measured win
rate that matches each live signal. With fewer than 30 trades in a group
the number is marked `INSUFFICIENT_SAMPLE` and should not be trusted.

### How to read the numbers

- The win rate depends on the target. TP1 is 1R, TP2 is 1.5R and TP3 is
  2R, so a higher target always has a lower win rate. Compare
  `profit_factor` and `expectancy_per_trade`, not win rate alone.
- Results are a simulation on past data with fees and slippage. They are
  not a prediction of future trades.
- The simulation takes one trade at a time and checks the stop first when
  one candle touches both levels, which is the cautious choice.
- Sanity check: on a trendless random walk this backtest gives a win rate
  below 50%, as it should when there is no edge.

### Running it regularly

The scheduler runs one task at a time, so a long backtest would block
data collection. Run it from the operating system instead, for example a
weekly cron job:

    0 3 * * 0  cd /path/to/project && python main.py backtest --last-days 180
