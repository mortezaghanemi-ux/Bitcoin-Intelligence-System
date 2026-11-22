# Bitcoin Intelligence System

An independent Bitcoin intelligence platform combining:

1. Bitcoin 7-Day Probabilistic Forecast
2. Bitcoin Trading Signal Analysis
3. Market Data Collection
4. Technical and Market Structure Analysis
5. Historical Backtesting
6. Performance Evaluation
7. Interactive Dashboard

## Main Principles

- Data quality is mandatory.
- All timestamps use UTC.
- Historical records are preserved.
- No look-ahead bias.
- Forecast and Trader engines operate independently.
- Trading signals are for manual review only.
- No automatic trading or order execution.
- No exchange account connection is required.

## Project Status

Development phase: Initial Architecture

## Commands

- `python main.py collect` collects and validates candles.
- `python main.py trader` generates a manual-review signal. It shows
  the measured historical win rate for that kind of signal.
- `python main.py import-history FILE.csv` imports Kraken history.
- `python main.py backtest` measures past win rate (see
  `backtesting/README.md`).
- `python main.py dashboard` starts the read-only dashboard.

Win rates shown by the system are measured on past data by simulation.
They are not predictions.

## Technology

- Python
- Pandas
- NumPy
- GitHub Actions
- HTML / CSS / JavaScript

## Main Modules

- `config/`
- `collectors/`
- `validation/`
- `indicators/`
- `market_analysis/`
- `forecast/`
- `trader/`
- `backtesting/`
- `dashboard/`
- `reports/`
- `tests/`
## Automatic scheduling and persistent data

`.github/workflows/main-runtime.yml` runs the scheduler every 5 minutes
(`schedule`) and on demand (`workflow_dispatch`). GitHub runners are
ephemeral, so each run restores `data/{raw,processed,forecast,trader,backtests}`
from the `runtime-data` branch, runs one scheduler cycle, and commits any
changes back to that branch (even if the cycle failed). The scheduler's
history file is part of that data, so each task runs only when its own
interval in `config/schedules.yaml` is due.

Notes: GitHub may delay scheduled runs by several minutes, and disables
scheduled workflows after 60 days without repository activity (the data
commits count as activity). Nothing here places orders.

### Public dashboard

After every run the workflow publishes a read-only copy of the
dashboard to GitHub Pages (`dashboard/static_site.py`). One-time setup:
repository Settings → Pages → Source: **GitHub Actions**. The address is
`https://<owner>.github.io/<repo>/`. It shows the same data as the local
dashboard and never places orders.
