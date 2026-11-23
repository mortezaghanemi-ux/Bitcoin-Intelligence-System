"""Import historical Kraken 5-minute candles for backtesting.

Kraken's public OHLC API only returns the latest ~720 candles, which is
about 2.5 days of 5m data. That is far too little for a meaningful
win-rate measurement. Kraken publishes full history as downloadable
OHLCVT CSV files; this module converts such a 5m file into the JSONL
format the backtesting loader reads, and derives 15m, 1h and 4h candles
from it so all timeframes are consistent.

Output goes to data/history/ohlcv/ and never touches data/raw/.

CSV columns (no header): timestamp(epoch seconds), open, high, low,
close, volume, trades.
"""

from __future__ import annotations

import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HISTORY_DIR = PROJECT_ROOT / "data" / "history" / "ohlcv"

SOURCE_NAME = "kraken_history"

BASE_MINUTES = 5

DERIVED_TIMEFRAMES = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

MAXIMUM_FILLED_GAP_CANDLES = 12


class HistoryImportError(RuntimeError):
    """Raised when historical data cannot be imported."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HistoryImportError(f"CSV file not found: {path}")

    try:
        frame = pd.read_csv(
            path,
            header=None,
            usecols=range(6),
            names=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

    except Exception as exc:
        raise HistoryImportError(
            f"Cannot read CSV file: {exc}"
        ) from exc

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna()

    if frame.empty:
        raise HistoryImportError("CSV file has no valid rows.")

    frame["timestamp"] = frame["timestamp"].astype("int64")

    frame = (
        frame.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if (frame["timestamp"] % (BASE_MINUTES * 60)).any():
        raise HistoryImportError(
            "Timestamps are not aligned to 5-minute boundaries. "
            "Use the 5-minute Kraken CSV file."
        )

    return frame


def _fill_gaps(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill short gaps with flat zero-volume candles.

    Kraken omits candles when nothing traded. Short gaps are filled
    with the previous close. A gap longer than the limit splits the
    data and only the most recent continuous segment is kept.
    """

    step = BASE_MINUTES * 60
    stamps = frame["timestamp"].to_numpy()
    jumps = (stamps[1:] - stamps[:-1]) // step

    long_breaks = [
        index + 1
        for index, jump in enumerate(jumps)
        if jump - 1 > MAXIMUM_FILLED_GAP_CANDLES
    ]

    dropped_rows = 0

    if long_breaks:
        start = long_breaks[-1]
        dropped_rows = start
        frame = frame.iloc[start:].reset_index(drop=True)

    index = pd.RangeIndex(
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1] + step,
        step,
    )

    full = frame.set_index("timestamp").reindex(index)

    missing = int(full["close"].isna().sum())

    full["close"] = full["close"].ffill()

    for column in ("open", "high", "low"):
        full[column] = full[column].fillna(full["close"])

    full["volume"] = full["volume"].fillna(0.0)

    full.index.name = "timestamp"

    return full.reset_index(), {
        "filled_candles": missing,
        "long_gaps_dropped_rows": dropped_rows,
        "long_gap_count": len(long_breaks),
    }


def _resample(
    base: pd.DataFrame,
    minutes: int,
) -> pd.DataFrame:
    """Aggregate 5m candles, keeping only complete buckets."""

    frame = base.copy()
    frame["open_time"] = pd.to_datetime(
        frame["timestamp"], unit="s", utc=True
    )
    frame = frame.set_index("open_time")

    rule = f"{minutes}min"

    grouped = frame.resample(
        rule, origin="epoch", label="left", closed="left"
    )

    result = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    result["count"] = grouped["close"].count()

    needed = minutes // BASE_MINUTES

    return result[result["count"] == needed].drop(columns="count")


def _records(
    frame: pd.DataFrame,
    pair: str,
    timeframe: str,
    minutes: int,
) -> list[dict[str, Any]]:
    rows = []

    delta = pd.Timedelta(minutes=minutes)

    for open_time, row in frame.iterrows():
        rows.append(
            {
                "schema_version": 1,
                "source": SOURCE_NAME,
                "pair": pair,
                "timeframe": timeframe,
                "open_time": open_time.isoformat(),
                "close_time": (open_time + delta).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    return rows


def _write_dataset(
    base: pd.DataFrame,
    directory: Path,
    pair: str,
) -> dict[str, Any]:
    """Resample a continuous 5m frame and write all four timeframes."""

    outputs = {
        "5m": _resample(base, 5),
        "15m": _resample(base, 15),
        "1h": _resample(base, 60),
        "4h": _resample(base, 240),
    }

    # All timeframes must cover the same period with whole 4h blocks,
    # otherwise the multi-timeframe snapshots would disagree.
    common_start = outputs["4h"].index[0]
    common_end = outputs["4h"].index[-1] + pd.Timedelta(hours=4)

    directory.mkdir(parents=True, exist_ok=True)

    minutes_by_timeframe = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
    summary = {}

    for timeframe, data in outputs.items():
        minutes = minutes_by_timeframe[timeframe]
        data = data[
            (data.index >= common_start)
            & (data.index + pd.Timedelta(minutes=minutes) <= common_end)
        ]

        path = directory / f"kraken_{pair}_{timeframe}.jsonl"

        with path.open("w", encoding="utf-8") as file:
            for record in _records(data, pair, timeframe, minutes):
                file.write(json.dumps(record) + "\n")

        summary[timeframe] = len(data)

    return {
        "candles": summary,
        "period_start": common_start,
        "period_end": common_end,
    }


def import_kraken_history(
    csv_path: str | Path,
    *,
    pair: str = "XBTUSD",
    output_dir: str | Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    """Convert a Kraken 5m CSV into 5m/15m/1h/4h JSONL files."""

    if not isinstance(pair, str) or not pair.isalnum():
        raise HistoryImportError("Invalid trading pair.")

    pair = pair.upper()

    frame = _read_csv(Path(csv_path))

    if start is not None:
        frame = frame[frame["timestamp"] >= int(start.timestamp())]

    if end is not None:
        frame = frame[frame["timestamp"] < int(end.timestamp())]

    if len(frame) < 100:
        raise HistoryImportError(
            "Fewer than 100 candles remain; nothing useful to import."
        )

    frame = frame.reset_index(drop=True)

    base, gap_report = _fill_gaps(frame)

    directory = Path(output_dir or DEFAULT_HISTORY_DIR)
    written = _write_dataset(base, directory, pair)
    summary = written["candles"]
    common_start = written["period_start"]
    common_end = written["period_end"]

    return {
        "status": "IMPORTED",
        "pair": pair,
        "output_dir": str(directory),
        "candles": summary,
        "period_start": common_start.isoformat(),
        "period_end": common_end.isoformat(),
        "days": round((common_end - common_start).total_seconds() / 86400, 2),
        "gaps": gap_report,
        "source": SOURCE_NAME,
    }


def import_kraken_history_segments(
    csv_path: str | Path,
    *,
    pair: str = "XBTUSD",
    output_dir: str | Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    minimum_days: float = 14.0,
) -> dict[str, Any]:
    """Import every continuous stretch of history, not just the newest.

    Kraken's file has holes (outages, thin trading in early years). A
    hole longer than the fill limit cannot be bridged without inventing
    prices, so the data is split there. Each stretch of at least
    minimum_days becomes its own folder that can be backtested on its
    own; backtests never straddle a hole.
    """

    if not isinstance(pair, str) or not pair.isalnum():
        raise HistoryImportError("Invalid trading pair.")

    pair = pair.upper()
    frame = _read_csv(Path(csv_path))

    if start is not None:
        frame = frame[frame["timestamp"] >= int(start.timestamp())]

    if end is not None:
        frame = frame[frame["timestamp"] < int(end.timestamp())]

    frame = frame.reset_index(drop=True)

    if len(frame) < 100:
        raise HistoryImportError(
            "Fewer than 100 candles remain; nothing useful to import."
        )

    step = BASE_MINUTES * 60
    stamps = frame["timestamp"].to_numpy()
    jumps = (stamps[1:] - stamps[:-1]) // step

    cuts = [0]
    cuts += [
        index + 1
        for index, jump in enumerate(jumps)
        if jump - 1 > MAXIMUM_FILLED_GAP_CANDLES
    ]
    cuts.append(len(frame))

    root = Path(output_dir or DEFAULT_HISTORY_DIR.parent / "segments")
    root.mkdir(parents=True, exist_ok=True)

    segments = []
    skipped_short = 0
    filled_total = 0

    for begin, finish in zip(cuts[:-1], cuts[1:]):
        part = frame.iloc[begin:finish].reset_index(drop=True)
        span_days = (
            part["timestamp"].iloc[-1] - part["timestamp"].iloc[0]
        ) / 86400

        if len(part) < 100 or span_days < minimum_days:
            skipped_short += 1
            continue

        base, gaps = _fill_gaps(part)
        stamp = pd.to_datetime(
            part["timestamp"].iloc[0], unit="s", utc=True
        ).strftime("%Y%m%d")
        folder = root / f"segment_{stamp}"

        try:
            written = _write_dataset(base, folder, pair)
        except (ValueError, IndexError):
            skipped_short += 1
            continue

        filled_total += gaps["filled_candles"]
        days = (
            written["period_end"] - written["period_start"]
        ).total_seconds() / 86400

        segments.append(
            {
                "name": folder.name,
                "start": written["period_start"].isoformat(),
                "end": written["period_end"].isoformat(),
                "days": round(days, 2),
                "candles": written["candles"],
            }
        )

    if not segments:
        raise HistoryImportError(
            f"No continuous stretch of at least {minimum_days} days."
        )

    index = {
        "pair": pair,
        "source": SOURCE_NAME,
        "segments": segments,
    }
    (root / "segments.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    return {
        "status": "IMPORTED",
        "pair": pair,
        "output_dir": str(root),
        "segment_count": len(segments),
        "total_days": round(sum(item["days"] for item in segments), 2),
        "skipped_short_stretches": skipped_short,
        "filled_candles": filled_total,
        "first_start": segments[0]["start"],
        "last_end": segments[-1]["end"],
    }
