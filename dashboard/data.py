
"""Read-only dashboard data access for Bitcoin Intelligence System."""

from __future__ import annotations

import json
import math

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from reports.engine import (
    DEFAULT_REPORT_DIRECTORY,
    ReportEngineError,
    list_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDLE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "market"
    / "ohlcv"
)

DEFAULT_SIGNAL_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "trader"
    / "signal_history.jsonl"
)

SUPPORTED_TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

DEFAULT_CANDLE_LIMIT = 100
DEFAULT_SIGNAL_LIMIT = 20

# Freshness limits are display warnings, not data validation rules.
DEFAULT_FRESHNESS_MINUTES = {
    "market_data": 20,
    "market": 30,
    "forecast": 480,
    "trading": 30,
    "data_quality": 60,
}


class DashboardDataError(RuntimeError):
    """Raised when dashboard data cannot be read safely."""


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _parse_utc_timestamp(
    value: Any,
    field: str = "timestamp",
) -> datetime:
    """Parse a timezone-aware UTC timestamp."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise DashboardDataError(
                f"Invalid {field}."
            ) from exc

    if not isinstance(value, datetime):
        raise DashboardDataError(
            f"{field} must be a datetime or ISO timestamp."
        )

    if value.tzinfo is None:
        raise DashboardDataError(
            f"{field} must be timezone-aware UTC."
        )

    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError) as exc:
        raise DashboardDataError(
            f"Invalid timezone in {field}."
        ) from exc

    if offset is None or offset != timedelta(0):
        raise DashboardDataError(
            f"{field} must use UTC."
        )

    return value.astimezone(timezone.utc)


def _validate_limit(
    value: Any,
    name: str,
    maximum: int = 1000,
) -> int:
    """Validate a positive result limit."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ValueError(
            f"{name} must be an integer between 1 and {maximum}."
        )

    return value


def _finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    """Validate a finite numeric value."""

    if isinstance(value, bool):
        raise DashboardDataError(
            f"{field} must be numeric."
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DashboardDataError(
            f"{field} must be numeric."
        ) from exc

    if not math.isfinite(number):
        raise DashboardDataError(
            f"{field} must be finite."
        )

    if minimum is not None and number < minimum:
        raise DashboardDataError(
            f"{field} is below its permitted minimum."
        )

    return number


def _read_jsonl(
    path: Path,
    *,
    maximum_records: int | None = None,
) -> list[dict[str, Any]]:
    """Read JSON Lines records without modifying the source file."""

    if not path.exists():
        return []

    if not path.is_file():
        raise DashboardDataError(
            "Dashboard data path is not a file."
        )

    records = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DashboardDataError(
                        f"Invalid JSONL record at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise DashboardDataError(
                        f"Invalid record structure at line {line_number}."
                    )

                records.append(record)

                if (
                    maximum_records is not None
                    and len(records) > maximum_records
                ):
                    records.pop(0)

    except DashboardDataError:
        raise

    except (OSError, UnicodeError) as exc:
        raise DashboardDataError(
            "Unable to read dashboard data."
        ) from exc

    return records


def _validate_candle(
    record: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Validate a stored candle before exposing it to the dashboard."""

    required = {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    }

    if not required.issubset(record):
        return None

    try:
        open_time = _parse_utc_timestamp(
            record["open_time"],
            "candle.open_time",
        )

        close_time = _parse_utc_timestamp(
            record["close_time"],
            "candle.close_time",
        )

        open_price = _finite_number(
            record["open"],
            "candle.open",
            minimum=0,
        )

        high = _finite_number(
            record["high"],
            "candle.high",
            minimum=0,
        )

        low = _finite_number(
            record["low"],
            "candle.low",
            minimum=0,
        )

        close = _finite_number(
            record["close"],
            "candle.close",
            minimum=0,
        )

        volume = _finite_number(
            record["volume"],
            "candle.volume",
            minimum=0,
        )

    except DashboardDataError:
        return None

    if close_time <= open_time:
        return None

    if close_time > now:
        return None

    if (
        high < max(open_price, close, low)
        or low > min(open_price, close, high)
    ):
        return None

    source = record.get("source")

    if not isinstance(source, str) or not source.strip():
        return None

    return {
        "open_time": open_time.isoformat(),
        "close_time": close_time.isoformat(),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "source": source.strip(),
    }


def load_market_candles(
    *,
    pair: str = "XBTUSD",
    timeframe: str = "5m",
    candle_directory: str | Path | None = None,
    limit: int = DEFAULT_CANDLE_LIMIT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Load recent closed candles from the raw Kraken JSONL history.

    Invalid, incomplete and unclosed records are never exposed.
    """

    if (
        not isinstance(pair, str)
        or not pair.strip()
        or not pair.isalnum()
    ):
        raise ValueError("Invalid trading pair.")

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}."
        )

    limit = _validate_limit(
        limit,
        "limit",
    )

    checked_at = (
        _parse_utc_timestamp(now, "now")
        if now is not None
        else _utc_now()
    )

    directory = Path(
        candle_directory
        if candle_directory is not None
        else DEFAULT_CANDLE_DIRECTORY
    )

    filename = (
        f"kraken_{pair.upper()}_{timeframe}.jsonl"
    )

    path = directory / filename

    records = _read_jsonl(path)

    valid_candles = []

    for record in records:

        if record.get("pair") != pair.upper():
            continue

        if record.get("timeframe") != timeframe:
            continue

        if record.get("source") != "kraken":
            continue

        candle = _validate_candle(
            record,
            now=checked_at,
        )

        if candle is not None:
            valid_candles.append(candle)

    # Keep the most recent version of each opening timestamp.
    unique = {}

    for candle in valid_candles:
        unique[candle["open_time"]] = candle

    candles = sorted(
        unique.values(),
        key=lambda item: item["open_time"],
    )

    candles = candles[-limit:]

    latest = (
        candles[-1]
        if candles
        else None
    )

    latest_close = (
        _parse_utc_timestamp(
            latest["close_time"],
            "latest.close_time",
        )
        if latest
        else None
    )

    age_minutes = (
        round(
            (checked_at - latest_close).total_seconds() / 60,
            2,
        )
        if latest_close
        else None
    )

    maximum_age = (
        SUPPORTED_TIMEFRAMES[timeframe]
        + DEFAULT_FRESHNESS_MINUTES["market_data"]
    )

    if not candles:
        status = "UNAVAILABLE"
    elif age_minutes > maximum_age:
        status = "STALE"
    else:
        status = "AVAILABLE"

    return {
        "status": status,
        "pair": pair.upper(),
        "timeframe": timeframe,
        "candles": candles,
        "latest_candle": latest,
        "records_returned": len(candles),
        "last_updated": (
            latest["close_time"]
            if latest
            else None
        ),
        "age_minutes": age_minutes,
        "source": "kraken" if latest else None,
    }


def _load_reports(
    report_directory: str | Path | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load standardized report history."""

    directory = (
        Path(report_directory)
        if report_directory is not None
        else DEFAULT_REPORT_DIRECTORY
    )

    try:
        reports = list_reports(
            output_directory=directory,
        )

    except ReportEngineError:
        return [], "Report history is unavailable or invalid."

    return reports, None


def _latest_report(
    reports: list[dict[str, Any]],
    report_type: str,
) -> dict[str, Any] | None:
    """Return the newest report of a requested type."""

    matches = [
        report
        for report in reports
        if report.get("report_type") == report_type
        and report.get("status") == "SUCCESS"
        and isinstance(report.get("payload"), dict)
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda item: (
            item.get("generated_at", ""),
            item.get("report_id", ""),
        ),
    )


def _report_section(
    report: dict[str, Any] | None,
    *,
    now: datetime,
    maximum_age_minutes: int,
) -> dict[str, Any]:
    """Create a dashboard section from a stored report."""

    if report is None:
        return {
            "status": "UNAVAILABLE",
            "generated_at": None,
            "age_minutes": None,
            "data": None,
        }

    try:
        generated_at = _parse_utc_timestamp(
            report["generated_at"],
            "report.generated_at",
        )

    except (KeyError, DashboardDataError):
        return {
            "status": "INVALID",
            "generated_at": None,
            "age_minutes": None,
            "data": None,
        }

    age_minutes = (
        now - generated_at
    ).total_seconds() / 60

    if age_minutes < 0:
        status = "INVALID"
    elif age_minutes > maximum_age_minutes:
        status = "STALE"
    else:
        status = "AVAILABLE"

    return {
        "status": status,
        "generated_at": generated_at.isoformat(),
        "age_minutes": round(age_minutes, 2),
        "data": report["payload"],
    }


def _is_dashboard_signal(
    signal: Any,
) -> bool:
    """Allow only actionable signals that require manual review."""

    if not isinstance(signal, dict):
        return False

    if signal.get("signal") not in {"LONG", "SHORT"}:
        return False

    if signal.get("status") != "REQUIRES_MANUAL_REVIEW":
        return False

    execution = signal.get("execution")

    if not isinstance(execution, dict):
        return False

    if (
        execution.get("manual_review_only") is not True
        or execution.get("auto_trading") is not False
        or execution.get("exchange_connection") is not False
        or execution.get("order_execution") is not False
    ):
        return False

    quality = signal.get("data_quality")

    if not isinstance(quality, dict):
        return False

    if (
        quality.get("status") != "VALIDATED"
        or quality.get("closed_candles_only") is not True
        or quality.get("future_data_used") is not False
    ):
        return False

    return True


def _active_profile_name() -> str | None:
    """Name of the strategy profile the live trader uses."""

    try:
        from trader.strategy_profile import load_profiles

        return load_profiles()[1]
    except Exception:
        return None


def load_trading_signals(
    *,
    history_path: str | Path | None = None,
    limit: int = DEFAULT_SIGNAL_LIMIT,
    now: datetime | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """
    Load actionable trading signals only.

    Rejected signals and NO_TRADE records are excluded. When
    profile_name is given, only signals issued by that strategy
    profile are kept (older signals came from rules that are no
    longer active).
    """

    limit = _validate_limit(
        limit,
        "limit",
        maximum=500,
    )

    checked_at = (
        _parse_utc_timestamp(now, "now")
        if now is not None
        else _utc_now()
    )

    path = Path(
        history_path
        if history_path is not None
        else DEFAULT_SIGNAL_HISTORY_PATH
    )

    records = _read_jsonl(path)

    candidates = []

    for record in records:

        if not _is_dashboard_signal(record):
            continue

        if profile_name is not None:
            parameters = record.get("parameters")
            issued_by = (
                parameters.get("profile")
                if isinstance(parameters, dict)
                else None
            )

            if issued_by != profile_name:
                continue

        try:
            timestamp = _parse_utc_timestamp(
                record.get("analysis_timestamp"),
                "signal.analysis_timestamp",
            )
        except DashboardDataError:
            continue

        if timestamp > checked_at:
            continue

        candidates.append(record)

    candidates.sort(
        key=lambda item: item["analysis_timestamp"],
        reverse=True,
    )

    signals = candidates[:limit]

    latest_timestamp = (
        _parse_utc_timestamp(
            signals[0]["analysis_timestamp"],
            "signal.analysis_timestamp",
        )
        if signals
        else None
    )

    age_minutes = (
        round(
            (checked_at - latest_timestamp).total_seconds() / 60,
            2,
        )
        if latest_timestamp
        else None
    )

    if not signals:
        status = "UNAVAILABLE"
    elif age_minutes > DEFAULT_FRESHNESS_MINUTES["trading"]:
        status = "STALE"
    else:
        status = "AVAILABLE"

    return {
        "status": status,
        "signals": signals,
        "records_returned": len(signals),
        "last_updated": (
            latest_timestamp.isoformat()
            if latest_timestamp
            else None
        ),
        "age_minutes": age_minutes,
        "manual_review_only": True,
        "auto_trading": False,
        "exchange_connection": False,
    }


def load_performance(
    *,
    performance_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Load the latest saved backtest performance report, read-only."""

    from backtesting.performance import load_latest_performance

    report = load_latest_performance(performance_directory)

    if report is None:
        return {
            "status": "UNAVAILABLE",
            "report": None,
            "message": "No backtest has been run yet.",
        }

    return {
        "status": "AVAILABLE",
        "report": report,
        "message": (
            "Measured on past data by simulation; "
            "not a prediction."
        ),
    }


def _attach_historical_performance(
    signals: list[dict[str, Any]],
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return signal copies with the matching measured win rate."""

    from backtesting.performance import lookup_signal_performance

    return [
        {
            **signal,
            "historical_performance": lookup_signal_performance(
                report,
                signal,
            ),
        }
        for signal in signals
    ]


def _section_status(
    section: dict[str, Any],
) -> str:
    """Safely extract a dashboard section status."""

    status = section.get("status")

    if status in {
        "AVAILABLE",
        "STALE",
        "UNAVAILABLE",
        "INVALID",
    }:
        return status

    return "INVALID"


def _load_live_performance(
    *,
    history_path,
    candle_directory,
    pair,
    backtest,
    now,
) -> dict[str, Any]:
    from trader.live_tracking import build_live_performance

    directory = (
        Path(candle_directory)
        if candle_directory is not None
        else DEFAULT_CANDLE_DIRECTORY
    )

    try:
        return build_live_performance(
            history_path=(
                history_path
                if history_path is not None
                else DEFAULT_SIGNAL_HISTORY_PATH
            ),
            candle_path=directory / f"kraken_{pair.upper()}_5m.jsonl",
            backtest=backtest,
            now=now,
        )
    except Exception as error:  # never break the dashboard
        return {
            "status": "INVALID",
            "message": f"Live tracking failed: {error}",
        }


def load_dashboard_data(
    *,
    pair: str = "XBTUSD",
    candle_timeframe: str = "5m",
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
    signal_limit: int = DEFAULT_SIGNAL_LIMIT,
    candle_directory: str | Path | None = None,
    history_path: str | Path | None = None,
    report_directory: str | Path | None = None,
    performance_directory: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Build a read-only dashboard snapshot.

    Reads existing raw candles, signal history and standardized reports.
    It does not collect data, generate analysis, create forecasts,
    generate trading signals or execute orders.
    """

    checked_at = (
        _parse_utc_timestamp(now, "now")
        if now is not None
        else _utc_now()
    )

    market_data = load_market_candles(
        pair=pair,
        timeframe=candle_timeframe,
        candle_directory=candle_directory,
        limit=candle_limit,
        now=checked_at,
    )

    trading = load_trading_signals(
        history_path=history_path,
        limit=signal_limit,
        now=checked_at,
        profile_name=_active_profile_name(),
    )

    performance = load_performance(
        performance_directory=performance_directory,
    )

    live_performance = _load_live_performance(
        history_path=history_path,
        candle_directory=candle_directory,
        pair=pair,
        backtest=performance.get("report"),
        now=checked_at,
    )

    trading["signals"] = _attach_historical_performance(
        trading["signals"],
        performance["report"],
    )

    reports, report_error = _load_reports(
        report_directory
    )

    market = _report_section(
        _latest_report(reports, "market"),
        now=checked_at,
        maximum_age_minutes=DEFAULT_FRESHNESS_MINUTES["market"],
    )

    forecast = _report_section(
        _latest_report(reports, "forecast"),
        now=checked_at,
        maximum_age_minutes=DEFAULT_FRESHNESS_MINUTES["forecast"],
    )

    trading_report = _report_section(
        _latest_report(reports, "trading"),
        now=checked_at,
        maximum_age_minutes=DEFAULT_FRESHNESS_MINUTES["trading"],
    )

    data_quality = _report_section(
        _latest_report(reports, "data_quality"),
        now=checked_at,
        maximum_age_minutes=DEFAULT_FRESHNESS_MINUTES["data_quality"],
    )

    statuses = {
        "market_data": market_data["status"],
        "market": _section_status(market),
        "forecast": _section_status(forecast),
        "trading": trading["status"],
        "data_quality": _section_status(data_quality),
    }

    if report_error:
        statuses["reports"] = "UNAVAILABLE"
    else:
        statuses["reports"] = "AVAILABLE"

    if any(
        status == "INVALID"
        for status in statuses.values()
    ):
        overall_status = "DEGRADED"
    elif any(
        status == "UNAVAILABLE"
        for status in statuses.values()
    ):
        overall_status = "PARTIAL"
    elif any(
        status == "STALE"
        for status in statuses.values()
    ):
        overall_status = "STALE"
    else:
        overall_status = "AVAILABLE"

    return {
        "schema_version": 1,
        "generated_at": checked_at.isoformat(),
        "asset": "BTC",
        "pair": pair.upper(),
        "status": overall_status,
        "system_status": statuses,
        "market_data": market_data,
        "market": market,
        "forecast": forecast,
        "trading": {
            **trading,
            "latest_report": trading_report,
        },
        "data_quality": data_quality,
        "performance": performance,
        "live_performance": live_performance,
        "limitations": [
            "Dashboard data is read-only.",
            "Win rates are measured on past data by simulation and are not predictions.",
            "Unavailable data is not replaced with zero or fabricated values.",
            "Stale data is identified explicitly.",
            "Only validated actionable trading candidates are displayed.",
            "Trading signals require independent manual review.",
            "Automatic trading and exchange order execution are disabled.",
        ],
    }


if __name__ == "__main__":
    try:
        snapshot = load_dashboard_data()

        print(
            json.dumps(
                snapshot,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

    except (DashboardDataError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
