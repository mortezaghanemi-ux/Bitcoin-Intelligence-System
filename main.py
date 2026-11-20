
"""Main entry point for the Bitcoin Intelligence System.

This module coordinates existing project components.

Supported commands:
    check-config
    collect
    market
    forecast
    trader
    dashboard

All trading operations are analysis-only.
No exchange connection or order execution is performed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent

LOGGER = logging.getLogger("bitcoin_intelligence")


class ApplicationError(RuntimeError):
    """Raised when an application workflow fails."""


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _validate_pair(value: str) -> str:
    """Validate a supported Bitcoin market pair."""

    if not isinstance(value, str):
        raise ApplicationError(
            "Trading pair must be a string."
        )

    normalized = value.strip().upper()

    if (
        not normalized
        or not normalized.isalnum()
        or len(normalized) > 20
    ):
        raise ApplicationError(
            "Invalid trading pair."
        )

    return normalized


def _validate_positive_number(
    value: Any,
    name: str,
) -> float:
    """Validate a positive finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ApplicationError(
            f"{name} must be a positive finite number."
        )

    return float(value)


def _print_json(value: Any) -> None:
    """Print structured results as strict JSON."""

    try:
        print(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=str,
            )
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise ApplicationError(
            "Unable to serialize application result."
        ) from exc


def _load_project_config() -> dict[str, Any]:
    """Load and validate all required project configuration."""

    from config.loader import (
        ConfigurationError,
        load_config,
    )

    try:
        configuration = load_config()

    except ConfigurationError as exc:
        raise ApplicationError(
            f"Project configuration is invalid: {exc}"
        ) from exc

    if not isinstance(configuration, dict):
        raise ApplicationError(
            "Configuration loader returned an invalid result."
        )

    return configuration


def run_config_check() -> dict[str, Any]:
    """Validate project configuration without running workflows."""

    configuration = _load_project_config()

    settings = configuration.get(
        "settings",
        {},
    ).get(
        "system",
        {},
    )

    trader = configuration.get(
        "trader",
        {},
    ).get(
        "trader",
        {},
    )

    execution = trader.get(
        "execution",
        {},
    )

    if (
        execution.get("auto_trading") is not False
        or execution.get("exchange_connection") is not False
        or execution.get("manual_review_only") is not True
    ):
        raise ApplicationError(
            "Trading safety configuration verification failed."
        )

    return {
        "status": "SUCCESS",
        "project": settings.get(
            "name",
            "Bitcoin-Intelligence-System",
        ),
        "version": settings.get(
            "version",
            "unknown",
        ),
        "timezone": settings.get(
            "timezone",
        ),
        "configuration_files_loaded": len(configuration),
        "safety": {
            "manual_review_only": True,
            "automatic_trading": False,
            "exchange_connection": False,
            "order_execution": False,
        },
    }


def run_collection(
    pair: str = "XBTUSD",
    timeframes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collect and validate closed Kraken candles."""

    from collectors.pipeline import (
        KrakenPipelineError,
        run_kraken_pipeline,
    )

    configuration = _load_project_config()

    schedules = configuration.get(
        "schedules",
        {},
    ).get(
        "schedules",
        {},
    )

    candle_settings = schedules.get(
        "candles",
        {},
    )

    if candle_settings.get("enabled") is not True:
        raise ApplicationError(
            "Candle collection is disabled in schedules.yaml."
        )

    configured_timeframes = candle_settings.get(
        "timeframes"
    )

    if (
        not isinstance(configured_timeframes, list)
        or not configured_timeframes
    ):
        raise ApplicationError(
            "No valid candle timeframes are configured."
        )

    selected_timeframes = (
        list(timeframes)
        if timeframes is not None
        else list(configured_timeframes)
    )

    if (
        not selected_timeframes
        or any(
            not isinstance(item, str)
            or item not in configured_timeframes
            for item in selected_timeframes
        )
    ):
        raise ApplicationError(
            "Requested timeframes must be enabled in schedules.yaml."
        )

    normalized_pair = _validate_pair(pair)

    results = []
    failures = []

    for timeframe in selected_timeframes:
        try:
            result = run_kraken_pipeline(
                pair=normalized_pair,
                timeframe=timeframe,
            )

            if not isinstance(result, dict):
                raise ApplicationError(
                    f"Invalid collector result for {timeframe}."
                )

            status = result.get("status")

            if status not in {"PASS", "FAIL"}:
                raise ApplicationError(
                    f"Unexpected collection status for {timeframe}: "
                    f"{status}"
                )

            item = {
                "timeframe": timeframe,
                "status": status,
                "collected_records": result.get(
                    "collected_records",
                    0,
                ),
                "accepted_records": len(
                    result.get("accepted", [])
                ),
                "rejected_records": len(
                    result.get("rejected", [])
                ),
                "message": result.get("message"),
            }

            results.append(item)

            if status != "PASS":
                failures.append(timeframe)

        except (
            KrakenPipelineError,
            ValueError,
            TypeError,
            OSError,
            ApplicationError,
        ) as exc:
            failures.append(timeframe)

            results.append(
                {
                    "timeframe": timeframe,
                    "status": "ERROR",
                    "error": str(exc),
                }
            )

    return {
        "status": (
            "SUCCESS"
            if not failures
            else "FAILED"
        ),
        "pair": normalized_pair,
        "timestamp": _utc_now().isoformat(),
        "timeframes": results,
        "failed_timeframes": failures,
        "execution": {
            "closed_candles_only": True,
            "validated_data_only": True,
            "automatic_trading": False,
            "order_execution": False,
        },
    }


def _collect_validated_analysis_data(
    pair: str,
    timeframe: str,
) -> Any:
    """Collect candles and return only accepted validation data."""

    from collectors.pipeline import (
        KrakenPipelineError,
        run_kraken_pipeline,
    )

    try:
        result = run_kraken_pipeline(
            pair=pair,
            timeframe=timeframe,
        )

    except (
        KrakenPipelineError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        raise ApplicationError(
            f"Market data collection failed: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ApplicationError(
            "Collector returned an invalid result."
        )

    if result.get("status") != "PASS":
        raise ApplicationError(
            "Candle validation did not pass. "
            "Market engines will not receive this data."
        )

    accepted = result.get("accepted")

    if accepted is None or accepted.empty:
        raise ApplicationError(
            "No accepted closed candles are available."
        )

    return accepted.copy(deep=True)


def run_market_analysis(
    pair: str = "XBTUSD",
    timeframe: str = "1h",
) -> dict[str, Any]:
    """Collect validated candles, analyze the market and save a report."""

    from market_analysis.engine import (
        MarketAnalysisEngineError,
        analyze_market,
    )

    from reports.market_report import (
        MarketReportError,
        generate_market_report,
    )

    normalized_pair = _validate_pair(pair)
    timestamp = _utc_now()

    candles = _collect_validated_analysis_data(
        normalized_pair,
        timeframe,
    )

    try:
        analysis = analyze_market(
            data=candles,
            analysis_time=timestamp,
        )

        report = generate_market_report(
            market_analysis=analysis,
        )

    except (
        MarketAnalysisEngineError,
        MarketReportError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        raise ApplicationError(
            f"Market analysis workflow failed: {exc}"
        ) from exc

    return {
        "status": "SUCCESS",
        "pair": normalized_pair,
        "timeframe": timeframe,
        "analysis": analysis,
        "report": report,
        "execution": {
            "validated_data_only": True,
            "closed_candles_only": True,
            "trading_signal_generated": False,
            "order_execution": False,
        },
    }


def run_forecast(
    pair: str = "XBTUSD",
    timeframe: str = "1h",
) -> dict[str, Any]:
    """Run the seven-day forecast and persist its report."""

    from forecast.engine import (
        ForecastEngineError,
        generate_forecast,
    )

    from reports.forecast_report import (
        ForecastReportError,
        generate_forecast_report,
    )

    normalized_pair = _validate_pair(pair)
    timestamp = _utc_now()

    candles = _collect_validated_analysis_data(
        normalized_pair,
        timeframe,
    )

    try:
        from market_analysis.engine import (
            MarketAnalysisEngineError,
            analyze_market,
        )

        market_analysis = analyze_market(
            data=candles,
            analysis_time=timestamp,
        )

        forecast = generate_forecast(
            market_analysis=market_analysis,
            forecast_time=timestamp,
        )

        report = generate_forecast_report(
            forecast
        )

    except (
        MarketAnalysisEngineError,
        ForecastEngineError,
        ForecastReportError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        raise ApplicationError(
            f"Forecast workflow failed: {exc}"
        ) from exc

    return {
        "status": "SUCCESS",
        "pair": normalized_pair,
        "timeframe": timeframe,
        "market_analysis": market_analysis,
        "forecast": forecast,
        "report": report,
        "execution": {
            "validated_data_only": True,
            "closed_candles_only": True,
            "forecast_horizon_days": forecast[
                "forecast_horizon"
            ]["days"],
            "statistically_calibrated": forecast[
                "model"
            ]["statistically_calibrated"],
            "order_execution": False,
        },
    }


def run_trader(
    pair: str = "XBTUSD",
    leverage: float = 1.0,
    account_equity: float | None = None,
) -> dict[str, Any]:
    """Run the manual-review-only trading signal runtime."""

    from trader.runtime import (
        TradingRuntimeError,
        run_trading_runtime,
    )

    normalized_pair = _validate_pair(pair)

    normalized_leverage = _validate_positive_number(
        leverage,
        "leverage",
    )

    if normalized_leverage > 2:
        raise ApplicationError(
            "Leverage cannot exceed the project maximum of 2."
        )

    if account_equity is not None:
        account_equity = _validate_positive_number(
            account_equity,
            "account_equity",
        )

    try:
        result = run_trading_runtime(
            pair=normalized_pair,
            leverage=normalized_leverage,
            account_equity=account_equity,
        )

    except (
        TradingRuntimeError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        raise ApplicationError(
            f"Trading analysis runtime failed: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ApplicationError(
            "Trading runtime returned an invalid result."
        )

    if result.get("runtime_status") != "SUCCESS":
        raise ApplicationError(
            "Trading runtime did not report SUCCESS."
        )

    execution = result.get("execution", {})

    if (
        execution.get("manual_review_only") is not True
        or execution.get("auto_trading") is not False
        or execution.get("exchange_connection") is not False
        or execution.get("order_execution") is not False
    ):
        raise ApplicationError(
            "Trading safety verification failed."
        )

    return result


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> int:
    """Start the read-only dashboard server."""

    from dashboard.app import (
        DashboardServerError,
        run_dashboard_server,
    )

    try:
        run_dashboard_server(
            host=host,
            port=port,
        )

    except (
        DashboardServerError,
        ValueError,
        OSError,
    ) as exc:
        raise ApplicationError(
            f"Dashboard startup failed: {exc}"
        ) from exc

    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the application command-line interface."""

    parser = argparse.ArgumentParser(
        prog="bitcoin-intelligence",
        description=(
            "Bitcoin Intelligence System command-line interface."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
        help="Application logging level.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "check-config",
        help="Validate project configuration and safety settings.",
    )

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect and validate configured Kraken candle timeframes.",
    )

    collect_parser.add_argument(
        "--pair",
        default="XBTUSD",
    )

    collect_parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
    )

    for command, description in (
        ("market", "Run market analysis and save a report."),
        ("forecast", "Generate the seven-day forecast and save a report."),
    ):
        workflow_parser = subparsers.add_parser(
            command,
            help=description,
        )

        workflow_parser.add_argument(
            "--pair",
            default="XBTUSD",
        )

        workflow_parser.add_argument(
            "--timeframe",
            default="1h",
        )

    trader_parser = subparsers.add_parser(
        "trader",
        help="Generate a manual-review-only trading signal.",
    )

    trader_parser.add_argument(
        "--pair",
        default="XBTUSD",
    )

    trader_parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
    )

    trader_parser.add_argument(
        "--account-equity",
        type=float,
        default=None,
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start the read-only local dashboard.",
    )

    dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run one requested application workflow."""

    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(
            logging,
            arguments.log_level,
        ),
        format=(
            "%(asctime)s UTC "
            "%(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    try:
        if arguments.command == "check-config":
            result = run_config_check()

        elif arguments.command == "collect":
            result = run_collection(
                pair=arguments.pair,
                timeframes=arguments.timeframes,
            )

        elif arguments.command == "market":
            result = run_market_analysis(
                pair=arguments.pair,
                timeframe=arguments.timeframe,
            )

        elif arguments.command == "forecast":
            result = run_forecast(
                pair=arguments.pair,
                timeframe=arguments.timeframe,
            )

        elif arguments.command == "trader":
            result = run_trader(
                pair=arguments.pair,
                leverage=arguments.leverage,
                account_equity=arguments.account_equity,
            )

        elif arguments.command == "dashboard":
            return run_dashboard(
                host=arguments.host,
                port=arguments.port,
            )

        else:
            parser.error(
                "Unsupported application command."
            )
            return 2

        _print_json(result)

        return (
            0
            if result.get("status") == "SUCCESS"
            else 1
        )

    except ApplicationError as exc:
        LOGGER.error("%s", exc)
        return 1

    except KeyboardInterrupt:
        LOGGER.warning(
            "Application interrupted by user."
        )
        return 130

    except Exception:
        LOGGER.exception(
            "Unexpected application failure."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
