
"""Local HTTP server for the Bitcoin Intelligence Dashboard.

This module serves the dashboard frontend and read-only JSON API.

It uses only Python standard-library HTTP components.
It does not collect market data, generate signals or execute trades.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dashboard.data import (
    DashboardDataError,
    load_dashboard_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

MAX_REQUEST_LIMIT = 500

STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/style.css": "style.css",
    "/script.js": "script.js",
}

LOGGER = logging.getLogger("bitcoin_dashboard")


class DashboardServerError(RuntimeError):
    """Raised when the dashboard server cannot start safely."""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a response using strict JSON rules."""

    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")

    except (TypeError, ValueError, OverflowError) as exc:
        raise DashboardServerError(
            "Unable to serialize dashboard response."
        ) from exc


def _error_payload(
    message: str,
    *,
    status: str = "ERROR",
) -> dict[str, Any]:
    """Build a consistent API error response."""

    return {
        "status": status,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "error": message,
    }


def _parse_limit(
    query: dict[str, list[str]],
    name: str,
    default: int,
    maximum: int,
) -> int:
    """Read and validate an integer query parameter."""

    values = query.get(name)

    if not values:
        return default

    if len(values) != 1:
        raise ValueError(
            f"{name} must be provided only once."
        )

    try:
        value = int(values[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if value < 1 or value > maximum:
        raise ValueError(
            f"{name} must be between 1 and {maximum}."
        )

    return value


def _single_query_value(
    query: dict[str, list[str]],
    name: str,
    default: str,
) -> str:
    """Read one non-empty query parameter."""

    values = query.get(name)

    if not values:
        return default

    if len(values) != 1:
        raise ValueError(
            f"{name} must be provided only once."
        )

    value = values[0].strip()

    if not value:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return value


def _build_dashboard_snapshot(
    query_string: str,
) -> dict[str, Any]:
    """Build a dashboard snapshot from validated query parameters."""

    query = parse_qs(
        query_string,
        keep_blank_values=True,
        strict_parsing=False,
    )

    allowed_parameters = {
        "pair",
        "timeframe",
        "candle_limit",
        "signal_limit",
    }

    unknown = set(query) - allowed_parameters

    if unknown:
        raise ValueError(
            "Unsupported query parameter: "
            + ", ".join(sorted(unknown))
        )

    pair = _single_query_value(
        query,
        "pair",
        "XBTUSD",
    ).upper()

    if (
        not pair.isalnum()
        or len(pair) > 30
    ):
        raise ValueError(
            "Invalid trading pair."
        )

    timeframe = _single_query_value(
        query,
        "timeframe",
        "5m",
    )

    supported_timeframes = {
        "5m",
        "15m",
        "1h",
        "4h",
        "1d",
    }

    if timeframe not in supported_timeframes:
        raise ValueError(
            "Unsupported timeframe."
        )

    candle_limit = _parse_limit(
        query,
        "candle_limit",
        default=100,
        maximum=MAX_REQUEST_LIMIT,
    )

    signal_limit = _parse_limit(
        query,
        "signal_limit",
        default=20,
        maximum=MAX_REQUEST_LIMIT,
    )

    return load_dashboard_data(
        pair=pair,
        candle_timeframe=timeframe,
        candle_limit=candle_limit,
        signal_limit=signal_limit,
    )


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handle static dashboard files and read-only API requests."""

    server_version = "BitcoinIntelligenceDashboard/1.0"
    sys_version = ""

    def log_message(
        self,
        format_string: str,
        *args: Any,
    ) -> None:
        """Write concise access logs."""

        LOGGER.info(
            "%s - %s",
            self.address_string(),
            format_string % args,
        )

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: HTTPStatus,
        content_type: str,
    ) -> None:
        """Send an HTTP response with safe default headers."""

        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type,
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Cache-Control",
            "no-store, no-cache, must-revalidate",
        )

        self.send_header(
            "X-Content-Type-Options",
            "nosniff",
        )

        self.send_header(
            "X-Frame-Options",
            "DENY",
        )

        self.send_header(
            "Referrer-Policy",
            "no-referrer",
        )

        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "frame-ancestors 'none'"
            ),
        )

        self.end_headers()

        self.wfile.write(body)

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        """Send a JSON response."""

        try:
            body = _json_bytes(payload)

        except DashboardServerError:
            body = json.dumps(
                _error_payload(
                    "Unable to create API response."
                )
            ).encode("utf-8")

            status = HTTPStatus.INTERNAL_SERVER_ERROR

        self._send_bytes(
            body,
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _serve_static_file(
        self,
        request_path: str,
    ) -> None:
        """Serve one explicitly allowed frontend file."""

        filename = STATIC_FILES.get(request_path)

        if filename is None:
            self._send_json(
                _error_payload("Resource not found."),
                status=HTTPStatus.NOT_FOUND,
            )
            return

        file_path = DASHBOARD_DIRECTORY / filename

        if (
            not file_path.is_file()
            or file_path.parent.resolve()
            != DASHBOARD_DIRECTORY.resolve()
        ):
            self._send_json(
                _error_payload(
                    "Dashboard frontend file is unavailable."
                ),
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content = file_path.read_bytes()

        except OSError:
            LOGGER.exception(
                "Unable to read dashboard frontend file."
            )

            self._send_json(
                _error_payload(
                    "Unable to read dashboard frontend."
                ),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        content_type, _ = mimetypes.guess_type(
            str(file_path)
        )

        if content_type is None:
            content_type = "application/octet-stream"

        if filename.endswith((".html", ".css", ".js")):
            content_type += "; charset=utf-8"

        self._send_bytes(
            content,
            status=HTTPStatus.OK,
            content_type=content_type,
        )

    def do_GET(self) -> None:
        """Handle read-only GET requests."""

        parsed = urlparse(self.path)

        if parsed.path == "/api/dashboard":

            try:
                snapshot = _build_dashboard_snapshot(
                    parsed.query
                )

            except ValueError as exc:
                self._send_json(
                    _error_payload(str(exc)),
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            except DashboardDataError:
                LOGGER.exception(
                    "Dashboard data loading failed."
                )

                self._send_json(
                    _error_payload(
                        "Dashboard data is currently unavailable."
                    ),
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            except (OSError, RuntimeError):
                LOGGER.exception(
                    "Unexpected dashboard API failure."
                )

                self._send_json(
                    _error_payload(
                        "Dashboard service encountered an internal error."
                    ),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            self._send_json(snapshot)
            return

        self._serve_static_file(
            parsed.path
        )

    def do_HEAD(self) -> None:
        """Support HEAD requests without exposing response bodies."""

        parsed = urlparse(self.path)

        if parsed.path not in STATIC_FILES:
            self._send_json(
                _error_payload("Resource not found."),
                status=HTTPStatus.NOT_FOUND,
            )
            return

        filename = STATIC_FILES[parsed.path]
        file_path = DASHBOARD_DIRECTORY / filename

        if not file_path.is_file():
            self._send_json(
                _error_payload(
                    "Dashboard frontend file is unavailable."
                ),
                status=HTTPStatus.NOT_FOUND,
            )
            return

        content_type, _ = mimetypes.guess_type(
            str(file_path)
        )

        if content_type is None:
            content_type = "application/octet-stream"

        if filename.endswith((".html", ".css", ".js")):
            content_type += "; charset=utf-8"

        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            content_type,
        )
        self.send_header(
            "Content-Length",
            str(file_path.stat().st_size),
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()

    def do_POST(self) -> None:
        """Reject all write operations."""

        self._send_json(
            _error_payload(
                "This dashboard is read-only. POST is not supported."
            ),
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def do_PUT(self) -> None:
        """Reject all write operations."""

        self.do_POST()

    def do_DELETE(self) -> None:
        """Reject all write operations."""

        self.do_POST()


class DashboardHTTPServer(ThreadingHTTPServer):
    """Threaded dashboard HTTP server."""

    daemon_threads = True
    allow_reuse_address = True


def run_dashboard_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Start the dashboard HTTP server."""

    if (
        not isinstance(host, str)
        or not host.strip()
    ):
        raise ValueError(
            "host must be a non-empty string."
        )

    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65535
    ):
        raise ValueError(
            "port must be an integer between 1 and 65535."
        )

    try:
        server = DashboardHTTPServer(
            (host, port),
            DashboardRequestHandler,
        )

    except OSError as exc:
        raise DashboardServerError(
            f"Unable to start dashboard server: {exc}"
        ) from exc

    LOGGER.info(
        "Dashboard server started at http://%s:%s",
        host,
        port,
    )

    try:
        server.serve_forever(
            poll_interval=0.5
        )

    except KeyboardInterrupt:
        LOGGER.info(
            "Dashboard server shutdown requested."
        )

    finally:
        server.server_close()

        LOGGER.info(
            "Dashboard server stopped."
        )


def main() -> int:
    """Parse command-line arguments and start the server."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only Bitcoin Intelligence Dashboard."
        )
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Bind address. Defaults to localhost only.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="HTTP port. Defaults to 8000.",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )

    arguments = parser.parse_args()

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
        run_dashboard_server(
            host=arguments.host,
            port=arguments.port,
        )

    except (DashboardServerError, ValueError) as exc:
        LOGGER.error(
            "Dashboard startup failed: %s",
            exc,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
