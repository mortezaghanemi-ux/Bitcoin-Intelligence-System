
"""Bitcoin spot ETF flow data collector."""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import re

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "etf"
    / "bitcoin_etf_flows.csv"
)

DEFAULT_TIMEOUT = 20

REQUIRED_FIELDS = {
    "date",
    "fund",
    "ticker",
    "net_flow_usd",
    "source",
}

DEFAULT_SOURCE_NAME = "farside"


class ETFCollectorError(RuntimeError):
    """Raised when ETF flow collection fails."""


def _parse_date(value: Any) -> date:
    """Parse an ETF observation date."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ETFCollectorError(
                "ETF datetime values must be timezone-aware."
            )

        return value.astimezone(timezone.utc).date()

    if isinstance(value, date):
        return value

    if not isinstance(value, str) or not value.strip():
        raise ETFCollectorError(
            "ETF date must be a non-empty string or date."
        )

    text = value.strip()

    for date_format in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(
                text,
                date_format,
            ).date()
        except ValueError:
            continue

    raise ETFCollectorError(
        f"Unsupported ETF date format: {text}"
    )


def _parse_amount(value: Any) -> float:
    """Parse a finite USD flow amount."""

    if isinstance(value, bool):
        raise ETFCollectorError(
            "ETF flow amount cannot be boolean."
        )

    if isinstance(value, (int, float)):
        amount = float(value)

    elif isinstance(value, str):
        text = value.strip()

        if not text or text in {"-", "—", "N/A", "n/a"}:
            raise ETFCollectorError(
                "ETF flow amount is missing."
            )

        negative = text.startswith("(") and text.endswith(")")

        if negative:
            text = text[1:-1].strip()

        text = (
            text.replace("$", "")
            .replace(",", "")
            .replace(" ", "")
        )

        multiplier = 1.0

        if text.endswith(("B", "b")):
            multiplier = 1_000_000_000
            text = text[:-1]
        elif text.endswith(("M", "m")):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.endswith(("K", "k")):
            multiplier = 1_000
            text = text[:-1]

        try:
            amount = float(text) * multiplier
        except ValueError as exc:
            raise ETFCollectorError(
                f"Invalid ETF flow amount: {value}"
            ) from exc

        if negative:
            amount = -amount

    else:
        raise ETFCollectorError(
            "ETF flow amount must be numeric."
        )

    if not math.isfinite(amount):
        raise ETFCollectorError(
            "ETF flow amount must be finite."
        )

    return amount


def _normalize_text(
    value: Any,
    field: str,
) -> str:
    """Validate and normalize a text field."""

    if not isinstance(value, str) or not value.strip():
        raise ETFCollectorError(
            f"ETF field '{field}' must be a non-empty string."
        )

    return value.strip()


def normalize_etf_record(
    record: dict[str, Any],
    source: str = DEFAULT_SOURCE_NAME,
) -> dict[str, Any]:
    """Normalize one ETF flow observation."""

    if not isinstance(record, dict):
        raise ETFCollectorError(
            "ETF record must be a dictionary."
        )

    observation_date = _parse_date(
        record.get("date")
    )

    fund = _normalize_text(
        record.get("fund"),
        "fund",
    )

    ticker = _normalize_text(
        record.get("ticker"),
        "ticker",
    ).upper()

    net_flow = _parse_amount(
        record.get("net_flow_usd")
    )

    source_name = _normalize_text(
        record.get("source", source),
        "source",
    )

    return {
        "date": observation_date.isoformat(),
        "fund": fund,
        "ticker": ticker,
        "net_flow_usd": net_flow,
        "source": source_name,
    }


def normalize_etf_records(
    records: list[dict[str, Any]],
    source: str = DEFAULT_SOURCE_NAME,
) -> list[dict[str, Any]]:
    """Normalize observations and reject duplicate fund dates."""

    if not isinstance(records, list):
        raise ETFCollectorError(
            "ETF records must be a list."
        )

    normalized = []
    seen = set()

    for record in records:
        item = normalize_etf_record(
            record,
            source=source,
        )

        identity = (
            item["date"],
            item["ticker"],
        )

        if identity in seen:
            raise ETFCollectorError(
                "Duplicate ETF observation for "
                f"{item['ticker']} on {item['date']}."
            )

        seen.add(identity)
        normalized.append(item)

    return sorted(
        normalized,
        key=lambda item: (
            item["date"],
            item["ticker"],
        ),
    )


def _validate_source_url(url: str) -> str:
    """Require a valid public HTTP or HTTPS source URL."""

    if not isinstance(url, str) or not url.strip():
        raise ETFCollectorError(
            "ETF source URL must be provided."
        )

    parsed = urlparse(url.strip())

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ETFCollectorError(
            "ETF source URL must be a valid HTTP or HTTPS URL."
        )

    if parsed.hostname.lower() in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ETFCollectorError(
            "Local ETF source URLs are not allowed."
        )

    return url.strip()


def parse_csv_content(
    content: str,
    source: str = DEFAULT_SOURCE_NAME,
) -> list[dict[str, Any]]:
    """Parse CSV content with explicit required columns."""

    if not isinstance(content, str) or not content.strip():
        raise ETFCollectorError(
            "ETF CSV content is empty."
        )

    try:
        reader = csv.DictReader(
            io.StringIO(content)
        )

        if not reader.fieldnames:
            raise ETFCollectorError(
                "ETF CSV header is missing."
            )

        normalized_headers = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if isinstance(name, str)
        }

        aliases = {
            "date": ("date", "day"),
            "fund": ("fund", "name"),
            "ticker": ("ticker", "symbol"),
            "net_flow_usd": (
                "net_flow_usd",
                "net flow",
                "net flow ($mm)",
                "flow",
            ),
        }

        selected = {}

        for target, candidates in aliases.items():
            for candidate in candidates:
                if candidate in normalized_headers:
                    selected[target] = normalized_headers[candidate]
                    break

            if target not in selected:
                raise ETFCollectorError(
                    f"ETF CSV is missing column '{target}'."
                )

        records = []

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ETFCollectorError(
                    f"Malformed ETF CSV row {row_number}."
                )

            if not any(
                str(value or "").strip()
                for value in row.values()
            ):
                continue

            records.append(
                {
                    "date": row[selected["date"]],
                    "fund": row[selected["fund"]],
                    "ticker": row[selected["ticker"]],
                    "net_flow_usd": row[
                        selected["net_flow_usd"]
                    ],
                    "source": source,
                }
            )

        return normalize_etf_records(
            records,
            source=source,
        )

    except csv.Error as exc:
        raise ETFCollectorError(
            f"Unable to parse ETF CSV: {exc}"
        ) from exc


class BitcoinETFCollector:
    """
    Collect Bitcoin spot ETF flow observations.

    The source URL is configurable because public ETF data providers
    can change their CSV formats and access policies.

    The collector never invents missing observations.
    """

    def __init__(
        self,
        source_url: str,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
        source_name: str = DEFAULT_SOURCE_NAME,
    ) -> None:
        self.source_url = _validate_source_url(
            source_url
        )

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ETFCollectorError(
                "timeout must be a positive finite number."
            )

        self.timeout = float(timeout)
        self.output_path = Path(output_path)
        self.source_name = _normalize_text(
            source_name,
            "source_name",
        )

        self.session = session or requests.Session()

    def fetch_csv(self) -> str:
        """Download raw ETF CSV content."""

        try:
            response = self.session.get(
                self.source_url,
                timeout=self.timeout,
                headers={
                    "Accept": "text/csv,text/plain,*/*",
                    "User-Agent": "Bitcoin-Intelligence-System/1.0",
                },
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise ETFCollectorError(
                f"ETF source request failed: {exc}"
            ) from exc

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if "json" in content_type:
            raise ETFCollectorError(
                "ETF source returned JSON instead of CSV."
            )

        content = response.text

        if not isinstance(content, str) or not content.strip():
            raise ETFCollectorError(
                "ETF source returned empty content."
            )

        return content

    def collect(self) -> list[dict[str, Any]]:
        """Fetch and normalize ETF flow observations."""

        content = self.fetch_csv()

        records = parse_csv_content(
            content,
            source=self.source_name,
        )

        if not records:
            raise ETFCollectorError(
                "ETF source returned no valid observations."
            )

        return records

    def save(
        self,
        records: list[dict[str, Any]],
    ) -> Path:
        """
        Append new observations without overwriting existing data.

        Existing date/ticker combinations are not duplicated.
        Conflicting values for an existing observation are rejected.
        """

        normalized = normalize_etf_records(
            records,
            source=self.source_name,
        )

        if not normalized:
            raise ETFCollectorError(
                "No ETF observations to save."
            )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = {}

        if self.output_path.exists():
            try:
                with self.output_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    reader = csv.DictReader(file)

                    if (
                        reader.fieldnames is None
                        or set(reader.fieldnames) != REQUIRED_FIELDS
                    ):
                        raise ETFCollectorError(
                            "Existing ETF CSV has an invalid schema."
                        )

                    for row in reader:
                        item = normalize_etf_record(
                            row,
                            source=self.source_name,
                        )

                        identity = (
                            item["date"],
                            item["ticker"],
                        )

                        if identity in existing:
                            raise ETFCollectorError(
                                "Existing ETF CSV contains duplicates."
                            )

                        existing[identity] = item

            except OSError as exc:
                raise ETFCollectorError(
                    f"Unable to read existing ETF data: {exc}"
                ) from exc

        new_records = []

        for item in normalized:
            identity = (
                item["date"],
                item["ticker"],
            )

            previous = existing.get(identity)

            if previous is not None:
                if previous != item:
                    raise ETFCollectorError(
                        "Conflicting ETF observation for "
                        f"{item['ticker']} on {item['date']}."
                    )

                continue

            new_records.append(item)

        if not new_records:
            return self.output_path

        write_header = not self.output_path.exists() or (
            self.output_path.stat().st_size == 0
        )

        try:
            with self.output_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "date",
                        "fund",
                        "ticker",
                        "net_flow_usd",
                        "source",
                    ],
                    extrasaction="raise",
                )

                if write_header:
                    writer.writeheader()

                writer.writerows(new_records)

        except (OSError, csv.Error) as exc:
            raise ETFCollectorError(
                f"Unable to save ETF observations: {exc}"
            ) from exc

        LOGGER.info(
            "Saved %s new Bitcoin ETF observations to %s",
            len(new_records),
            self.output_path,
        )

        return self.output_path

    def run(self) -> dict[str, Any]:
        """Fetch, validate, and append ETF flow observations."""

        records = self.collect()
        output = self.save(records)

        return {
            "status": "SUCCESS",
            "source": self.source_name,
            "record_count": len(records),
            "output_path": str(output),
            "collected_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "data": records,
        }


def collect_bitcoin_etf_flows(
    source_url: str | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Convenience function for collecting Bitcoin ETF flows."""

    configured_url = (
        source_url
        or os.environ.get("BITCOIN_ETF_SOURCE_URL")
    )

    if not configured_url:
        raise ETFCollectorError(
            "Set BITCOIN_ETF_SOURCE_URL or provide source_url."
        )

    collector = BitcoinETFCollector(
        source_url=configured_url,
        output_path=output_path,
        timeout=timeout,
    )

    return collector.run()
