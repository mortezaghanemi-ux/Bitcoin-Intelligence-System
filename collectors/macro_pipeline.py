"""FRED macroeconomic collection pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from collectors.fred import (
    FREDCollectorError,
    collect_fred_observations,
)

from config.macro_series_loader import (
    MacroConfigurationError,
    load_macro_series_config,
)


class MacroPipelineError(RuntimeError):
    """Raised when the macroeconomic pipeline fails."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "macro"
    / "fred"
)


def run_fred_pipeline(
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    collection_time: datetime | None = None,
    timeout: float = 20.0,
    page_size: int = 100000,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Load macro configuration and collect enabled FRED series.

    API credentials are passed directly to the collector.
    They are never included in the returned result.

    Raw observations are stored by the FRED collector
    using append-only JSONL storage.

    This pipeline does not generate forecasts or signals.
    """

    # 1. Load and validate macro configuration.

    try:
        config = load_macro_series_config(
            config_path=config_path,
        )

    except MacroConfigurationError as exc:
        raise MacroPipelineError(
            f"Invalid macroeconomic configuration: {exc}"
        ) from exc

    # 2. Resolve enabled series identifiers.

    series = config.get("series")

    if not isinstance(series, list) or not series:
        raise MacroPipelineError(
            "No enabled FRED series are configured."
        )

    series_ids = []

    for item in series:

        if not isinstance(item, dict):
            raise MacroPipelineError(
                "Invalid FRED series configuration."
            )

        series_id = item.get("id")

        if (
            not isinstance(series_id, str)
            or not series_id
        ):
            raise MacroPipelineError(
                "Invalid FRED series identifier."
            )

        if item.get("enabled") is not True:
            raise MacroPipelineError(
                "Disabled series must not enter the collection pipeline."
            )

        series_ids.append(series_id)

    # 3. Resolve observation date range.

    collection = config.get("collection")

    if not isinstance(collection, dict):
        raise MacroPipelineError(
            "Invalid macro collection configuration."
        )

    observation_start = collection.get(
        "observation_start"
    )

    observation_end = collection.get(
        "observation_end"
    )

    # 4. Resolve output directory.

    destination = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    # 5. Collect observations.

    try:
        records = collect_fred_observations(
            series_ids=series_ids,
            observation_start=observation_start,
            observation_end=observation_end,
            output_dir=destination,
            collection_time=collection_time,
            timeout=timeout,
            page_size=page_size,
            api_key=api_key,
        )

    except (
        FREDCollectorError,
        ValueError,
        OSError,
    ) as exc:
        raise MacroPipelineError(
            "FRED macroeconomic collection failed."
        ) from exc

    # 6. Validate collector response.

    if not isinstance(records, list):
        raise MacroPipelineError(
            "FRED collector returned an invalid result."
        )

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            raise MacroPipelineError(
                f"Invalid FRED record at index {index}."
            )

        if record.get("source") != "fred":
            raise MacroPipelineError(
                f"Unexpected data source at index {index}."
            )

        if record.get("series_id") not in series_ids:
            raise MacroPipelineError(
                f"Unexpected FRED series at index {index}."
            )

        if record.get("value_status") not in (
            "VALID",
            "MISSING",
        ):
            raise MacroPipelineError(
                f"Invalid FRED value status at index {index}."
            )

    # 7. Build collection summary.

    counts_by_series = {
        series_id: 0
        for series_id in series_ids
    }

    for record in records:
        counts_by_series[
            record["series_id"]
        ] += 1

    return {
        "status": (
            "SUCCESS"
            if records
            else "NO_NEW_DATA"
        ),
        "provider": "fred",
        "timezone": "UTC",
        "requested_series": series_ids,
        "collected_records": len(records),
        "counts_by_series": counts_by_series,
        "records": records,
        "output_file": str(
            destination / "fred_observations.jsonl"
        ),
        "message": (
            "FRED observations collected successfully."
            if records
            else "No new FRED observations were available."
        ),
    }


if __name__ == "__main__":
    try:
        result = run_fred_pipeline()

    except MacroPipelineError as exc:
        raise SystemExit(
            f"MACRO PIPELINE ERROR: {exc}"
        ) from exc

    print("FRED PIPELINE:", result["status"])
    print("Requested series:", len(result["requested_series"]))
    print("Collected records:", result["collected_records"])