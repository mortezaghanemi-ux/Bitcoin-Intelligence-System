"""Build a static, read-only copy of the dashboard for public hosting.

The live dashboard asks a local server for JSON. For GitHub Pages the
same JSON is generated once per timeframe at build time and written
next to the page, so no server is needed. Nothing here collects data,
generates signals or places orders.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from dashboard.data import load_dashboard_data

DASHBOARD_DIRECTORY = Path(__file__).resolve().parent

# The daily timeframe has no collected data, so it is not published.
STATIC_TIMEFRAMES = ("5m", "15m", "1h", "4h")
DEFAULT_STATIC_TIMEFRAME = "1h"


def _static_index(html: str, version: str = "0") -> str:
    # The version query makes browsers fetch new assets after a deploy.
    html = html.replace(
        'href="/style.css"', f'href="style.css?v={version}"'
    )
    html = html.replace(
        'src="/script.js"', f'src="script.js?v={version}"'
    )
    html = html.replace(
        '<option value="1d">روزانه</option>', ""
    )
    html = html.replace(
        '<option value="5m" selected>', '<option value="5m">'
    )
    html = html.replace(
        '<option value="1h">', '<option value="1h" selected>'
    )
    html = html.replace(
        "</head>",
        '    <meta name="dashboard-static" content="1">\n</head>',
        1,
    )

    return html


def build_static_dashboard(
    output_directory: str | Path,
    *,
    pair: str = "XBTUSD",
    candle_limit: int = 100,
    signal_limit: int = 100,
    **data_paths: Any,
) -> dict[str, Any]:
    """Write index.html, assets and one JSON snapshot per timeframe."""

    output = Path(output_directory)
    data_directory = output / "data"
    data_directory.mkdir(parents=True, exist_ok=True)

    index = (DASHBOARD_DIRECTORY / "index.html").read_text(
        encoding="utf-8"
    )
    (output / "index.html").write_text(
        _static_index(index, str(int(time.time()))), encoding="utf-8"
    )

    for asset in ("style.css", "script.js"):
        shutil.copyfile(DASHBOARD_DIRECTORY / asset, output / asset)

    (output / ".nojekyll").write_text("", encoding="utf-8")

    written = []

    for timeframe in STATIC_TIMEFRAMES:
        snapshot = load_dashboard_data(
            pair=pair,
            candle_timeframe=timeframe,
            candle_limit=candle_limit,
            signal_limit=signal_limit,
            **data_paths,
        )
        path = data_directory / f"{timeframe}.json"
        path.write_text(
            json.dumps(
                snapshot,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
        )
        written.append(path.name)

    return {"output": str(output), "snapshots": written}


if __name__ == "__main__":
    import sys

    print(
        build_static_dashboard(
            sys.argv[1] if len(sys.argv) > 1 else "site"
        )
    )
