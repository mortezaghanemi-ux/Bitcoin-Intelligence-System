import json

from dashboard.static_site import build_static_dashboard


def test_static_build_writes_page_and_snapshots(tmp_path):
    result = build_static_dashboard(
        tmp_path / "site",
        candle_directory=tmp_path / "candles",
        history_path=tmp_path / "history.jsonl",
        report_directory=tmp_path / "reports",
        performance_directory=tmp_path / "perf",
    )

    site = tmp_path / "site"
    html = (site / "index.html").read_text(encoding="utf-8")

    assert 'name="dashboard-static"' in html
    assert 'href="style.css"' in html
    assert 'src="script.js"' in html
    assert 'value="1d"' not in html
    assert '<option value="1h" selected>' in html
    assert (site / ".nojekyll").exists()
    assert "STATIC_MODE" in (site / "script.js").read_text(
        encoding="utf-8"
    )
    assert result["snapshots"] == [
        "5m.json", "15m.json", "1h.json", "4h.json"
    ]

    snapshot = json.loads((site / "data" / "1h.json").read_text())

    assert "trading" in snapshot
    assert snapshot["performance"]["status"] == "UNAVAILABLE"
