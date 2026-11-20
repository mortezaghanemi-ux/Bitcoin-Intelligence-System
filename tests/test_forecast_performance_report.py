
"""Tests for the forecast performance report builder and storage adapter."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from forecast.performance import ForecastPerformanceError
from reports.engine import ReportEngineError
from reports.forecast_performance_report import (
    ForecastPerformanceReportError,
    _validate_analysis,
    build_forecast_performance_report,
    generate_forecast_performance_report,
)


UTC = timezone.utc

GENERATED_AT = datetime(
    2026,
    11,
    20,
    7,
    0,
    tzinfo=UTC,
)


def make_evaluation(
    forecast_id="BTC-REPORT-001",
    actual_direction="BULLISH",
    predicted_direction="BULLISH",
    confidence_score=80.0,
    horizon_days=7,
    brier_score=0.25,
    log_loss=0.50,
    directional_accuracy=1.0,
):
    """Create a valid completed forecast evaluation."""

    return {
        "evaluation_id": f"{forecast_id}-EVALUATION",
        "forecast_id": forecast_id,
        "status": "SUCCESS",
        "evaluated_at": "2026-11-15T12:00:00+00:00",
        "forecast_timestamp": "2026-11-01T12:00:00+00:00",
        "forecast_horizon_end": "2026-11-08T12:00:00+00:00",
        "actual_outcome": {
            "timestamp": "2026-11-08T12:00:00+00:00",
            "reference_price": 95000.0,
            "outcome_price": 100000.0,
            "percentage_change": 5.263158,
            "direction": actual_direction,
            "neutral_threshold_pct": 1.0,
        },
        "predicted_outcome": {
            "probabilities": {
                "BULLISH": 70.0,
                "BEARISH": 20.0,
                "NEUTRAL": 10.0,
            },
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
        },
        "metrics": {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "directional_accuracy": directional_accuracy,
            "directional_correct": (
                actual_direction == predicted_direction
            ),
        },
        "metadata": {
            "asset": "BTC",
            "horizon_days": horizon_days,
            "method": "multiclass_probability_evaluation",
            "statistical_calibration_assessed": False,
        },
    }


def make_evaluations():
    """Return multiple valid evaluations with different outcomes."""

    return [
        make_evaluation(
            forecast_id="BTC-REPORT-001",
            actual_direction="BULLISH",
            predicted_direction="BULLISH",
            confidence_score=85.0,
            horizon_days=7,
            brier_score=0.20,
            log_loss=0.40,
            directional_accuracy=1.0,
        ),
        make_evaluation(
            forecast_id="BTC-REPORT-002",
            actual_direction="BEARISH",
            predicted_direction="BEARISH",
            confidence_score=60.0,
            horizon_days=7,
            brier_score=0.30,
            log_loss=0.60,
            directional_accuracy=1.0,
        ),
        make_evaluation(
            forecast_id="BTC-REPORT-003",
            actual_direction="NEUTRAL",
            predicted_direction="BULLISH",
            confidence_score=45.0,
            horizon_days=14,
            brier_score=0.80,
            log_loss=1.20,
            directional_accuracy=0.0,
        ),
    ]


# =========================================================
# Successful report construction
# =========================================================

def test_builds_valid_forecast_performance_report():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    assert report["report_type"] == "forecast_performance"
    assert report["source"] == "forecast.performance"
    assert report["metadata"]["asset"] == "BTC"
    assert report["metadata"]["analysis_type"] == "FORECAST_PERFORMANCE"


def test_report_contains_generated_timestamp():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    timestamp = datetime.fromisoformat(
        report["generated_at"]
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset().total_seconds() == 0


def test_report_contains_expected_metadata():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    metadata = report["metadata"]

    assert metadata["evaluation_count"] == 3
    assert metadata["method"] == "historical_evaluation_analysis"
    assert metadata["statistical_calibration_assessed"] is False
    assert metadata["input_records_modified"] is False


def test_report_contains_sorted_forecast_ids():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    assert report["metadata"]["forecast_ids"] == [
        "BTC-REPORT-001",
        "BTC-REPORT-002",
        "BTC-REPORT-003",
    ]


def test_report_contains_complete_analysis():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    analysis = report["payload"]["analysis"]

    assert analysis["status"] == "SUCCESS"
    assert analysis["analysis_type"] == "FORECAST_PERFORMANCE"
    assert analysis["evaluation_count"] == 3
    assert isinstance(analysis["summary"], dict)
    assert isinstance(analysis["by_horizon_days"], dict)
    assert isinstance(analysis["by_confidence"], dict)
    assert isinstance(analysis["by_actual_direction"], dict)
    assert isinstance(analysis["weaknesses"], list)


def test_report_summary_contains_correct_metrics():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    summary = report["payload"]["analysis"]["summary"]

    assert summary["evaluation_count"] == 3
    assert summary["mean_brier_score"] == pytest.approx(
        (0.20 + 0.30 + 0.80) / 3
    )
    assert summary["mean_log_loss"] == pytest.approx(
        (0.40 + 0.60 + 1.20) / 3
    )
    assert summary["directional_accuracy"] == pytest.approx(
        2 / 3
    )
    assert summary["correct_predictions"] == 2
    assert summary["incorrect_predictions"] == 1


def test_report_contains_interpretation_notice():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    notice = report["payload"]["interpretation_notice"]

    assert isinstance(notice, str)
    assert "historical forecast outcomes" in notice
    assert "do not guarantee future accuracy" in notice
    assert "statistical calibration" in notice


def test_report_preserves_horizon_groups():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    groups = report["payload"]["analysis"]["by_horizon_days"]

    assert groups["7"]["evaluation_count"] == 2
    assert groups["14"]["evaluation_count"] == 1


def test_report_preserves_confidence_groups():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    groups = report["payload"]["analysis"]["by_confidence"]

    assert groups["LOW"]["evaluation_count"] == 1
    assert groups["MODERATE"]["evaluation_count"] == 1
    assert groups["HIGH"]["evaluation_count"] == 0
    assert groups["VERY_HIGH"]["evaluation_count"] == 1


def test_report_preserves_actual_direction_groups():
    report = build_forecast_performance_report(
        make_evaluations()
    )

    groups = report["payload"]["analysis"]["by_actual_direction"]

    assert groups["BULLISH"]["evaluation_count"] == 1
    assert groups["BEARISH"]["evaluation_count"] == 1
    assert groups["NEUTRAL"]["evaluation_count"] == 1


# =========================================================
# Input validation
# =========================================================

@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "invalid",
        b"invalid",
        {},
        123,
    ],
)
def test_rejects_invalid_evaluation_collections(invalid_input):
    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report(invalid_input)


def test_rejects_empty_evaluation_collection():
    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([])


def test_rejects_non_dictionary_evaluation():
    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report(
            [
                make_evaluation(),
                "invalid-record",
            ]
        )


def test_rejects_failed_evaluation():
    record = make_evaluation()
    record["status"] = "FAILED"

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


def test_rejects_duplicate_forecast_ids():
    first = make_evaluation(
        forecast_id="BTC-DUPLICATE",
    )
    second = make_evaluation(
        forecast_id="BTC-DUPLICATE",
    )

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report(
            [first, second]
        )


def test_rejects_invalid_forecast_direction():
    record = make_evaluation()
    record["actual_outcome"]["direction"] = "UNKNOWN"

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


def test_rejects_inconsistent_directional_accuracy():
    record = make_evaluation(
        actual_direction="BULLISH",
        predicted_direction="BEARISH",
        directional_accuracy=1.0,
    )

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


def test_rejects_invalid_confidence_score():
    record = make_evaluation(
        confidence_score=101.0,
    )

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


def test_rejects_invalid_horizon_days():
    record = make_evaluation(
        horizon_days=0,
    )

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


def test_rejects_non_finite_brier_score():
    record = make_evaluation(
        brier_score=float("nan"),
    )

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


def test_rejects_negative_log_loss():
    record = make_evaluation(
        log_loss=-0.1,
    )

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        build_forecast_performance_report([record])


# =========================================================
# Analysis output validation
# =========================================================

def make_valid_analysis():
    """Create a valid analyzer output for direct validation tests."""

    return {
        "status": "SUCCESS",
        "analysis_type": "FORECAST_PERFORMANCE",
        "evaluation_count": 1,
        "summary": {
            "evaluation_count": 1,
            "mean_brier_score": 0.25,
            "mean_log_loss": 0.5,
            "directional_accuracy": 1.0,
            "correct_predictions": 1,
            "incorrect_predictions": 0,
        },
        "by_horizon_days": {
            "7": {
                "evaluation_count": 1,
            },
        },
        "by_confidence": {
            "HIGH": {
                "evaluation_count": 1,
            },
        },
        "by_actual_direction": {
            "BULLISH": {
                "evaluation_count": 1,
            },
        },
        "weaknesses": [],
        "metadata": {
            "asset": "BTC",
        },
        "forecast_ids": [
            "BTC-REPORT-001",
        ],
    }


def test_validate_analysis_accepts_valid_analysis():
    analysis = make_valid_analysis()

    result = _validate_analysis(analysis)

    assert result is analysis


@pytest.mark.parametrize(
    "invalid_analysis",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_validate_analysis_rejects_non_dictionary(invalid_analysis):
    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(invalid_analysis)


def test_validate_analysis_rejects_failed_status():
    analysis = make_valid_analysis()
    analysis["status"] = "FAILED"

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_wrong_analysis_type():
    analysis = make_valid_analysis()
    analysis["analysis_type"] = "FORECAST"

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_missing_required_field():
    analysis = make_valid_analysis()
    del analysis["forecast_ids"]

    with pytest.raises(
        ForecastPerformanceReportError,
        match="missing fields",
    ):
        _validate_analysis(analysis)


@pytest.mark.parametrize(
    "count",
    [
        0,
        -1,
        True,
        1.5,
        "1",
        None,
    ],
)
def test_validate_analysis_rejects_invalid_evaluation_count(count):
    analysis = make_valid_analysis()
    analysis["evaluation_count"] = count

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_summary_count_mismatch():
    analysis = make_valid_analysis()
    analysis["summary"]["evaluation_count"] = 2

    with pytest.raises(
        ForecastPerformanceReportError,
        match="does not match",
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_non_dictionary_summary():
    analysis = make_valid_analysis()
    analysis["summary"] = []

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_missing_summary_metric():
    analysis = make_valid_analysis()
    del analysis["summary"]["mean_brier_score"]

    with pytest.raises(
        ForecastPerformanceReportError,
        match="missing mean_brier_score",
    ):
        _validate_analysis(analysis)


@pytest.mark.parametrize(
    "field",
    [
        "by_horizon_days",
        "by_confidence",
        "by_actual_direction",
    ],
)
def test_validate_analysis_rejects_invalid_group(field):
    analysis = make_valid_analysis()
    analysis[field] = []

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_non_list_weaknesses():
    analysis = make_valid_analysis()
    analysis["weaknesses"] = {}

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


def test_validate_analysis_rejects_non_dictionary_metadata():
    analysis = make_valid_analysis()
    analysis["metadata"] = []

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


@pytest.mark.parametrize(
    "forecast_ids",
    [
        None,
        "BTC-001",
        [],
        ["BTC-001", "BTC-002"],
        ["BTC-001", "BTC-001"],
        [""],
        [None],
        [123],
    ],
)
def test_validate_analysis_rejects_invalid_forecast_ids(forecast_ids):
    analysis = make_valid_analysis()
    analysis["forecast_ids"] = forecast_ids

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        _validate_analysis(analysis)


# =========================================================
# Input immutability and iterable handling
# =========================================================

def test_build_report_does_not_modify_input_records():
    records = make_evaluations()
    original = deepcopy(records)

    build_forecast_performance_report(records)

    assert records == original


def test_build_report_accepts_generator():
    records = make_evaluations()

    report = build_forecast_performance_report(
        record for record in records
    )

    assert report["metadata"]["evaluation_count"] == 3


def test_build_report_accepts_tuple():
    report = build_forecast_performance_report(
        tuple(make_evaluations())
    )

    assert report["metadata"]["evaluation_count"] == 3


# =========================================================
# Report persistence
# =========================================================

def test_generate_report_saves_json_file(tmp_path):
    result = generate_forecast_performance_report(
        make_evaluations(),
        output_directory=tmp_path,
    )

    assert result["status"] == "SUCCESS"
    assert result["report_type"] == "forecast_performance"
    assert result["storage"]["saved"] is True
    assert result["storage"]["path"].endswith(".json")


def test_generate_report_creates_readable_history(tmp_path):
    result = generate_forecast_performance_report(
        make_evaluations(),
        output_directory=tmp_path,
    )

    engine = __import__(
        "reports.engine",
        fromlist=["ReportEngine"],
    ).ReportEngine(tmp_path)

    history = engine.get_history(
        report_type="forecast_performance"
    )

    assert len(history) == 1
    assert history[0]["report_id"] == result["report_id"]
    assert history[0]["report_type"] == "forecast_performance"


def test_generate_report_rejects_invalid_records_before_saving(tmp_path):
    record = make_evaluation()
    record["status"] = "FAILED"

    with pytest.raises(
        ForecastPerformanceReportError
    ):
        generate_forecast_performance_report(
            [record],
            output_directory=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_generate_report_wraps_storage_errors(monkeypatch, tmp_path):
    class FailingReportEngine:
        def __init__(self, output_directory):
            self.output_directory = output_directory

        def generate(self, **kwargs):
            raise ReportEngineError(
                "simulated storage failure"
            )

    monkeypatch.setattr(
        "reports.forecast_performance_report.ReportEngine",
        FailingReportEngine,
    )

    with pytest.raises(
        ForecastPerformanceReportError,
        match="Unable to save",
    ):
        generate_forecast_performance_report(
            make_evaluations(),
            output_directory=tmp_path,
        )
