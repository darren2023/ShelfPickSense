"""Rule collision baseline tests."""

from __future__ import annotations

from pathlib import Path

from fixtures import make_fixture_record


def test_external_collision_baseline_runs_on_fixture(tmp_path: Path):
    from analysis.records import load_record
    from analysis.rule_baseline import (
        RULE_COLLISION_BASELINE_NAME,
        evaluate_external_collision_baseline,
        predict_record_with_external_collision,
    )

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    preds = predict_record_with_external_collision(record, pose_frame_interval=2)

    assert len(preds) == len(record.frames())
    assert all("collision_alarm_collisions" in p for p in preds)

    report = evaluate_external_collision_baseline([record], data_dir=str(fixture_dir), pose_frame_interval=2)
    assert report.model_name == RULE_COLLISION_BASELINE_NAME
    assert report.extra["source_frame_count"] == len(record.frames())
    assert report.extra["frame_count"] == 5
    assert report.extra["skipped_pose_frame_count"] == 5
    assert report.extra["source"] == "box_human_det/services/event_engine/collision.py"
    assert 0.0 <= report.picking.macro_f1 <= 1.0


def test_external_collision_pose_frame_interval_changes_eval_sample_count(tmp_path: Path):
    from analysis.records import load_record
    from analysis.rule_baseline import evaluate_external_collision_baseline

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)

    report_interval_1 = evaluate_external_collision_baseline([record], data_dir=str(fixture_dir), pose_frame_interval=1)
    report_interval_3 = evaluate_external_collision_baseline([record], data_dir=str(fixture_dir), pose_frame_interval=3)

    assert report_interval_1.extra["frame_count"] == 10
    assert report_interval_1.extra["skipped_pose_frame_count"] == 0
    assert report_interval_3.extra["frame_count"] == 4
    assert report_interval_3.extra["skipped_pose_frame_count"] == 6


def test_benchmark_includes_external_collision_baseline(tmp_path: Path):
    from analysis.benchmark import run_benchmark
    from analysis.rule_baseline import RULE_COLLISION_BASELINE_NAME

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    output_dir = tmp_path / "benchmark_out"
    result = run_benchmark(
        train_data_dir=fixture_dir,
        output_dir=output_dir,
        model_names=["sklearn_dummy"],
        jobs=1,
    )

    assert result.baseline_report is not None
    assert result.baseline_report.model_name == RULE_COLLISION_BASELINE_NAME
    assert (output_dir / RULE_COLLISION_BASELINE_NAME / "eval_report.json").is_file()

    baseline_rows = [row for row in result.comparison if row.get("is_baseline")]
    ml_rows = [row for row in result.comparison if not row.get("is_baseline")]
    assert len(baseline_rows) == 1
    assert len(ml_rows) == 1
    assert "beats_baseline" in ml_rows[0]

    report_md = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "rule_collision" in report_md


def test_cli_eval_rule_uses_external_collision_baseline(tmp_path: Path):
    from analysis.cli import main

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    output_dir = tmp_path / "rule_eval"
    assert main(["eval-rule", "--data-dir", str(fixture_dir), "--output", str(output_dir)]) == 0
    assert (output_dir / "eval_report.json").is_file()
    assert any(output_dir.glob("eval_predictions*.json"))

