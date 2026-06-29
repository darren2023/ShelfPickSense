"""规则基线评测测试。"""

from __future__ import annotations

from pathlib import Path

from fixtures import make_fixture_record


def test_rule_baseline_runs_on_fixture(tmp_path: Path):
    from analysis.records import load_record
    from analysis.rule_baseline import RULE_BASELINE_NAME, evaluate_rule_baseline, predict_record_with_rules

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    preds = predict_record_with_rules(record)

    assert len(preds) == len(record.frames())
    alarm_frames = [p for p in preds if p["is_picking"]]
    assert alarm_frames, "规则基线应在碰撞持续后触发报警"

    report = evaluate_rule_baseline([record], data_dir=str(fixture_dir))
    assert report.model_name == RULE_BASELINE_NAME
    assert report.extra["frame_count"] == len(record.frames())
    assert 0.0 <= report.picking.macro_f1 <= 1.0


def test_benchmark_includes_rule_baseline(tmp_path: Path):
    from analysis.benchmark import run_benchmark
    from analysis.rule_baseline import RULE_BASELINE_NAME

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    output_dir = tmp_path / "benchmark_out"
    result = run_benchmark(
        train_data_dir=fixture_dir,
        output_dir=output_dir,
        model_names=["sklearn_dummy"],
        jobs=1,
    )

    assert result.baseline_report is not None
    assert result.baseline_report.model_name == RULE_BASELINE_NAME
    assert (output_dir / RULE_BASELINE_NAME / "eval_report.json").is_file()

    baseline_rows = [row for row in result.comparison if row.get("is_baseline")]
    ml_rows = [row for row in result.comparison if not row.get("is_baseline")]
    assert len(baseline_rows) == 1
    assert len(ml_rows) == 1
    assert "beats_baseline" in ml_rows[0]

    report_md = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## 规则基线" in report_md
    assert RULE_BASELINE_NAME in report_md
