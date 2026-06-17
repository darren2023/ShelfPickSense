from pathlib import Path

import pytest

from fixtures import make_fixture_record


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    return make_fixture_record(tmp_path / "record_001")


def test_load_and_train_eval(fixture_data_dir: Path, tmp_path: Path):
    from analysis.evaluation import evaluate_model
    from analysis.records import load_record
    from analysis.train import train_model

    record = load_record(fixture_data_dir)
    assert record.record_id == "record_001"
    assert len(record.box_tokens) == 2
    assert record.labels.frame_labels[6].is_picking
    assert record.labels.frame_labels[6].confirmed_box_tokens == ["S1:A1"]

    model_dir = tmp_path / "model"
    result = train_model(fixture_data_dir, model_dir)
    assert result.frame_count == 10
    assert result.positive_frames == 3

    report = evaluate_model(model_dir, fixture_data_dir)
    assert report.picking.f1 >= 0.5
    assert report.extra["frame_count"] == 10


def test_no_event_review_all_negative(tmp_path: Path):
    from analysis.labels import build_labels_from_event_review
    from analysis.records import load_record
    from fixtures import make_fixture_record

    record_dir = make_fixture_record(tmp_path / "no_review")
    (record_dir / "event_review.json").unlink()

    record = load_record(record_dir)
    assert all(not lbl.is_picking for lbl in record.labels.frame_labels.values())


def test_feature_registry_extensible(fixture_data_dir: Path):
    from analysis.features.base import FeatureContext, FeatureExtractor
    from analysis.features.registry import FeatureRegistry, default_registry
    from analysis.records import load_record

    class DummyExtractor(FeatureExtractor):
        name = "dummy"

        def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
            return {"value": float(ctx.frame.frame_idx)}

    reg = default_registry()
    reg.register(DummyExtractor())
    record = load_record(fixture_data_dir)
    frame = record.frames()[0]
    feat = reg.extract_frame_features(record, frame)
    assert "dummy.value" in feat.features


def test_benchmark_runs_multiple_models(fixture_data_dir: Path, tmp_path: Path):
    from analysis.benchmark import run_benchmark

    output_dir = tmp_path / "benchmark"
    result = run_benchmark(
        train_data_dir=fixture_data_dir,
        output_dir=output_dir,
        model_names=["sklearn_rf", "sklearn_logistic"],
    )

    assert [r.model_name for r in result.reports] == ["sklearn_rf", "sklearn_logistic"]
    assert len(result.comparison) == 2
    assert (output_dir / "sklearn_rf" / "eval_report.json").is_file()
    assert (output_dir / "sklearn_logistic" / "eval_report.json").is_file()
    assert (output_dir / "benchmark_summary.json").is_file()
