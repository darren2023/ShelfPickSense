"""Optuna 调参测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import make_fixture_record


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_evaluate_params_cv(tmp_path: Path, model_name: str):
    from analysis.dataset import build_dataset
    from analysis.features.registry import default_registry
    from analysis.records import load_record
    from analysis.tuning import evaluate_params_cv, suggest_xgboost_params

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    record = load_record(fixture_dir)
    dataset = build_dataset([record], default_registry())

    class _Trial:
        def suggest_int(self, name, low, high, step=1):
            return (low + high) // 2

        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2.0

    params = suggest_xgboost_params(_Trial()) if model_name == "xgboost" else None
    if model_name == "lightgbm":
        from analysis.tuning import suggest_lightgbm_params

        params = suggest_lightgbm_params(_Trial())

    combined, picking, box = evaluate_params_cv(
        dataset,
        model_name=model_name,
        params=params,
        cv_folds=2,
    )
    assert 0.0 <= combined <= 1.0
    assert 0.0 <= picking <= 1.0
    assert box is not None
    assert 0.0 <= box <= 1.0


def test_tune_model_runs(tmp_path: Path):
    from analysis.tuning import tune_model

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    output_dir = tmp_path / "tuned_xgb"
    result = tune_model(
        fixture_dir,
        output_dir,
        model_name="xgboost",
        n_trials=2,
        cv_folds=2,
    )

    assert result.best_params
    assert result.best_value >= 0.0
    assert (output_dir / "tune_result.json").is_file()
    assert (output_dir / "optuna_trials.json").is_file()
    assert (output_dir / "meta.json").is_file()

    meta = __import__("json").loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("clf_params") == result.best_params


def test_cli_tune(tmp_path: Path):
    from analysis.cli import main

    fixture_dir = make_fixture_record(tmp_path / "record_001")
    output_dir = tmp_path / "cli_tuned"
    assert (
        main(
            [
                "tune",
                "--data-dir",
                str(fixture_dir),
                "--output",
                str(output_dir),
                "--model",
                "lightgbm",
                "--trials",
                "2",
                "--cv-folds",
                "2",
            ]
        )
        == 0
    )
    assert (output_dir / "tune_result.json").is_file()
