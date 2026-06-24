"""多特征 benchmark 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.feature_benchmark import (
    FeatureBenchmarkSetSpec,
    _resolve_config_path,
    resolve_feature_selection,
)


def test_resolve_config_path_supports_project_root_and_plan_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = tmp_path / "project"
    plan_dir = project_root / "configs"
    plan_dir.mkdir(parents=True)
    feature_file = plan_dir / "selected_features.example.json"
    feature_file.write_text('{"frame_features": ["a"]}', encoding="utf-8")
    monkeypatch.chdir(project_root)

    by_plan_dir = _resolve_config_path("selected_features.example.json", base_dir=plan_dir)
    assert by_plan_dir == feature_file

    by_project_root = _resolve_config_path("configs/selected_features.example.json", base_dir=plan_dir)
    assert by_project_root.resolve() == feature_file.resolve()


def test_resolve_config_path_raises_with_tried_paths(tmp_path: Path):
    plan_dir = tmp_path / "configs"
    plan_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="已尝试"):
        _resolve_config_path("missing.json", base_dir=plan_dir)


def test_resolve_feature_selection_uses_feature_config_file(tmp_path: Path):
    config_path = tmp_path / "features.json"
    config_path.write_text(
        json.dumps({"frame_features": ["a"], "box_features": ["x"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    selection = resolve_feature_selection(
        FeatureBenchmarkSetSpec(name="set1", feature_config=str(config_path)),
        base_dir=tmp_path,
    )
    assert selection is not None
    assert selection.frame_features == ["a"]
    assert selection.box_features == ["x"]
