"""特征选择配置测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.features.selection import FeatureSelection, load_feature_selection


def test_feature_selection_keeps_order_and_filters():
    selection = FeatureSelection(
        frame_features=["b", "a"],
        box_features=["y"],
    )
    assert selection.select_frame(["a", "b", "c"]) == ["b", "a"]
    assert selection.select_box(["x", "y"]) == ["y"]


def test_feature_selection_none_uses_all():
    selection = FeatureSelection(frame_features=None, box_features=None)
    available = ["a", "b"]
    assert selection.select_frame(available) == available
    assert selection.select_box(available) == available


def test_feature_selection_rejects_unknown_names():
    selection = FeatureSelection(frame_features=["missing"])
    with pytest.raises(ValueError, match="frame_features 包含未知特征"):
        selection.select_frame(["a", "b"])


def test_load_feature_selection_supports_aliases(tmp_path: Path):
    config_path = tmp_path / "features.json"
    config_path.write_text(
        json.dumps({"frame": ["a"], "box": ["x"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    selection = load_feature_selection(config_path)
    assert selection is not None
    assert selection.frame_features == ["a"]
    assert selection.box_features == ["x"]
    assert selection.source_path.endswith("features.json")


def test_load_feature_selection_empty_path_returns_none():
    assert load_feature_selection(None) is None
    assert load_feature_selection("") is None
