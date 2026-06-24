"""特征选择配置：通过 JSON 文件指定训练/导出使用的特征子集。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureSelection:
    """帧级与货框级特征名白名单。"""

    frame_features: list[str] | None = None
    box_features: list[str] | None = None
    source_path: str = ""

    def select_frame(self, available: list[str]) -> list[str]:
        return _select_names(available, self.frame_features, "frame_features")

    def select_box(self, available: list[str]) -> list[str]:
        return _select_names(available, self.box_features, "box_features")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "frame_features": self.frame_features,
            "box_features": self.box_features,
        }


def _select_names(available: list[str], selected: list[str] | None, key: str) -> list[str]:
    if selected is None:
        return list(available)
    available_set = set(available)
    missing = [name for name in selected if name not in available_set]
    if missing:
        raise ValueError(f"{key} 包含未知特征: {', '.join(missing)}")
    return list(selected)


def _optional_name_list(data: dict[str, Any], *keys: str) -> list[str] | None:
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} 必须是字符串列表")
        return list(dict.fromkeys(value))
    return None


def load_feature_selection(path: str | Path | None) -> FeatureSelection | None:
    """从 JSON 配置文件加载特征选择。"""

    if not path:
        return None
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("特征选择配置必须是 JSON 对象")
    return FeatureSelection(
        frame_features=_optional_name_list(data, "frame_features", "frame"),
        box_features=_optional_name_list(data, "box_features", "box"),
        source_path=str(config_path.resolve()),
    )
