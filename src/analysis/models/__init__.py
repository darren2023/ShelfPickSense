"""模型基类与 sklearn 实现。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.dataset import Dataset


@dataclass
class PickingPrediction:
    record_id: str
    frame_idx: int
    is_picking: bool
    picking_prob: float
    predicted_box_tokens: list[str] = field(default_factory=list)


class PickingModel(ABC):
    """取货检测 + 货框识别模型接口。"""

    name: str = "base"

    @abstractmethod
    def fit(self, dataset: Dataset) -> None: ...

    @abstractmethod
    def predict_frame(self, x: np.ndarray, *, record_id: str, frame_idx: int) -> PickingPrediction: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> PickingModel: ...


def _make_picking_clf(model_type: str) -> Pipeline:
    if model_type == "logistic":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        est = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def _make_box_clf(model_type: str) -> Pipeline:
    if model_type == "logistic":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        est = RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


@dataclass
class SklearnPickingModel(PickingModel):
    """两阶段模型：帧级取货检测 + 货框二分类（正样本帧内）。"""

    model_type: str = "random_forest"
    picking_clf: Pipeline | None = None
    box_clf: Pipeline | None = None
    frame_feature_names: list[str] = field(default_factory=list)
    box_feature_names: list[str] = field(default_factory=list)
    box_score_threshold: float = 0.5
    name: str = "sklearn_two_stage"

    def fit(self, dataset: Dataset) -> None:
        self.frame_feature_names = list(dataset.frame_feature_names)
        self.box_feature_names = list(dataset.box_feature_names)

        x_pick = np.vstack([s.x for s in dataset.frame_samples]) if dataset.frame_samples else np.empty((0, 0))
        y_pick = np.array([int(s.is_picking) for s in dataset.frame_samples], dtype=np.int32)
        self.picking_clf = _make_picking_clf(self.model_type)
        if len(y_pick) > 0 and len(np.unique(y_pick)) > 1:
            self.picking_clf.fit(x_pick, y_pick)
        elif len(y_pick) > 0:
            self.picking_clf.fit(x_pick, y_pick)
        else:
            raise ValueError("训练集无帧样本")

        x_box = np.vstack([s.x for s in dataset.box_samples]) if dataset.box_samples else np.empty((0, 0))
        y_box = np.array([int(s.is_target) for s in dataset.box_samples], dtype=np.int32)
        self.box_clf = _make_box_clf(self.model_type)
        if len(y_box) > 0 and len(np.unique(y_box)) > 1:
            self.box_clf.fit(x_box, y_box)
        elif len(y_box) > 0:
            self.box_clf.fit(x_box, y_box)

    def predict_frame(self, x: np.ndarray, *, record_id: str, frame_idx: int) -> PickingPrediction:
        if self.picking_clf is None:
            raise RuntimeError("模型尚未训练")
        x2 = x.reshape(1, -1)
        prob = float(self.picking_clf.predict_proba(x2)[0, 1]) if hasattr(self.picking_clf, "predict_proba") else 0.0
        is_picking = bool(self.picking_clf.predict(x2)[0])
        return PickingPrediction(
            record_id=record_id,
            frame_idx=frame_idx,
            is_picking=is_picking,
            picking_prob=prob,
            predicted_box_tokens=[],
        )

    def predict_boxes_for_frame(
        self,
        box_features: list[tuple[str, np.ndarray]],
    ) -> list[str]:
        if self.box_clf is None or not box_features:
            return []
        tokens: list[str] = []
        for token, x in box_features:
            x2 = x.reshape(1, -1)
            if hasattr(self.box_clf, "predict_proba"):
                prob = float(self.box_clf.predict_proba(x2)[0, 1])
            else:
                prob = float(self.box_clf.predict(x2)[0])
            if prob >= self.box_score_threshold:
                tokens.append(token)
        return tokens

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": self.model_type,
            "name": self.name,
            "frame_feature_names": self.frame_feature_names,
            "box_feature_names": self.box_feature_names,
            "box_score_threshold": self.box_score_threshold,
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(self.picking_clf, path / "picking_clf.pkl")
        joblib.dump(self.box_clf, path / "box_clf.pkl")

    @classmethod
    def load(cls, path: Path) -> SklearnPickingModel:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        model = cls(
            model_type=meta.get("model_type", "random_forest"),
            frame_feature_names=list(meta.get("frame_feature_names") or []),
            box_feature_names=list(meta.get("box_feature_names") or []),
            box_score_threshold=float(meta.get("box_score_threshold", 0.5)),
            name=meta.get("name", "sklearn_two_stage"),
        )
        model.picking_clf = joblib.load(path / "picking_clf.pkl")
        box_path = path / "box_clf.pkl"
        model.box_clf = joblib.load(box_path) if box_path.is_file() else None
        return model


MODEL_REGISTRY: dict[str, type[SklearnPickingModel]] = {
    "sklearn_rf": SklearnPickingModel,
    "sklearn_logistic": SklearnPickingModel,
}


def create_model(model_name: str, **kwargs: Any) -> SklearnPickingModel:
    if model_name == "sklearn_logistic":
        return SklearnPickingModel(model_type="logistic", name=model_name, **kwargs)
    return SklearnPickingModel(model_type="random_forest", name=model_name or "sklearn_rf", **kwargs)
