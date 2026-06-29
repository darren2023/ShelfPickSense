"""模型基类与 sklearn 实现。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis.dataset import Dataset


class _ArrayLGBMClassifier:
    """LightGBM 包装：始终用 numpy 数组训练/预测，避免 Pipeline 特征名警告。"""

    def __init__(self, **kwargs: Any) -> None:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("需要安装 lightgbm，请运行: uv sync") from exc
        self._clf = LGBMClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Any = None) -> _ArrayLGBMClassifier:
        self._clf.fit(np.asarray(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(np.asarray(X))

    @property
    def classes_(self) -> np.ndarray:
        return self._clf.classes_


SUPPORTED_MODEL_NAMES = [
    "sklearn_rf",
    "sklearn_logistic",
    "sklearn_extra_trees",
    "sklearn_gradient_boosting",
    "sklearn_hist_gradient_boosting",
    "sklearn_ada_boost",
    "sklearn_svm_rbf",
    "sklearn_linear_svm",
    "sklearn_knn",
    "sklearn_decision_tree",
    "sklearn_dummy",
    "xgboost",
    "lightgbm",
]


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


def _make_classifier(model_type: str, *, for_box: bool = False) -> Pipeline:
    if model_type == "logistic":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "extra_trees":
        est = ExtraTreesClassifier(
            n_estimators=80 if for_box else 120,
            max_depth=10 if for_box else 12,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        est = GradientBoostingClassifier(
            n_estimators=80 if for_box else 120,
            max_depth=3,
            random_state=42,
        )
    elif model_type == "hist_gradient_boosting":
        est = HistGradientBoostingClassifier(
            max_iter=80 if for_box else 120,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=42,
        )
    elif model_type == "ada_boost":
        est = AdaBoostClassifier(
            n_estimators=60 if for_box else 100,
            learning_rate=0.5,
            random_state=42,
        )
    elif model_type == "svm_rbf":
        est = SVC(
            C=2.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "linear_svm":
        est = SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "knn":
        est = KNeighborsClassifier(n_neighbors=3, weights="distance")
    elif model_type == "decision_tree":
        est = DecisionTreeClassifier(
            max_depth=6 if for_box else 8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "dummy":
        est = DummyClassifier(strategy="prior")
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("需要安装 xgboost，请运行: uv sync") from exc
        est = XGBClassifier(
            n_estimators=80 if for_box else 120,
            max_depth=4 if for_box else 6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    elif model_type == "lightgbm":
        est = _ArrayLGBMClassifier(
            n_estimators=80 if for_box else 120,
            max_depth=6 if for_box else 8,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
    elif model_type == "random_forest":
        est = RandomForestClassifier(
            n_estimators=80 if for_box else 100,
            max_depth=10 if for_box else 12,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def _positive_probability(clf: Pipeline, x: np.ndarray) -> float:
    if not hasattr(clf, "predict_proba"):
        if hasattr(clf, "decision_function"):
            score = np.ravel(clf.decision_function(x))[0]
            return float(1.0 / (1.0 + np.exp(-score)))
        return float(clf.predict(x)[0])
    probabilities = clf.predict_proba(x)[0]
    classes = list(getattr(clf, "classes_", []))
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    return 0.0


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
        self.picking_clf = _make_classifier(self.model_type)
        if len(y_pick) > 0 and len(np.unique(y_pick)) > 1:
            self.picking_clf.fit(x_pick, y_pick)
        elif len(y_pick) > 0:
            self.picking_clf.fit(x_pick, y_pick)
        else:
            raise ValueError("训练集无帧样本")

        x_box = np.vstack([s.x for s in dataset.box_samples]) if dataset.box_samples else np.empty((0, 0))
        y_box = np.array([int(s.is_target) for s in dataset.box_samples], dtype=np.int32)
        self.box_clf = _make_classifier(self.model_type, for_box=True)
        if len(y_box) > 0 and len(np.unique(y_box)) > 1:
            self.box_clf.fit(x_box, y_box)
        elif len(y_box) > 0:
            self.box_clf.fit(x_box, y_box)

    def predict_frame(self, x: np.ndarray, *, record_id: str, frame_idx: int) -> PickingPrediction:
        if self.picking_clf is None:
            raise RuntimeError("模型尚未训练")
        x2 = x.reshape(1, -1)
        prob = _positive_probability(self.picking_clf, x2)
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
            prob = _positive_probability(self.box_clf, x2)
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
    name: SklearnPickingModel for name in SUPPORTED_MODEL_NAMES
}


def create_model(model_name: str, **kwargs: Any) -> SklearnPickingModel:
    model_name = model_name or "sklearn_rf"
    model_types = {
        "sklearn_rf": "random_forest",
        "sklearn_logistic": "logistic",
        "sklearn_extra_trees": "extra_trees",
        "sklearn_gradient_boosting": "gradient_boosting",
        "sklearn_hist_gradient_boosting": "hist_gradient_boosting",
        "sklearn_ada_boost": "ada_boost",
        "sklearn_svm_rbf": "svm_rbf",
        "sklearn_linear_svm": "linear_svm",
        "sklearn_knn": "knn",
        "sklearn_decision_tree": "decision_tree",
        "sklearn_dummy": "dummy",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
    }
    if model_name not in model_types:
        raise ValueError(f"未知模型: {model_name}，可用模型: {', '.join(SUPPORTED_MODEL_NAMES)}")
    return SklearnPickingModel(model_type=model_types[model_name], name=model_name, **kwargs)
