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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from analysis.box_layout import BoxNumericCode, resolve_box_tokens_by_layout
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
    predicted_layout_layer: int = 0
    predicted_layout_column: int = 0
    predicted_box_tokens: list[str] = field(default_factory=list)


class PickingModel(ABC):
    """取货检测 + 货框布局识别模型接口。"""

    name: str = "base"

    @abstractmethod
    def fit(self, dataset: Dataset) -> None: ...

    @abstractmethod
    def predict_frame(
        self,
        x: np.ndarray,
        *,
        record_id: str,
        frame_idx: int,
        box_layout: dict[str, BoxNumericCode] | None = None,
    ) -> PickingPrediction: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> PickingModel: ...


def _make_classifier(model_type: str, *, for_layout: bool = False) -> Pipeline:
    if model_type == "logistic":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "extra_trees":
        est = ExtraTreesClassifier(
            n_estimators=80 if for_layout else 120,
            max_depth=10 if for_layout else 12,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        est = GradientBoostingClassifier(
            n_estimators=80 if for_layout else 120,
            max_depth=3,
            random_state=42,
        )
    elif model_type == "hist_gradient_boosting":
        est = HistGradientBoostingClassifier(
            max_iter=80 if for_layout else 120,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=42,
        )
    elif model_type == "ada_boost":
        est = AdaBoostClassifier(
            n_estimators=60 if for_layout else 100,
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
            max_depth=6 if for_layout else 8,
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
            n_estimators=80 if for_layout else 120,
            max_depth=4 if for_layout else 6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    elif model_type == "lightgbm":
        est = _ArrayLGBMClassifier(
            n_estimators=80 if for_layout else 120,
            max_depth=6 if for_layout else 8,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
    elif model_type == "random_forest":
        est = RandomForestClassifier(
            n_estimators=80 if for_layout else 100,
            max_depth=10 if for_layout else 12,
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


def _fit_classifier(clf: Pipeline, x: np.ndarray, y: np.ndarray) -> Pipeline:
    if len(y) == 0:
        raise ValueError("训练标签为空")
    unique = np.unique(y)
    if len(unique) < 2:
        constant = int(unique[0])
        safe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="constant", constant=constant)),
        ])
        safe.fit(x, y)
        return safe
    clf.fit(x, y)
    return clf


@dataclass
class SklearnPickingModel(PickingModel):
    """两阶段模型：帧级 is_picking 检测 + 帧级 layout_layer/layout_column 预测。"""

    model_type: str = "random_forest"
    picking_clf: Pipeline | None = None
    layout_layer_clf: Pipeline | None = None
    layout_column_clf: Pipeline | None = None
    layout_layer_encoder: LabelEncoder | None = None
    layout_column_encoder: LabelEncoder | None = None
    frame_feature_names: list[str] = field(default_factory=list)
    box_feature_names: list[str] = field(default_factory=list)
    name: str = "sklearn_two_stage"

    def fit(self, dataset: Dataset) -> None:
        self.frame_feature_names = list(dataset.frame_feature_names)
        self.box_feature_names = list(dataset.box_feature_names)

        x_pick = np.vstack([s.x for s in dataset.frame_samples]) if dataset.frame_samples else np.empty((0, 0))
        y_pick = np.array([int(s.is_picking) for s in dataset.frame_samples], dtype=np.int32)
        if len(y_pick) == 0:
            raise ValueError("训练集无帧样本")
        self.picking_clf = _fit_classifier(_make_classifier(self.model_type), x_pick, y_pick)

        layout_samples = [
            s
            for s in dataset.frame_samples
            if s.is_picking and s.target_layout_layer > 0 and s.target_layout_column > 0
        ]
        if layout_samples:
            x_layout = np.vstack([s.x for s in layout_samples])
            y_layer = np.array([int(s.target_layout_layer) for s in layout_samples], dtype=np.int32)
            y_column = np.array([int(s.target_layout_column) for s in layout_samples], dtype=np.int32)
            self.layout_layer_encoder = LabelEncoder()
            self.layout_column_encoder = LabelEncoder()
            y_layer_enc = self.layout_layer_encoder.fit_transform(y_layer)
            y_column_enc = self.layout_column_encoder.fit_transform(y_column)
            self.layout_layer_clf = _fit_classifier(
                _make_classifier(self.model_type, for_layout=True),
                x_layout,
                y_layer_enc,
            )
            self.layout_column_clf = _fit_classifier(
                _make_classifier(self.model_type, for_layout=True),
                x_layout,
                y_column_enc,
            )

    def predict_layout(self, x: np.ndarray) -> tuple[int, int]:
        if self.layout_layer_clf is None or self.layout_column_clf is None:
            return 0, 0
        if self.layout_layer_encoder is None or self.layout_column_encoder is None:
            return 0, 0
        x2 = x.reshape(1, -1)
        layer_enc = int(self.layout_layer_clf.predict(x2)[0])
        column_enc = int(self.layout_column_clf.predict(x2)[0])
        layer = int(self.layout_layer_encoder.inverse_transform([layer_enc])[0])
        column = int(self.layout_column_encoder.inverse_transform([column_enc])[0])
        return layer, column

    def predict_frame(
        self,
        x: np.ndarray,
        *,
        record_id: str,
        frame_idx: int,
        box_layout: dict[str, BoxNumericCode] | None = None,
    ) -> PickingPrediction:
        if self.picking_clf is None:
            raise RuntimeError("模型尚未训练")
        x2 = x.reshape(1, -1)
        prob = _positive_probability(self.picking_clf, x2)
        is_picking = bool(self.picking_clf.predict(x2)[0])

        predicted_layer = 0
        predicted_column = 0
        predicted_tokens: list[str] = []
        if is_picking and self.layout_layer_clf is not None and self.layout_column_clf is not None:
            predicted_layer, predicted_column = self.predict_layout(x)
            if box_layout:
                predicted_tokens = resolve_box_tokens_by_layout(
                    box_layout,
                    layer=predicted_layer,
                    column=predicted_column,
                )

        return PickingPrediction(
            record_id=record_id,
            frame_idx=frame_idx,
            is_picking=is_picking,
            picking_prob=prob,
            predicted_layout_layer=predicted_layer,
            predicted_layout_column=predicted_column,
            predicted_box_tokens=predicted_tokens,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": 2,
            "model_type": self.model_type,
            "name": self.name,
            "frame_feature_names": self.frame_feature_names,
            "box_feature_names": self.box_feature_names,
            "stage1_target": "is_picking",
            "stage2_targets": ["target_layout_layer", "target_layout_column"],
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(self.picking_clf, path / "picking_clf.pkl")
        if self.layout_layer_clf is not None:
            joblib.dump(self.layout_layer_clf, path / "layout_layer_clf.pkl")
        if self.layout_column_clf is not None:
            joblib.dump(self.layout_column_clf, path / "layout_column_clf.pkl")
        if self.layout_layer_encoder is not None:
            joblib.dump(self.layout_layer_encoder, path / "layout_layer_encoder.pkl")
        if self.layout_column_encoder is not None:
            joblib.dump(self.layout_column_encoder, path / "layout_column_encoder.pkl")

    @classmethod
    def load(cls, path: Path) -> SklearnPickingModel:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        model = cls(
            model_type=meta.get("model_type", "random_forest"),
            frame_feature_names=list(meta.get("frame_feature_names") or []),
            box_feature_names=list(meta.get("box_feature_names") or []),
            name=meta.get("name", "sklearn_two_stage"),
        )
        model.picking_clf = joblib.load(path / "picking_clf.pkl")
        layer_path = path / "layout_layer_clf.pkl"
        column_path = path / "layout_column_clf.pkl"
        model.layout_layer_clf = joblib.load(layer_path) if layer_path.is_file() else None
        model.layout_column_clf = joblib.load(column_path) if column_path.is_file() else None
        layer_enc_path = path / "layout_layer_encoder.pkl"
        column_enc_path = path / "layout_column_encoder.pkl"
        model.layout_layer_encoder = joblib.load(layer_enc_path) if layer_enc_path.is_file() else None
        model.layout_column_encoder = joblib.load(column_enc_path) if column_enc_path.is_file() else None
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
