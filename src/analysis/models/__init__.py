"""模型基类与 sklearn 实现。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from analysis.box_layout import (
    BoxNumericCode,
    denormalize_layout_column,
    denormalize_layout_layer,
    record_layout_denorm_bounds,
    resolve_box_tokens_by_layout,
)
from analysis.dataset import Dataset


class _ArrayLGBMRegressor:
    def __init__(self, **kwargs: Any) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("需要安装 lightgbm，请运行: uv sync") from exc
        self._reg = LGBMRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Any = None) -> _ArrayLGBMRegressor:
        self._reg.fit(np.asarray(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._reg.predict(np.asarray(X))


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
    predicted_layout_layer_norm: float = 0.0
    predicted_layout_column_norm: float = 0.0
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


def _make_regressor(model_type: str, *, for_layout: bool = False) -> Pipeline:
    if model_type == "logistic":
        est = Ridge(alpha=1.0)
    elif model_type == "extra_trees":
        est = ExtraTreesRegressor(
            n_estimators=80 if for_layout else 120,
            max_depth=10 if for_layout else 12,
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        est = GradientBoostingRegressor(
            n_estimators=80 if for_layout else 120,
            max_depth=3,
            random_state=42,
        )
    elif model_type == "hist_gradient_boosting":
        est = HistGradientBoostingRegressor(
            max_iter=80 if for_layout else 120,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=42,
        )
    elif model_type == "ada_boost":
        est = GradientBoostingRegressor(
            n_estimators=60 if for_layout else 100,
            learning_rate=0.5,
            random_state=42,
        )
    elif model_type in ("svm_rbf", "linear_svm"):
        est = SVR(C=2.0 if model_type == "svm_rbf" else 1.0, gamma="scale")
    elif model_type == "knn":
        est = KNeighborsRegressor(n_neighbors=3, weights="distance")
    elif model_type == "decision_tree":
        est = DecisionTreeRegressor(
            max_depth=6 if for_layout else 8,
            min_samples_leaf=2,
            random_state=42,
        )
    elif model_type == "dummy":
        est = DummyRegressor(strategy="mean")
    elif model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("需要安装 xgboost，请运行: uv sync") from exc
        est = XGBRegressor(
            n_estimators=80 if for_layout else 120,
            max_depth=4 if for_layout else 6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=0,
        )
    elif model_type == "lightgbm":
        est = _ArrayLGBMRegressor(
            n_estimators=80 if for_layout else 120,
            max_depth=6 if for_layout else 8,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1,
        )
    elif model_type == "random_forest":
        est = RandomForestRegressor(
            n_estimators=80 if for_layout else 100,
            max_depth=10 if for_layout else 12,
            random_state=42,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("reg", est)])


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


def _fit_regressor(reg: Pipeline, x: np.ndarray, y: np.ndarray) -> Pipeline:
    if len(y) == 0:
        raise ValueError("训练标签为空")
    reg.fit(x, y)
    return reg


@dataclass
class SklearnPickingModel(PickingModel):
    """两阶段模型：帧级 is_picking + 帧级 layout_layer/column 归一化回归。"""

    model_type: str = "random_forest"
    picking_clf: Pipeline | None = None
    layout_layer_reg: Pipeline | None = None
    layout_column_reg: Pipeline | None = None
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
            if s.is_picking and s.target_layout_layer_norm > 0 and s.target_layout_column_norm > 0
        ]
        if layout_samples:
            x_layout = np.vstack([s.x for s in layout_samples])
            y_layer = np.array([float(s.target_layout_layer_norm) for s in layout_samples], dtype=np.float64)
            y_column = np.array([float(s.target_layout_column_norm) for s in layout_samples], dtype=np.float64)
            self.layout_layer_reg = _fit_regressor(
                _make_regressor(self.model_type, for_layout=True),
                x_layout,
                y_layer,
            )
            self.layout_column_reg = _fit_regressor(
                _make_regressor(self.model_type, for_layout=True),
                x_layout,
                y_column,
            )

    def predict_layout_norm(self, x: np.ndarray) -> tuple[float, float]:
        if self.layout_layer_reg is None or self.layout_column_reg is None:
            return 0.0, 0.0
        x2 = x.reshape(1, -1)
        layer_norm = float(self.layout_layer_reg.predict(x2)[0])
        column_norm = float(self.layout_column_reg.predict(x2)[0])
        return max(0.0, layer_norm), max(0.0, column_norm)

    def predict_layout(
        self,
        x: np.ndarray,
        *,
        box_layout: dict[str, BoxNumericCode] | None = None,
    ) -> tuple[float, float, int, int]:
        layer_norm, column_norm = self.predict_layout_norm(x)
        if not box_layout:
            return layer_norm, column_norm, 0, 0
        max_layer, max_column = record_layout_denorm_bounds(box_layout)
        layer = denormalize_layout_layer(layer_norm, max_layer)
        column = denormalize_layout_column(column_norm, max_column)
        return layer_norm, column_norm, layer, column

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

        layer_norm = 0.0
        column_norm = 0.0
        predicted_layer = 0
        predicted_column = 0
        predicted_tokens: list[str] = []
        if is_picking and self.layout_layer_reg is not None and self.layout_column_reg is not None:
            layer_norm, column_norm, predicted_layer, predicted_column = self.predict_layout(
                x,
                box_layout=box_layout,
            )
            if box_layout and predicted_layer > 0 and predicted_column > 0:
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
            predicted_layout_layer_norm=layer_norm,
            predicted_layout_column_norm=column_norm,
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
            "stage2_targets": ["target_layout_layer_norm", "target_layout_column_norm"],
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(self.picking_clf, path / "picking_clf.pkl")
        if self.layout_layer_reg is not None:
            joblib.dump(self.layout_layer_reg, path / "layout_layer_reg.pkl")
        if self.layout_column_reg is not None:
            joblib.dump(self.layout_column_reg, path / "layout_column_reg.pkl")

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
        layer_path = path / "layout_layer_reg.pkl"
        column_path = path / "layout_column_reg.pkl"
        if not layer_path.is_file():
            layer_path = path / "layout_layer_clf.pkl"
        if not column_path.is_file():
            column_path = path / "layout_column_clf.pkl"
        model.layout_layer_reg = joblib.load(layer_path) if layer_path.is_file() else None
        model.layout_column_reg = joblib.load(column_path) if column_path.is_file() else None
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
