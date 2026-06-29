"""Optuna 超参数搜索（初步支持 xgboost / lightgbm）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from loguru import logger
from sklearn.model_selection import GroupKFold

from analysis.dataset import Dataset, build_dataset, filter_empty_skeleton_frames
from analysis.evaluation import compute_picking_metrics
from analysis.features.registry import FeatureRegistry, default_registry
from analysis.features.selection import FeatureSelection
from analysis.models import TUNABLE_MODEL_NAMES, _make_classifier, resolve_model_type
from analysis.records import RecordData, load_all_records
from analysis.train import TrainResult, train_model_from_dataset

COMBINED_METRIC_NAME = "combined_macro_f1"
PICKING_WEIGHT = 0.7
BOX_WEIGHT = 0.3


@dataclass
class TuneResult:
    model_name: str
    model_path: str
    data_dir: str
    best_params: dict[str, Any]
    best_value: float
    metric: str
    n_trials: int
    cv_folds: int
    picking_cv_score: float
    box_cv_score: float | None
    train_result: dict[str, Any]
    tuned_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 20.0, log=True)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": pos_weight,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }


def suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42,
        "verbosity": -1,
    }


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "xgboost":
        return suggest_xgboost_params(trial)
    if model_name == "lightgbm":
        return suggest_lightgbm_params(trial)
    raise ValueError(f"模型 {model_name} 暂不支持 Optuna 调参，可用: {', '.join(TUNABLE_MODEL_NAMES)}")


def _group_codes(values: list[str]) -> np.ndarray:
    _, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int32)


def _effective_cv_folds(groups: np.ndarray, cv_folds: int) -> int:
    unique_groups = len(np.unique(groups))
    if unique_groups < 2:
        return 1
    return max(2, min(cv_folds, unique_groups))


def _cv_binary_macro_f1(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_type: str,
    params: dict[str, Any],
    cv_folds: int,
    for_box: bool,
) -> float:
    if len(y) == 0:
        return 0.0
    if len(np.unique(y)) < 2:
        return float(y.mean())

    n_splits = _effective_cv_folds(groups, cv_folds)
    if n_splits <= 1:
        clf = _make_classifier(model_type, for_box=for_box, params=params)
        clf.fit(x, y)
        pred = clf.predict(x)
        return compute_picking_metrics(y.astype(bool).tolist(), pred.astype(bool).tolist()).macro_f1

    splitter = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, val_idx in splitter.split(x, y, groups):
        clf = _make_classifier(model_type, for_box=for_box, params=params)
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[val_idx])
        metrics = compute_picking_metrics(
            y[val_idx].astype(bool).tolist(),
            pred.astype(bool).tolist(),
        )
        scores.append(metrics.macro_f1)
    return float(np.mean(scores))


def evaluate_params_cv(
    dataset: Dataset,
    *,
    model_name: str,
    params: dict[str, Any],
    cv_folds: int = 5,
) -> tuple[float, float, float | None]:
    """返回 combined_score, picking_macro_f1, box_macro_f1。"""
    model_type = resolve_model_type(model_name)

    x_pick = np.vstack([s.x for s in dataset.frame_samples])
    y_pick = np.array([int(s.is_picking) for s in dataset.frame_samples], dtype=np.int32)
    groups_pick = _group_codes([s.record_id for s in dataset.frame_samples])
    picking_score = _cv_binary_macro_f1(
        x_pick,
        y_pick,
        groups_pick,
        model_type=model_type,
        params=params,
        cv_folds=cv_folds,
        for_box=False,
    )

    box_score: float | None = None
    if dataset.box_samples:
        x_box = np.vstack([s.x for s in dataset.box_samples])
        y_box = np.array([int(s.is_target) for s in dataset.box_samples], dtype=np.int32)
        groups_box = _group_codes([s.record_id for s in dataset.box_samples])
        box_score = _cv_binary_macro_f1(
            x_box,
            y_box,
            groups_box,
            model_type=model_type,
            params=params,
            cv_folds=cv_folds,
            for_box=True,
        )
        combined = PICKING_WEIGHT * picking_score + BOX_WEIGHT * box_score
    else:
        combined = picking_score

    return combined, picking_score, box_score


def run_optuna_tune(
    dataset: Dataset,
    *,
    model_name: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    timeout: float | None = None,
    seed: int = 42,
    study_name: str | None = None,
) -> tuple[dict[str, Any], float, float, float | None, optuna.Study]:
    if model_name not in TUNABLE_MODEL_NAMES:
        raise ValueError(f"模型 {model_name} 暂不支持 Optuna 调参，可用: {', '.join(TUNABLE_MODEL_NAMES)}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name or f"{model_name}_tuning",
        sampler=sampler,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        combined, picking_score, box_score = evaluate_params_cv(
            dataset,
            model_name=model_name,
            params=params,
            cv_folds=cv_folds,
        )
        trial.set_user_attr("picking_macro_f1", picking_score)
        if box_score is not None:
            trial.set_user_attr("box_macro_f1", box_score)
        return combined

    logger.info(
        "开始 Optuna 调参: model={}, trials={}, cv_folds={}, frames={}, box_samples={}",
        model_name,
        n_trials,
        cv_folds,
        dataset.frame_count,
        len(dataset.box_samples),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    if study.best_trial is None:
        raise RuntimeError("Optuna 未产生有效 trial")

    best_params = dict(study.best_params)
    best_value = float(study.best_value)
    picking_score = float(study.best_trial.user_attrs.get("picking_macro_f1", best_value))
    box_score_raw = study.best_trial.user_attrs.get("box_macro_f1")
    box_score = float(box_score_raw) if box_score_raw is not None else None
    logger.info(
        "Optuna 调参完成: best_value={:.4f}, picking_cv={:.4f}, box_cv={}",
        best_value,
        picking_score,
        f"{box_score:.4f}" if box_score is not None else "N/A",
    )
    return best_params, best_value, picking_score, box_score, study


def tune_model(
    data_dir: Path,
    output_dir: Path,
    *,
    model_name: str = "xgboost",
    n_trials: int = 50,
    cv_folds: int = 5,
    timeout: float | None = None,
    seed: int = 42,
    registry: FeatureRegistry | None = None,
    feature_selection: FeatureSelection | None = None,
    filter_empty_skeleton: bool = True,
) -> TuneResult:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_records(data_dir)
    reg = registry or default_registry()
    dataset = build_dataset(records, reg, feature_selection=feature_selection)
    skipped = 0
    if filter_empty_skeleton:
        dataset, skipped = filter_empty_skeleton_frames(dataset, records)

    best_params, best_value, picking_cv, box_cv, study = run_optuna_tune(
        dataset,
        model_name=model_name,
        n_trials=n_trials,
        cv_folds=cv_folds,
        timeout=timeout,
        seed=seed,
    )

    train_result, _ = train_model_from_dataset(
        dataset,
        records=records,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        skipped_empty_skeleton_frames=skipped,
        clf_params=best_params,
    )

    tune_result = TuneResult(
        model_name=model_name,
        model_path=str(output_dir.resolve()),
        data_dir=str(data_dir.resolve()),
        best_params=best_params,
        best_value=best_value,
        metric=COMBINED_METRIC_NAME,
        n_trials=len(study.trials),
        cv_folds=cv_folds,
        picking_cv_score=picking_cv,
        box_cv_score=box_cv,
        train_result=train_result.to_dict(),
        tuned_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "tune_result.json").write_text(
        json.dumps(tune_result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_study_summary(study, output_dir / "optuna_trials.json")
    logger.info("调参结果已保存: {}", output_dir / "tune_result.json")
    return tune_result


def _save_study_summary(study: optuna.Study, output_path: Path) -> None:
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "picking_macro_f1": trial.user_attrs.get("picking_macro_f1"),
            "box_macro_f1": trial.user_attrs.get("box_macro_f1"),
        }
        rows.append(row)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
