"""模型训练。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from analysis.dataset import build_dataset
from analysis.features.registry import FeatureRegistry, default_registry
from analysis.models import create_model
from analysis.records import load_all_records


@dataclass
class TrainResult:
    model_name: str
    model_path: str
    data_dir: str
    record_ids: list[str]
    frame_count: int
    positive_frames: int
    box_samples: int
    trained_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def train_model(
    data_dir: Path,
    output_dir: Path,
    *,
    model_name: str = "sklearn_rf",
    registry: FeatureRegistry | None = None,
) -> TrainResult:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    logger.info("加载训练数据: {}", data_dir)
    records = load_all_records(data_dir)
    logger.info("训练记录加载完成: records={}", len(records))
    reg = registry or default_registry()
    logger.debug("开始构建训练样本")
    dataset = build_dataset(records, reg)
    logger.info(
        "训练样本构建完成: frames={}, positive_frames={}, box_samples={}",
        dataset.frame_count,
        dataset.positive_frame_count,
        len(dataset.box_samples),
    )

    model = create_model(model_name)
    logger.info("开始拟合模型: {}", model.name)
    model.fit(dataset)
    logger.info("模型拟合完成: {}", model.name)
    model.save(output_dir)
    logger.info("模型已保存: {}", output_dir)

    result = TrainResult(
        model_name=model.name,
        model_path=str(output_dir.resolve()),
        data_dir=str(data_dir.resolve()),
        record_ids=[r.record_id for r in records],
        frame_count=dataset.frame_count,
        positive_frames=dataset.positive_frame_count,
        box_samples=len(dataset.box_samples),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "train_result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("训练结果已写入: {}", output_dir / "train_result.json")
    return result
