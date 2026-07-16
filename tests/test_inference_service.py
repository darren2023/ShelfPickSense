from __future__ import annotations

import json
from pathlib import Path

from fixtures import make_fixture_record


def test_inference_service_predicts_frame(tmp_path: Path):
    from analysis.inference_service import InferenceService, InferenceServiceConfig
    from analysis.records import load_record
    from analysis.train import train_model

    record_dir = make_fixture_record(tmp_path / "record_001")
    model_dir = tmp_path / "model"
    train_model(record_dir, model_dir)
    record = load_record(record_dir)

    config = InferenceServiceConfig(
        model_dir=str(model_dir),
        annotation_path=str(record_dir / "annotation.json"),
        infer_width=record.infer_width,
        infer_height=record.infer_height,
        record_id="camera_01",
    )
    service = InferenceService(config)
    frame = record.frames()[5]

    result = service.predict(
        {
            "record_id": "camera_01",
            "frame_idx": frame.frame_idx,
            "timestamp_sec": frame.timestamp_sec,
            "persons": frame.persons,
        }
    )

    assert result["record_id"] == "camera_01"
    assert result["frame_idx"] == frame.frame_idx
    assert "is_picking" in result
    assert "predicted_box_tokens" in result
    assert service.reset({"record_id": "camera_02"}) == {"status": "ok", "record_id": "camera_02"}


def test_inference_service_accepts_skeleton_row_list_payload(tmp_path: Path):
    from analysis.inference_service import InferenceService, InferenceServiceConfig
    from analysis.records import load_record
    from analysis.train import train_model

    record_dir = make_fixture_record(tmp_path / "record_001")
    model_dir = tmp_path / "model"
    train_model(record_dir, model_dir)
    record = load_record(record_dir)
    frame_idx = 6
    rows = (
        record.skeleton[record.skeleton["frame_idx"] == frame_idx]
        .copy()
        .to_dict(orient="records")
    )
    assert rows
    duplicate = dict(rows[0])
    duplicate["person_track_id"] = 2
    rows.append(duplicate)

    service = InferenceService(
        InferenceServiceConfig(
            model_dir=str(model_dir),
            annotation_path=str(record_dir / "annotation.json"),
            infer_width=record.infer_width,
            infer_height=record.infer_height,
            record_id="camera_01",
        )
    )

    result = service.predict(rows)

    assert result["record_id"] == "camera_01"
    assert result["frame_idx"] == frame_idx
    assert "is_picking" in result
    assert "predicted_box_tokens" in result


def test_inference_service_config_from_file(tmp_path: Path):
    from analysis.inference_service import InferenceServiceConfig

    config_path = tmp_path / "service.json"
    config_path.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": 9000,
                "model_dir": "models/rf",
                "annotation_path": "data/demo/annotation.json",
                "infer_width": 852,
                "infer_height": 480,
                "record_id": "camera_01",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = InferenceServiceConfig.from_file(config_path)

    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.model_dir == "models/rf"
    assert config.infer_width == 852
    assert config.infer_height == 480
