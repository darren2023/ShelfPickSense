"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from analysis.benchmark import DEFAULT_MODEL_NAMES, run_benchmark
from analysis.evaluation import compare_reports, evaluate_model, save_report
from analysis.feature_benchmark import load_feature_benchmark_plan, regenerate_feature_benchmark_report, run_feature_benchmarks
from analysis.features.selection import load_feature_selection
from analysis.models import SUPPORTED_MODEL_NAMES
from analysis.realtime import RealtimePickingPredictor
from analysis.rule_baseline import RealtimeRulePredictor, evaluate_rule_baseline
from analysis.train import train_model


def configure_logging(args: argparse.Namespace) -> None:
    """配置 loguru，默认输出到 stderr。"""
    level = str(getattr(args, "log_level", "INFO") or "INFO").upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
    )
    log_file = str(getattr(args, "log_file", "") or "").strip()
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="20 MB",
            retention=5,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        )


def _log_json(data: Any, *, indent: int = 2) -> None:
    """输出 JSON 结果（raw 模式，不带 loguru 前缀）。"""
    logger.opt(raw=True).info(json.dumps(data, ensure_ascii=False, indent=indent) + "\n")


def _log_jsonl(line: str) -> None:
    """输出单行 JSON（用于 infer 流式结果）。"""
    logger.opt(raw=True).info(line + "\n")


def _cmd_train(args: argparse.Namespace) -> int:
    logger.info(
        "开始训练模型: model={}, data_dir={}, output={}, feature_config={}",
        args.model,
        args.data_dir,
        args.output,
        args.feature_config or "",
    )
    result = train_model(
        Path(args.data_dir),
        Path(args.output),
        model_name=args.model,
        feature_selection=load_feature_selection(args.feature_config),
        filter_empty_skeleton=not args.keep_empty_skeleton_frames,
        feature_jobs=args.feature_jobs,
        feature_frame_stride=args.feature_frame_stride,
    )
    logger.info(
        "训练完成: model={}, frames={}, positive_frames={}, box_samples={}, skipped_empty_skeleton={}",
        result.model_name,
        result.frame_count,
        result.positive_frames,
        result.box_samples,
        result.skipped_empty_skeleton_frames,
    )
    _log_json(result.to_dict())
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    logger.info("开始评测模型: model={}, data_dir={}", args.model, args.data_dir)
    predictions_out = Path(args.predictions) if args.predictions else Path(args.model) / "eval_predictions.json"
    report = evaluate_model(
        Path(args.model),
        Path(args.data_dir),
        predictions_output_path=predictions_out,
    )
    out = Path(args.report) if args.report else Path(args.model) / "eval_report.json"
    save_report(report, out)
    logger.info(
        "评测完成: model={}, macro_f1={:.4f}, picking_f1={:.4f}, recall={:.4f}, precision={:.4f}, box_f1={:.4f}",
        report.model_name,
        report.picking.macro_f1,
        report.picking.f1,
        report.picking.recall,
        report.picking.precision,
        report.box.micro_f1,
    )
    _log_json(report.to_dict())
    logger.info("报告已保存: {}", out)
    logger.info("预测结果已保存: {}", predictions_out)
    return 0


def _cmd_eval_rule(args: argparse.Namespace) -> int:
    from analysis.benchmark import prediction_filename_for_records
    from analysis.records import load_all_records

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("开始规则基线评测: data_dir={}, output={}", data_dir, output_dir)

    records = load_all_records(data_dir)
    predictions_out = (
        Path(args.predictions)
        if args.predictions
        else output_dir / prediction_filename_for_records(records)
    )
    report = evaluate_rule_baseline(
        records,
        data_dir=str(data_dir.resolve()),
        predictions_output_path=predictions_out,
    )
    report_path = Path(args.report) if args.report else output_dir / "eval_report.json"
    save_report(report, report_path)
    logger.info(
        "规则基线评测完成: macro_f1={:.4f}, picking_f1={:.4f}, recall={:.4f}, box_f1={:.4f}",
        report.picking.macro_f1,
        report.picking.f1,
        report.picking.recall,
        report.box.micro_f1,
    )
    _log_json(report.to_dict())
    logger.info("报告已保存: {}", report_path)
    logger.info("预测结果已保存: {}", predictions_out)
    return 0


def _infer_rule_output_path(
    output: str,
    record_id: str,
    *,
    multi_record: bool,
) -> Path:
    """解析 infer-rule 的 JSONL 输出路径。"""
    out = Path(output)
    if multi_record and out.suffix != ".jsonl":
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{record_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _run_infer_rule_record(
    record,
    *,
    infer_width: float | None,
    infer_height: float | None,
    fps: float,
    max_frames: int,
    realtime: bool,
    video: str,
    out_file,
) -> int:
    """对单条记录执行规则逐帧推理，返回处理的帧数。"""
    predictor = RealtimeRulePredictor.from_record_dir(
        record.record_dir,
        infer_width=infer_width if infer_width is not None else record.infer_width,
        infer_height=infer_height if infer_height is not None else record.infer_height,
        video_fps=fps,
    )

    frames = record.frames()
    if max_frames > 0:
        frames = frames[:max_frames]
    frame_interval = 1.0 / fps if realtime and fps > 0 else 0.0
    logger.info(
        "开始规则逐帧推理: record={}, video={}, frames={}, realtime={}, fps={}",
        record.record_id,
        video or "",
        len(frames),
        realtime,
        fps,
    )

    for frame in frames:
        pred = predictor.predict_frame(
            frame.persons,
            frame_idx=frame.frame_idx,
            timestamp_sec=frame.timestamp_sec,
        )
        line = json.dumps(pred.to_dict(), ensure_ascii=False)
        if out_file:
            out_file.write(line + "\n")
        _log_jsonl(line)
        if frame_interval > 0:
            time.sleep(frame_interval)
    return len(frames)


def _cmd_infer_rule(args: argparse.Namespace) -> int:
    from analysis.records import load_all_records

    input_path = Path(args.record_dir)
    records = load_all_records(input_path)
    multi_record = len(records) > 1
    merge_output = bool(args.output) and multi_record and Path(args.output).suffix == ".jsonl"

    logger.info(
        "规则逐帧推理: input={}, records={}, output={}, merge_output={}",
        input_path,
        len(records),
        args.output or "",
        merge_output,
    )

    merge_file = None
    total_frames = 0
    try:
        if merge_output:
            merge_path = Path(args.output)
            merge_path.parent.mkdir(parents=True, exist_ok=True)
            merge_file = merge_path.open("w", encoding="utf-8")

        for record in records:
            out_file = merge_file
            close_after = False
            if out_file is None and args.output:
                out_path = _infer_rule_output_path(
                    args.output,
                    record.record_id,
                    multi_record=multi_record,
                )
                out_file = out_path.open("w", encoding="utf-8")
                close_after = True

            try:
                total_frames += _run_infer_rule_record(
                    record,
                    infer_width=args.infer_width,
                    infer_height=args.infer_height,
                    fps=args.fps,
                    max_frames=args.max_frames,
                    realtime=args.realtime,
                    video=args.video,
                    out_file=out_file,
                )
            finally:
                if close_after and out_file:
                    out_file.close()
    finally:
        if merge_file:
            merge_file.close()

    logger.info("规则逐帧推理完成: records={}, frames={}", len(records), total_frames)
    return 0


def _run_infer_collision_record(
    record,
    CollisionProcessor,
    *,
    infer_width: float | None,
    infer_height: float | None,
    fps: float,
    max_frames: int,
    realtime: bool,
    video: str,
    alarm_min_consecutive_frames: int,
    alarm_cooldown_frames: int,
    pose_frame_interval: int,
    out_file,
) -> int:
    """对单条记录执行 collision.py 逐帧推理，返回处理的帧数。"""
    import numpy as np

    # 确保 box_index 存在
    if not record.box_index:
        from analysis.annotation import build_box_index
        record.box_index = build_box_index(
            record.annotation,
            infer_w=record.infer_width,
            infer_h=record.infer_height,
        )

    # 将 BoxIndex 转换为 CollisionProcessor 需要的格式
    boxes = []
    for token, box_info in record.box_index.items():
        # 将 polygon 转换为 OpenCV contour 格式 (n, 1, 2)
        polygon = np.array(box_info.polygon, dtype=np.float32).reshape(-1, 1, 2)
        boxes.append({
            "token": token,
            "shelf_code": box_info.shelf_code,
            "box_id": box_info.box_id,
            "orig_contour": polygon,
        })

    processor = CollisionProcessor(
        boxes,
        alarm_min_consecutive_frames=alarm_min_consecutive_frames,
        alarm_cooldown_frames=alarm_cooldown_frames,
        video_fps=fps,
    )

    frames = record.frames()
    if max_frames > 0:
        frames = frames[:max_frames]
    frame_interval = 1.0 / fps if realtime and fps > 0 else 0.0

    # 构建帧索引到帧数据的映射
    frames_by_idx = {f.frame_idx: f for f in frames}

    # 确定帧索引范围
    if frames:
        min_frame = min(f.frame_idx for f in frames)
        max_frame = max(f.frame_idx for f in frames)
    else:
        min_frame = 1
        max_frame = 1

    processed_count = 0
    logger.info(
        "开始 collision.py 逐帧推理: record={}, video={}, frame_range={}-{}, frames={}, pose_frame_interval={}, realtime={}, fps={}",
        record.record_id,
        video or "",
        min_frame,
        max_frame,
        len(frames),
        pose_frame_interval,
        realtime,
        fps,
    )

    # 按帧索引顺序处理，填充缺失帧
    for frame_idx in range(min_frame, max_frame + 1):
        # 根据 pose_frame_interval 跳帧处理（基于帧索引而非数组索引）
        if (frame_idx - 1) % pose_frame_interval != 0:
            continue

        frame = frames_by_idx.get(frame_idx)

        if frame is not None:
            # 有骨架数据，正常处理
            pose_frame = {
                "frame_idx": frame.frame_idx,
                "persons": frame.persons,
            }
            result = processor.process(pose_frame)

            pred = {
                "record_id": record.record_id,
                "frame_idx": frame.frame_idx,
                "is_picking": bool(result.get("alarm_collisions")),
                "picking_prob": 1.0 if result.get("alarm_collisions") else 0.0,
                "predicted_box_tokens": list(result.get("alarm_collisions") or result.get("collisions") or []),
                "collision_collisions": list(result.get("collisions") or []),
                "collision_alarm_collisions": list(result.get("alarm_collisions") or []),
            }
        else:
            # 无骨架数据，输出空结果
            pred = {
                "record_id": record.record_id,
                "frame_idx": frame_idx,
                "is_picking": False,
                "picking_prob": 0.0,
                "predicted_box_tokens": [],
                "collision_collisions": [],
                "collision_alarm_collisions": [],
            }

        processed_count += 1

        line = json.dumps(pred, ensure_ascii=False)
        if out_file:
            out_file.write(line + "\n")
        _log_jsonl(line)
        if frame_interval > 0:
            time.sleep(frame_interval)
    return processed_count


def _cmd_infer_collision(args: argparse.Namespace) -> int:
    """调用 box_human_det/services/event_engine/collision.py 进行逐帧推理。"""
    import sys
    from pathlib import Path

    # 添加 box_human_det 到 sys.path
    box_human_det_path = Path(__file__).parent.parent.parent.parent / "box_human_det"
    box_human_det_path = box_human_det_path.resolve()
    if str(box_human_det_path) not in sys.path:
        sys.path.insert(0, str(box_human_det_path))
    logger.info("已添加到 sys.path: {}", box_human_det_path)

    try:
        from services.event_engine.collision import CollisionProcessor
    except ImportError as e:
        logger.error("无法导入 collision.py: {}", e)
        return 1

    from analysis.records import load_all_records

    input_path = Path(args.record_dir)
    records = load_all_records(input_path)
    multi_record = len(records) > 1
    merge_output = bool(args.output) and multi_record and Path(args.output).suffix == ".jsonl"

    logger.info(
        "collision.py 逐帧推理: input={}, records={}, output={}, merge_output={}",
        input_path,
        len(records),
        args.output or "",
        merge_output,
    )

    merge_file = None
    total_frames = 0
    try:
        if merge_output:
            merge_path = Path(args.output)
            merge_path.parent.mkdir(parents=True, exist_ok=True)
            merge_file = merge_path.open("w", encoding="utf-8")

        for record in records:
            out_file = merge_file
            close_after = False
            if out_file is None:
                # 确定输出路径
                if args.output:
                    # 用户指定了输出路径
                    out_path = _infer_rule_output_path(
                        args.output,
                        record.record_id,
                        multi_record=multi_record,
                    )
                else:
                    # 未指定输出时，自动在每个 record 目录中生成 collision_infer.jsonl
                    out_path = record.record_dir / "collision_infer.jsonl"

                out_file = out_path.open("w", encoding="utf-8")
                close_after = True
                logger.info("输出文件: {}", out_path)

            try:
                total_frames += _run_infer_collision_record(
                    record,
                    CollisionProcessor,
                    infer_width=args.infer_width,
                    infer_height=args.infer_height,
                    fps=args.fps,
                    max_frames=args.max_frames,
                    realtime=args.realtime,
                    video=args.video,
                    alarm_min_consecutive_frames=args.alarm_min_consecutive_frames,
                    alarm_cooldown_frames=args.alarm_cooldown_frames,
                    pose_frame_interval=args.pose_frame_interval,
                    out_file=out_file,
                )
            finally:
                if close_after and out_file:
                    out_file.close()
    finally:
        if merge_file:
            merge_file.close()

    logger.info("collision.py 逐帧推理完成: records={}, frames={}", len(records), total_frames)
    return 0


def _cmd_viz_frames(args: argparse.Namespace) -> int:
    from analysis.frame_viz import write_frame_viz_html

    record_dir = Path(args.record_dir)
    output_path = Path(args.output)
    predictions = Path(args.predictions) if args.predictions else None
    logger.info(
        "生成帧可视化: record={}, frames={}.., max={}, output={}",
        record_dir,
        args.start_frame,
        args.max_frames,
        output_path,
    )
    out = write_frame_viz_html(
        record_dir,
        output_path,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        predictions_path=predictions,
    )
    logger.info("可视化页面已生成: {}", out.resolve())
    return 0


def _cmd_viz_feature_curves(args: argparse.Namespace) -> int:
    from analysis.feature_curves_viz import write_feature_curves_html

    features_dir = Path(args.features_dir)
    output_path = Path(args.output) if args.output else None
    out = write_feature_curves_html(features_dir, output_path)
    logger.info("feature curves html generated: {}", out.resolve())
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from analysis.evaluation import BoxMetrics, ModelEvaluation, PickingMetrics

    logger.info("开始对比评测报告: count={}", len(args.reports))
    reports: list[ModelEvaluation] = []
    for p in args.reports:
        data = json.loads(Path(p).read_text(encoding="utf-8"))

        reports.append(
            ModelEvaluation(
                model_name=data["model_name"],
                data_dir=data["data_dir"],
                record_ids=data["record_ids"],
                picking=PickingMetrics(**data["picking"]),
                box=BoxMetrics(**data["box"]),
                evaluated_at=data.get("evaluated_at", ""),
                extra=data.get("extra", {}),
            )
        )
    rows = compare_reports(reports)
    logger.info("对比完成: count={}", len(rows))
    _log_json(rows)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    logger.info(
        "开始批量 benchmark: models={}, jobs={}, train_data={}, eval_data={}, output={}",
        args.models,
        args.jobs,
        args.data_dir,
        args.eval_data_dir or args.data_dir,
        args.output,
    )
    result = run_benchmark(
        train_data_dir=Path(args.data_dir),
        eval_data_dir=Path(args.eval_data_dir) if args.eval_data_dir else None,
        output_dir=Path(args.output),
        model_names=args.models,
        jobs=args.jobs,
        feature_selection=load_feature_selection(args.feature_config),
        filter_empty_skeleton=not args.keep_empty_skeleton_frames,
        feature_frame_stride=args.feature_frame_stride,
    )
    logger.info("benchmark 完成: models={}, output={}", len(result.model_names), result.output_dir)
    _log_json(result.comparison)
    logger.info("批量对比报告已保存: {}", Path(args.output) / "benchmark_summary.json")
    return 0


def _cmd_benchmark_features(args: argparse.Namespace) -> int:
    plan = load_feature_benchmark_plan(Path(args.plan))
    if args.data_dir:
        plan.train_data_dir = Path(args.data_dir)
    if args.eval_data_dir:
        plan.eval_data_dir = Path(args.eval_data_dir)
    if args.output:
        plan.output_dir = Path(args.output)
    if args.models:
        plan.model_names = list(args.models)
    if args.jobs:
        plan.jobs = int(args.jobs)
    if args.feature_frame_stride:
        plan.feature_frame_stride = int(args.feature_frame_stride)

    if args.report_only:
        logger.info(
            "重新生成多特征 benchmark 报告: sets={}, output={}",
            len(plan.sets),
            plan.output_dir,
        )
        result = regenerate_feature_benchmark_report(plan)
        logger.info("多特征 benchmark 报告已重新生成: {}", result.report_path)
    else:
        logger.info(
            "开始多特征 benchmark: sets={}, models={}, jobs={}, feature_frame_stride={}, train_data={}, eval_data={}, output={}",
            len(plan.sets),
            plan.model_names,
            plan.jobs,
            plan.feature_frame_stride,
            plan.train_data_dir,
            plan.eval_data_dir or plan.train_data_dir,
            plan.output_dir,
        )
        result = run_feature_benchmarks(plan)
        logger.info("多特征 benchmark 完成: sets={}, output={}", len(result.sets), result.output_dir)

    summary = [
        {
            "name": item.name,
            "best_model": item.best_model,
            "best_macro_f1": item.best_macro_f1,
            "output_dir": item.output_dir,
        }
        for item in result.sets
    ]
    _log_json(summary)
    logger.info("多特征 benchmark 汇总已保存: {}", result.summary_path)
    logger.info("多特征 benchmark 报告已保存: {}", result.report_path)
    return 0


def _json_safe_rows(rows: list[dict]) -> list[dict]:
    safe_rows = []
    for row in rows:
        safe_rows.append(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, list | dict) else v
                for k, v in row.items()
            }
        )
    return safe_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_export_output_dir(data_dir: Path) -> Path:
    from analysis.records import is_record_dir

    data_dir = Path(data_dir)
    if is_record_dir(data_dir):
        return data_dir
    return data_dir / "features"


def _cmd_export_features(args: argparse.Namespace) -> int:
    import pandas as pd

    from analysis.dataset import load_dataset

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else _default_export_output_dir(data_dir)
    logger.info(
        "开始提取特征: data_dir={}, output={}, format={}, feature_config={}",
        data_dir,
        out_dir,
        args.format,
        args.feature_config or "",
    )
    feature_selection = load_feature_selection(args.feature_config)
    dataset = load_dataset(
        data_dir,
        feature_selection=feature_selection,
        feature_jobs=args.feature_jobs,
        feature_frame_stride=args.feature_frame_stride,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    for sample in dataset.frame_samples:
        row = {
            "record_id": sample.record_id,
            "frame_idx": sample.frame_idx,
            "person_track_id": sample.person_track_id,
            "shelf_code": sample.shelf_code,
            "is_picking": sample.is_picking,
            "target_layout_shelf_side": sample.target_layout_shelf_side,
            "target_layout_layer_norm": sample.target_layout_layer_norm,
            "target_layout_column_norm": sample.target_layout_column_norm,
        }
        row.update(dict(zip(dataset.frame_feature_names, sample.x.tolist(), strict=True)))
        frame_rows.append(row)

    box_rows = []
    for sample in dataset.box_samples:
        row = {
            "record_id": sample.record_id,
            "frame_idx": sample.frame_idx,
            "box_token": sample.box_token,
            "box_code": sample.box_code,
            "is_target": sample.is_target,
            "target_layout_shelf_side": sample.target_layout_shelf_side,
            "target_layout_layer_norm": sample.target_layout_layer_norm,
            "target_layout_column_norm": sample.target_layout_column_norm,
        }
        row.update(dict(zip(dataset.box_feature_names, sample.x.tolist(), strict=True)))
        box_rows.append(row)

    frame_df = pd.DataFrame(frame_rows)
    box_df = pd.DataFrame(box_rows)
    formats = ["parquet", "csv", "jsonl"] if args.format == "all" else [args.format]
    output_files: dict[str, dict[str, str]] = {}

    for output_format in formats:
        frame_path = out_dir / f"frame_features.{output_format}"
        box_path = out_dir / f"box_features.{output_format}"
        if output_format == "parquet":
            frame_df.to_parquet(frame_path, index=False)
            box_df.to_parquet(box_path, index=False)
        elif output_format == "csv":
            pd.DataFrame(_json_safe_rows(frame_rows)).to_csv(frame_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(_json_safe_rows(box_rows)).to_csv(box_path, index=False, encoding="utf-8-sig")
        elif output_format == "jsonl":
            _write_jsonl(frame_path, frame_rows)
            _write_jsonl(box_path, box_rows)
        output_files[output_format] = {
            "frame_features_path": str(frame_path),
            "box_features_path": str(box_path),
        }

    meta_path = out_dir / "features_meta.json"
    primary_format = formats[0]

    meta = {
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "output_format": args.format,
        "output_files": output_files,
        "frame_features_path": output_files[primary_format]["frame_features_path"],
        "box_features_path": output_files[primary_format]["box_features_path"],
        "frame_count": len(dataset.frame_samples),
        "box_sample_count": len(dataset.box_samples),
        "frame_feature_names": dataset.frame_feature_names,
        "box_feature_names": dataset.box_feature_names,
        "feature_selection": feature_selection.to_dict() if feature_selection else None,
        "feature_frame_stride": args.feature_frame_stride,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "特征提取完成: frames={}, box_samples={}, output={}",
        meta["frame_count"],
        meta["box_sample_count"],
        out_dir,
    )
    _log_json(meta)
    for output_format, paths in output_files.items():
        logger.info("{} 帧级特征已保存: {}", output_format, paths["frame_features_path"])
        logger.info("{} 货框特征已保存: {}", output_format, paths["box_features_path"])
    logger.info("特征元数据已保存: {}", meta_path)
    return 0


def _cmd_analyze_features(args: argparse.Namespace) -> int:
    from analysis.feature_correlation import analyze_exported_feature_correlations, analyze_feature_correlations

    logger.info(
        "开始特征相关性分析: data_dir={}, features_dir={}, output={}, method={}, threshold={}, feature_config={}",
        args.data_dir or "",
        args.features_dir or "",
        args.output,
        args.method,
        args.threshold,
        args.feature_config or "",
    )
    feature_selection = load_feature_selection(args.feature_config)
    if args.features_dir:
        result = analyze_exported_feature_correlations(
            Path(args.features_dir),
            Path(args.output),
            method=args.method,
            threshold=args.threshold,
            top_n=args.top_n,
            feature_selection=feature_selection,
        )
    else:
        result = analyze_feature_correlations(
            Path(args.data_dir),
            Path(args.output),
            method=args.method,
            threshold=args.threshold,
            top_n=args.top_n,
            feature_selection=feature_selection,
            feature_jobs=args.feature_jobs,
            feature_frame_stride=args.feature_frame_stride,
        )
    logger.info(
        "特征相关性分析完成: frames={}, box_samples={}, output={}",
        result.frame_count,
        result.box_sample_count,
        result.output_dir,
    )
    _log_json(result.to_dict())
    logger.info("相关性分析结果已保存: {}", result.output_dir)
    return 0


def _cmd_infer_frame(args: argparse.Namespace) -> int:
    from analysis.records import load_record

    record = load_record(Path(args.record_dir))
    predictor = RealtimePickingPredictor(record_id=record.record_id)
    predictor.load_model(Path(args.model))
    predictor.set_infer_size(
        args.infer_width if args.infer_width is not None else record.infer_width,
        args.infer_height if args.infer_height is not None else record.infer_height,
    )
    predictor.annotation = record.annotation

    frames = record.frames()
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    frame_interval = 1.0 / args.fps if args.realtime and args.fps > 0 else 0.0
    logger.info(
        "开始模拟视频流逐帧推理: record={}, video={}, frames={}, realtime={}, fps={}",
        record.record_id,
        args.video or "",
        len(frames),
        args.realtime,
        args.fps,
    )

    out_file = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")

    try:
        for frame in frames:
            pred = predictor.predict_frame(
                frame.persons,
                frame_idx=frame.frame_idx,
                timestamp_sec=frame.timestamp_sec,
            )
            line = json.dumps(pred.to_dict(), ensure_ascii=False)
            if out_file:
                out_file.write(line + "\n")
            _log_jsonl(line)
            if frame_interval > 0:
                time.sleep(frame_interval)
    finally:
        if out_file:
            out_file.close()
    return 0


def _cmd_serve_inference(args: argparse.Namespace) -> int:
    from analysis.inference_service import InferenceServiceConfig, serve

    config = InferenceServiceConfig.from_file(Path(args.config))
    serve(config)
    return 0


def _cmd_box_layout(args: argparse.Namespace) -> int:
    import csv
    import sys

    from analysis.annotation import load_annotation
    from analysis.box_layout import (
        compute_shelf_bottom_bounds_from_rows,
        list_box_layout_rows,
        render_box_layout_svg,
    )
    from analysis.constants import ANNOTATION_FILE
    from analysis.records import discover_record_dirs

    output_format = str(args.format or "json").lower()
    explicit_output = Path(args.output) if args.output else None
    explicit_viz = Path(args.viz_output) if args.viz_output else None

    def write_one(*, ann_path: Path, record_dir: Path | None) -> bool:
        if not ann_path.is_file():
            logger.error("未找到 annotation 文件: {}", ann_path)
            return False

        annotation = load_annotation(ann_path)
        rows = list_box_layout_rows(annotation)
        bottom_bounds = compute_shelf_bottom_bounds_from_rows(rows)
        payload = {
            "annotation": str(ann_path),
            "box_count": len(rows),
            "shelf_bottom_bounds": {key: value.to_dict() for key, value in bottom_bounds.items()},
            "boxes": [row.to_dict() for row in rows],
        }

        output_path = explicit_output
        viz_path = explicit_viz
        if record_dir is not None:
            record_dir = record_dir.resolve()
        if record_dir is not None and output_path is None:
            ext = {"json": "json", "csv": "csv", "table": "txt"}.get(output_format, output_format)
            output_path = record_dir / f"box_layout.{ext}"
        if record_dir is not None and viz_path is None:
            viz_path = record_dir / "box_layout.svg"

        if viz_path is not None:
            render_box_layout_svg(annotation, output_path=viz_path)
            logger.info("box_layout 可视化已写入: {}", viz_path)

        if output_format == "json":
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("货框布局已写入: {} boxes={}", output_path, len(rows))
            else:
                _log_json(payload)
            return True

        if output_format == "csv":
            fieldnames = list(payload["boxes"][0].keys()) if rows else []
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(payload["boxes"])
                logger.info("货框布局已写入: {} boxes={}", output_path, len(rows))
            else:
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(payload["boxes"])
            return True

        if output_format == "table":
            header = (
                f"{'token':<12} {'box_id':>6} {'side':>4} {'layer':>5} {'col':>4} "
                f"{'cx':>8} {'cy':>8} {'ann_L':>5} {'ann_C':>5}"
            )
            logger.info("annotation={} box_count={}", ann_path, len(rows))
            logger.info(header)
            for row in rows:
                logger.info(
                    "{:<12} {:>6} {:>4} {:>5} {:>4} {:>8.1f} {:>8.1f} {:>5} {:>5}",
                    row.token,
                    row.box_id,
                    row.shelf_side,
                    row.layer,
                    row.column,
                    row.centroid_x,
                    row.centroid_y,
                    row.annotation_layer if row.annotation_layer is not None else "-",
                    row.annotation_column if row.annotation_column is not None else "-",
                )
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                lines = [header]
                for row in rows:
                    lines.append(
                        f"{row.token:<12} {row.box_id:>6} {row.shelf_side:>4} {row.layer:>5} {row.column:>4} "
                        f"{row.centroid_x:>8.1f} {row.centroid_y:>8.1f} "
                        f"{row.annotation_layer if row.annotation_layer is not None else '-':>5} "
                        f"{row.annotation_column if row.annotation_column is not None else '-':>5}"
                    )
                output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                logger.info("货框布局已写入: {}", output_path)
            return True

        logger.error("不支持的输出格式: {}", args.format)
        return False

    if args.record_dir:
        input_dir = Path(args.record_dir).resolve()
        record_dirs = discover_record_dirs(input_dir)
        if not record_dirs and (input_dir / ANNOTATION_FILE).is_file():
            record_dirs = [input_dir]
        if not record_dirs:
            logger.error("未发现有效记录目录: {}", input_dir)
            return 1
        if len(record_dirs) > 1 and (explicit_output is not None or explicit_viz is not None):
            logger.error("批量处理多个记录目录时不支持 --output 或 --viz-output，请使用默认每条记录目录输出")
            return 1
        ok = True
        for record_dir in record_dirs:
            ok = write_one(ann_path=record_dir / ANNOTATION_FILE, record_dir=record_dir) and ok
        logger.info("box_layout 批量处理完成: records={}", len(record_dirs))
        return 0 if ok else 1

    ann_path = Path(args.annotation).resolve()
    return 0 if write_one(ann_path=ann_path, record_dir=None) else 1



def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别（默认 INFO）",
    )
    parser.add_argument("--log-file", default="", help="日志文件路径（默认不写文件）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="货架取货行为分析：训练与评测",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--data-dir", required=True, help="数据目录（含多条记录或单条记录）")
    p_train.add_argument("--output", required=True, help="模型输出目录")
    p_train.add_argument(
        "--model",
        default="sklearn_rf",
        choices=SUPPORTED_MODEL_NAMES,
        help="模型类型",
    )
    p_train.add_argument("--feature-config", default="", help="特征选择 JSON 配置路径（默认使用全部特征）")
    p_train.add_argument("--feature-jobs", type=int, default=1, help="特征提取并行 record 数（默认 1）")
    p_train.add_argument("--feature-frame-stride", type=int, default=1, help="特征提取帧采样间隔：每 N 个骨架帧取 1 帧（默认 1）")
    p_train.add_argument(
        "--keep-empty-skeleton-frames",
        action="store_true",
        help="保留无骨架帧参与训练（默认在特征提取后过滤无骨架帧）",
    )
    _add_logging_args(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="评测模型")
    p_eval.add_argument("--data-dir", required=True, help="评测数据目录")
    p_eval.add_argument("--model", required=True, help="已训练模型目录")
    p_eval.add_argument("--report", default="", help="评测报告输出路径（默认写入模型目录）")
    p_eval.add_argument("--predictions", default="", help="预测结果输出路径（默认写入模型目录）")
    _add_logging_args(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    p_eval_rule = sub.add_parser("eval-rule", help="用规则碰撞方法评测数据（无需训练模型）")
    p_eval_rule.add_argument("--data-dir", required=True, help="评测数据目录")
    p_eval_rule.add_argument("--output", required=True, help="评测结果输出目录")
    p_eval_rule.add_argument("--report", default="", help="评测报告路径（默认 <output>/eval_report.json）")
    p_eval_rule.add_argument("--predictions", default="", help="预测结果路径（默认 <output>/eval_predictions_*.json）")
    _add_logging_args(p_eval_rule)
    p_eval_rule.set_defaults(func=_cmd_eval_rule)

    p_cmp = sub.add_parser("compare", help="对比多份评测报告")
    p_cmp.add_argument("reports", nargs="+", help="eval_report.json 路径列表")
    _add_logging_args(p_cmp)
    p_cmp.set_defaults(func=_cmd_compare)

    p_bench = sub.add_parser("benchmark", help="批量训练、评测并对比多个模型")
    p_bench.add_argument("--data-dir", required=True, help="训练数据目录")
    p_bench.add_argument(
        "--eval-data-dir",
        default="",
        help="评测数据目录（默认与训练数据相同）",
    )
    p_bench.add_argument("--output", required=True, help="批量输出目录")
    p_bench.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        choices=SUPPORTED_MODEL_NAMES,
        help="需要批量运行的模型列表",
    )
    p_bench.add_argument("--jobs", type=int, default=8, help="并行运行的模型数量（默认 8）")
    p_bench.add_argument("--feature-config", default="", help="特征选择 JSON 配置路径（默认使用全部特征）")
    p_bench.add_argument("--feature-frame-stride", type=int, default=1, help="特征提取帧采样间隔：每 N 个骨架帧取 1 帧（默认 1）")
    p_bench.add_argument(
        "--keep-empty-skeleton-frames",
        action="store_true",
        help="保留无骨架帧参与训练（默认在特征提取后过滤无骨架帧）",
    )
    _add_logging_args(p_bench)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_bench_features = sub.add_parser("benchmark-features", help="按多组特征配置批量运行 benchmark")
    p_bench_features.add_argument("--plan", required=True, help="批量 benchmark JSON 配置路径")
    p_bench_features.add_argument("--data-dir", default="", help="覆盖配置中的训练数据目录")
    p_bench_features.add_argument("--eval-data-dir", default="", help="覆盖配置中的评测数据目录")
    p_bench_features.add_argument("--output", default="", help="覆盖配置中的输出目录")
    p_bench_features.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=SUPPORTED_MODEL_NAMES,
        help="覆盖配置中的模型列表",
    )
    p_bench_features.add_argument("--jobs", type=int, default=0, help="覆盖配置中的并行模型数")
    p_bench_features.add_argument("--feature-frame-stride", type=int, default=0, help="覆盖配置中的特征提取帧采样间隔：每 N 个骨架帧取 1 帧")
    p_bench_features.add_argument(
        "--report-only",
        action="store_true",
        help="仅基于已有 benchmark 结果重新生成 Markdown 报告，不重新训练或评测模型",
    )
    _add_logging_args(p_bench_features)
    p_bench_features.set_defaults(func=_cmd_benchmark_features)

    p_export = sub.add_parser("export-features", help="从记录提取特征并保存到文件")
    p_export.add_argument("--data-dir", required=True, help="数据目录（含多条记录或单条记录）")
    p_export.add_argument(
        "--output",
        default="",
        help="特征输出目录；默认单记录输出到该 record 目录，多记录父目录输出到其 features 子目录",
    )
    p_export.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "csv", "jsonl", "all"],
        help="特征文件格式（默认 parquet；可选 csv/jsonl/all）",
    )
    p_export.add_argument("--feature-config", default="", help="特征选择 JSON 配置路径（默认导出全部特征）")
    p_export.add_argument("--feature-jobs", type=int, default=1, help="特征提取并行 record 数（默认 1）")
    p_export.add_argument("--feature-frame-stride", type=int, default=1, help="特征提取帧采样间隔：每 N 个骨架帧取 1 帧（默认 1）")
    _add_logging_args(p_export)
    p_export.set_defaults(func=_cmd_export_features)

    p_analyze = sub.add_parser("analyze-features", help="分析输入记录的特征相关性")
    source_group = p_analyze.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--data-dir", default="", help="数据目录（含多条记录或单条记录）")
    source_group.add_argument("--features-dir", default="", help="export-features 已导出的特征目录")
    p_analyze.add_argument("--output", required=True, help="相关性分析输出目录")
    p_analyze.add_argument(
        "--method",
        default="pearson",
        choices=["pearson", "spearman", "kendall"],
        help="相关性计算方法（默认 pearson）",
    )
    p_analyze.add_argument("--threshold", type=float, default=0.9, help="高相关特征对阈值（默认 0.9）")
    p_analyze.add_argument("--top-n", type=int, default=100, help="最多输出高相关特征对数量（默认 100）")
    p_analyze.add_argument("--feature-config", default="", help="特征选择 JSON 配置路径（默认分析全部特征）")
    p_analyze.add_argument("--feature-jobs", type=int, default=1, help="从原始 data-dir 提取特征时的并行 record 数（默认 1）")
    p_analyze.add_argument("--feature-frame-stride", type=int, default=1, help="从原始 data-dir 提取特征时的帧采样间隔：每 N 个骨架帧取 1 帧（默认 1）")
    _add_logging_args(p_analyze)
    p_analyze.set_defaults(func=_cmd_analyze_features)

    p_infer = sub.add_parser("infer-frame", help="用已抽取骨架记录模拟视频流逐帧推理")
    p_infer.add_argument("--model", required=True, help="已训练模型目录")
    p_infer.add_argument("--record-dir", required=True, help="记录目录（读取 skeleton.parquet 和 annotation.json）")
    p_infer.add_argument("--video", default="", help="原始视频路径，仅用于日志标识")
    p_infer.add_argument("--infer-width", type=float, default=None, help="推理坐标宽度")
    p_infer.add_argument("--infer-height", type=float, default=None, help="推理坐标高度")
    p_infer.add_argument("--fps", type=float, default=25.0, help="模拟流帧率")
    p_infer.add_argument("--max-frames", type=int, default=0, help="最多推理帧数，0 表示全部")
    p_infer.add_argument("--realtime", action="store_true", help="按 fps sleep，模拟真实时间流")
    p_infer.add_argument("--output", default="", help="JSONL 输出文件路径")
    _add_logging_args(p_infer)
    p_infer.set_defaults(func=_cmd_infer_frame)

    p_serve = sub.add_parser("serve-inference", help="启动实时推理 HTTP 服务")
    p_serve.add_argument("--config", required=True, help="推理服务 JSON 配置路径")
    _add_logging_args(p_serve)
    p_serve.set_defaults(func=_cmd_serve_inference)

    p_infer_rule = sub.add_parser("infer-rule", help="用规则碰撞方法模拟视频流逐帧推理")
    p_infer_rule.add_argument(
        "--record-dir",
        required=True,
        help="记录目录或包含多条记录的父目录（需含 skeleton.parquet 与 annotation.json）",
    )
    p_infer_rule.add_argument("--video", default="", help="原始视频路径，仅用于日志标识")
    p_infer_rule.add_argument("--infer-width", type=float, default=None, help="推理坐标宽度")
    p_infer_rule.add_argument("--infer-height", type=float, default=None, help="推理坐标高度")
    p_infer_rule.add_argument("--fps", type=float, default=25.0, help="模拟流帧率")
    p_infer_rule.add_argument("--max-frames", type=int, default=0, help="最多推理帧数，0 表示全部")
    p_infer_rule.add_argument("--realtime", action="store_true", help="按 fps sleep，模拟真实时间流")
    p_infer_rule.add_argument(
        "--output",
        default="",
        help="JSONL 输出：单条记录为文件路径；多条记录时为目录（逐条生成 {record_id}.jsonl）或合并 .jsonl 文件",
    )
    _add_logging_args(p_infer_rule)
    p_infer_rule.set_defaults(func=_cmd_infer_rule)

    p_infer_collision = sub.add_parser("infer-collision", help="调用 box_human_det/collision.py 模拟视频流逐帧推理")
    p_infer_collision.add_argument(
        "--record-dir",
        required=True,
        help="记录目录或包含多条记录的父目录（需含 skeleton.parquet 与 annotation.json）",
    )
    p_infer_collision.add_argument("--video", default="", help="原始视频路径，仅用于日志标识")
    p_infer_collision.add_argument("--infer-width", type=float, default=None, help="推理坐标宽度")
    p_infer_collision.add_argument("--infer-height", type=float, default=None, help="推理坐标高度")
    p_infer_collision.add_argument("--fps", type=float, default=15.0, help="模拟流帧率")
    p_infer_collision.add_argument("--max-frames", type=int, default=0, help="最多推理帧数，0 表示全部")
    p_infer_collision.add_argument("--realtime", action="store_true", help="按 fps sleep，模拟真实时间流")
    p_infer_collision.add_argument(
        "--pose-frame-interval",
        type=int,
        default=1,
        help="姿态估计帧间隔，每隔 N 帧处理一次（默认 1，即每帧都处理）",
    )
    p_infer_collision.add_argument(
        "--alarm-min-consecutive-frames",
        type=int,
        default=3,
        help="报警最小连续帧数（默认 3）",
    )
    p_infer_collision.add_argument(
        "--alarm-cooldown-frames",
        type=int,
        default=0,
        help="报警冷却帧数（默认 0）",
    )
    p_infer_collision.add_argument(
        "--output",
        default="",
        help="JSONL 输出：单条记录为文件路径；多条记录时为目录（逐条生成 {record_id}.jsonl）或合并 .jsonl 文件",
    )
    _add_logging_args(p_infer_collision)
    p_infer_collision.set_defaults(func=_cmd_infer_collision)

    p_viz = sub.add_parser("viz-frames", help="生成交互 HTML，叠加绘制骨架与货框标注")
    p_viz.add_argument("--record-dir", required=True, help="记录目录")
    p_viz.add_argument("--output", required=True, help="输出 HTML 文件路径")
    p_viz.add_argument("--max-frames", type=int, default=10, help="最多可视化帧数（默认 10）")
    p_viz.add_argument("--start-frame", type=int, default=1, help="起始帧索引（默认 1）")
    p_viz.add_argument(
        "--predictions",
        default="",
        help="可选：infer-rule 输出的 JSONL，用于叠加碰撞/报警高亮",
    )
    _add_logging_args(p_viz)
    p_viz.set_defaults(func=_cmd_viz_frames)

    p_feature_curves = sub.add_parser("viz-feature-curves", help="generate feature curve HTML from frame_features.parquet")
    p_feature_curves.add_argument("--features-dir", required=True, help="export-features output directory")
    p_feature_curves.add_argument("--output", default="", help="output HTML path, default: features-dir/feature_curves.html")
    _add_logging_args(p_feature_curves)
    p_feature_curves.set_defaults(func=_cmd_viz_feature_curves)

    p_box_layout = sub.add_parser("box-layout", help="读取 annotation 并输出各货框数值布局")
    source_group = p_box_layout.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--annotation", default="", help="annotation.json 路径")
    source_group.add_argument("--record-dir", default="", help="记录目录（读取其中的 annotation.json）")
    p_box_layout.add_argument(
        "--format",
        default="json",
        choices=["json", "csv", "table"],
        help="输出格式（默认 json）",
    )
    p_box_layout.add_argument("--output", default="", help="输出文件路径（默认打印到 stdout）")
    p_box_layout.add_argument("--viz-output", default="", help="可选：输出 box_layout SVG 可视化文件")
    _add_logging_args(p_box_layout)
    p_box_layout.set_defaults(func=_cmd_box_layout)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
