"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

from analysis.benchmark import DEFAULT_MODEL_NAMES, run_benchmark
from analysis.evaluation import compare_reports, evaluate_model, save_report
from analysis.features.selection import load_feature_selection
from analysis.models import SUPPORTED_MODEL_NAMES
from analysis.realtime import RealtimePickingPredictor
from analysis.train import train_model


def configure_logging(args: argparse.Namespace) -> None:
    """配置 loguru。日志默认写 stderr，保留 stdout 给 JSON 结果。"""
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
    )
    logger.info(
        "训练完成: model={}, frames={}, positive_frames={}, box_samples={}",
        result.model_name,
        result.frame_count,
        result.positive_frames,
        result.box_samples,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
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
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {out}")
    print(f"预测结果已保存: {predictions_out}")
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
    print(json.dumps(rows, ensure_ascii=False, indent=2))
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
    )
    logger.info("benchmark 完成: models={}, output={}", len(result.model_names), result.output_dir)
    print(json.dumps(result.comparison, ensure_ascii=False, indent=2))
    print(f"\n批量对比报告已保存: {Path(args.output) / 'benchmark_summary.json'}")
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


def _cmd_export_features(args: argparse.Namespace) -> int:
    import pandas as pd

    from analysis.dataset import load_dataset

    logger.info(
        "开始提取特征: data_dir={}, output={}, format={}, feature_config={}",
        args.data_dir,
        args.output,
        args.format,
        args.feature_config or "",
    )
    feature_selection = load_feature_selection(args.feature_config)
    dataset = load_dataset(Path(args.data_dir), feature_selection=feature_selection)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    for sample in dataset.frame_samples:
        row = {
            "record_id": sample.record_id,
            "frame_idx": sample.frame_idx,
            "is_picking": sample.is_picking,
            "confirmed_box_tokens": sample.confirmed_box_tokens,
        }
        row.update(dict(zip(dataset.frame_feature_names, sample.x.tolist(), strict=True)))
        frame_rows.append(row)

    box_rows = []
    for sample in dataset.box_samples:
        row = {
            "record_id": sample.record_id,
            "frame_idx": sample.frame_idx,
            "box_token": sample.box_token,
            "is_target": sample.is_target,
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
        "data_dir": str(Path(args.data_dir)),
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
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "特征提取完成: frames={}, box_samples={}, output={}",
        meta["frame_count"],
        meta["box_sample_count"],
        out_dir,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    for output_format, paths in output_files.items():
        print(f"\n{output_format} 帧级特征已保存: {paths['frame_features_path']}")
        print(f"{output_format} 货框特征已保存: {paths['box_features_path']}")
    print(f"特征元数据已保存: {meta_path}")
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
        )
    logger.info(
        "特征相关性分析完成: frames={}, box_samples={}, output={}",
        result.frame_count,
        result.box_sample_count,
        result.output_dir,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n相关性分析结果已保存: {result.output_dir}")
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
            print(line)
            if frame_interval > 0:
                time.sleep(frame_interval)
    finally:
        if out_file:
            out_file.close()
    return 0


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
    _add_logging_args(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="评测模型")
    p_eval.add_argument("--data-dir", required=True, help="评测数据目录")
    p_eval.add_argument("--model", required=True, help="已训练模型目录")
    p_eval.add_argument("--report", default="", help="评测报告输出路径（默认写入模型目录）")
    p_eval.add_argument("--predictions", default="", help="预测结果输出路径（默认写入模型目录）")
    _add_logging_args(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

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
    _add_logging_args(p_bench)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_export = sub.add_parser("export-features", help="从记录提取特征并保存到文件")
    p_export.add_argument("--data-dir", required=True, help="数据目录（含多条记录或单条记录）")
    p_export.add_argument("--output", required=True, help="特征输出目录")
    p_export.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "csv", "jsonl", "all"],
        help="特征文件格式（默认 parquet；可选 csv/jsonl/all）",
    )
    p_export.add_argument("--feature-config", default="", help="特征选择 JSON 配置路径（默认导出全部特征）")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
