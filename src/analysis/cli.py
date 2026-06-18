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
    logger.info("开始训练模型: model={}, data_dir={}, output={}", args.model, args.data_dir, args.output)
    result = train_model(
        Path(args.data_dir),
        Path(args.output),
        model_name=args.model,
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
    from analysis.evaluation import ModelEvaluation

    logger.info("开始对比评测报告: count={}", len(args.reports))
    reports: list[ModelEvaluation] = []
    for p in args.reports:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        from analysis.evaluation import BoxMetrics, ModelEvaluation, PickingMetrics

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
    )
    logger.info("benchmark 完成: models={}, output={}", len(result.model_names), result.output_dir)
    print(json.dumps(result.comparison, ensure_ascii=False, indent=2))
    print(f"\n批量对比报告已保存: {Path(args.output) / 'benchmark_summary.json'}")
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
    _add_logging_args(p_bench)
    p_bench.set_defaults(func=_cmd_benchmark)

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
