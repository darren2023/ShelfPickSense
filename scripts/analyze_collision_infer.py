"""分析 collision_infer 结果，计算指标并打标签"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datetime import datetime


def load_jsonl(path: Path) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(path: Path) -> Dict:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(event_review_path: Path) -> Dict[int, Set[str]]:
    """加载 ground truth 数据"""
    data = load_json(event_review_path)
    verified_true = data.get('verified_true', [])

    # 构建 frame_idx -> confirmed_box_tokens 的映射
    ground_truth = defaultdict(set)
    for event in verified_true:
        frame_idx = event.get('frame_idx')
        confirmed_tokens = event.get('confirmed_box_tokens', [])
        if frame_idx and confirmed_tokens:
            ground_truth[frame_idx].update(confirmed_tokens)

    return dict(ground_truth)


def normalize_token(token: str, shelf_code: str = "MAP_6") -> str:
    """标准化 token 格式

    将各种格式统一为 shelf_code:box_id 格式
    - Box_1005 -> MAP_6:1005
    - MAP_6:1005 -> MAP_6:1005
    - 1005 -> MAP_6:1005
    """
    if not token:
        return token

    token = str(token).strip()

    # 已经是 shelf_code:box_id 格式
    if ':' in token:
        return token

    # Box_ 前缀格式
    if token.startswith('Box_'):
        box_id = token[4:]
        return f"{shelf_code}:{box_id}"

    # 纯数字格式
    if token.isdigit():
        return f"{shelf_code}:{token}"

    return token


def normalize_tokens(tokens: Set[str], shelf_code: str = "MAP_6") -> Set[str]:
    """标准化 token 集合"""
    return {normalize_token(t, shelf_code) for t in tokens}


def analyze_predictions(predictions: List[Dict], ground_truth: Dict[int, Set[str]]) -> Dict:
    """分析预测结果，基于 is_picking 进行分类"""
    # 构建 frame_idx -> predicted_tokens 的映射
    pred_by_frame = {p.get('frame_idx'): p for p in predictions}

    # 获取所有相关的帧
    all_frames = set(pred_by_frame.keys()) | set(ground_truth.keys())

    # 标准化 ground truth tokens
    normalized_ground_truth = {
        frame_idx: normalize_tokens(tokens, "MAP_6")
        for frame_idx, tokens in ground_truth.items()
    }

    # 初始化统计
    tp_frames = []      # True Positive 帧
    tp_mismatch_frames = []  # TP 不匹配帧（预测 picking 但 token 不匹配）
    fp_frames = []      # False Positive 帧
    fn_frames = []      # False Negative 帧
    tn_frames = []      # True Negative 帧

    # 详细的帧级别分析
    frame_analysis = []

    for frame_idx in sorted(all_frames):
        pred = pred_by_frame.get(frame_idx, {})
        gt_tokens = normalized_ground_truth.get(frame_idx, set())

        # Ground truth 状态：是否有 confirmed_box_tokens
        gt_is_picking = len(gt_tokens) > 0

        # 预测状态
        pred_is_picking = pred.get('is_picking', False)
        pred_alarms = set(pred.get('collision_alarm_collisions', []))
        pred_collisions = set(pred.get('collision_collisions', []))

        # 计算 token 匹配情况（集合相等）
        token_match = gt_tokens == pred_alarms

        # 根据 GT is_picking 状态进行分类
        if gt_is_picking:
            # Ground truth 为 picking 的情况
            if pred_is_picking:
                if token_match:
                    # 预测正确，完全匹配
                    frame_label = "TP"
                    tp_frames.append(frame_idx)
                else:
                    # 不匹配
                    frame_label = "TP_MISMATCH"
                    tp_mismatch_frames.append(frame_idx)
                    tp_frames.append(frame_idx)  # 仍然算作 TP，但标记为不匹配
            else:
                # GT 有 picking 但预测无 picking → FN
                frame_label = "FN"
                fn_frames.append(frame_idx)
        else:
            # Ground truth 为非 picking 的情况
            if pred_is_picking:
                # GT 无 picking 但预测有 picking → FP
                frame_label = "FP"
                fp_frames.append(frame_idx)
            else:
                # GT 无 picking 且预测无 picking → TN
                frame_label = "TN"
                tn_frames.append(frame_idx)

        frame_analysis.append({
            "frame_idx": frame_idx,
            "label": frame_label,
            "gt_is_picking": gt_is_picking,
            "gt_tokens": list(gt_tokens),
            "pred_is_picking": pred_is_picking,
            "pred_alarm_tokens": list(pred_alarms),
            "pred_collision_tokens": list(pred_collisions),
            # "token_match": token_match,
        })

    # 计算指标
    total_frames = len(all_frames)
    frame_tp = len(tp_frames)
    frame_tp_partial = len(tp_mismatch_frames)
    frame_fp = len(fp_frames)
    frame_fn = len(fn_frames)
    frame_tn = len(tn_frames)

    # 计算 precision, recall, f1
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    precision = frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0
    recall = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "metrics": {
            "frame_level": {
                "true_positives": frame_tp,
                "tp_partial_mismatch": frame_tp_partial,
                "false_positives": frame_fp,
                "false_negatives": frame_fn,
                "true_negatives": frame_tn,
                "total_frames": total_frames,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        },
        "frame_analysis": frame_analysis,
        "tp_frames": tp_frames,
        "tp_mismatch_frames": tp_mismatch_frames,
        "fp_frames": fp_frames,
        "fn_frames": fn_frames,
        "tn_frames": tn_frames,
    }


def save_labeled_results(frame_analysis: List[Dict], output_path: Path):
    """保存带标签的结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for fa in frame_analysis:
            f.write(json.dumps(fa, ensure_ascii=False) + '\n')


def save_metrics_report(analysis: Dict, output_path: Path, predictions_path: Path, ground_truth_path: Path):
    """保存指标报告"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "source_files": {
            "predictions": str(predictions_path),
            "ground_truth": str(ground_truth_path),
        },
        "metrics": analysis["metrics"],
        "summary": {
            "tp_frames_count": len(analysis["tp_frames"]),
            "fp_frames_count": len(analysis["fp_frames"]),
            "fn_frames_count": len(analysis["fn_frames"]),
            "tn_frames_count": len(analysis["tn_frames"]),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_report(analysis: Dict):
    """打印报告"""
    frame_metrics = analysis["metrics"]["frame_level"]

    print("\n" + "="*60)
    print("碰撞检测分析报告（帧级别）")
    print("="*60)

    print("\n【分类统计】")
    print(f"  TP (True Positive):        {frame_metrics['true_positives']}")
    print(f"  TP_MISMATCH (不匹配):     {frame_metrics.get('tp_partial_mismatch', 0)}")
    print(f"  FP (False Positive):       {frame_metrics['false_positives']}")
    print(f"  FN (False Negative):       {frame_metrics['false_negatives']}")
    print(f"  TN (True Negative):        {frame_metrics['true_negatives']}")
    print(f"  总帧数:                    {frame_metrics['total_frames']}")

    print("\n【性能指标】")
    print(f"  精度 (Precision):          {frame_metrics['precision']:.4f}")
    print(f"  召回率 (Recall):           {frame_metrics['recall']:.4f}")
    print(f"  F1 分数:                   {frame_metrics['f1_score']:.4f}")

    print("\n" + "="*60)


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="分析 collision_infer 结果，计算指标并打标签"
    )
    parser.add_argument(
        "predictions",
        help="预测结果文件路径（JSONL 或 JSON）"
    )
    parser.add_argument(
        "ground_truth",
        nargs='?',
        default=None,
        help="Ground truth 文件路径（event_review.json，默认与预测文件同目录）"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="输出目录（默认与预测文件同目录的 analysis_results 子目录）"
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    predictions_path = Path(args.predictions)

    # 如果没有指定 ground_truth，使用同一目录下的 event_review.json
    if args.ground_truth:
        ground_truth_path = Path(args.ground_truth)
    else:
        ground_truth_path = predictions_path.parent / "event_review.json"

    if not predictions_path.exists():
        print(f"错误: 预测文件不存在: {predictions_path}")
        exit(1)

    if not ground_truth_path.exists():
        print(f"错误: Ground truth 文件不存在: {ground_truth_path}")
        print(f"提示: 请确保 {ground_truth_path.name} 与预测文件在同一目录")
        exit(1)

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = predictions_path.parent / "analysis_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print(f"加载预测结果: {predictions_path}")
    if predictions_path.suffix == '.jsonl':
        predictions = load_jsonl(predictions_path)
    else:
        predictions = load_json(predictions_path)

    print(f"加载 Ground Truth: {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)

    print(f"预测帧数: {len(predictions)}")
    print(f"GT 正样本帧数: {len(ground_truth)}")

    # 分析预测结果
    analysis = analyze_predictions(predictions, ground_truth)

    # 打印报告
    print_report(analysis)

    # 保存带标签的结果
    labeled_path = output_dir / "labeled_results.jsonl"
    save_labeled_results(analysis["frame_analysis"], labeled_path)
    print(f"\n带标签结果已保存: {labeled_path}")

    # 保存指标报告
    report_path = output_dir / "metrics_report.json"
    save_metrics_report(analysis, report_path, predictions_path, ground_truth_path)
    print(f"指标报告已保存: {report_path}")
