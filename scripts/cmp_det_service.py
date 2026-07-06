"""对比 output.jsonl 和 rule-baseline-prod 的 JSON 文件"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def save_diff_report(output_path, report_data):
    """保存差异报告到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n差异报告已保存到: {output_path}")


def load_jsonl(path):
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(path):
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_files(jsonl_path, json_path, output_dir=None):
    """对比两个文件"""
    jsonl_data = load_jsonl(jsonl_path)
    json_data = load_json(json_path)

    print(f"=== 文件基本信息 ===")
    print(f"JSONL 文件: {jsonl_path}")
    print(f"  - 行数: {len(jsonl_data)}")
    print(f"  - 字段: {list(jsonl_data[0].keys()) if jsonl_data else 'N/A'}")

    print(f"\nJSON 文件: {json_path}")
    print(f"  - 行数: {len(json_data)}")
    print(f"  - 字段: {list(json_data[0].keys()) if json_data else 'N/A'}")

    # 对比字段
    jsonl_fields = set(jsonl_data[0].keys()) if jsonl_data else set()
    json_fields = set(json_data[0].keys()) if json_data else set()

    print(f"\n=== 字段对比 ===")
    print(f"仅 JSONL 有: {jsonl_fields - json_fields}")
    print(f"仅 JSON 有: {json_fields - jsonl_fields}")
    print(f"共同字段: {jsonl_fields & json_fields}")

    # 对比 record_id
    print(f"\n=== record_id 对比 ===")
    jsonl_record_ids = set(d.get('record_id') for d in jsonl_data)
    json_record_ids = set(d.get('record_id') for d in json_data)
    print(f"JSONL record_id: {jsonl_record_ids}")
    print(f"JSON record_id: {json_record_ids}")

    # 对比帧索引范围
    print(f"\n=== 帧索引对比 ===")
    jsonl_frames = [d.get('frame_idx') for d in jsonl_data]
    json_frames = [d.get('frame_idx') for d in json_data]
    print(f"JSONL 帧范围: {min(jsonl_frames)} - {max(jsonl_frames)} (共 {len(jsonl_frames)} 帧)")
    print(f"JSON 帧范围: {min(json_frames)} - {max(json_frames)} (共 {len(json_frames)} 帧)")

    # 检查 picking 帧数
    jsonl_picking = sum(1 for d in jsonl_data if d.get('is_picking'))
    json_picking = sum(1 for d in json_data if d.get('is_picking'))
    print(f"\n=== picking 检测对比 ===")
    print(f"JSONL picking 帧数: {jsonl_picking}")
    print(f"JSON picking 帧数: {json_picking}")

    # 检查有碰撞的帧
    jsonl_collision = sum(1 for d in jsonl_data if d.get('collision_collisions'))
    json_collision = sum(1 for d in json_data if d.get('rule_collisions'))
    print(f"\n=== 碰撞检测对比 ===")
    print(f"JSONL 有碰撞帧数: {jsonl_collision}")
    print(f"JSON 有碰撞帧数: {json_collision}")

    # 检查报警帧
    jsonl_alarm = sum(1 for d in jsonl_data if d.get('collision_alarm_collisions'))
    json_alarm = sum(1 for d in json_data if d.get('rule_alarm_collisions'))
    print(f"\n=== 报警检测对比 ===")
    print(f"JSONL 报警帧数: {jsonl_alarm}")
    print(f"JSON 报警帧数: {json_alarm}")

    # 详细对比共同字段值
    print(f"\n=== 共同字段值对比 ===")

    # 构建帧索引映射
    jsonl_by_frame = {d.get('frame_idx'): d for d in jsonl_data}
    json_by_frame = {d.get('frame_idx'): d for d in json_data}

    # 获取共同的帧
    common_frames = set(jsonl_by_frame.keys()) & set(json_by_frame.keys())
    print(f"共同帧数: {len(common_frames)}")

    # 对比 is_picking
    picking_diff = []
    for frame_idx in common_frames:
        jsonl_val = jsonl_by_frame[frame_idx].get('is_picking')
        json_val = json_by_frame[frame_idx].get('is_picking')
        if jsonl_val != json_val:
            picking_diff.append((frame_idx, jsonl_val, json_val))

    print(f"\nis_picking 差异帧数: {len(picking_diff)}")
    if picking_diff and len(picking_diff) <= 10:
        print(f"  差异详情: {picking_diff[:10]}")

    # 对比 picking_prob
    prob_diff = []
    for frame_idx in common_frames:
        jsonl_val = jsonl_by_frame[frame_idx].get('picking_prob')
        json_val = json_by_frame[frame_idx].get('picking_prob')
        # 处理 None 和 0.0 的比较
        if json_val is None and jsonl_val == 0.0:
            continue
        if jsonl_val != json_val:
            prob_diff.append((frame_idx, jsonl_val, json_val))

    print(f"\npicking_prob 差异帧数: {len(prob_diff)}")
    if prob_diff and len(prob_diff) <= 10:
        print(f"  差异详情: {prob_diff[:10]}")

    # 对比 predicted_box_tokens
    tokens_diff = []
    for frame_idx in common_frames:
        jsonl_val = set(jsonl_by_frame[frame_idx].get('predicted_box_tokens', []))
        json_val = set(json_by_frame[frame_idx].get('predicted_box_tokens', []))
        if jsonl_val != json_val:
            tokens_diff.append((frame_idx, jsonl_val, json_val))

    print(f"\npredicted_box_tokens 差异帧数: {len(tokens_diff)}")
    if tokens_diff and len(tokens_diff) <= 10:
        print(f"  差异详情: {tokens_diff[:10]}")

    # 对比碰撞字段（字段名不同但含义相同）
    collision_diff = []
    for frame_idx in common_frames:
        jsonl_val = set(jsonl_by_frame[frame_idx].get('collision_collisions', []))
        json_val = set(json_by_frame[frame_idx].get('rule_collisions', []))
        if jsonl_val != json_val:
            collision_diff.append((frame_idx, jsonl_val, json_val))

    print(f"\ncollision 差异帧数: {len(collision_diff)}")
    if collision_diff and len(collision_diff) <= 10:
        print(f"  差异详情: {collision_diff[:10]}")

    # 对比报警字段
    alarm_diff = []
    for frame_idx in common_frames:
        jsonl_val = set(jsonl_by_frame[frame_idx].get('collision_alarm_collisions', []))
        json_val = set(json_by_frame[frame_idx].get('rule_alarm_collisions', []))
        if jsonl_val != json_val:
            alarm_diff.append((frame_idx, jsonl_val, json_val))

    print(f"\nalarm 差异帧数: {len(alarm_diff)}")
    if alarm_diff and len(alarm_diff) <= 10:
        print(f"  差异详情: {alarm_diff[:10]}")

    # 详细分析差异类型
    print(f"\n=== 差异详细分析 ===")

    # 分析 picking_prob 差异原因
    jsonl_prob_values = set(d.get('picking_prob') for d in jsonl_data)
    json_prob_values = set(d.get('picking_prob') for d in json_data)
    print(f"\npicking_prob 值分布:")
    print(f"  JSONL: {jsonl_prob_values}")
    print(f"  JSON: {json_prob_values}")

    # 分析 is_picking 差异的具体帧
    if picking_diff:
        print(f"\nis_picking 差异详情 (前20个):")
        for frame_idx, jsonl_val, json_val in picking_diff[:20]:
            jsonl_coll = jsonl_by_frame[frame_idx].get('collision_collisions', [])
            jsonl_alarm = jsonl_by_frame[frame_idx].get('collision_alarm_collisions', [])
            json_coll = json_by_frame[frame_idx].get('rule_collisions', [])
            json_alarm = json_by_frame[frame_idx].get('rule_alarm_collisions', [])
            print(f"  帧 {frame_idx}: JSONL={jsonl_val} JSON={json_val}")
            print(f"    JSONL collision={jsonl_coll}, alarm={jsonl_alarm}")
            print(f"    JSON rule_coll={json_coll}, alarm={json_alarm}")

    # 分析 alarm 差异的具体帧
    if alarm_diff:
        print(f"\nalarm 差异详情 (前20个):")
        for frame_idx, jsonl_val, json_val in alarm_diff[:20]:
            jsonl_picking = jsonl_by_frame[frame_idx].get('is_picking')
            json_picking = json_by_frame[frame_idx].get('is_picking')
            print(f"  帧 {frame_idx}: JSONL={list(jsonl_val)} JSON={list(json_val)}")
            print(f"    JSONL picking={jsonl_picking}, JSON picking={json_picking}")

    # 生成差异报告
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "files": {
                "jsonl": str(jsonl_path),
                "json": str(json_path)
            },
            "summary": {
                "jsonl_frames": len(jsonl_data),
                "json_frames": len(json_data),
                "common_frames": len(common_frames),
                "jsonl_picking_count": sum(1 for d in jsonl_data if d.get('is_picking')),
                "json_picking_count": sum(1 for d in json_data if d.get('is_picking')),
            },
            "differences": {
                "is_picking": [
                    {"frame_idx": f, "jsonl": j, "json": r}
                    for f, j, r in picking_diff
                ],
                "collision": [
                    {"frame_idx": f, "jsonl": list(j), "json": list(r)}
                    for f, j, r in collision_diff
                ],
                "alarm": [
                    {"frame_idx": f, "jsonl": list(j), "json": list(r)}
                    for f, j, r in alarm_diff
                ],
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"diff_report_{timestamp}.json"
        save_diff_report(report_path, report)


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="对比 JSONL 和 JSON 文件的差异"
    )
    parser.add_argument(
        "jsonl",
        help="JSONL 文件路径"
    )
    parser.add_argument(
        "json",
        help="JSON 文件路径"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="差异报告输出目录（默认为输出文件所在目录的 diff_reports 子目录）"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成差异报告文件"
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    json_path = Path(args.json)

    if not jsonl_path.exists():
        print(f"错误: JSONL 文件不存在: {jsonl_path}")
        sys.exit(1)

    if not json_path.exists():
        print(f"错误: JSON 文件不存在: {json_path}")
        sys.exit(1)

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = jsonl_path.parent / "diff_reports"

    # 如果不需要生成报告，传入 None
    final_output_dir = None if args.no_report else output_dir

    compare_files(jsonl_path, json_path, output_dir=final_output_dir)
