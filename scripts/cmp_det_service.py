"""对比 collision_infer.jsonl 和 collision_infer_base.json，自动解析 record 目录"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List


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


def find_records(data_dir: Path) -> List[Path]:
    """查找所有 record 目录"""
    records = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('record_'):
            records.append(item)
    return sorted(records)


def compare_record(record_dir: Path) -> Dict:
    """对比单个 record 目录中的两个文件"""
    jsonl_path = record_dir / 'collision_infer.jsonl'
    json_path = record_dir / 'collision_infer_base.json'

    result = {
        'record_id': record_dir.name,
        'status': 'error',
        'error': None,
        'summary': None,
        'differences': None,
    }

    # 检查文件是否存在
    if not jsonl_path.exists():
        result['error'] = 'collision_infer.jsonl 不存在'
        return result

    if not json_path.exists():
        result['error'] = 'collision_infer_base.json 不存在'
        return result

    try:
        jsonl_data = load_jsonl(jsonl_path)
        json_data = load_json(json_path)

        # 构建帧索引映射
        jsonl_by_frame = {d.get('frame_idx'): d for d in jsonl_data}
        json_by_frame = {d.get('frame_idx'): d for d in json_data}

        # 获取共同的帧
        common_frames = set(jsonl_by_frame.keys()) & set(json_by_frame.keys())

        # 对比 is_picking
        picking_diff = []
        for frame_idx in sorted(common_frames):
            jsonl_val = jsonl_by_frame.get(frame_idx, {}).get('is_picking')
            json_val = json_by_frame.get(frame_idx, {}).get('is_picking')
            if jsonl_val != json_val:
                picking_diff.append({
                    'frame_idx': frame_idx,
                    'collision_infer': jsonl_val,
                    'collision_infer_base': json_val,
                })

        # 对比 alarm
        alarm_diff = []
        for frame_idx in sorted(common_frames):
            jsonl_val = set(jsonl_by_frame.get(frame_idx, {}).get('collision_alarm_collisions', []))
            json_val = set(json_by_frame.get(frame_idx, {}).get('collision_alarm_collisions', []))
            if jsonl_val != json_val:
                alarm_diff.append({
                    'frame_idx': frame_idx,
                    'collision_infer': list(jsonl_val),
                    'collision_infer_base': list(json_val),
                })

        # 对比 collisions
        collision_diff = []
        for frame_idx in sorted(common_frames):
            jsonl_val = set(jsonl_by_frame.get(frame_idx, {}).get('collision_collisions', []))
            json_val = set(json_by_frame.get(frame_idx, {}).get('collision_collisions', []))
            if jsonl_val != json_val:
                collision_diff.append({
                    'frame_idx': frame_idx,
                    'collision_infer': list(jsonl_val),
                    'collision_infer_base': list(json_val),
                })

        # 统计 picking 帧数
        jsonl_picking_count = sum(1 for d in jsonl_data if d.get('is_picking'))
        json_picking_count = sum(1 for d in json_data if d.get('is_picking'))

        result['status'] = 'success'
        result['summary'] = {
            'jsonl_frames': len(jsonl_data),
            'json_frames': len(json_data),
            'common_frames': len(common_frames),
            'jsonl_picking_count': jsonl_picking_count,
            'json_picking_count': json_picking_count,
            'jsonl_alarm_count': sum(1 for d in jsonl_data if d.get('collision_alarm_collisions')),
            'json_alarm_count': sum(1 for d in json_data if d.get('collision_alarm_collisions')),
        }
        result['differences'] = {
            'is_picking_count': len(picking_diff),
            'is_picking_details': picking_diff[:100],  # 限制详情数量
            'alarm_count': len(alarm_diff),
            'alarm_details': alarm_diff[:100],
            'collision_count': len(collision_diff),
            'collision_details': collision_diff[:100],
        }

    except Exception as e:
        result['error'] = str(e)

    return result


def save_report(report: Dict, output_path: Path):
    """保存差异报告到 JSON 文件"""
    report['generated_at'] = datetime.now().isoformat()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_summary(results: List[Dict]):
    """打印汇总信息"""
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{'='*60}")
    print("批量对比汇总")
    print(f"{'='*60}")
    print(f"总计: {total}")
    print(f"成功: {success}")
    print(f"错误: {errors}")

    if success > 0:
        print(f"\n{'='*60}")
        print("差异统计")
        print(f"{'='*60}")

        total_picking_diff = 0
        total_alarm_diff = 0
        total_collision_diff = 0

        for r in results:
            if r['status'] == 'success' and r['differences']:
                total_picking_diff += r['differences'].get('is_picking_count', 0)
                total_alarm_diff += r['differences'].get('alarm_count', 0)
                total_collision_diff += r['differences'].get('collision_count', 0)

        print(f"is_picking 差异帧数总计: {total_picking_diff}")
        print(f"alarm 差异帧数总计: {total_alarm_diff}")
        print(f"collision 差异帧数总计: {total_collision_diff}")

        # 打印每个 record 的简要统计
        print(f"\n{'='*60}")
        print("各 record 统计")
        print(f"{'='*60}")
        print(f"{'Record':<15} {'JSONL帧':<8} {'JSON帧':<8} {'Picking差异':<12} {'Alarm差异':<12}")
        print(f"{'-'*60}")

        for r in results:
            if r['status'] == 'success':
                record_id = r['record_id']
                summary = r.get('summary', {})
                diff = r.get('differences', {})
                print(f"{record_id:<15} {summary['jsonl_frames']:<8} {summary['json_frames']:<8} "
                      f"{diff['is_picking_count']:<12} {diff['alarm_count']:<12}")
            else:
                print(f"{r['record_id']:<15} 错误: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="批量对比 collision_infer.jsonl 和 collision_infer_base.json"
    )
    parser.add_argument(
        "data_dir",
        help="包含 record 目录的数据目录"
    )
    parser.add_argument(
        "--record",
        default=None,
        help="指定单个 record ID（如 record_001），不指定则处理所有 record"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成差异报告文件"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)

    # 获取要处理的 record 列表
    if args.record:
        record_dirs = [data_dir / args.record]
    else:
        record_dirs = find_records(data_dir)

    if not record_dirs:
        print(f"错误: 未找到 record 目录")
        sys.exit(1)

    print(f"找到 {len(record_dirs)} 个 record 目录")
    print('='*60)

    results = []
    for record_dir in record_dirs:
        print(f"处理: {record_dir.name}")
        result = compare_record(record_dir)
        results.append(result)

        if result['status'] == 'success':
            diff = result['differences']
            summary = result['summary']
            print(f"  JSONL帧: {summary['jsonl_frames']}, JSON帧: {summary['json_frames']}")
            print(f"  Picking差异: {diff['is_picking_count']}, Alarm差异: {diff['alarm_count']}, Collision差异: {diff['collision_count']}")

            # 保存报告到 record 目录
            if not args.no_report:
                report_path = record_dir / 'diff_report.json'
                save_report(result, report_path)
                print(f"  报告已保存: {report_path}")
        else:
            print(f"  错误: {result['error']}")

    # 打印汇总
    print_summary(results)


if __name__ == "__main__":
    main()
