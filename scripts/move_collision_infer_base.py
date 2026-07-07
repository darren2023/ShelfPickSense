"""将 rule-baseline-prod-test 中的推理结果移动到对应的 record 目录"""

import json
import shutil
from pathlib import Path
from typing import Dict, List


def load_manifest(manifest_path: Path) -> Dict:
    """加载 _manifest.json 文件"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_record_name(infer_size_record_dir: str) -> str:
    """从 infer_size_record_dir 路径中提取 record 名称

    例如: D:\\work\\workspace\\git-repo\\ShelfPickSense\\data\\data28-merged\\Train\\record_001
    返回: record_001
    """
    path = Path(infer_size_record_dir)
    return path.name


def build_target_path(record_name: str, base_path: Path) -> Path:
    """构建目标路径

    Args:
        record_name: record 名称，如 record_001
        base_path: 基础路径，如 D:\\...\\data28-merged\\Train

    Returns:
        完整的目标路径
    """
    return base_path / record_name


def move_files(manifest_path: Path, target_base_path: Path, source_dir: Path | None = None, dry_run: bool = True) -> List[Dict]:
    """移动文件到对应目录

    Args:
        manifest_path: _manifest.json 文件路径
        target_base_path: 目标基础路径
        source_dir: 源目录（如果提供，将使用此目录而非 manifest 中的 path）
        dry_run: 是否仅模拟运行（不实际移动文件）

    Returns:
        移动操作列表
    """
    manifest = load_manifest(manifest_path)
    records = manifest.get('records', [])

    operations = []
    success_count = 0
    skip_count = 0
    error_count = 0

    for record in records:
        file_name = record.get('file')
        infer_size_record_dir = record.get('infer_size_record_dir')

        if not all([file_name, infer_size_record_dir]):
            print(f"跳过: 字段不完整 - {record.get('record_id')}")
            skip_count += 1
            continue

        # 使用源目录（如果提供），否则使用 manifest 中的路径
        if source_dir:
            source_path = source_dir / file_name
        else:
            source_path_str = record.get('path')
            if not source_path_str:
                print(f"跳过: 未提供源目录且 manifest 中无 path - {record.get('record_id')}")
                skip_count += 1
                continue
            source_path = Path(source_path_str)

        record_name = extract_record_name(infer_size_record_dir)
        target_dir = build_target_path(record_name, target_base_path)
        target_path = target_dir / 'collision_infer_base.json'

        operation = {
            'record_id': record.get('record_id'),
            'source': str(source_path),
            'target': str(target_path),
            'status': 'pending',
        }

        # 检查源文件是否存在
        if not source_path.exists():
            operation['status'] = 'error'
            operation['error'] = '源文件不存在'
            error_count += 1
            operations.append(operation)
            print(f"错误: 源文件不存在 - {source_path}")
            continue

        # 检查目标目录是否存在，不存在则创建
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        # 如果目标文件已存在，先备份
        if target_path.exists():
            backup_path = target_path.with_suffix('.bak')
            if not dry_run:
                shutil.move(str(target_path), str(backup_path))
            print(f"备份已存在文件: {target_path} -> {backup_path}")

        # 执行移动
        if not dry_run:
            try:
                shutil.move(str(source_path), str(target_path))
                operation['status'] = 'success'
                success_count += 1
                print(f"移动成功: {source_path} -> {target_path}")
            except Exception as e:
                operation['status'] = 'error'
                operation['error'] = str(e)
                error_count += 1
                print(f"移动失败: {source_path} -> {error}")
        else:
            operation['status'] = 'dry_run'
            print(f"[DRY RUN] 将移动: {source_path} -> {target_path}")
            success_count += 1

        operations.append(operation)

    print(f"\n统计:")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")
    print(f"  总计: {len(records)}")

    return operations


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="将 rule-baseline-prod-test 中的推理结果移动到对应的 record 目录"
    )
    parser.add_argument(
        '--manifest',
        default=r'D:\project\git\CV\box_human_yanfeng\ShelfPickSense\data\rule-baseline-prod-test\_manifest.json',
        help='_manifest.json 文件路径'
    )
    parser.add_argument(
        '--source-dir',
        default=r'D:\project\git\CV\box_human_yanfeng\ShelfPickSense\data\rule-baseline-prod-test',
        help='源文件所在目录（包含 *.json 文件）'
    )
    parser.add_argument(
        '--target-base',
        default=r'D:\project\git\CV\box_human_yanfeng\ShelfPickSense\data\data28-merged\data28-merged\Train',
        help='目标基础路径'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际移动文件'
    )
    parser.add_argument(
        '--output-ops',
        help='将操作记录输出到指定文件（JSON 格式）'
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    source_dir = Path(args.source_dir)
    target_base_path = Path(args.target_base)

    print(f"源 manifest: {manifest_path}")
    print(f"源目录: {source_dir}")
    print(f"目标基础路径: {target_base_path}")
    print(f"模式: {'DRY RUN (不实际移动)' if args.dry_run else '实际移动'}")
    print()

    operations = move_files(
        manifest_path,
        target_base_path,
        source_dir=source_dir,
        dry_run=args.dry_run
    )

    if args.output_ops:
        output_path = Path(args.output_ops)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(operations, f, ensure_ascii=False, indent=2)
        print(f"\n操作记录已保存: {output_path}")


if __name__ == '__main__':
    main()
