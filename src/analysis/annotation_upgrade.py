"""Upgrade legacy annotation.json files to shelves[] format."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analysis.constants import ANNOTATION_FILE
from analysis.records import discover_record_dirs


@dataclass
class AnnotationUpgradeStats:
    record_id: str
    input_path: str
    output_path: str
    upgraded: bool
    shelf_code: str
    box_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"annotation root must be an object: {path}")
    return data


def _infer_shelf_code(annotation: dict[str, Any], *, shelf_code: str = "") -> str:
    explicit = str(shelf_code or "").strip()
    if explicit:
        return explicit
    source_info = annotation.get("source_info")
    if isinstance(source_info, dict):
        for key in ("shelf_code", "camera_name"):
            value = str(source_info.get(key) or "").strip()
            if value:
                return value
    for key in ("shelf_code", "camera_name"):
        value = str(annotation.get(key) or "").strip()
        if value:
            return value
    return ""


def upgrade_annotation_payload(
    annotation: dict[str, Any],
    *,
    shelf_code: str = "",
) -> tuple[dict[str, Any], bool, str, int]:
    """Return annotation in shelves[] format.

    Legacy annotations with top-level boxes[] are converted to one shelf. Existing
    shelves[] annotations are returned unchanged.
    """
    if isinstance(annotation.get("shelves"), list) and annotation.get("shelves"):
        box_count = sum(
            len(shelf.get("boxes") or [])
            for shelf in annotation["shelves"]
            if isinstance(shelf, dict)
        )
        first_code = ""
        for shelf in annotation["shelves"]:
            if isinstance(shelf, dict):
                first_code = str(shelf.get("shelf_code") or "").strip()
                if first_code:
                    break
        return dict(annotation), False, first_code, box_count

    boxes = annotation.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        return dict(annotation), False, "", 0

    resolved_shelf_code = _infer_shelf_code(annotation, shelf_code=shelf_code)
    if not resolved_shelf_code:
        raise ValueError(
            "legacy annotation has top-level boxes[] but no shelf_code. "
            "Pass --shelf-code or add source_info.shelf_code before upgrading."
        )

    upgraded: dict[str, Any] = {}
    if isinstance(annotation.get("annotation_size"), dict):
        upgraded["annotation_size"] = annotation["annotation_size"]

    shelf: dict[str, Any] = {"shelf_code": resolved_shelf_code}
    source_info = annotation.get("source_info")
    if isinstance(source_info, dict) and source_info.get("camera_name"):
        shelf["shelf_name"] = str(source_info.get("camera_name"))
    if annotation.get("shelf_corners") is not None:
        shelf["shelf_corners"] = annotation.get("shelf_corners")
    if annotation.get("grid_shape") is not None:
        shelf["grid_shape"] = annotation.get("grid_shape")

    shelf_boxes: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        item = dict(box)
        item.pop("shelf_code", None)
        shelf_boxes.append(item)
    shelf["boxes"] = shelf_boxes
    upgraded["shelves"] = [shelf]

    return upgraded, True, resolved_shelf_code, len(shelf_boxes)


def upgrade_record_annotation(
    record_dir: Path,
    *,
    annotation_filename: str = ANNOTATION_FILE,
    output_path: Path | None = None,
    in_place: bool = False,
    backup: bool = True,
    shelf_code: str = "",
) -> AnnotationUpgradeStats:
    record_dir = Path(record_dir).resolve()
    input_path = record_dir / annotation_filename
    if not input_path.is_file():
        raise FileNotFoundError(f"annotation file not found: {input_path}")

    annotation = _read_json(input_path)
    payload, upgraded, resolved_shelf_code, box_count = upgrade_annotation_payload(
        annotation,
        shelf_code=shelf_code,
    )

    default_output = record_dir / f"{input_path.stem}_v2{input_path.suffix}"
    out = input_path if in_place else (Path(output_path) if output_path else default_output)
    if in_place and backup:
        shutil.copy2(input_path, input_path.with_suffix(input_path.suffix + ".bak"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return AnnotationUpgradeStats(
        record_id=record_dir.name,
        input_path=str(input_path),
        output_path=str(out),
        upgraded=upgraded,
        shelf_code=resolved_shelf_code,
        box_count=box_count,
    )


def upgrade_annotations(
    data_dir: Path,
    *,
    annotation_filename: str = ANNOTATION_FILE,
    output_dir: Path | None = None,
    in_place: bool = False,
    backup: bool = True,
    shelf_code: str = "",
) -> list[AnnotationUpgradeStats]:
    record_dirs = discover_record_dirs(Path(data_dir))
    if not record_dirs:
        raise FileNotFoundError(f"no valid record directories found under: {data_dir}")
    results: list[AnnotationUpgradeStats] = []
    for record_dir in record_dirs:
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir) / Path(record_dir).name / annotation_filename
        results.append(
            upgrade_record_annotation(
                record_dir,
                annotation_filename=annotation_filename,
                output_path=output_path,
                in_place=in_place,
                backup=backup,
                shelf_code=shelf_code,
            )
        )
    return results
