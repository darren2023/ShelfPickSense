# 数据格式说明

训练和评测以“记录目录”为基本单位。一个有效记录目录至少包含：

```text
record_xxx/
  annotation.json
  annotation_v2.json  # 可选，存在时优先于 annotation.json
  skeleton.parquet
  event_review.json   # 可选
```

## annotation.json

`annotation.json` 是货架货框标注，支持 `shelves[].boxes[]` 格式，也兼容 legacy 顶层 `boxes[]`。

核心字段：

- `shelves[].shelf_code`：货架编码。
- `shelves[].boxes[].box_id`：货框编码。
- `video_polygon`：货框多边形坐标。
- `video_polygon_norm`：可选，归一化多边形坐标。
- `annotation_size.width / height`：标注坐标尺寸。
- `video_polygon_norm`：相对 `annotation_size` 的归一化坐标，缩放货框时优先使用。

货框坐标会按 pose 推理尺寸变换。默认推理尺寸为 `852x480`，与 `640x360` 标注下的 skeleton 坐标系一致。记录目录若包含 `manifest.json`，会优先读取其中的 `infer_width` / `infer_height`。

货框 token 规则与采集项目保持一致：

- 有 `shelf_code` 时：`<shelf_code>:<box_id>`
- 无 `shelf_code` 时：`Box_<box_id>`

旧版 annotation 可能使用顶层 `boxes[]`、`shelf_corners`、`grid_shape`。可用升级脚本转换为 `shelves[]` 格式：

```powershell
uv run python scripts/upgrade_annotation_json.py --data-dir .\data\data28-merged\data28-merged\Train\record_026
```

默认会在记录目录生成 `annotation_v2.json`，加载记录时会优先使用该文件。确认无误后也可以覆盖原文件并备份：

```powershell
uv run python scripts/upgrade_annotation_json.py --data-dir .\data\data28-merged\data28-merged\Train --in-place
```

示例：

```json
{
  "annotation_size": {"width": 640, "height": 480},
  "shelves": [
    {
      "shelf_code": "S1",
      "boxes": [
        {
          "box_id": "A1",
          "video_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]
        }
      ]
    }
  ]
}
```

## skeleton.parquet

`skeleton.parquet` 是逐帧人体骨架数据。每行表示一帧中的一个人。

常用字段：

- `frame_idx`：帧索引。
- `source_frame_idx`：源视频帧索引，可选。
- `timestamp_sec`：时间戳秒数。
- `person_id`：人体 ID。
- `person_track_id`：跟踪 ID，可选。
- `bbox_x1 / bbox_y1 / bbox_x2 / bbox_y2`：人体框，可选。
- `kpt_0_x / kpt_0_y / kpt_0_score` ... `kpt_16_x / kpt_16_y / kpt_16_score`：COCO17 关键点。

COCO17 中与当前特征强相关的关键点：

- `kpt_5 / kpt_6`：左右肩。
- `kpt_9 / kpt_10`：左右手腕。

如果某帧在 `skeleton.parquet` 中无有效关键点（全为空或置信度过低），训练时默认会过滤该帧；导出特征与评测仍会保留这些帧。

## event_review.json

`event_review.json` 是人工复核结果，用于构建监督信号。当前特征提取和训练只支持 `schema: 2`。

核心字段：

- `schema`：必须为 `2`。
- `verified_true`：人工确认的取货事件列表。
- `verified_true[].frame_idx`：取货事件所在帧。
- `verified_true[].is_pick`：必须为 `true`。
- `verified_true[].shelf_code` / `verified_true[].box_id`：目标货框，可推导出 canonical token，二者都必须非空。
- `verified_true[].person_track_id`：取货人 track id。

示例：

```json
{
  "schema": 2,
  "status": "completed",
  "verified_true": [
    {
      "event_type": "collision",
      "frame_idx": 120,
      "is_pick": true,
      "person_track_id": 2,
      "shelf_code": "S1",
      "box_id": "A1"
    }
  ]
}
```

规则：

- 如果没有 `event_review.json`，则该记录所有帧都视为非取货。
- 如果某帧不在 `verified_true` 中，则该帧视为非取货。
- 旧版字段 `tokens`、`box_tokens`、`token`、`box_token`、`confirmed_box_tokens`、`confirmed_box_token` 已不再被特征提取支持；遇到旧格式会报错并要求先升级 review 文件。
