# 数据格式说明

训练和评测以“记录目录”为基本单位。一个有效记录目录至少包含：

```text
record_xxx/
  annotation.json
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

货框 token 规则与采集项目保持一致：

- 有 `shelf_code` 时：`<shelf_code>:<box_id>`
- 无 `shelf_code` 时：`Box_<box_id>`

例如：

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

`skeleton.parquet` 是逐帧人体骨骼数据。每行表示一帧中的一个人。

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

若某帧在 `skeleton.parquet` 中无有效关键点（全为空或置信度过低），训练时会默认过滤该帧；导出特征与评测仍保留这些帧。

## event_review.json

`event_review.json` 是人工复核结果，用于构建监督信号。

核心字段：

- `verified_true`：人工确认的取货事件列表。
- `verified_true[].frame_idx`：取货事件所在帧。
- `verified_true[].confirmed_box_tokens`：人工确认的取货货框 token 列表。

示例：

```json
{
  "verified_true": [
    {
      "event_type": "collision",
      "frame_idx": 120,
      "box_tokens": ["S1:A1"],
      "confirmed_box_tokens": ["S1:A1"]
    }
  ]
}
```

规则：

- 如果没有 `event_review.json`，则该记录所有帧都视为非取货。
- 如果某帧不在 `verified_true` 中，则该帧视为非取货。
- 如果 `confirmed_box_tokens` 缺失，该帧仍可作为取货帧，但货框识别评测不会把它作为有效货框真值。
