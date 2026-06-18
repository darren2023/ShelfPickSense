# 实时逐帧推理

实时推理面向外部应用集成。核心类是：

```python
from analysis.realtime import RealtimePickingPredictor
```

## 外部应用集成

推荐方式是先创建推理器，再按需加载模型、设置尺寸和标注。

```python
from analysis.realtime import RealtimePickingPredictor

predictor = RealtimePickingPredictor(record_id="camera_01")

# 模型按需加载
predictor.load_model("models/rf")

# 推理坐标尺寸按需设置
predictor.set_infer_size(640, 480)

# annotation 按需通过赋值设定
predictor.annotation = annotation_dict

# 每帧输入骨架数据
prediction = predictor.predict_frame(
    skeleton_persons,
    frame_idx=123,
    timestamp_sec=4.92,
)

print(prediction.to_dict())
```

输出示例：

```json
{
  "record_id": "camera_01",
  "frame_idx": 123,
  "is_picking": true,
  "picking_prob": 0.87,
  "predicted_box_tokens": ["S1:A1"]
}
```

## 骨架输入格式

`predict_frame()` 支持三类输入。

### persons 列表

```python
skeleton_persons = [
    {
        "person_id": 0,
        "keypoints": [
            [x, y, score],
            # ... 共 17 个 COCO 关键点
        ],
        "bbox": [x1, y1, x2, y2],
    }
]

prediction = predictor.predict_frame(
    skeleton_persons,
    frame_idx=123,
    timestamp_sec=4.92,
)
```

### 单帧 dict

```python
frame = {
    "frame_idx": 123,
    "timestamp_sec": 4.92,
    "persons": skeleton_persons,
}

prediction = predictor.predict_frame(frame)
```

也兼容 `skeletons` 字段：

```python
frame = {
    "frame_idx": 123,
    "skeletons": skeleton_persons,
}
```

### 单个人体 dict

```python
person = {
    "person_id": 0,
    "keypoints": [[x, y, score], ...],
}

prediction = predictor.predict_frame(person, frame_idx=123)
```

## annotation 设置

可以直接赋值：

```python
predictor.annotation = annotation_dict
```

也可以从文件加载：

```python
predictor.load_annotation("data/demo/annotation.json")
```

如果标注坐标需要指定推理尺寸：

```python
predictor.load_annotation(
    "data/demo/annotation.json",
    infer_width=640,
    infer_height=480,
)
```

或者：

```python
predictor.set_annotation(
    annotation_dict,
    infer_width=640,
    infer_height=480,
)
```

## 生命周期

实时应用中建议：

1. 应用启动时创建 `RealtimePickingPredictor`。
2. 根据模型配置调用 `load_model()`。
3. 根据当前货架/摄像头调用 `set_infer_size()` 和设置 `annotation`。
4. 每帧收到骨架数据后调用 `predict_frame()`。
5. 如果货架标注或模型切换，只更新 annotation 或重新 `load_model()`，无需重建对象。

## 模拟视频流 CLI

CLI 不直接解码视频，而是使用已抽取好的记录目录，按 `skeleton.parquet` 的帧顺序模拟视频流。

```bash
uv run python main.py infer-frame \
  --model models/rf \
  --record-dir data/demo \
  --video demo.mp4 \
  --fps 25 \
  --output outputs/realtime_predictions.jsonl
```

参数：

- `--model`：已训练模型目录。
- `--record-dir`：包含 `skeleton.parquet` 和 `annotation.json` 的记录目录。
- `--video`：原始视频路径，仅用于日志标识。
- `--fps`：模拟流帧率。
- `--max-frames`：最多推理帧数，`0` 表示全部。
- `--realtime`：按 `fps` sleep，模拟真实时间流。
- `--output`：JSONL 输出路径。

输出为 JSONL，每行是一帧推理结果：

```json
{"record_id":"demo","frame_idx":1,"is_picking":false,"picking_prob":0.12,"predicted_box_tokens":[]}
{"record_id":"demo","frame_idx":2,"is_picking":true,"picking_prob":0.91,"predicted_box_tokens":["S1:A1"]}
```

## 注意事项

- 调用 `predict_frame()` 前必须先加载模型。
- 调用 `predict_frame()` 前必须设置 annotation。
- 调用 `predict_frame()` 前必须设置有效的 `infer_width` 和 `infer_height`。
- 输入骨架坐标系需要与 annotation 缩放后的推理坐标系一致。
