## 文件夹说明：
- `annotation.json` 是货架信息，包括 shelves\boxes 两个层级
- `skeleton.parquet` 是人体骨骼数据，包含了每个人的骨骼坐标信息
- `event_review.json` 
   - 记录了所有取货的事件
   - 核心字段是verified_true中的frame_idx和confirmed_box_tokens字段
   - frame_idx表示视频帧的索引
   - confirmed_box_tokens是一个列表，通过这些token可以在annotation.json中找到对应的货架信息。
   - 如果没有这个文件，则说明视频中所有帧都没有取货事件。
   - 帧索引不在event_review.json中，说明该帧没有取货事件。

## 模型功能
- 根据文件夹中的文件，忽略其它文件
- 构建监督模型
    - skeleton.parquet、annotation.json作为输入
    - 根据event_review.json构建监督信号
    - 根据人体骨骼数据和货架信息分析当前是否在取货，取货的货架是哪个

## 环境配置
使用 uv 管理依赖：
```bash
uv sync
```