# 特征提取 — 问题清单与 TODO

> 审查日期：2026-07-03（第二轮，聚焦逻辑自洽与坐标对齐）  
> 项目定位：新特征提取 + 新规则探索（**不要求复现 visual-dps 现场**）  
> 审查范围：`records.py` / `annotation.py` 坐标系、`features/*`、`rule_collision.py`  
> 实测数据：`data/demo` 28 条记录

---

## 坐标对齐结论（640×360 标注 vs 骨架规格）

### 设计意图（当前代码逻辑）

```
annotation_size (640×360)     skeleton 坐标系 (默认 852×480)
        │                              │
        │  video_polygon_norm (0~1)    │  kpt_*_x/y 直接读 parquet
        └──────────┬───────────────────┘
                   ▼
        infer_polygon = norm × infer_w/h
        build_box_index(infer_w, infer_h)
```

- `annotation_size` **仅描述标注画布**，不等于骨架坐标系。
- 骨架坐标系由 `resolve_infer_frame_size()` 解析，优先级：
  1. `manifest.json` 的 `infer_width` / `infer_height`
  2. `annotation.json` 根的 `infer_width` / `infer_height`
  3. 启发式：`640×360` 标注 → **固定 `852×480`**（`constants.DEFAULT_POSE_INFER_*`）
- 货框缩放**优先** `video_polygon_norm`，与 `annotation_size` 数值解耦。

### 实测（data/demo）

| 项 | 结果 |
|----|------|
| 标注尺寸 | 全部 `640×360` |
| 解析 infer 尺寸 | 全部 `852×480`（无 manifest，走启发式） |
| `video_polygon_norm` | 28/28 记录、全部货框均有 |
| 骨架 bbox_y2 最大值 | ≈ `480.3`，与 infer_h 基本一致 |
| 关键点越界比例 | 平均 **4.4%**（个别帧 y > 480 或 x > 852，属噪声/遮挡） |
| 取货帧 `rule.any_collision` | 抽样均有命中（demo 数据在正确 infer 下可用） |

**结论**：demo 数据在「无 manifest、有 norm 多边形、infer=852×480」条件下，**货框与骨架基本在同一坐标系**。  
用户关心的「标注 640×360、骨架不是这个规格」属于**预期行为**，代码已用 norm + infer 尺寸做变换。

### 坐标系相关的真实漏洞

- [ ] **P0：`manifest.json` 误把 `annotation_size` 当作 infer 尺寸**  
  - 位置：`records._infer_from_manifest()`  
  - 当 manifest 无 `infer_width/infer_height`，但有嵌套 `annotation.annotation_size` 时，返回 `640×360`  
  - 实测：`record_001` 在错误 infer 下，1070 个取货帧仅 585 个仍能触发 `rule.any_collision`（**丢失 45%**）  
  - 修复：manifest 回退不应使用 `annotation_size`；或强制要求显式 `infer_width/infer_height`

- [ ] **P0：infer 尺寸完全依赖元数据，不校验骨架实际范围**  
  - `resolve_infer_frame_size()` 不读取 skeleton 的 kpt/bbox 范围  
  - 若 manifest 缺失且标注不是 `640×360`（如 `640×480`），会走 `round(480×aspect)` 而非采集真实尺寸  
  - 修复：加载时根据 skeleton bbox/kpt 分位数推断或校验，与解析 infer 不一致时告警

- [ ] **P1：仅 `video_polygon`、无 `video_polygon_norm` 时缩放脆弱**  
  - demo 数据 100% 有 norm，尚未踩坑  
  - 若 `video_polygon` 已是 infer 空间坐标（部分采集工具会这样），再按 `ann_w/ann_h → infer` 缩放会**二次缩放**  
  - 修复：检测 polygon 坐标量级（对比 ann_size / infer_size），或强制要求 norm

- [ ] **P2：layout 在标注空间、碰撞在 infer 空间**  
  - `build_box_layout(frame_width=ann_w)` 用 `640` 判左右货架  
  - `spatial`/`rule` 碰撞在 `852×480`  
  - 属设计分离，但层列布局与碰撞距离跨空间，文档需写清

---

## 代码 Bug（与现场无关）

- [ ] **P0：`_box_hand_collision_flags` 忽略 `hand_points` 参数**  
  - 文件：`features/rule_engine.py` L224  
  - 循环应使用传入的 `hand_points`，而非重新 `collect_rule_hand_points()`

- [ ] **P0：无 `person_track_id` 时跨帧特征恒为 0**  
  - `find_person_by_track(None)` 直接返回 `None`  
  - 影响：`temporal.*_move_*`、`consecutive_hit_*`、`rule.window_hit_*`  
  - demo 数据 track_id 覆盖率 100%，但字段为可选，新数据会踩坑  
  - 修复：`consecutive_hit` / `window_hit_count` / `side_movement_norm` 复用 `tracked_person_at_offset` 的 anchor 回退

- [ ] **P1：`spatial.extract_per_box` 无主人物时特征键缺失**  
  - 缺 `left_wrist_dist_norm` 等；`to_vector` 补 0 掩盖缺失

- [ ] **P2：`spatial.extract_per_box` 每货框重复收集手腕/踝点**  
  - 性能问题，可提至循环外

---

## 逻辑自洽性（仓库内部）

以下在「探索新规则」前提下**不是致命 bug**，但混用时会自相矛盾，建议在特征选择或文档中区分。

- [ ] **P1：`spatial`/`temporal` 与 `rule` 命中定义不同**  
  - `spatial`/`temporal`：手腕硬边界（无 margin）  
  - `rule`/`rule_collision`：手腕+前臂、软边界 margin、最近货框  
  - 同帧可出现 `spatial.wrist_inside=1` 且 `rule.hand_collision=0`  
  - 建议：特征选择时按组使用，或统一命名前缀（`hard_*` / `soft_*`）

- [ ] **P1：主人物选择不一致**  
  - `skeleton`：两手腕 min score 最大者  
  - 其他模块：`select_primary_track_id()`  
  - 建议：统一 primary person 策略

- [ ] **P1：归一化尺度不统一**  
  - `skeleton`：x/y 分别除 `infer_width/height`  
  - `spatial`/`temporal`/`rule`：距离除 `max(w,h)`  
  - 建议：统一或文档标明

- [ ] **P1：两套时序语义不同**  
  - `temporal.consecutive_hit_*`：严格连续 streak，缺 track 即断  
  - `rule.window_hit_*`：M-of-N 滑窗，缺帧 `continue`  
  - 建议：视为不同特征，勿混为「连续命中」冗余

- [ ] **P2：关键点置信度阈值不一致**  
  - `spatial`/`temporal`/`rule`：`MIN_KEYPOINT_SCORE=0.3`  
  - `skeleton._pt()`：不过滤低分

---

## 数据与训练管线

- [ ] **P1：`box_samples` 仅在 `is_picking` 帧生成**  
  - `dataset.build_dataset()` 只在取货帧追加 per-box 样本  
  - 货框分类器见不到「非取货帧的负样本」  
  - 评估是否影响货框 Micro-F1，考虑全帧采样

- [ ] **P2：`skeleton.infer_width/height` 作为特征输入**  
  - 每帧写入 `skeleton.infer_width/height`  
  - 跨记录训练时若 infer 尺寸一致则为常数列，无信息；若不一致可能泄漏 record 身份

- [ ] **P3：测试缺口**  
  - [ ] manifest 误回退 `annotation_size`  
  - [ ] 无 `person_track_id` 跨帧特征  
  - [ ] infer 尺寸与 skeleton 范围校验  
  - [ ] 仅 `video_polygon`、无 norm 的缩放  
  - fixture 为 `640×480` + `person_track_id=1`，与 demo 真实分布不同

---

## 文档修正

- [ ] 更新 `README.md` / `docs/usage.md`：本项目为**新规则探索**，非现场复现  
- [ ] 更新 `docs/data-format.md`：明确 `annotation_size ≠ skeleton 坐标系`，说明 infer 解析优先级与 `manifest` 陷阱  
- [ ] 建议每条记录提供 `manifest.json` 写明真实 `infer_width/infer_height`

---

## 建议实施顺序

1. **修复 manifest 回退** + **加载时 skeleton/infer 一致性校验/告警**（坐标 P0）  
2. **修复 `_box_hand_collision_flags`**、**track_id 缺失回退**（代码 P0）  
3. **补充坐标对齐测试**（manifest 陷阱、640×360→852×480、越界率）  
4. **内部语义**：特征选择分组或统一 primary person / 归一化（P1，按建模需要）  
5. **文档**（P3）

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/analysis/records.py` | infer 尺寸解析（`resolve_infer_frame_size`） |
| `src/analysis/annotation.py` | 货框 norm 缩放（`infer_polygon_points`） |
| `src/analysis/constants.py` | `DEFAULT_POSE_INFER_WIDTH/HEIGHT`（852×480） |
| `src/analysis/features/rule_engine.py` | 软边界规则特征 |
| `src/analysis/features/spatial.py` | 硬边界空间特征 |
| `src/analysis/features/tracking.py` | track 匹配 |
| `src/analysis/dataset.py` | 训练样本构建 |
| `docs/data-format.md` | 数据格式说明 |
