# 特征说明

本文档说明 `src/analysis/features/` 中默认注册的特征。默认特征由 `default_registry()` 组合，导出、训练、benchmark、实时推理都会使用同一套注册表，除非通过 `--feature-config` 指定特征白名单。

## 基本约定

特征分为两类：

- **帧级特征**：描述单帧整体状态，用于判断当前帧是否正在取货。导出到 `frame_features.*`，标签为 `is_picking`。
- **货框级特征**：描述单帧内某一个货框和人体的关系，用于判断取货目标货框。导出到 `box_features.*`，标签为 `is_target`。

注册表会自动给特征名加上提取器前缀，格式为：

```text
<extractor>.<feature_name>
```

例如：

- `skeleton.person_count`
- `spatial.wrist_inside`
- `temporal.consecutive_hit_3`
- `rule.window_hit_3_6`

特征值统一为 `float`。布尔特征使用 `1.0` 表示命中或存在，`0.0` 表示未命中或不存在。距离类特征通常已归一化，越小表示越接近；进入货框内部时距离为 `0.0`。

## 人物与时间窗口约定

部分特征会展开到固定人物槽位：

```text
p0, p1, p2
```

槽位来自 `sorted_persons()`，最多保留 3 个人。排序优先使用 `person_track_id` / `track_id` / `person_id`，缺失 track 的人物排在后面。没有人物的槽位会填充默认值，例如 `present=0.0`、距离为 `1.0`、位移为 `0.0`。

部分特征使用主人物 `primary`。主人物由 `select_primary_track_id()` 选择：优先选择能与上一帧关联、且手腕置信度较高的人；如果没有可靠 track，在空间和规则货框特征中会回退到当前帧排序第一的人。

常量展开如下：

- 连续命中窗口：`3`、`5`、`7`
- 手腕位移 offset：`1`、`3`、`5`、`7`
- 脚部位移 offset：`1`、`2`、`3`、`5`、`7`
- 规则滑窗命中组合：`3/6`、`3/7`、`5/7`

## `skeleton`：骨骼统计特征

`SkeletonFeatureExtractor` 只输出帧级特征，主要描述当前帧的人数、画面尺寸、最佳人物的手腕位置和人体锚点。

| 特征 | 含义 |
| --- | --- |
| `skeleton.person_count` | 当前帧检测到的人数。 |
| `skeleton.wrist_min_score` | 最佳人物左右手腕置信度中的较小值。最佳人物按该值最大者选择。 |
| `skeleton.left_wrist_x_norm` / `skeleton.left_wrist_y_norm` | 最佳人物左手腕坐标，分别除以 `infer_width` / `infer_height`。 |
| `skeleton.right_wrist_x_norm` / `skeleton.right_wrist_y_norm` | 最佳人物右手腕坐标，分别除以 `infer_width` / `infer_height`。 |
| `skeleton.wrist_spread` | 左右手腕距离，除以 `max(infer_width, infer_height)`。 |
| `skeleton.anchor_x_norm` / `skeleton.anchor_y_norm` | 人体锚点坐标。优先使用双肩中心，否则使用可见关键点均值。 |

无人物时，除人数和画面尺寸外，其余骨骼特征填 `0.0`。

## `spatial`：人体与货框空间关系

`BoxSpatialFeatureExtractor` 同时输出帧级和货框级特征。它使用硬边界判断：关键点在货框多边形内部即命中，不额外扩大边界。

### 帧级特征

主人物到任意货框的距离和内部命中：

| 模式 | 含义 |
| --- | --- |
| `spatial.primary_<side>_min_box_dist_norm` | 主人物某侧关键点到最近货框的归一化距离。 |
| `spatial.primary_<side>_inside_any_box` | 主人物某侧关键点是否进入任意货框。 |

其中 `<side>` 为：

```text
left_wrist, right_wrist, left_foot, right_foot
```

固定人物槽位特征：

| 模式 | 含义 |
| --- | --- |
| `spatial.p<i>_present` | 第 `i` 个人物槽位是否存在。 |
| `spatial.p<i>_track_id` | 第 `i` 个人物槽位的 track id，缺失为 `0.0`。 |
| `spatial.p<i>_<side>_min_box_dist_norm` | 第 `i` 个人某侧关键点到最近货框的归一化距离。 |
| `spatial.p<i>_<side>_inside_any_box` | 第 `i` 个人某侧关键点是否进入任意货框。 |

全帧聚合特征：

| 特征 | 含义 |
| --- | --- |
| `spatial.min_wrist_box_dist_norm` | 全帧所有手腕到所有货框的最小归一化距离。 |
| `spatial.any_wrist_inside_box` | 是否有任意手腕进入任意货框。 |
| `spatial.boxes_with_wrist_inside` | 有手腕进入的货框数量。 |
| `spatial.min_foot_box_dist_norm` | 全帧所有脚踝到所有货框的最小归一化距离。 |
| `spatial.any_foot_inside_box` | 是否有任意脚踝进入任意货框。 |
| `spatial.boxes_with_foot_inside` | 有脚踝进入的货框数量。 |

### 货框级特征

主人物相对当前货框的关键点特征：

| 模式 | 含义 |
| --- | --- |
| `spatial.<part>_dist_norm` | 主人物某个关键点到当前货框的归一化距离。 |
| `spatial.<part>_inside` | 主人物某个关键点是否进入当前货框。 |

其中 `<part>` 为：

```text
left_wrist, right_wrist, left_foot, right_foot
```

全人物相对当前货框的聚合特征：

| 特征 | 含义 |
| --- | --- |
| `spatial.wrist_min_dist_norm` | 所有手腕到当前货框的最小归一化距离。 |
| `spatial.wrist_inside` | 是否有任意手腕进入当前货框。 |
| `spatial.wrist_mean_dist_norm` | 所有手腕到当前货框的平均归一化距离。 |
| `spatial.centroid_dist_norm` | 最近手腕到当前货框中心点的归一化距离。 |
| `spatial.foot_min_dist_norm` | 所有脚踝到当前货框的最小归一化距离。 |
| `spatial.foot_inside` | 是否有任意脚踝进入当前货框。 |
| `spatial.foot_mean_dist_norm` | 所有脚踝到当前货框的平均归一化距离。 |

如果没有可用手腕或脚踝，对应距离默认为 `1.0`，inside 默认为 `0.0`。

## `layout`：货框几何布局特征

`BoxLayoutFeatureExtractor` 只输出货框级特征。它基于 `box_layout` 推导货框所在货架侧、层、列，以及货架整体层列统计。这里的 `layout_layer` / `layout_column` 是几何推导结果，不一定等同于原始标注字段。

| 特征 | 含义 |
| --- | --- |
| `layout.shelf_side` | 货架侧编码。通常用于区分左右货架。 |
| `layout.layout_layer` | 当前货框所在层。 |
| `layout.layout_column` | 当前货框所在列。 |
| `layout.shelf_layer_count` | 当前货架侧推导出的层数。 |
| `layout.shelf_column_count_mean` | 当前货架侧各层列数的平均值。 |

## `temporal`：跨帧时序特征

`TemporalFeatureExtractor` 只输出帧级特征。它把当前帧与历史帧关联起来，描述命中持续性、手腕位移、脚部位置和脚部位移。

### 主人物特征

连续命中特征：

| 模式 | 含义 |
| --- | --- |
| `temporal.consecutive_hit_<n>` | 主人物从当前帧向前是否连续至少 `n` 帧手腕进入任意货框。 |

`<n>` 为 `3`、`5`、`7`。

手腕位移特征：

| 模式 | 含义 |
| --- | --- |
| `temporal.left_wrist_move_<offset>` | 主人物左手腕相对前 `offset` 帧的归一化位移距离。 |
| `temporal.right_wrist_move_<offset>` | 主人物右手腕相对前 `offset` 帧的归一化位移距离。 |

`<offset>` 为 `1`、`3`、`5`、`7`。

脚部位置特征：

| 特征 | 含义 |
| --- | --- |
| `temporal.left_foot_x_norm` / `temporal.left_foot_y_norm` | 主人物左脚坐标，除以 `max(infer_width, infer_height)`。 |
| `temporal.right_foot_x_norm` / `temporal.right_foot_y_norm` | 主人物右脚坐标，除以 `max(infer_width, infer_height)`。 |
| `temporal.foot_avg_x_norm` / `temporal.foot_avg_y_norm` | 左右脚平均点坐标；只有一侧可见时使用该侧。 |

脚部位移特征：

| 模式 | 含义 |
| --- | --- |
| `temporal.<foot>_dx_<offset>` | 当前脚点相对前 `offset` 帧的 x 方向归一化位移。 |
| `temporal.<foot>_dy_<offset>` | 当前脚点相对前 `offset` 帧的 y 方向归一化位移。 |
| `temporal.<foot>_dist_<offset>` | 当前脚点相对前 `offset` 帧的归一化位移距离。 |

`<foot>` 为 `left_foot`、`right_foot`、`foot_avg`；`<offset>` 为 `1`、`2`、`3`、`5`、`7`。

### 固定人物槽位特征

固定人物槽位会输出主人物同类的时序特征，并带有 `p<i>_` 前缀：

| 模式 | 含义 |
| --- | --- |
| `temporal.p<i>_present` | 第 `i` 个人物槽位是否存在。 |
| `temporal.p<i>_track_id` | 第 `i` 个人物槽位的 track id。 |
| `temporal.p<i>_consecutive_hit_<n>` | 第 `i` 个人是否连续至少 `n` 帧手腕进入任意货框。 |
| `temporal.p<i>_left_wrist_move_<offset>` | 第 `i` 个人左手腕相对历史帧位移。 |
| `temporal.p<i>_right_wrist_move_<offset>` | 第 `i` 个人右手腕相对历史帧位移。 |
| `temporal.p<i>_<foot>_x_norm` / `temporal.p<i>_<foot>_y_norm` | 第 `i` 个人脚部位置。 |
| `temporal.p<i>_<foot>_dx_<offset>` / `dy` / `dist` | 第 `i` 个人脚部相对历史帧位移。 |

`<i>` 为 `0`、`1`、`2`。

### 全人物聚合特征

| 模式 | 含义 |
| --- | --- |
| `temporal.any_track_consecutive_hit_<n>` | 当前帧任意人物是否连续至少 `n` 帧手腕进入任意货框。 |

## `rule`：规则碰撞特征

`RuleEngineFeatureExtractor` 同时输出帧级和货框级特征。它与 `rule_collision` 的规则基线对齐，使用软边界、前臂外推点和 M-of-N 滑窗逻辑。与 `spatial` 不同，`rule` 不只看手腕是否严格落入货框内部，还会考虑：

- 手腕置信度和肘部置信度阈值；
- 手腕点与肘腕方向外推出的前臂点；
- 基于肩宽和最小像素值计算的货框软边界 margin；
- 最近货框归属；
- 历史滑窗内的命中次数。

### 帧级特征

当前帧碰撞统计：

| 特征 | 含义 |
| --- | --- |
| `rule.any_collision` | 当前帧任意人物是否碰撞任意货框。 |
| `rule.collision_count` | 当前帧被碰撞的货框数量。 |
| `rule.primary_any_collision` | 主人物是否碰撞任意货框。 |
| `rule.primary_collision_count` | 主人物碰撞的货框数量。 |

主人物滑窗特征：

| 模式 | 含义 |
| --- | --- |
| `rule.window_hit_<min_hits>_<window>` | 主人物在最近 `window` 帧内是否至少命中 `min_hits` 帧。 |
| `rule.window_hits_<window>` | 主人物在最近 `window` 帧内的命中帧数。 |

当前实现使用组合：

```text
window_hit_3_6, window_hit_3_7, window_hit_5_7
window_hits_6, window_hits_7
```

固定人物槽位特征：

| 模式 | 含义 |
| --- | --- |
| `rule.p<i>_present` | 第 `i` 个人物槽位是否存在。 |
| `rule.p<i>_track_id` | 第 `i` 个人物槽位的 track id。 |
| `rule.p<i>_any_collision` | 第 `i` 个人当前帧是否碰撞任意货框。 |
| `rule.p<i>_collision_count` | 第 `i` 个人当前帧碰撞的货框数量。 |
| `rule.p<i>_window_hit_<min_hits>_<window>` | 第 `i` 个人在滑窗内是否满足命中帧数阈值。 |

全人物聚合特征：

| 特征 | 含义 |
| --- | --- |
| `rule.any_track_window_hit_3_6` | 当前帧任意人物在最近 6 帧内是否至少命中 3 帧。 |

### 货框级特征

| 特征 | 含义 |
| --- | --- |
| `rule.frame_collision` | 当前货框是否被当前帧任意人物碰撞。 |
| `rule.wrist_collision` | 主人物手腕点是否命中当前货框的软边界。 |
| `rule.forearm_collision` | 主人物前臂外推点是否命中当前货框的软边界。 |
| `rule.hand_collision` | `wrist_collision` 或 `forearm_collision` 是否命中。 |
| `rule.max_signed_dist_norm` | 主人物手部候选点到当前货框的最大有符号距离，除以 `max(infer_width, infer_height)`；内部为正，外部为负。 |
| `rule.nearest_collision` | 当前货框是否为某个主人物手部候选点在软边界内归属的最近货框。 |
| `rule.window_hit_<min_hits>_<window>` | 主人物在滑窗内对当前货框是否满足命中帧数阈值。 |
| `rule.window_hits_<window>` | 主人物在滑窗内对当前货框的命中帧数。 |

## 特征选择示例

可以在 JSON 配置中只保留部分特征：

```json
{
  "frame_features": [
    "skeleton.person_count",
    "spatial.any_wrist_inside_box",
    "temporal.consecutive_hit_3",
    "rule.window_hit_3_6"
  ],
  "box_features": [
    "spatial.wrist_min_dist_norm",
    "spatial.wrist_inside",
    "layout.layout_layer",
    "layout.layout_column",
    "rule.hand_collision",
    "rule.window_hit_3_6"
  ]
}
```

列表项既可以写完整特征名，也可以写提取器类别名前缀。类别名会展开为当前可用的全部同类特征，例如：

```json
{
  "frame_features": ["skeleton", "temporal.consecutive_hit_3"],
  "box_features": ["layout", "rule"]
}
```

其中 `skeleton` 会匹配所有 `skeleton.*` 帧级特征，`layout` 会匹配所有 `layout.*` 货框级特征。类别项和完整特征名可以混用，重复项会自动去重。

使用方式：

```bash
uv run python main.py train \
  --data-dir data/demo \
  --output models/selected \
  --feature-config configs/selected_features.json
```

如果某一类不写，例如只写 `frame_features`，则另一类会保留全部特征。写入未知特征名会直接报错，避免配置拼写错误被静默忽略。

## 使用建议

- `spatial` 是硬边界空间特征，适合表达“手腕/脚是否真的进入货框”。
- `rule` 是软边界规则特征，适合复用规则基线的碰撞语义，通常比 `spatial` 更宽松。
- `temporal` 适合识别取货动作的持续性和运动趋势，但依赖跨帧关联；没有稳定 track 时会使用肩点锚点做有限回退。
- `layout` 适合货框分类阶段，帮助模型学习目标货框的货架位置先验。
