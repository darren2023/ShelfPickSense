# 模型训练完整步骤

本文说明如何使用 `main.py train` 从零完成一次模型训练，包括数据准备、命令参数、跳帧采样、训练产物与训练后评测。

所有命令建议在项目根目录执行：

```bash
uv run python main.py <子命令> ...
```

等价入口：`python main.py ...`（根目录 `main.py`）。

---

## 1. 环境准备

### 1.1 要求

- Python 3.13+
- 推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖

### 1.2 安装依赖

```bash
uv sync
```

若使用 `xgboost` / `lightgbm` 等可选模型，需确保对应包装已安装（`uv sync` 会按 `pyproject.toml` 拉取）。

---

## 2. 准备训练数据

### 2.1 记录目录结构

训练以 **record 目录** 为单位。`--data-dir` 可以是：

- **单条记录目录**，或
- **包含多条记录的父目录**（其下每个子目录若含 `skeleton.parquet` + `annotation.json` 即视为一条 record）

每条有效记录至少包含：

```text
record_xxx/
  skeleton.parquet      # 逐帧人体骨骼（必需）
  annotation.json       # 货框 ROI 标注（必需）
  event_review.json     # 人工复核标注（推荐，用于 is_picking / 目标货框标签）
  manifest.json         # 可选，infer 宽高等元数据
```

字段说明见 [data-format.md](data-format.md)。

### 2.2 监督标签来源

- **帧级标签 `is_picking`**：来自 `event_review.json` 的 `verified_true` 等字段
- **货框级标签 `is_target`**：在取货正样本帧上，由 `confirmed_box_tokens` 与货框 layout 编码对齐得到

无 `event_review.json` 时无法构建有效监督信号，训练会缺少正样本标签。

### 2.3 数据目录示例

```text
data/data28/
  Train/
    record_001/
      skeleton.parquet
      annotation.json
      event_review.json
    record_002/
      ...
  Test/
    ...
```

训练时使用 `Train`，评测时使用 `Test`（路径按实际项目调整）。

---

## 3. 最简训练

```bash
uv run python main.py train \
  --data-dir data/demo \
  --output models/rf \
  --model sklearn_rf
```

执行成功后，模型写入 `models/rf/`。

---

## 4. 完整参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-dir` | 是 | — | 训练数据目录（单 record 或多 record 父目录） |
| `--output` | 是 | — | 模型输出目录（不存在会自动创建） |
| `--model` | 否 | `sklearn_rf` | 分类器类型，见下文「支持的模型」 |
| `--feature-config` | 否 | 空（全特征） | 特征白名单 JSON 路径 |
| `--feature-jobs` | 否 | `1` | 特征提取时并行处理的 record 数 |
| `--feature-frame-stride` | 否 | `1` | 骨架帧采样间隔：每 N 帧取 1 帧提特征 |
| `--keep-empty-skeleton-frames` | 否 | 关闭 | 指定后保留无骨架帧参与训练 |
| `--log-level` | 否 | `INFO` | 日志级别 |
| `--log-file` | 否 | 空 | 日志写入文件路径 |

### 4.1 支持的模型

- `sklearn_rf`（默认）
- `sklearn_logistic`
- `sklearn_extra_trees`
- `sklearn_gradient_boosting`
- `sklearn_hist_gradient_boosting`
- `sklearn_ada_boost`
- `sklearn_svm_rbf`
- `sklearn_linear_svm`
- `sklearn_knn`
- `sklearn_decision_tree`
- `sklearn_dummy`
- `xgboost`
- `lightgbm`

---

## 5. 推荐训练命令示例

### 5.1 指定特征子集

复制并修改 `configs/selected_features.example.json`，然后：

```bash
uv run python main.py train \
  --data-dir data/data28/Train \
  --output models/selected_rf \
  --model sklearn_rf \
  --feature-config configs/selected_features.example.json
```

配置格式：

```json
{
  "frame_features": ["skeleton", "spatial.any_wrist_inside_box"],
  "box_features": ["spatial.wrist_min_dist_norm", "rule"]
}
```

- 可写完整特征名（如 `skeleton.person_count`）
- 可写前缀（如 `skeleton` 展开为全部 `skeleton.*` 帧级特征）

特征含义见 [features.md](features.md)。

### 5.2 跳帧采样（模拟现场隔帧）

若 pose 管线写入的 parquet 为**全量逐帧**，而现场推理为隔帧，可在训练时下采样：

```bash
uv run python main.py train \
  --data-dir data/data28/Train \
  --output models/strided_rf \
  --model sklearn_rf \
  --feature-frame-stride 2
```

含义：

- 对每条 record 内骨架帧列表做 `frames[::N]`，只对子集提取特征并进入训练集
- 时序特征（`temporal.*`、`skeleton.bbox_motion_*`、`rule.window_*` 等）的历史 lookup 与**同一采样序列**对齐，不会使用被 stride 跳过的中间帧

**注意：**

| 数据情况 | 建议 stride |
|----------|-------------|
| parquet 本身已隔帧（仅 1,3,5…） | `1`（勿再 stride，避免二次下采样） |
| parquet 全量、现场隔帧间隔 N | `N` |
| 与现场一致的全量 parquet + 全帧推理 | `1` |

### 5.3 多 record 并行提特征

```bash
uv run python main.py train \
  --data-dir data/data28/Train \
  --output models/rf \
  --model sklearn_rf \
  --feature-jobs 4
```

单条 record 内部仍按帧串行提取（时序特征依赖同 record 内的采样序列）。

### 5.4 保留无骨架帧

默认会过滤无有效肩/腕关键点的帧。若需保留：

```bash
uv run python main.py train \
  --data-dir data/demo \
  --output models/rf \
  --model sklearn_rf \
  --keep-empty-skeleton-frames
```

---

## 6. 训练内部流程

`train` 子命令等价于以下流水线（**不会**另存一份特征数据集文件，仅在内存中构建样本）：

```text
load_all_records(data_dir)
  → 读取 skeleton.parquet / annotation.json / event_review.json
build_dataset(..., feature_frame_stride=N)
  → 按 stride 对骨架帧采样
  → 提取帧级 / 货框级特征（FeatureRegistry）
filter_empty_skeleton_frames()          # 默认开启
  → 剔除无有效骨架的样本行
model.fit(dataset)
  → picking_clf：帧级是否取货
  → box_clf：正样本帧上哪个货框为目标
model.save(output_dir)
  → 写入 pkl + meta.json + train_result.json
```

---

## 7. 训练输出

```text
models/rf/
  meta.json              # 模型类型、特征名列表等元数据
  picking_clf.pkl        # 取货二分类器
  box_clf.pkl            # 货框目标分类器
  train_result.json      # 训练统计
```

`train_result.json` 主要字段：

| 字段 | 说明 |
|------|------|
| `frame_count` | 参与训练的帧级样本数 |
| `positive_frames` | 取货正样本帧数 |
| `box_samples` | 货框级样本数 |
| `skipped_empty_skeleton_frames` | 过滤掉的无骨架帧数 |
| `fit_seconds` / `save_seconds` | 拟合与保存耗时 |
| `record_ids` | 使用的 record 列表 |

---

## 8. 训练后评测

训练完成后在测试集上评测。**若训练使用了 `--feature-frame-stride`，评测需传相同参数**，以保证采样序列与时序特征与训练一致：

```bash
uv run python main.py eval \
  --data-dir data/data28/Test \
  --model models/rf_stride2 \
  --feature-frame-stride 2
```

未使用 stride 训练时，省略该参数即可（默认 `1`，评测全帧）：

```bash
uv run python main.py eval \
  --data-dir data/data28/Test \
  --model models/rf
```

输出默认在模型目录：

```text
models/rf/
  eval_report.json
  eval_predictions_<record_id>.json
```

也可指定路径：

```bash
uv run python main.py eval \
  --data-dir data/data28/Test \
  --model models/rf \
  --report outputs/eval_report.json \
  --predictions outputs/predictions.json
```

指标含义见 [metrics.md](metrics.md)。

---

## 9. 批量训练与对比（可选）

一次训练多个模型并对比指标：

```bash
uv run python main.py benchmark \
  --data-dir data/data28/Train \
  --eval-data-dir data/data28/Test \
  --output models/benchmark \
  --models sklearn_rf sklearn_logistic sklearn_extra_trees \
  --feature-frame-stride 2 \
  --jobs 4
```

多特征配置批量实验见 [usage.md](usage.md) 中「多特征配置批量 Benchmark」与 `configs/feature_benchmark.example.json`。

输出 `benchmark_report.md` 中各列指标定义见 [metrics.md](metrics.md)。

---

## 10. 调试与排查

### 10.1 查看详细日志

```bash
uv run python main.py train \
  --data-dir data/demo \
  --output models/rf \
  --model sklearn_rf \
  --log-level DEBUG
```

### 10.2 检查 parquet 是否隔帧

训练前可用 `tests/parquet_stride_check.py` 扫描 `frame_idx` 间隔：

```powershell
uv run python tests/parquet_stride_check.py data/data28/Train
```

- **dense**：连续逐帧（`gap=1`），训练时 `--feature-frame-stride 1`
- **regular_stride**：固定间隔隔帧（如仅 1,3,5…，`inferred_stride=2`），训练时通常 `--feature-frame-stride 1`
- **irregular**：间隔不一致（多为检测丢帧或片段拼接），需人工判断

JSON 输出与 CI 门禁：

```powershell
uv run python tests/parquet_stride_check.py data/data28/Train --json
uv run python tests/parquet_stride_check.py data/data28/Train --fail-on-subsampled
```

### 10.3 导出特征检查（可选）

`train` 不落盘特征表；若需人工查看特征值：

```bash
uv run python main.py export-features \
  --data-dir data/demo \
  --output outputs/features \
  --feature-frame-stride 2 \
  --format csv
```

### 10.4 常见问题

| 现象 | 可能原因 |
|------|----------|
| `未找到有效记录目录` | `--data-dir` 下缺少 `skeleton.parquet` 或 `annotation.json` |
| `positive_frames=0` | 缺少 `event_review.json` 或无 `verified_true` 标注 |
| 训练很慢 | 特征多、record 多；可试 `--feature-config` 减特征、`--feature-jobs` 并行 |
| 训练 stride 与现场不一致 | 检查 parquet 是否全量；调整 `--feature-frame-stride` 与现场隔帧间隔一致 |
| xgboost/lightgbm 报错 | 运行 `uv sync` 安装可选依赖 |

---

## 11. 相关文档

| 文档 | 内容 |
|------|------|
| [data-format.md](data-format.md) | 数据格式与字段 |
| [features.md](features.md) | 特征列表与含义 |
| [usage.md](usage.md) | 训练、评测、benchmark、实时推理等完整用法 |
| [realtime.md](realtime.md) | 训练模型接入实时推理 |

---

## 12. 一键复制：典型工作流

```bash
# 1. 安装依赖
uv sync

# 2. 训练（全量特征 + 隔帧 stride=2 + 4 路并行提特征）
uv run python main.py train \
  --data-dir data/data28/Train \
  --output models/rf_stride2 \
  --model sklearn_rf \
  --feature-frame-stride 2 \
  --feature-jobs 4

# 3. 评测（stride 与训练一致）
uv run python main.py eval \
  --data-dir data/data28/Test \
  --model models/rf_stride2 \
  --feature-frame-stride 2

# 4. 查看训练统计
cat models/rf_stride2/train_result.json
```

（Windows PowerShell 下查看 JSON 可用 `Get-Content models/rf_stride2/train_result.json`。）
