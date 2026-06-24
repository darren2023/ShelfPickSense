# 训练、评测与 Benchmark

所有命令都建议通过 `uv run python main.py` 执行。

## 安装依赖

```bash
uv sync
```

## 训练单个模型

```bash
uv run python main.py train \
  --data-dir data/demo \
  --output models/rf \
  --model sklearn_rf
```

参数：

- `--data-dir`：训练数据目录，可以是单条记录目录，也可以是多条记录的父目录。
- `--output`：模型输出目录。
- `--model`：模型名称。

当前支持模型：

- `sklearn_rf`
- `sklearn_logistic`
- `sklearn_extra_trees`
- `sklearn_gradient_boosting`
- `sklearn_svm_rbf`
- `sklearn_knn`

训练输出：

```text
models/rf/
  meta.json
  picking_clf.pkl
  box_clf.pkl
  train_result.json
```

## 导出特征

从记录目录提取训练流程使用的帧级特征和货框级特征，并保存为文件：

```bash
uv run python main.py export-features \
  --data-dir data/demo \
  --output outputs/features
```

输出：

```text
outputs/features/
  frame_features.parquet
  box_features.parquet
  features_meta.json
```

`frame_features.parquet` 包含 `record_id`、`frame_idx`、`is_picking`、`confirmed_box_tokens` 和帧级特征列。
`box_features.parquet` 包含 `record_id`、`frame_idx`、`box_token`、`is_target` 和货框级特征列。
`features_meta.json` 保存特征名列表、样本数量和输出路径。

默认格式是 `parquet`。如果需要更直观查看，可以指定：

```bash
uv run python main.py export-features \
  --data-dir data/demo \
  --output outputs/features \
  --format csv
```

支持的格式：

- `parquet`：默认格式，适合后续程序读取。
- `csv`：表格文本，适合用 Excel 或编辑器查看。
- `jsonl`：每行一个样本对象，适合流式处理或快速抽样查看。
- `all`：同时输出 parquet、CSV 和 JSONL。

## 特征相关性分析

分析输入目录中所有记录的帧级特征和货框级特征：

```bash
uv run python main.py analyze-features \
  --data-dir data/demo \
  --output outputs/correlations
```

也支持直接分析 `export-features` 已导出的特征目录：

```bash
uv run python main.py analyze-features \
  --features-dir outputs/features \
  --output outputs/correlations
```

也可以直接运行脚本：

```bash
uv run python scripts/analyze_feature_correlation.py \
  --data-dir data/demo \
  --output outputs/correlations
```

输出：

```text
outputs/correlations/
  frame_feature_samples.csv
  frame_feature_correlation.csv
  frame_target_correlation.csv
  frame_high_correlation_pairs.csv
  frame_pca_explained_variance.csv
  frame_pca_loadings.csv
  frame_pca_projection.csv
  frame_low_value_constant_features.csv
  frame_low_value_low_target_correlation.csv
  frame_low_value_redundant_pairs.csv
  box_feature_samples.csv
  box_feature_correlation.csv
  box_target_correlation.csv
  box_high_correlation_pairs.csv
  box_pca_explained_variance.csv
  box_pca_loadings.csv
  box_pca_projection.csv
  box_low_value_constant_features.csv
  box_low_value_low_target_correlation.csv
  box_low_value_redundant_pairs.csv
  correlation_report.md
  correlation_summary.json
  figures/
    frame_target_correlation_top.svg
    box_target_correlation_top.svg
    frame_feature_correlation_heatmap.svg
    box_feature_correlation_heatmap.svg
    frame_high_correlation_pairs.svg
    box_high_correlation_pairs.svg
    frame_pca_explained_variance.svg
    box_pca_explained_variance.svg
    frame_pca_scatter.svg
    box_pca_scatter.svg
```

说明：

- `correlation_report.md`：图文版 Markdown 报告，汇总样本数、Top 相关特征、热力图和高相关特征对。
- `frame_target_correlation.csv`：帧级特征与 `is_picking` 的相关性，按绝对值降序排序。
- `box_target_correlation.csv`：货框级特征与 `is_target` 的相关性，按绝对值降序排序。
- `*_feature_correlation.csv`：特征两两相关矩阵。
- `*_high_correlation_pairs.csv`：超过阈值的高相关特征对，默认阈值为 `0.9`。
- `*_pca_explained_variance.csv`：PCA 每个主成分的解释方差和累计解释方差。
- `*_pca_loadings.csv`：各特征在主成分上的载荷。
- `*_pca_projection.csv`：样本在 PC1/PC2 上的二维投影，可用于观察类别分布。
- `*_low_value_constant_features.csv`：常量或近常量特征候选。
- `*_low_value_low_target_correlation.csv`：与目标标签几乎无相关的特征候选。
- `*_low_value_redundant_pairs.csv`：高度冗余的特征对，以及建议优先删除的一侧。
- `figures/*.svg`：报告引用的柱状图和相关矩阵热力图。

可选参数：

```bash
--data-dir data/demo          # 原始记录目录
--features-dir outputs/features  # 已导出的特征目录，和 --data-dir 二选一
--method pearson   # 可选 pearson/spearman/kendall，默认 pearson
--threshold 0.9    # 高相关特征对阈值
--top-n 100        # 最多输出的高相关特征对数量
```

## 评测模型

```bash
uv run python main.py eval \
  --data-dir data/demo \
  --model models/rf
```

默认输出：

```text
models/rf/eval_report.json
models/rf/eval_predictions.json
```

可以自定义输出路径：

```bash
uv run python main.py eval \
  --data-dir data/demo \
  --model models/rf \
  --report outputs/eval_report.json \
  --predictions outputs/eval_predictions.json
```

`eval_predictions.json` 保存逐帧预测明细：

- `record_id`
- `frame_idx`
- `true_is_picking`
- `pred_is_picking`
- `picking_prob`
- `true_box_tokens`
- `predicted_box_tokens`
- `box_exact_match`

## 评测指标

帧级取货检测指标：

- `accuracy`
- `precision / recall / f1`：以“取货”为正类。
- `negative_precision / negative_recall / negative_f1`：以“非取货”为负类。
- `macro_f1`：正类 F1 与负类 F1 的平均值。
- `balanced_accuracy`：正类 recall 与负类 recall 的平均值。

货框识别指标：

- `exact_match_ratio`
- `any_hit_ratio`
- `micro_precision`
- `micro_recall`
- `micro_f1`

样本不均衡时，建议优先关注：

- `macro_f1`
- `balanced_accuracy`
- `picking_recall`
- `box_micro_f1`

## 批量 Benchmark

默认运行全部支持模型：

```bash
uv run python main.py benchmark \
  --data-dir data/demo \
  --output models/benchmark \
  --jobs 4
```

指定模型子集：

```bash
uv run python main.py benchmark \
  --data-dir data/demo \
  --output models/benchmark \
  --models sklearn_rf sklearn_svm_rbf \
  --jobs 2
```

训练集和评测集分开：

```bash
uv run python main.py benchmark \
  --data-dir data/train \
  --eval-data-dir data/eval \
  --output models/benchmark \
  --jobs 4
```

benchmark 输出：

```text
models/benchmark/
  benchmark_summary.json
  sklearn_rf/
    meta.json
    picking_clf.pkl
    box_clf.pkl
    train_result.json
    eval_report.json
    eval_predictions_<record_name>.json
  ...
```

说明：

- benchmark 会复用训练数据加载和训练 Dataset 构建，减少重复数据处理。
- 每个模型仍会独立训练和独立推理评测。
- 对比摘要按 `macro_f1` 降序排序。

## Benchmark 训练测试报告

当输入目录已经按训练集和测试集分好时，可以使用：

```text
data/split/
  Train/
    record_001/
      skeleton.parquet
      annotation.json
      event_review.json
  Test/
    record_101/
      skeleton.parquet
      annotation.json
      event_review.json
```

直接传入 `Train` 与 `Test` 文件夹运行所有模型训练与测试：

```bash
uv run python main.py benchmark \
  --data-dir data/split/Train \
  --eval-data-dir data/split/Test \
  --output models/train_test \
  --jobs 4
```

指定模型子集：

```bash
uv run python main.py benchmark \
  --data-dir data/split/Train \
  --eval-data-dir data/split/Test \
  --output models/train_test \
  --models sklearn_rf sklearn_logistic \
  --jobs 2
```

输出：

```text
models/train_test/
  benchmark_report.md
  benchmark_summary.json
  sklearn_rf/
    train_result.json
    eval_report.json
    eval_predictions_*.json
  ...
```

`benchmark_report.md` 会汇总 Train 数据规模、Test 集各模型指标，并按 `macro_f1` 给出推荐模型与结论。结论会同时列出 `balanced_accuracy`、取货 recall 和货框 `micro_f1`，避免只看单一指标。

## 日志

所有命令支持：

```bash
--log-level DEBUG
--log-file logs/run.log
```

示例：

```bash
uv run python main.py benchmark \
  --data-dir data/demo \
  --output models/benchmark \
  --jobs 4 \
  --log-level DEBUG \
  --log-file logs/benchmark.log
```
