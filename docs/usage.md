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
