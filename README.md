# 货架取货行为分析模型

本项目基于人体骨骼数据和货架标注，训练并评测“是否正在取货、取货货框是哪一个”的机器学习模型。

## 能力范围

- 从 `skeleton.parquet` 和 `annotation.json` 构建特征。
- 从 `event_review.json` 的人工复核结果构建监督信号。
- 支持多个 sklearn 模型训练、评测、批量 benchmark。
- 支持将记录特征导出为 parquet 文件，便于外部分析或复用。
- 支持 macro-F1、balanced accuracy、货框 micro-F1 等不均衡样本友好指标。
- 训练前默认过滤无骨架帧，减轻空帧对样本均衡的干扰。
- Benchmark 自动对比规则碰撞基线（`rule_baseline`），判断 ML 模型是否超过规则方法。
- Optuna 超参数搜索（`tune` 命令，支持 xgboost / lightgbm）。
- 支持逐帧实时推理，便于外部应用直接集成 `RealtimePickingPredictor`。
- 特征提取模块采用注册表模式，便于新增特征。

## 环境配置

开发语言：Python 3.13+

```bash
uv sync
```

## 快速开始

训练单个模型：

```bash
uv run python main.py train --data-dir data/demo --output models/rf --model sklearn_rf
```

Optuna 调参并训练 XGBoost / LightGBM：

```bash
uv run python main.py tune --data-dir data/demo --output models/xgb_tuned --model xgboost --trials 50
```

评测模型并保存逐帧预测结果：

```bash
uv run python main.py eval --data-dir data/demo --model models/rf
```

用规则碰撞方法评测（无需训练模型）：

```bash
uv run python main.py eval-rule --data-dir data/demo --output outputs/rule_baseline
```

批量运行多个模型进行对比：

```bash
uv run python main.py benchmark --data-dir data/demo --output models/benchmark --jobs 4
```

使用 `Train/` 和 `Test/` 目录训练测试所有模型并生成结论报告：

```bash
uv run python main.py benchmark --data-dir data/split/Train --eval-data-dir data/split/Test --output models/train_test --jobs 4
```

从记录提取特征并保存：

```bash
uv run python main.py export-features --data-dir data/demo --output outputs/features
```

也可以输出更便于查看的 CSV 或 JSONL：

```bash
uv run python main.py export-features --data-dir data/demo --output outputs/features --format csv
```

导出、训练、benchmark 和特征分析都支持通过 JSON 配置只使用已选择的特征：

```bash
uv run python main.py train --data-dir data/demo --output models/selected --feature-config configs/selected_features.json
```

批量对比多组特征配置的 benchmark：

```bash
uv run python main.py benchmark-features --plan configs/feature_benchmark.example.json
```

分析输入记录的特征相关性：

```bash
uv run python main.py analyze-features --data-dir data/demo --output outputs/correlations
```

也可以分析已导出的特征目录：

```bash
uv run python main.py analyze-features --features-dir outputs/features --output outputs/correlations
```

模拟视频流逐帧实时推理：

```bash
uv run python main.py infer-frame --model models/rf --record-dir data/demo --video demo.mp4 --output outputs/realtime.jsonl
```

用规则碰撞方法逐帧推理：

```bash
uv run python main.py infer-rule --record-dir data/demo/record_001 --output outputs/rule_stream.jsonl
```

## 文档

- [数据格式说明](docs/data-format.md)
- [训练、评测与 benchmark](docs/usage.md)
- [实时推理集成](docs/realtime.md)

## 项目结构

```text
src/analysis/
  annotation.py         # annotation.json 解析与货框 token
  records.py            # skeleton.parquet / annotation / event_review 加载
  labels.py             # 人工复核监督信号
  dataset.py            # 训练样本构建与无骨架帧过滤
  rule_collision.py     # 规则碰撞检测（与 event_engine 对齐）
  rule_baseline.py      # 规则基线评测
  features/             # 可扩展特征提取（含 rule_engine 规则特征）
  models/               # sklearn 两阶段模型
  train.py              # 训练流程
  tuning.py             # Optuna 超参数搜索（xgboost / lightgbm）
  evaluation.py         # 指标、预测保存、报告生成
  benchmark.py          # 多模型批量训练评测与基线对比
  feature_benchmark.py  # 多特征配置批量 benchmark
  realtime.py           # 外部应用实时逐帧推理
  cli.py                # 命令行入口
```