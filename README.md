# 货架取货行为分析模型

本项目基于人体骨骼数据和货架标注，训练并评测“是否正在取货、取货货框是哪一个”的机器学习模型。

## 能力范围

- 从 `skeleton.parquet` 和 `annotation.json` 构建特征。
- 从 `event_review.json` 的人工复核结果构建监督信号。
- 支持多个 sklearn 模型训练、评测、批量 benchmark。
- 支持 macro-F1、balanced accuracy、货框 micro-F1 等不均衡样本友好指标。
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

评测模型并保存逐帧预测结果：

```bash
uv run python main.py eval --data-dir data/demo --model models/rf
```

批量运行多个模型进行对比：

```bash
uv run python main.py benchmark --data-dir data/demo --output models/benchmark --jobs 4
```

模拟视频流逐帧实时推理：

```bash
uv run python main.py infer-frame --model models/rf --record-dir data/demo --video demo.mp4 --output outputs/realtime.jsonl
```

## 文档

- [数据格式说明](docs/data-format.md)
- [训练、评测与 benchmark](docs/usage.md)
- [实时推理集成](docs/realtime.md)

## 项目结构

```text
src/analysis/
  annotation.py      # annotation.json 解析与货框 token
  records.py         # skeleton.parquet / annotation / event_review 加载
  labels.py          # 人工复核监督信号
  dataset.py         # 训练样本构建
  features/          # 可扩展特征提取
  models/            # sklearn 两阶段模型
  train.py           # 训练流程
  evaluation.py      # 指标、预测保存、报告生成
  benchmark.py       # 多模型批量训练评测
  realtime.py        # 外部应用实时逐帧推理
  cli.py             # 命令行入口
```