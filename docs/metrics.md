# 评测指标说明

本文说明 `eval`、`benchmark`、`benchmark_report.md` 与 `eval_report.json` 中各项指标的**定义、计算方式与阅读建议**。实现见 `src/analysis/evaluation.py`。

---

## 1. 评测任务

系统包含两个子任务：

| 任务 | 说明 | 真值来源 |
|------|------|----------|
| **帧级取货检测** | 当前帧是否正在取货 | `event_review.json` → `label.is_picking` |
| **货框定位** | 取货帧上目标货框 token 是否正确 | `label.confirmed_box_tokens`（如 `S1:A1`） |

---

## 2. 评测范围

### 2.1 帧级取货

- 对评测集中**每一帧**（或 stride 采样后的每一帧）比较 `true_is_picking` vs `pred_is_picking`。
- 使用 `--feature-frame-stride N` 时，**只对采样序列中的帧计分**，与训练对齐。

### 2.2 货框

- **仅在**「真值为取货 **且** `confirmed_box_tokens` 非空」的帧上计算。
- 比较真值框集合与预测框集合 `predicted_box_tokens`（集合相等性 / token 级 TP/FP/FN）。

### 2.3 规则基线

- `benchmark` 与 `benchmark-features` 均使用外部 `rule_collision`（`box_human_det/services/event_engine/collision.py`）作为规则基线。
- 规则基线对**实际推理帧**计分；`pose_frame_interval > 1` 时跳过的帧不参与指标（与 ML stride 评测口径对齐）。

---

## 3. 训练数据概览（benchmark_report 专用）

出现在 `benchmark_report.md` 的「训练数据概览」一节，描述**训练集**规模（非 Test 集）：

| 字段 | 含义 |
|------|------|
| **训练记录数** | 参与训练的 record 条数 |
| **训练帧数** | 进入 `picking_clf` 的帧级样本数（stride 采样 + 默认过滤无骨架帧后） |
| **正样本帧数** | 训练帧中 `is_picking=true` 的数量 |
| **正样本比例** | `正样本帧数 / 训练帧数` |
| **货框训练样本数** | 进入 `box_clf` 的样本数：每个「帧 × 人 × 货框 token」一行 |

---

## 4. 帧级取货指标

**正类 = 取货**，**负类 = 非取货**。混淆矩阵：

| | 预测取货 | 预测非取货 |
|---|---------|-----------|
| **真取货** | TP | FN |
| **真非取货** | FP | TN |

### 4.1 取货 Precision / Recall / F1

| 指标 | 公式 | 含义 |
|------|------|------|
| **取货 Precision** | TP / (TP + FP) | 预测为取货中，真正取货的比例（误报越少越高） |
| **取货 Recall** | TP / (TP + FN) | 真实取货中被检出的比例（漏报越少越高） |
| **取货 F1** | 2·P·R / (P + R) | Precision 与 Recall 的调和平均 |

报告中「取货 F1 / 取货 Recall / 取货 Precision」即上表三项。

### 4.2 非取货类指标（JSON 中有，报告表一般省略）

| 字段 | 含义 |
|------|------|
| `negative_precision` | TN / (TN + FN) |
| `negative_recall` | TN / (TN + FP) |
| `negative_f1` | 非取货类的 F1 |

### 4.3 Macro-F1

```
Macro-F1 = (取货 F1 + 非取货 F1) / 2
```

- **作用**：正负类 F1 的算术平均，缓解「非取货占绝大多数」时 Accuracy 虚高的问题。
- **报告用途**：模型排序、推荐结论、相对基线 Δ **均基于 Macro-F1**。

### 4.4 Balanced Accuracy（Balanced Acc）

```
Balanced Acc = (取货 Recall + 非取货 Recall) / 2
             = (TP/(TP+FN) + TN/(TN+FP)) / 2
```

正负类召回率的平均，衡量两类是否均衡判对。

### 4.5 Accuracy（JSON 中有，报告表一般省略）

```
Accuracy = (TP + TN) / 总帧数
```

非取货帧远多于取货帧时易偏高，**不建议作为主要指标**。

### 4.6 support 与原始计数

| 字段 | 含义 |
|------|------|
| `support_positive` | 真取货帧数 |
| `support_negative` | 真非取货帧数 |
| `tp` / `fp` / `fn` / `tn` | 混淆矩阵四项计数 |

---

## 5. 货框指标

仅在「真值取货且有 `confirmed_box_tokens`」的帧上统计。每帧比较两个 **token 集合**。

### 5.1 货框精确匹配（exact_match_ratio）

```
精确匹配率 = 预测框集合 == 真值框集合 的帧数 / 参与评测的取货帧数
```

最严格：token 集合必须完全一致。

### 5.2 任意命中（any_hit_ratio，JSON 中有）

```
任意命中率 = 预测框与真值框有交集 的帧数 / 参与评测的取货帧数
```

比精确匹配宽松，比 Micro-F1 更偏「帧级是否沾边」。

### 5.3 货框 Micro-F1（box_micro_f1）

将所有参与帧的框 token **展平**后，按 token 计数：

```
TP = Σ |真值框 ∩ 预测框|
FP = Σ |预测框 − 真值框|
FN = Σ |真值框 − 预测框|

Micro-Precision = TP / (TP + FP)
Micro-Recall    = TP / (TP + FN)
Micro-F1        = 2·Micro-P·Micro-R / (Micro-P + Micro-R)
```

允许「部分框对、部分框错」，适合多框或单框 token 识别质量评估。

---

## 6. Benchmark 报告专用字段

### 6.1 相对基线 Δ / 超过基线

| 字段 | 含义 |
|------|------|
| **相对基线 Δ** | 该模型 Macro-F1 − 规则基线 Macro-F1 |
| **超过基线** | Δ > 0 为「是」；基线行标「基线」 |

### 6.2 模型计算耗时

| 字段 | 含义 |
|------|------|
| **特征数据来源** | `extract` = 现场提取；`cache` = 读 npz 缓存 |
| **特征数据耗时** | 全体模型共享的一次特征提取/加载时间 |
| **特征帧采样间隔** | `--feature-frame-stride`（对列表 `frames[::N]` 采样） |
| **训练拟合(s)** | `model.fit` 耗时 |
| **保存(s)** | 模型写入磁盘耗时 |
| **评测(s)** | Test 集逐帧预测 + 算指标耗时 |
| **总耗时(s)** | 单模型 fit → eval 总耗时 |

### 6.3 结论段落

- 按 **Macro-F1** 降序选推荐模型。
- 若与第二名差距很小，报告会提示结合推理速度、稳定性再选。
- 说明相对规则基线的 Macro-F1 提升幅度。

---

## 7. stride 与基线注意事项

| 场景 | 行为 |
|------|------|
| ML 训练 + eval/benchmark 传相同 `--feature-frame-stride` | 训练与评测帧级、时序特征对齐 |
| 规则基线（`rule_collision`） | 外部 `collision.py`；`pose_frame_interval` 与 ML stride 对齐，**仅对实际 process 的帧计分** |
| `benchmark` 与 `benchmark-features` | 均使用 `rule_collision` 作为规则基线 |

---

## 8. 阅读建议

| 业务关注点 | 建议优先看 |
|------------|------------|
| 综合排序（报告默认） | **Macro-F1** |
| 类别是否均衡 | **Balanced Acc** |
| 不能漏检取货 | **取货 Recall** |
| 不能误报取货 | **取货 Precision** |
| 框 token 识别质量 | **货框 Micro-F1** |
| 框必须完全点对 | **货框精确匹配** |

样本极不均衡时，**优先 Macro-F1、Balanced Acc、取货 Recall、货框 Micro-F1**，而非 Accuracy。

---

## 9. 输出文件对照

| 文件 | 内容 |
|------|------|
| `eval_report.json` | 单模型完整指标（含 JSON 独有字段） |
| `eval_predictions.json` | 逐帧真值/预测明细 |
| `benchmark_report.md` | 多模型汇总表 + 结论（人类可读） |
| `benchmark_summary.json` | 完整 JSON，含 `comparison` 数组 |
| `train_result.json` | 训练规模（帧数、正样本、货框样本等） |

`eval_report.json` 的 `extra.feature_frame_stride` 可确认评测是否使用了 stride 采样。
