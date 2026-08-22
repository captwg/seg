# 染色体图像质量评分模型（未增强版）

本目录只包含数据增强前的 `Traditional Quality Ranker V4 — Hierarchical 46` 固定标定模型。
明确不包含 `enhanced_ace_v2_white_background` 的模型、参考分布或输出。

## 内容

- `models/hierarchical_46_fixed_ranker.joblib`：冻结的 4,455 张训练参考分布和七项权重。
- `scripts/score_new_dataset.py`：目录批量评分入口。
- `scripts/build_and_score.py`：分项定义、对象特征和固定百分位评分实现。
- `traditional_quality_ranker/features.py`：灰度、清晰度、背景和形态等基础特征预处理。
- `test_images/`：6 张双重冻结测试样例；既属于评分模型 test，也属于分割模型 test，且已排除与分割 train/val 的精确和近重复。

## 使用

```bash
python scripts/score_new_dataset.py test_images outputs/scores.csv --workers 2
```

输出的 `score` 是相对于冻结训练参考分布的无 GT 代理质量百分位，不是 Dice、AP 或分割成功概率。

模型 SHA256：

```text
49e4c31a4614d08efcde6ccbb4ef5eecd76062daf6f8690ec8243ff19a21e04c
```
