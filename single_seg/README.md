# 染色体单解实例分割发布包

正式基线为 MaskDINO R50，固定使用训练 step 59999 的 checkpoint。该包只做单解实例分割，不包含尚未生产验证的局部多假设 Q3–Q6 后处理。

## 内容

- `model_code/maskdino/`：MaskDINO 模型结构、像素解码器、Transformer 解码器和后处理代码。
- `weights/model_0059999.pth`：正式 step59999 checkpoint。
- `preprocessing.py`：与训练/验证一致的 RGB 转换及最短边 512、最长边 512 缩放。
- `infer.py`：单图或目录推理，输出掩膜 NPZ、实例 ID PNG、叠加图和 JSON。
- `configs/`：R50 模型结构及独立推理配置，不含训练数据绝对路径。
- `test_images/`：6 张从双重冻结测试交集中选择的图片，不含标注。
- `provenance/leakage_audit.json`：划分、样例选择和泄露审计。

## 运行环境

已验证环境：

```text
/home/server1/miniconda3/envs/detectron_121/bin/python
```

MaskDINO/Detectron2 通常需要与 CUDA 匹配的源码安装，不能只依赖通用 `pip install detectron2`。
首次迁移到新环境时，还需在 `model_code/maskdino/modeling/pixel_decoder/ops/` 下执行 `sh make.sh`，编译多尺度可变形注意力算子；本包保留源码，不携带含原机器绝对路径的旧构建缓存。

## 推理

```bash
export PYTHONUTF8=1
export PYTHONIOENCODING=UTF-8
/home/server1/miniconda3/envs/detectron_121/bin/python infer.py test_images outputs
```

可调参数：

```bash
python infer.py /path/to/images /path/to/output --device cuda --score-threshold 0.5
```

checkpoint SHA256：

```text
0f9cf0d7f977fcb6c5b6a3b7551bfe8b1983e38eefbbb512c7bad753465c1b4b
```

注意：历史 COCO 清单按实际图片 SHA256 存在少量跨划分重复，详见泄露审计。因此不得把历史全 test 指标称为完全无泄露结果。本发布包的 6 张测试图片已单独排除这些重复，并且全部属于评分模型 frozen test。
