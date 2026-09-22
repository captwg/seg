# 染色体分割与质量评分

这个仓库提供两项彼此独立的推理能力：

1. **`single_seg/` — 单解实例分割。** 基于 MaskDINO R50，使用固定的 step 59999 checkpoint 为一张图或一个目录生成染色体实例掩膜、实例编号图、叠加预览图和 JSON 结果。
2. **`quality_ranker/` — V4 传统图像质量评分。** 使用冻结的 4,455 张参考图像分布，对输入图像给出 0–100 的跨批次可比质量代理分数和七个组成项。

`quality_ranker/` 保留自原 `captwg/quility_ranker` 的发布提交；本仓库是其后续统一入口。它与根目录已有的 `rank/` 历史审计包并存：日常评分请使用 `quality_ranker/`，`rank/` 只保留旧版本的审计证据和样例。

## 目录与模型

| 路径 | 用途 | 权重和下载方式 |
| --- | --- | --- |
| `single_seg/` | MaskDINO R50 单解实例分割 | `single_seg/weights/model_0059999.pth`，由 Git LFS 管理。克隆后执行 `git lfs pull --include="single_seg/weights/model_0059999.pth"` 下载。 |
| `quality_ranker/` | 冻结 V4 质量评分运行时 | `quality_ranker/models/hierarchical_46_fixed_ranker.joblib`，普通 Git 文件，常规 `git clone` 即会下载。 |
| `rank/` | 旧版评分器、审计记录和无标注样例 | 不作为当前评分入口。 |
| `MANIFEST.sha256` | 统一发布包的完整性清单 | 用 `sha256sum -c MANIFEST.sha256` 校验。 |

分割 checkpoint 固定在训练 step **59999**；其 SHA256 为：

```text
0f9cf0d7f977fcb6c5b6a3b7551bfe8b1983e38eefbbb512c7bad753465c1b4b
```

V4 评分模型的 SHA256 为：

```text
49e4c31a4614d08efcde6ccbb4ef5eecd76062daf6f8690ec8243ff19a21e04c
```

## 先下载权重并校验

GitHub 的网页 ZIP 不会下载 Git LFS 中的真实 checkpoint，因此请使用 Git 克隆。下面的命令只下载分割所需的 LFS 权重；评分模型会随常规克隆一同取得。

```bash
git clone git@github.com:captwg/seg.git
cd seg

git lfs install
git lfs pull --include="single_seg/weights/model_0059999.pth"

# 确认该文件不是 Git LFS 文本指针，并校验两份模型。
git lfs ls-files
sha256sum single_seg/weights/model_0059999.pth
sha256sum quality_ranker/models/hierarchical_46_fixed_ranker.joblib

# 校验整个发布包；需要已完成上面的 git lfs pull。
sha256sum -c MANIFEST.sha256
```

若已克隆仓库但权重缺失，可在仓库根目录重复下面一条命令：

```bash
git lfs pull --include="single_seg/weights/model_0059999.pth"
```

`git lfs ls-files` 应列出 `single_seg/weights/model_0059999.pth`。若 `sha256sum` 的结果与上面的值不同，请删除该工作副本后重新克隆和拉取，不要混用其他 checkpoint。

## 快速开始

分割和评分依赖不同的 Python 环境，建议分别建立环境。评分可以独立运行，完全不需要下载或安装 MaskDINO；分割也不需要质量评分模型。

### 1. V4 质量评分

评分器需要 Python 3.10+。以下命令在仓库根目录执行：

```bash
python -m venv .venv-ranker
. .venv-ranker/bin/activate
pip install -r quality_ranker/requirements.txt

python quality_ranker/scripts/score_directory.py \
  /path/to/input_images \
  /path/to/output/scores.csv \
  --workers 8
```

输入目录会递归搜索 `JPG`、`JPEG`、`PNG`、`TIF`、`TIFF` 和 `BMP`。命令不会训练、重新标定、修改输入文件或进行图像增强；输出 CSV 的 `rank_in_batch` 只表示本次命令内的排序。

主要列包括：

| CSV 列 | 含义 |
| --- | --- |
| `score` | 0–100 的冻结 V4 综合代理分数，可用于跨批次比较。 |
| `estimated_chromosome_count`、`count_absolute_error` | 传统连通域估计及相对 46 的误差，不是人工标注计数。 |
| `*_score`、`*_contribution` | 七个组成项的百分位分数及其对总分的加权贡献。 |
| `relative_path`、`path`、`sha256` | 输入来源和内容身份，便于可复现比对。 |

综合分由下列固定参考分布的百分位数加权得到；参考集包含 4,455 张训练参考图像：

| 组成项 | 权重 |
| --- | ---: |
| segmentation proxy | 40% |
| individual clarity | 22% |
| morphology and distribution | 14% |
| gray visibility | 10% |
| focus | 6% |
| background illumination | 6% |
| original acquisition specification | 2% |

该分数是**无 GT 的传统图像质量代理**，不是 Dice、AP、真实分割成功概率，也不能证明图像增强后每条染色体都被保留。若要评价增强，应在同一冻结模型上按原图文件名、路径或 SHA256 配对前后图像，并额外审查实例数量、断裂、粘连和可视化结果。

评分器的更细说明和其独立完整性清单位于 [`quality_ranker/README.md`](quality_ranker/README.md) 与 [`quality_ranker/MANIFEST.sha256`](quality_ranker/MANIFEST.sha256)。

### 2. MaskDINO 单解实例分割

该模型依赖与 CUDA 对应的 PyTorch、torchvision 和 Detectron2。`requirements.txt` 列出 Python 包名称，但 Detectron2 的安装包必须匹配你的 PyTorch 与 CUDA 版本；不应把一个不匹配的通用 wheel 当作可用环境。首次在新环境使用时还必须编译多尺度可变形注意力算子。

```bash
# 在已配置好匹配的 PyTorch / torchvision / Detectron2 的环境中：
pip install -r single_seg/requirements.txt

cd single_seg/model_code/maskdino/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../../..

# CUDA 可用时默认使用 cuda；输入可以是一张图或目录。
python single_seg/infer.py single_seg/test_images /path/to/segmentation_outputs

# 强制 CPU，或调整实例置信度阈值：
python single_seg/infer.py /path/to/images /path/to/outputs \
  --device cpu --score-threshold 0.5
```

在已验证的发布环境中，Python 位于：

```text
/home/server1/miniconda3/envs/detectron_121/bin/python
```

该路径仅记录已验证环境，不是运行本仓库所必需的绝对路径。使用 GPU 时建议保留默认 `--device cuda`；CPU 可运行但推理会显著较慢。推理前处理会将图像按配置转换为 RGB，并用最短边/最长边均为 512 的规则缩放；输入目录递归支持与评分器相同的六类格式。

每个输入图会生成：

| 输出 | 内容 |
| --- | --- |
| `*_instances.npz` | `masks`（布尔实例掩膜）、`scores` 和 `boxes`（`xyxy`）。 |
| `*_instance_ids.png` | `uint16` 实例编号图；0 为背景，1 起为实例 ID。 |
| `*_overlay.png` | 用不同颜色叠加实例轮廓的人工检查预览图。 |
| `*.json` | 原图尺寸、阈值、实例数、每个实例的分数和框。 |
| `summary.json` | 本次目录推理的所有图像摘要。 |

请保留原图和 `*_overlay.png`，并在人为审核中关注漏检、粘连和断裂。`instance_count` 是本模型在所选阈值下的预测数，不是人工真值染色体数。

分割包的运行细节见 [`single_seg/README.md`](single_seg/README.md)。

## 完整性、范围与历史审计

根目录 `MANIFEST.sha256` 覆盖统一发布包中的运行时文件、模型、样例和说明；`quality_ranker/MANIFEST.sha256` 另外覆盖评分器原始最小运行时。运行校验前，必须先下载 LFS checkpoint。

该仓库不含 COCO 标注、训练图、验证图、训练数据清单或原始训练数据绝对路径。分割发布中的 6 张无标注测试图片经过独立审计：它们同时属于分割测试与评分 frozen test，且未发现与分割 train/val 的精确重复或近重复。历史 COCO 清单仍有 1 个 train/val 和 3 个 train/test 的精确内容冲突，因此历史完整 test 的指标不能称为完全无泄露。证据见：

- [`single_seg/provenance/leakage_audit.json`](single_seg/provenance/leakage_audit.json)
- [`rank/reports/leakage_audit.json`](rank/reports/leakage_audit.json)

`single_seg/model_code/MASKDINO_LICENSE.txt` 保留了 MaskDINO 的许可证文本。运行本仓库前，请同时核对所安装的 PyTorch、Detectron2 与 CUDA 组件的许可证和兼容性。
