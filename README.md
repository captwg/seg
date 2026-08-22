# 染色体模型发布目录

- `rank/`：未进行 ACE/白背景数据增强的 V4 固定标定质量评分模型。
- `single_seg/`：MaskDINO R50 单解分割结构、step59999 权重、预处理、推理脚本和 6 张无标注测试图片。
- `MANIFEST.sha256`：发布目录全部文件的相对路径与 SHA256。

数据泄露结论分两层：

1. 本发布包的 6 张测试图片通过了独立审计：均同时属于分割 test 和评分 frozen test，不与分割 train/val 精确重复，也未触发近重复筛查。
2. 历史分割 COCO 清单本身存在 1 个 train/val 和 3 个 train/test 精确内容冲突。因此旧的全测试集结果不能宣称为完全无泄露；详细证据见 `single_seg/provenance/leakage_audit.json`。

包内没有 COCO 标注、训练图、验证图、训练数据清单或训练数据绝对路径。
