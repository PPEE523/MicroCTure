# MicroCTure：任务一重做起点

2026-10-10 已完成目录清理。旧 M2 未实现稳定可靠的 OPCID、CHIN、CHID 三类识别，任务一需要重新建立数据处理、训练和评价流程。当前没有新的达标模型。

## 保留内容

| 路径                                                             | 用途                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------ |
| `micro-c数据/`                                                 | 全部原始接触矩阵、压缩原件和 Excel 标注，原位保留            |
| `data/metadata/`、`data/processed/`、`data/cache/`         | 既有来源信息、注释、划分和数值矩阵，原位保留                 |
| `data/legacy_experiments/`                                     | 从旧结果目录迁移的全部 CSV、JSON、NPZ 数值记录；仅作历史数据 |
| [课程实践方案](docs/基于深度学习的接触矩阵处理实践方案.pdf)       | 原始任务与报告要求                                           |
| [报告材料索引](report_materials/README.md)                        | 失败复盘、历史图表、方法记录、旧报告和文献                   |
| [历史审计记录](reports/README.md)                                 | 数据来源、旧处理过程和数值核验的出处                         |
| `configs/`、`scripts/prepare_data.py`                        | 数据来源、样本对应、人工观察和原始数据审计脚本               |
| [Conda 环境配置](environment.yml)、[环境说明](docs/ENVIRONMENT.md) | 当前 WSL 本地的独立数据分析与 GPU 训练环境                   |

原有 33 个原始数据及 `data/` 文件的 SHA-256 均与清理前一致。旧结果中的数值记录迁移时保持字节不变；重复展示副本只有在确认保留相同内容后才删除。完整路径对应和校验记录见 [清理清单](report_materials/cleanup_manifest.json)。

## 旧结果的使用边界

M2 三个种子的测试 macro-F1 为 **0.501 ± 0.099**，CHID 分别仅识别正确 **1/4、1/4、0/4**；只有 **24/53** 个测试结构的三种子预测一致。详见 [任务一失败复盘](report_materials/task1_failure_review.md)。

历史缓存、划分、图表、核验“通过”和旧报告均不能作为新任务一已经完成的依据。标注坐标约定仍为 `provisional`，需要核实。旧测试结果已经查看，重做时需要重新明确评价协议。

旧权重、旧执行流程与测试、展示网站、PPT、报告制作脚本、重复导出和 `.venv` 已删除。任务一必要的方法源码和旧依赖记录存于 `report_materials/method_snapshot/`，供报告追溯。Git 历史及原有 `.gitignore` 修改保留。

## 当前 WSL 环境

环境名为 `microcture`，位于 `/home/pppe/miniconda3/envs/microcture`，使用 Python 3.12 和 CUDA 版 PyTorch。

```bash
conda activate microcture
python scripts/check_environment.py
```

检查数据读取与 GPU 前后向计算，结果见 [环境检查记录](reports/environment_check.json)。它验证运行条件，不代表新任务一训练完成。

## 四模型对比代码

新入口 [scripts/run_task1_comparison.py](scripts/run_task1_comparison.py) 统一调用 `pca_lr`、`m2`、`m2plus`、`cnn_transformer`。数据准备和训练由你在本地运行；默认使用空间分组五折评估，神经模型各三个训练种子。已完成的模型/种子/折自动跳过，外层预测与三类指标汇总到 `results/task1_comparison_v1/`。

```bash
python scripts/run_task1_comparison.py check
python scripts/run_task1_comparison.py prepare
python scripts/run_task1_comparison.py train --model all
python scripts/run_task1_comparison.py summarize
```

逐模型命令、分组/预处理规则、资源需求、输出和回传说明见 [本地对比实验说明](docs/TASK1_COMPARISON.md)，参数见 [对比配置](configs/task1_comparison.json)。代码检查通过不代表训练结果达标。
