# 环境与复现指南

## 三种使用方式

| 目标 | 是否需要原始数据 | 入口 |
|---|---|---|
| 浏览方法与结果 | 否 | [交互作品页](../site/index.html)、[结果精选](../showcase/README.md) |
| 检查代码与展示文件 | 否 | 合成单元测试、快照与仓库检查 |
| 重跑完整实验 | 是 | 下列分阶段命令；数据见 [输入说明](../data/README.md) |

记录环境为 Linux、Python 3.14、CPU PyTorch。`requirements.txt` 是数据阶段实际环境；`requirements-training.txt` 是含训练依赖的实际环境；`requirements-build.txt` 保留构建依赖；`requirements.in` 是早期数据阶段的直接依赖清单，不含完整训练依赖。没有声称所有平台均可安装或已经验证。

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m unittest discover -s tests -v
python3 scripts/export_showcase.py --check
python3 scripts/check_repository.py
```

单元测试用临时目录与合成数据验证坐标、掩码、空间隔离、损失、筛选预算和判据，不下载真实数据。GitHub Actions 配置执行上述公开文件检查与合成测试；完整研究流水线不会在 CI 自动运行。

## 原始数据与预处理

按 [数据说明](../data/README.md) 放置 WT 两重复和 Excel，随后运行：

```bash
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/plot_examples.py
.venv/bin/python scripts/prepare_dataset.py
```

脚本通过自身路径定位项目根目录，不依赖终端当前目录。预处理分块扫描像素和局部接触带，不构建全基因组稠密矩阵。原始数据不可被模型输出覆盖。

## 结构分类、对照与消融

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/train_task1.py --threads 4
.venv/bin/python scripts/report_task1.py
.venv/bin/python scripts/check_task1.py
```

单独加载已有权重复现预测（必须先有完整本地运行产物）：

```bash
.venv/bin/python scripts/predict_task1.py --checkpoint results/task1/M2_42.pt --data data/cache/classification_v1/test.npz --output results/task1/reproduced_predictions.csv
```

## 候选发现：按依赖顺序复现

第二轮沿用第一轮扫描的窗口标识和注释信息，因此应先运行第一轮。以下命令也生成后续核查需要的表。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/discover_task2.py
.venv/bin/python scripts/report_task2.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/diagnose_discovery_v2.py
.venv/bin/python scripts/check_discovery_v2.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/discover_task2_v2.py
.venv/bin/python scripts/check_task2_v2.py
.venv/bin/python scripts/report_task2_v2.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/review_novelty.py
.venv/bin/python scripts/check_novelty_review.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/centering_sensitivity.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/supplement_centering.py
.venv/bin/python scripts/report_centering_combined.py
.venv/bin/python scripts/check_centering.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/audit_morphology.py
.venv/bin/python scripts/report_morphology_audit.py
.venv/bin/python scripts/check_morphology_audit.py
```

`configs/morphology_observations.json` 保存针对当前 21 个窗口的描述性图像审阅，不是可以无条件迁移到新数据的自动类型标签。换数据后必须重新审阅，不应复用这些结论。

## 多条件信号与差异分析

先放置 WT 与 Excel。GEO SOFT 元数据已随仓库提供。

```bash
.venv/bin/python scripts/download_conditions.py
.venv/bin/python scripts/prepare_conditions.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/analyze_conditions.py
.venv/bin/python scripts/check_conditions.py
.venv/bin/python scripts/report_conditions.py
```

下载器只获取配置明确指定的四个额外条件归档，支持续传并校验 gzip；后续脚本解压和核对坐标。它不会替你获取 WT 或课程 Excel。

## 超分辨与三维嵌入

任务四使用上面的条件差异流程。任务五需要任务二生成的两份 WT 100 bp 带状计数缓存，以及 `classification_split.csv`；任务六直接读取两个原始 WT Cooler。沿用现有依赖，无需新下载条件或安装图形库。

```bash
# 第一步：任务四重新验收
.venv/bin/python scripts/check_conditions.py

# 第二步：6 次超分辨训练 + 插值对照 + 明确标记的事后距离诊断
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/train_task5.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/diagnose_task5.py

# 第三步：全基因组聚合、MDS 与 6 次三维拟合
.venv/bin/python scripts/prepare_task6.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/train_task6.py

# 独立复算指标并生成报告
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/check_task56.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/report_task56.py
```

参数为 `configs/optional_tasks.json`；宏观结构域示意在 `configs/macrodomain_schematic.json`。保持参数、脚本、输入哈希不变时，可复用本地 checkpoint 重新评价；不自动覆盖不同版本的输出。现有运行是 CPU 实验，所有种子都必须报告。

本地入口为 `results/task5/index.html` 和 `results/task6/index.html`。任务六页面支持鼠标旋转、滚轮缩放和模型切换，不依赖网络。精选 HTML 随公开快照提供，可直接下载打开；GitHub 文件页本身不执行 HTML。

方法与评价口径见 [技术设计](PROJECT.md) 和 [结果摘要](RESULTS.md)。

## 运行记录与重新执行

- 完整输出写入 `results/`，缓存写入 `data/cache/`，两者不提交 Git。
- `reports/` 中的现有 JSON 是历史运行的审计记录，不表示刚克隆仓库后已经重新通过所有实验核验。依赖权重、原始数据或缓存的 `check_*.py` 必须在对应阶段运行后执行。
- 多个实验将配置、源码、输入大小或修改时间写入 manifest。文件签名不同会拒绝复用结果，这是避免混用版本的保护。输入改变应使用新的输出目录；不要删除哈希断言来强行复用。
- 跨平台和依赖差异可能影响数值或图像字节；展示快照的 SHA256 校验仅保证发布文件没有改变，不宣称训练在任何机器都逐字节一致。
- 本地保留的 `*_superseded_*`、失败序列化目录属于历史诊断产物，不参与当前结论，也不导出到公开精选。

## 更新公开结果

```bash
python3 scripts/export_showcase.py
.venv/bin/python scripts/build_portfolio.py
python3 scripts/check_repository.py
```

这些命令需要完整本地输出存在，浏览或验证已有快照则不需要。请在导出后审阅差异，确保 README 中的结论与结果一致。

封面默认使用公开的固定矩阵预览和坐标重绘，不需要原始数据。若需要从本地超分辨 NPZ 重新导出固定案例，使用 `scripts/build_portfolio.py --refresh-data`。该操作不重新训练模型。

```bash
python3 -m http.server 8000
# 打开 http://localhost:8000/site/
```

`.gitignore` 让阶段报告、旧版图集和本地运行产物留在工作区，不进入公开提交；文件不因此删除。公开展示统一使用 `showcase/`，已有作业图集仍在 `examples/` 和 `results/`。
