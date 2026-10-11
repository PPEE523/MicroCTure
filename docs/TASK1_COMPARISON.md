# 任务一：四模型对比的本地运行说明

这套代码已经建立统一输入、分组划分、训练入口和结果汇总。**模型尚未训练，也没有新的性能结果。** 原始数据和旧数值缓存保持原位，新流程不读取旧分类缓存、权重或划分。

## 四个模型与调用名称

| 调用名称 | 模型 | 默认结构 | 要回答的问题 |
| --- | --- | --- | --- |
| `pca_lr` | PCA + Logistic Regression | 两尺度、信号和遮罩全部展平；训练集 StandardScaler → PCA → 类别加权 LR | 当前数据上复杂模型是否比这一传统基线更好 |
| `m2` | 原双尺度 CNN | 共享三层 CNN，逐尺度 GAP，拼接后分类；27,891 参数 | 在新的共同协议下，原架构表现如何 |
| `m2plus` | 改进双尺度 CNN | 相同 CNN，逐尺度 4×4 空间网格，位置/尺度编码，逐 Token FFN，按位置展平分类；106,163 参数 | 保留空间特征的改进是否有效 |
| `cnn_transformer` | 双尺度 CNN-Transformer | 与 M2+ 相同空间表示和分类头，增加一层 4-head Attention；122,931 参数 | 在这个共同表示上，Attention 是否增加价值 |

M2 的默认网络已与历史定义核对，但新的数据划分、规范化和训练会产生新的数值结果。M2+ 同时改变了空间读出、FFN 和参数量，所以它胜过 M2 不能单独归因于 GAP。M2+ 对应 Transformer 的去 Attention 版本；Transformer 比 M2+ 多约 1.68 万参数，也须在报告中披露。PCA 基线的结果只说明这一具体基线的相对表现，不能证明深度学习在所有方法中“必要”。

Transformer 是小规模探索模型：每尺度 16 个 Token，总计 32 个，维度 64，一层 Attention。当前只有 344 个独立结构，不能预先认定它比 CNN 更好。

## 共同数据与评价协议

- 类别固定为 `OPCID / CHIN / CHID`，独立结构数为 `68 / 250 / 26`；使用 WT 37°C 的两个实验重复。
- 从原始 10 bp `.cool` 流式聚合到 100 bp。提取 6.4 kb 与 25.6 kb 窗口，均生成 `64×64` 输入，含接触信号和有效像素比例两通道。无需构造全基因组稠密矩阵。
- 环形距离带缓存仅保存原始计数；两个重复合计约 **191 MB**。末端不完整 100 bp bin 全部遮罩；圆环 coarse-bin 距离的末端近似记录在元数据中。
- 最大窗口重叠或不足 1 kb 间隔的标注合并为同一空间组。两个重复、相邻窗口和反转增强留在同一组，分组后做五折外层评估。原始矩阵不会按样本划分后独立成新原始文件。
- 每个外层训练折再取约 20% 的分组内层验证集。各模型、各训练种子共用相同内层划分；搜索只使用标签与位置。每个外层测试折及内层验证都要求每类至少 2 个结构，否则报错，不能退回有重叠的随机划分。
- OE 距离背景（包括零计数的机会数）、接触信号均值/标准差、PCA 的额外逐特征 StandardScaler 与 PCA 都只在**内层训练集**拟合。CNN 使用共同的逐尺度信号标准化；PCA 再进行其方法所需的逐特征标准化，包括展平的遮罩特征。外层测试在选择参数/最佳 epoch 后才提取并预测。
- 四模型统一使用训练样本的原图及两个坐标轴同时反转的图；两尺度和遮罩一同反转。内层验证和外层测试使用原图。
- 类别加权来自训练集。两个重复分别作为训练图像；评价时先平均同一结构的两重复概率，再算指标，支持数不会翻倍。
- PCA 搜索维度 `[8,16,32]` 与 `C=[0.1,1,10]`，以内层结构级 macro-F1 选择。CNN 默认 AdamW、学习率 `1e-3`、batch 32、最多 100 epoch、patience 15，同样以内层结构级 macro-F1 选最早最优 checkpoint。
- 选择后保留内层训练得到的模型，不把内层验证重新加入训练。外层评估不参与早停或选参数。
- 神经模型的训练种子为 `42,43,44`，PCA 是固定 seed 42 的单次确定性基线，不将重复的相同 PCA 拟合算成三次独立实验。

外层预测拼成 OOF（out-of-fold）结果，每个结构每个种子恰有一次留出预测。`comparison.csv` 的主 macro-F1 是各完整种子的 **OOF macro-F1 均值**，种子标准差、折间标准差分开记录。单种子的标准差为缺失值，不能写成“零波动”。保存各类 precision/recall/F1、混淆矩阵、实验重复差异，以及可计算的种子间预测一致率。

这是已查看过数据上的探索性分组交叉验证，不能称为新的盲测。坐标约定仍为 `provisional`，当前明确允许暂定的 0-based 左闭右开解释；正式报告应保留这一状态，在确认来源约定后再冻结最终协议。代码不会自动声称“达标”，三类召回与多种子稳定性需要根据跑出的结果判断。

## 一次性准备

在 **WSL 终端**执行：

```bash
cd /home/pppe/MicroCTure
source /home/pppe/miniconda3/etc/profile.d/conda.sh
conda activate microcture
python scripts/run_task1_comparison.py check
python scripts/run_task1_comparison.py prepare
```

`check` 只检查配置和模型规格，不训练、不扫描原始像素。`prepare` 由你运行，扫描两个 WT `.cool` 各一次、建立共同空间分组和五折划分，写入：

```text
data/cache/task1_comparison_v1/
  metadata.json
  folds.json
  WT_37C_rep1.band.npy
  WT_37C_rep2.band.npy
```

准备成功会打印 `structures: 344`、`folds: 5` 和每折三类计数，四个文件都应存在。准备只需成功运行一次，后续四模型共用缓存。已有缓存目录时程序拒绝覆盖；使用既有成功缓存直接训练即可。准备失败时仅清除本次新建的未完成目录，不触碰原始文件。

这一步主要消耗 CPU、磁盘吞吐和分组搜索时间，不使用 GPU；缓存约 191 MB，chunk 内存和数组开销另计。未测量这台电脑的完整扫描耗时，因此不给虚构的分钟数。

## 先跑一折确认速度

```bash
python scripts/run_task1_comparison.py train --model m2 --fold 0
```

折编号为 `0–4`。这条命令跑 M2 的第 0 折、三个种子；输出 epoch 日志及每次 `fit_seconds`，结果暂记为 `pending`。使用 GPU 自动选择 `auto`；指定 `--device cuda` 会在 CUDA 不可用时直接报错，`--device cpu` 可用于 CPU 运行。

完成后执行全折命令，会跳过这 3 次已完成的 M2 拟合。每次神经训练使用内层训练样本数、最多 100 epoch，实际耗时取决于早停。第 0 折三次 `fit_seconds` 的均值 × 15 可以估计某个神经模型的全套时间；不同架构分别测量更准确。

## 逐个调用四模型

以下命令默认各跑完整五折，按顺序执行：

```bash
# PCA + Logistic Regression（CPU，五折；每折九组内层候选）
python scripts/run_task1_comparison.py train --model pca_lr

# 原始 M2（五折 × 三个种子）
python scripts/run_task1_comparison.py train --model m2

# M2+（五折 × 三个种子）
python scripts/run_task1_comparison.py train --model m2plus

# CNN-Transformer（五折 × 三个种子）
python scripts/run_task1_comparison.py train --model cnn_transformer

# 重新生成对比表和图
python scripts/run_task1_comparison.py summarize
```

也可一次执行整组，包含汇总：

```bash
python scripts/run_task1_comparison.py train --model all
```

`all` 和四条独立命令的协议相同。它会按外层折遍历，训练该折所选模型，复用这折的预处理。完成的折自动跳过，无须额外的 resume 开关。不要并行开启多个训练进程：同一结果目录有互斥锁，GPU 也只需要一个训练任务。

中断后重跑原命令，**已完成的模型/种子/折跳过；中断中的那一次从头跑**。只保留每次神经训练的最佳权重，不保留每个 epoch 的权重。运行中的一次性目录位于 `/tmp/microcture_task1_fold_*`，正常结束/异常退出会清理。系统强杀或断电可能留下本次临时目录，此时先确认没有训练进程，保留完整 `fold_0` 等结果，可删除仅属于中断运行的 `/tmp` 目录。

## 参数与更改实验

默认配置为 [configs/task1_comparison.json](../configs/task1_comparison.json)。CLI 全局 `--config` 写在子命令前，例如：

```bash
python scripts/run_task1_comparison.py --config configs/task1_comparison.json train --model m2
```

8 GB GPU 先用默认 batch 32。若实际显存不足，把配置 `training.batch_size` 改为 16，并用新的结果目录跑四模型，保持共同训练设置：

```bash
python scripts/run_task1_comparison.py train --model all --run-dir results/task1_comparison_batch16
```

旧结果保留，不覆写。修改源代码、配置、运行环境或设备指定后，已有结果目录会拒绝混入不同协议，必须指定新的 `--run-dir`。仅改训练/模型设置可以继续使用相同原始计数缓存；修改数据/分组设置则准备新的缓存并在训练时指定同一路径：

```bash
python scripts/run_task1_comparison.py prepare --cache-dir data/cache/task1_comparison_v2
python scripts/run_task1_comparison.py train --model all --cache-dir data/cache/task1_comparison_v2 --run-dir results/task1_comparison_v2
python scripts/run_task1_comparison.py summarize --run-dir results/task1_comparison_v2
```

## 输出、成功判据与回传

默认结果目录为 `results/task1_comparison_v1/`：

```text
manifest.json                  # 配置、代码签名、数据签名、环境、计划模型/种子/折
folds/fold_0.json               # 该折训练/内层验证/外层测试ID和训练拟合参数
runs/<model>/seed_<seed>/fold_<fold>/
  selection.json               # 在外层预测前冻结的选择
  metrics.json                 # validation=内层；test=外层；耗时、参数量等
  predictions.csv              # 逐结构外层预测；同时记录两重复概率
  best.pt                      # 神经模型的唯一最佳权重
  history.csv                  # 神经训练的epoch轨迹
  model.joblib                 # PCA模型；替代best.pt/history.csv
summary.json                   # 每模型每种子完整OOF指标、缺失状态、稳定性
comparison.csv                 # 可放入报告的对比表
comparison.png                 # 四模型比较图
oof/                           # 完整种子的OOF预测与混淆矩阵
```

最后全部四行应为 `complete`：PCA 完成 5 个折，其他模型各完成 15 个折。`pending` 说明仍缺折或缺种子。这表示实验运行完整，不表示结构识别性能达标。以 CHID/OPCID/CHIN 的召回、F1、种子波动和预测一致率判断可靠性，不能只看总准确率。

跑完发回 `comparison.csv`、`summary.json` 和 `manifest.json` 即可，随后再按结果分析失误结构。若失败，发回终端错误结尾和具体命令；若要估算整套时间，先发 M2 第 0 折的 `metrics.json`。原始数据、缓存、最终权重、训练轨迹和数值结果都可复用且对报告有意义，因此保留；一次性临时文件由代码清理。

## 已执行的代码检查

助手仅执行配置/语法检查、合成输入前向和单元检查；23 项检查通过，没有进行真实数据训练、批量预处理或全基因组扫描。流程测试使用假拟合，测试中的 `1.0000` 分数不是实验性能。检查记录见 [代码检查证据](../reports/task1_comparison_code_check.json)。可复用合成测试命令：

```bash
python -B -m unittest discover -s tests -p 'test_task1*.py'
```
