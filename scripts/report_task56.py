"""Generate optional-task reports from completed, checked experiments."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import json
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optional_common import ROOT, read_csv, write_json
from train_task5 import profile


def curve_figure(directory):
    histories = sorted(directory.glob('*_history.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in histories:
        rows = json.loads(path.read_text())
        for ax, key in zip(axes, ['train_loss', 'validation_loss']):
            ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], label=path.stem.removesuffix('_history'))
            ax.set(xlabel='epoch', ylabel=key)
    axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(directory/'learning_curves.png', dpi=140); plt.close(fig)


def family_mean(rows, family, replicate, metric, task):
    selected = [r for r in rows if (r['model'] == family or r['model'].startswith(family+'_'))
                and int(r['replicate']) == replicate and (r['type'] == 'all' if task == 5 else r['partition'] == 'test')]
    values = [float(r[metric]) for r in selected]
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.


def format_number(pair, places=3):
    mean, std = pair
    return f'{mean:.{places}f} ± {std:.{places}f}' if std else f'{mean:.{places}f}'


def task5():
    out = ROOT/'results/task5'
    rows = read_csv(out/'summary.csv')
    table = ['| 方法 | rep1 PSNR ↑ | rep1 SSIM ↑ | rep2 PSNR ↑ | rep2 SSIM ↑ | rep1 边界偏差 bp ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    summary = {}
    for family in ['bilinear', 'bicubic', 'residual_cnn', 'plain_cnn']:
        pairs = [family_mean(rows, family, rep, metric, 5) for rep, metric in
                 [(1, 'psnr'), (1, 'ssim'), (2, 'psnr'), (2, 'ssim'), (1, 'boundary_shift_bp')]]
        table.append('| '+family+' | '+' | '.join(format_number(pair) for pair in pairs)+' |')
        summary[family] = dict(zip(['rep1_psnr', 'rep1_ssim', 'rep2_psnr', 'rep2_ssim', 'rep1_boundary_shift_bp'],
                                   [dict(mean=mean, seed_std=std) for mean, std in pairs]))
    text = '''# 任务五：接触矩阵超分辨重建

完成了 800 → 200 bp 的合成降分辨率实验：两种插值基线、残差 CNN、去残差 CNN 消融；两个 CNN 各 3 个种子，共 6 次真实训练。**图像指标改善，但尚未证明结构位置恢复稳定优于插值。**

## 输入与公平评价

每个已知结构使用 25.6 kb 固定窗口，覆盖包括最长结构在内的完整注释范围。100 bp 原始计数缓存先聚合为 128 × 128 的 200 bp 目标，再以有效像素均值进行 4 × 4 合并，得到 32 × 32 的 800 bp 输入。这里的平均接触强度与求和计数相差确定的分箱因子；不是对彩色图像降采样。

沿用空间分组：240 / 51 / 53 个结构，分区间 1 kb 隔离。训练和验证只用 rep1；测试的 53 个位置分别评价 rep1、rep2。覆盖阈值和 log1p 信号的 99.5% 分位尺度由 rep1 训练区拟合。rep2 使用同一归一化尺度与深度校正后的覆盖阈值。接触值变换到 [0,1] 并截断；截断比例记录于本地 `results/task5/manifest.json`。

残差模型为三层卷积（16 通道），在双线性上采样后预测修正量，输出与转置取均值得到对称矩阵；消融去掉输入到输出的残差加法。二者训练预算、学习率、划分、损失相同，但残差模型末层采用零初始化，因此这比较的是完整的残差训练方案，而非只改一个开关。最多 30 轮，验证集选择 checkpoint；不选最佳测试种子。插值和 CNN 均裁剪到相同范围。

PSNR 的固定动态范围为 1；SSIM 使用 7 × 7 均匀局部窗口，仅在窗口内全部像素有效时评价。只计上三角，排除距离不足 400 bp 的像素。各窗口等权平均，± 是 3 个训练种子的样本标准差，非生物学置信区间。

## 实测结果

TABLE

![重建指标](../results/task5/comparison.png)
![学习曲线](../results/task5/learning_curves.png)

[全部分类汇总](../results/task5/summary.csv)包含每类结果；完整逐窗口指标为本地 `results/task5/metrics.csv`（848 行）。残差连接整体改善训练稳定性，但不同种子仍有差异；不能仅用一个最优种子代表方法。

## 已知结构恢复与反例

以跨对角线接触剖面衡量形态；在注释 start/end 附近固定 ±5 个 200 bp bin，比较重建和目标剖面谷值位置，报告平均偏差。这是利用已知位置的辅助评价，不是从全基因组盲检测得到的定位精度，也不直接证明注释边界真实性。

残差 CNN 的边界偏差高于双三次插值，因此不能从 PSNR 的提升推出结构定位更好。当前记录满足模型、权重、矩阵、基线、图像和下游评价等交付要求；课程要求中的“稳定恢复形态与位置”仍只有部分支持。

![固定测试案例](../results/task5/examples.png)
![去距离背景的形态和接触剖面](../results/task5/structure_profiles.png)

展示对象是每类第一个测试结构，未按模型表现筛选。第二张图减去训练区平均距离背景以显露局部模式；剖面竖线表示暂定注释边界。

## 距离分层诊断

查看总体指标后增加了明确标记的事后诊断，未重新训练或改变选模：与仅输出训练平均距离背景的模型比较，并按 0.4–1.6、1.6–6.4、6.4–25.6 kb 分层。

rep1 的固定种子 42：相对双三次插值，残差 CNN 在近距离层 PSNR 从 14.079 提高到 25.493 dB；中距离从 36.882 到 37.834 dB；远距离则从 46.930 降到 46.460 dB。不能把整体约 10 dB 的改善理解为所有位置都改善。详见 [分层诊断表](../results/task5/distance_diagnostics.csv)。

## 交付与限制

本地 `results/task5/` 包含 6 个 `.pt` checkpoint、6 份训练历史、8 份重建 `.npz`、逐窗口结果及 HTML 入口。NPZ 保存归一化重建、逆变换后的每百万接触强度、目标、掩码、窗口索引和分辨率；逆变换无法恢复被截断的高值。

这是 Micro-C 自身的合成退化实验；目标也是含测量噪声的观测，不是无噪声真值。没有做真实 Hi-C → Micro-C、10 bp 重建或独立新实验验证。旧 WT 测试区域已在前序任务中被查看，本轮属于探索性研究。原始坐标约定仍为 provisional。

复现命令见 [复现指南](REPRODUCIBILITY.md)，数值验收见 [独立检查](../reports/task56_verification.json)。
'''.replace('TABLE', '\n'.join(table))
    (ROOT/'docs/任务五超分辨实验结果.md').write_text(text)
    # Diagnostic view exposes shape beyond the strong diagonal background.
    with np.load(out/'distance_only_diagnostic.npz') as z: background = z['background']
    windows = read_csv(out/'windows.csv')
    datasets = {name: np.load(out/f'{name}_reconstruction.npz') for name in ['bicubic', 'residual_cnn_42']}
    z = datasets['bicubic']; distance = np.abs(np.arange(128)[:, None]-np.arange(128)[None, :])
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for line, kind in enumerate(['OPCID', 'CHIN', 'CHID']):
        j = next(j for j, i in enumerate(z['window_indices']) if windows[i]['type'] == kind and windows[i]['replicate'] == '1')
        row = windows[z['window_indices'][j]]; mask = z['mask'][j]
        images = [z['target'][j], z['prediction'][j], datasets['residual_cnn_42']['prediction'][j]]
        for ax, matrix, name in zip(axes[line, :3], images, ['target', 'bicubic', 'residual CNN 42']):
            ax.imshow(np.where(mask, matrix-background[distance], np.nan), vmin=-.08, vmax=.08, cmap='coolwarm', origin='lower')
            ax.set_title(f'{row["structure_id"]}: {name}', fontsize=9)
        ax = axes[line, 3]
        for matrix, name in zip(images, ['target', 'bicubic', 'CNN 42']):
            ax.plot(np.arange(128)*.2, profile(matrix, mask), label=name)
        for end in ['start', 'end']: ax.axvline((int(row[end])-int(row['start_bin'])*100)/1000, color='gray', ls=':')
        ax.set(xlabel='kb from window start', ylabel='cross-boundary signal'); ax.legend(fontsize=7)
    fig.suptitle('Fixed examples: residual to training distance background (shared +/-0.08); profiles')
    fig.tight_layout(); fig.savefig(out/'structure_profiles.png', dpi=140); plt.close(fig)
    for dataset in datasets.values(): dataset.close()
    curve_figure(out)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>任务五：超分辨实验</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}p{line-height:1.7}</style>'
        '<h1>任务五：800 → 200 bp 合成超分辨</h1><p>图像质量改善，结构边界定位尚未优于插值；全部种子均报告。</p>'
        '<p><a href="summary.csv">分类汇总</a> · <a href="metrics.csv">逐窗口指标</a> · <a href="distance_diagnostics.csv">距离分层</a></p>'
        +''.join(f'<h2>{html.escape(title)}</h2><img src="{file}">' for title, file in
                  [('重建质量', 'comparison.png'), ('学习曲线', 'learning_curves.png'), ('固定案例', 'examples.png'), ('形态与位置', 'structure_profiles.png')]))
    return summary


def task6():
    out = ROOT/'results/task6'; rows = read_csv(out/'metrics.csv')
    table = ['| 方法 | rep1 距离 RMSE ↓ | rep2 距离 RMSE ↓ | rep1 接触相关 ↑ | rep2 接触相关 ↑ | rep1 距离层内相关 ↑ |',
             '|---|---:|---:|---:|---:|---:|']
    summary = {}
    for family in ['mds', 'geometry', 'graph']:
        pairs = [family_mean(rows, family, rep, metric, 6) for rep, metric in
                 [(1, 'log_distance_rmse'), (2, 'log_distance_rmse'), (1, 'contact_log_pearson'),
                  (2, 'contact_log_pearson'), (1, 'within_distance_pearson')]]
        table.append('| '+family+' | '+' | '.join(format_number(pair) for pair in pairs)+' |')
        summary[family] = dict(zip(['rep1_rmse', 'rep2_rmse', 'rep1_pearson', 'rep2_pearson', 'rep1_within_distance_pearson'],
                                   [dict(mean=mean, seed_std=std) for mean, std in pairs]))
    manifest = json.loads((out/'manifest.json').read_text())
    text = '''# 任务六：全基因组三维形态重建

完成两个 WT 重复的 5 kb 全基因组聚合、MDS 基线、距离几何优化和神经图解码器，共 7 组坐标（含 6 个训练 checkpoint）。**图模型降低了留出距离误差，但没有全面优于 MDS；当前无法确认复现了真实宏观结构域的三维形态。**

## 数据与评价划分

每重复 929 个 bin，覆盖 4,641,652 bp，最后一个 bin 为 1,652 bp。流式聚合只生成 929 × 929 稠密矩阵；对角线计数不重复，聚合前后原始上三角总计数守恒。后续按 bin 实际长度修正末端暴露量；rep2 按文库量对齐到 rep1。环形基因组距离用 bin 索引计算，末端不等长使这一距离近似而非精确。未进行 ICE 平衡，覆盖和可比对性偏差仍可能影响远离主体的点。

rep1 的非相邻上三角接触对按固定种子分为训练 / 验证 / 测试，数量分别为 COUNTS。测试边及其转置不进入图邻接、MDS 输入的已知边或任何拟合统计；缺失边以训练边按环形基因组距离的中位目标距离填补。rep2 不参与模型训练与选模，按同一边集合评价。

这是传导式图重建：全部模型共享同一染色体节点，留出的是接触边而非新细胞、新染色体或完全独立区域。测试边还共享端点，不能把数万条边解释为数万个生物学重复。

## 模型与共同目标

接触 C 转成目标距离 `(C + 0.5)^(-1/3)`，再用训练边的中位数定尺度。指数 1/3 是固定建模假设，没有声称由本数据估计出真实物理关系。

- **MDS**：对训练接触和训练距离背景填补矩阵进行经典三维 MDS，再仅以训练边校正全局距离尺度。
- **geometry**：从 MDS 加微小随机扰动出发，直接优化节点坐标。
- **graph**：MDS 初始化的残差图解码器，训练图邻接聚合、32 维节点嵌入、固定环形位置特征，输出三维坐标修正。它是训练在本染色体上的神经模型，不是可以零样本泛化的预训练 GNN。

geometry 与 graph 各 3 个种子，采用同一对数距离平方误差、相邻环形弹簧项、学习率、最大 500 轮和验证选模规则。邻接约束包含首尾连接，是先验而不是观察结果。已知宏观结构域标签不参与训练。

## 留出结果

TABLE

距离 RMSE 指对数距离的 RMSE；接触相关为逆变换预测接触与观测接触的 log1p Pearson。距离层内相关先在每个精确环形分离距离上计算相关，再按有效边数加权，减少天然距离衰减主导的解释。完整表另附 Spearman 和扣除训练距离背景后的相关。

图模型相比 MDS 的距离误差降低约三分之一，与直接距离几何优化接近；但接触排序及距离层内相关下降。这是损失目标的取舍，不能只报告有利的误差指标。三个 graph 种子的全对距离相关约 0.981–0.987，说明当前优化相对稳定，不代表构象已被生物学验证。

![三维对照](../results/task6/structures.png)
![反算接触图](../results/task6/contact_maps.png)
![训练曲线](../results/task6/learning_curves.png)

## 整体形态核查

参考 [Valens 等人的原始宏观结构域研究](https://doi.org/10.1038/sj.emboj.7600434)，图中使用 Ori、Right、Ter、Left 及两个较少结构化区域的**近似示意分段**；遗传图位置线性换算到本基因组，尚未序列级核实边界。mioC 位置仅提示 oriC 邻域，dnaA 来自现有基因表；Ter 标记为示意中点，不是假称精确 dif 位点。

对每个示意域计算回转半径，与同长度弧段的全部循环平移参照比较，结果见 [域形态表](../results/task6/domain_geometry.csv)。固定 graph_42 的 Ori、Right、Ter、Left 回转半径相对匹配弧段中位数约为 1.045、1.016、1.056、0.977，并没有提供明确的四域独立紧致性支持。参照间彼此相关，排名分数不作为 p 值。

因此，颜色分区、环状连线和整体图形本身不足以证明真实细胞内空间组织。本实验完成了模型、坐标、反算图、基线指标和可视化交付；“复现已知整体形态”的科学验证仍未满足。群体平均接触和简单幂律不能唯一决定单细胞三维构象，也没有长度为纳米的物理标定。

## 交付

- [留出评价表](../results/task6/metrics.csv)、[种子稳定性](../results/task6/seed_stability.csv)、[固定种子坐标示例](../results/task6/graph_42_coordinates.csv)。
- 本地 `results/task6/index.html`：可拖动旋转、缩放和切换 7 个模型的独立 HTML，无网络依赖。
- 本地 `results/task6/*_coordinates.csv`、`*_maps.npz`、`*.pt`、`*_history.json`：所有坐标、距离/接触图、权重和训练过程。
- [数据守恒审计](../reports/task6_data_audit.json)、[独立数值验收](../reports/task56_verification.json)。

复现步骤见 [复现指南](REPRODUCIBILITY.md)。
'''.replace('TABLE', '\n'.join(table)).replace('COUNTS', ' / '.join(str(manifest['pair_counts'][key]) for key in ['train', 'validation', 'test']))
    (ROOT/'docs/任务六三维重建实验结果.md').write_text(text)
    curve_figure(out)
    return summary


def main():
    summary = dict(task5=task5(), task6=task6(),
                   interpretation='Engineering deliverables complete; task5 boundary improvement and task6 biological morphology validation remain unsupported.')
    write_json(ROOT/'reports/task56_results.json', summary)
    print('Generated task 5/6 reports, learning curves and task 5 gallery')


if __name__ == '__main__':
    main()
