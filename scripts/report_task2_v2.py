"""Generate a complete v2 experiment report without selecting successful runs."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discover_task2_v2 import ROOT, read_csv
from discover_task2 import write_csv


def main():
    cfg = json.loads((ROOT / 'configs/discovery_v2.json').read_text())
    out = ROOT / cfg['output_dir']
    report = json.loads((out / 'report.json').read_text())
    frozen = json.loads((out / 'frozen_before_rep2_reload.json').read_text())
    recalls = read_csv(f"{cfg['output_dir']}/known_recall.csv")
    validation = read_csv(f"{cfg['output_dir']}/candidate_validation.csv")
    clusters = read_csv(f"{cfg['output_dir']}/clusters.csv")
    rows = read_csv(f"{cfg['output_dir']}/scan_rep1.csv")
    union = read_csv(f"{cfg['output_dir']}/unannotated_supported_union.csv")
    method_table = ['| 方法 | 未注释候选 | 通过重复门槛（全部） | 其中未注释 | 稳定新簇配置数 |', '|---|---:|---:|---:|---:|']
    for r in report['methods']:
        method_table.append(f"| {r['method']} | {r['unannotated']} | {r['replicate_supported']} | {r['unannotated_supported']} | {r['stable_cluster_configurations']} |")
    training_table = ['| 模型 | 实际轮数 | 最佳轮次 | 最佳早停集损失 | 内部评价集损失 |', '|---|---:|---:|---:|---:|']
    for r in report['training']:
        training_table.append(f"| {r['model']} | {r['epochs']} | {r['best_epoch']} | {r['validation_loss']:.5f} | {r['test_loss']:.5f} |")
    # Exact same candidate quotas and coverage within each evaluation domain.
    names = [r['method'] for r in report['methods']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, kind in zip(axes, ['OPCID', 'CHIN', 'CHID']):
        values = [next(float(r['recall']) for r in recalls if r['domain'] == 'test' and r['method'] == name and r['type'] == kind) for name in names]
        random_values = [float(r['recall']) for r in recalls if r['domain'] == 'test' and r['method'].startswith('random_') and r['type'] == kind]
        ax.bar(np.arange(len(names)), values, color=['#4678a5' if not name.startswith('dae') else '#d88038' for name in names])
        ax.axhspan(*np.quantile(random_values, [.025, .975]), color='gray', alpha=.15, label='20 random runs: 95% empirical range')
        ax.axhline(np.mean(random_values), color='gray', ls='--', label='Random mean')
        ax.set(xticks=np.arange(len(names)), xticklabels=names, ylim=(0, 1.05),
               title=f"{kind} (n={report['test_known_counts'][kind]})", ylabel='Known structure recall')
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    axes[0].legend(fontsize=6)
    fig.suptitle('Internal spatial evaluation: fixed 27 candidates / 307.2 kb coverage; previously inspected data')
    fig.savefig(out / 'known_recall_comparison.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    xx = np.arange(len(names))
    ax.bar(xx, [r['replicate_supported'] for r in report['methods']], color='#4678a5', label='All supported windows')
    ax.bar(xx, [r['unannotated_supported'] for r in report['methods']], color='#d88038', label='Unannotated subset')
    ax.set(xticks=xx, xticklabels=names, ylabel='Window count', title='Fixed 189 candidates / 2.1504 Mb coverage per method')
    ax.tick_params(axis='x', labelrotation=60, labelsize=8)
    ax.legend()
    fig.savefig(out / 'replication_comparison.png', dpi=140)
    plt.close(fig)
    # Empty output is explicit when no cluster passes, preserving the distinction
    # between replicated windows and supported candidate classes.
    candidate_new = []
    by_index = {int(r['window_index']): r for r in validation}
    for r in clusters:
        if r['stable_candidate_new_cluster'] != 'True':
            continue
        key = f"{r['method']}:{r['k']}:{r['seed']}"
        labels = frozen['clustering'][key]['labels']
        for i, label in zip(frozen['selected'][r['method']], labels):
            if label == int(r['cluster']) and by_index[i]['replicate_supported'] == 'True':
                candidate_new.append(dict(method=r['method'], k=r['k'], seed=r['seed'], cluster=r['cluster'], **by_index[i]))
    write_csv(out / 'candidate_new_structures.csv', candidate_new, ['method', 'k', 'seed', 'cluster'] + list(validation[0]))
    gallery = []
    for r in sorted(union, key=lambda r: -float(r['correlation'])):
        title = f"{r['window_id']} · r={float(r['correlation']):.3f} · 背景阈值={float(r['cutoff']):.3f} · {r['start_bp']} bp 起，宽 {r['window_length_bp']} bp"
        name = f"heatmaps/{r['window_id']}.png"
        gallery.append(f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)}</figcaption></figure>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>任务二第二轮训练与扫描</title>
<style>body{{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif}}img{{max-width:100%}}figure{{margin:25px 0}}</style>
<h1>任务二第二轮训练与重新扫描</h1>
<p>普通 AE、去噪 AE 各三个种子，14 个评分配置，168 次聚类。主配置稳定候选新簇：{report['primary_candidate_new_clusters']}；全部配置通过次数：{report['all_passing_cluster_configurations']}（不同配置不视为独立发现）。</p>
<p>全部方法的未注释复现窗口并集：{report['unannotated_supported_union']} 个；去除跨方法空间重叠后：{report['unannotated_supported_disjoint']} 个。这些是待核查窗口，不等同于新类型。WT 两重复已被查看，本轮属于探索性再分析。</p>
<p><a href="method_summary.csv">全部方法汇总</a> · <a href="known_recall.csv">等预算已知召回</a> · <a href="candidate_validation.csv">逐窗口验证</a> · <a href="clusters.csv">全部聚类结果</a> · <a href="unannotated_supported_disjoint.csv">非重叠未注释复现窗口</a> · <a href="candidate_new_structures.csv">通过新簇判据的窗口</a></p>
<img src="replication_comparison.png" alt="重复验证结果对照"><img src="known_recall_comparison.png" alt="空间评价区等预算召回对照">
<h2>未注释复现窗口（跨方法并集，包含重叠）</h2><p>双重复热图共用仅由 rep1 主配置确定的色标。跨原点窗口使用展开的分箱坐标；来源坐标约定仍为暂定。</p>
'''
    (out / 'index.html').write_text(page + '\n'.join(gallery) + '</html>')
    primary = next(r for r in report['methods'] if r['method'] == cfg['primary_method'])
    recall_table = ['| 方法（内部评价区） | OPCID | CHIN | CHID |', '|---|---:|---:|---:|']
    for name in names:
        values = [next(r for r in recalls if r['domain'] == 'test' and r['method'] == name and r['type'] == kind) for kind in ['OPCID', 'CHIN', 'CHID']]
        recall_table.append('| ' + name + ' | ' + ' | '.join(f"{r['recalled']}/{r['total']} ({float(r['recall']):.1%})" for r in values) + ' |')
    random_means = []
    for kind in ['OPCID', 'CHIN', 'CHID']:
        values = [float(r['recall']) for r in recalls if r['domain'] == 'test' and r['method'].startswith('random_') and r['type'] == kind]
        random_means.append(f'{np.mean(values):.1%}')
    recall_table.append('| 等预算随机参考均值（20 次） | ' + ' | '.join(random_means) + ' |')
    text = f'''# 任务二第二轮：训练与重新扫描结果

本轮已完成普通 AE、去噪 AE 各三个种子的六次训练，并用 14 个评分配置重新扫描和选择候选，执行 168 次聚类。第一轮和第二轮固定候选诊断均保留。

主配置预先固定为 `{cfg['primary_method']}`、K={cfg['primary_k']}、聚类种子 {cfg['primary_cluster_seed']}：189 个候选中 {primary['replicate_supported']} 个通过重复门槛，其中 {primary['unannotated_supported']} 个未注释；稳定候选新簇 **{report['primary_candidate_new_clusters']} 个**。所有配置通过稳定新簇判据的次数为 {report['all_passing_cluster_configurations']}。跨方法未注释复现窗口并集为 {report['unannotated_supported_union']} 个，按相关性降序消除空间重叠后保留 {report['unannotated_supported_disjoint']} 个，不能把重叠窗口或不同配置的结果当作独立发现重复计数。

入口：[浏览图表与窗口](../results/task2_v2/index.html)、[模型与候选汇总](../results/task2_v2/method_summary.csv)、[全部聚类结果](../results/task2_v2/clusters.csv)、[验证报告](../reports/task2_v2_verification.json)。

## 固定的实验规则

沿用 100 bp 分辨率、6.4/25.6 kb 窗口、1.6 kb 步长，扫描 {report['scan_windows']} 个窗口，{report['eligible_windows']} 个通过有效比例 ≥0.8、零比例 ≤0.8 的质量规则。

连续 200 kb 区块按编号 mod 10 划分：0–5 训练、6–7 早停、8–9 内部评价。跨区块或跨染色体原点的窗口不用于训练与内部评价，但可进入全基因组探索扫描。合格窗口数为训练 {report['partition_windows']['train']}、早停 {report['partition_windows']['validation']}、内部评价 {report['partition_windows']['test']}。

低覆盖统计只累计同一分区内、距离 200 bp–25.5 kb 的局部接触。阈值为训练区正覆盖中位数的 10%，应用于各区；训练距离期望仅使用两端都在训练区且有效的接触，包含零值机会。rep1、rep2 分别用各自训练区拟合背景。这样避免评价区接触间接改变训练掩码；同时分区边缘局部覆盖可能受截断影响。每窗口标准化是样本自身变换，不跨窗口估计。

神经模型完全同架构：8/16/32 通道卷积、16 维瓶颈、对称解码器。每种子配对使用完全相同的最多 2,000 个训练窗口、初始化、批次顺序、Adam 学习率 0.001、批量 64、最大 40 轮、patience=6。唯一训练消融为 DAE 输入额外对称遮挡 15% 非对角像素，同时将输入有效性通道对应位置置零；目标仍是原始 rep1，原目标不会被修改。早停都使用未额外遮挡的早停输入及相同加权重建损失。这是一种正则化，不是拥有真实无噪声标签的监督去噪。

形态基线使用标准化 log1p(O/E) 的平滑能量评分；其聚类表征为 16×16 池化形态的 8 维 PCA。AE/DAE 用重建残差评分，潜变量标准化后 PCA 到 8 维。所有 PCA 与潜变量缩放只拟合合格训练窗口。各尺度分数百分位只用训练窗口校准；融合为形态与残差百分位等权平均。高残差也可能表示噪声，不自动表示新类型。

每方法固定 49 个大窗口、140 个小窗口，先选大窗口再选小窗口，窗口间不重叠，总覆盖 2,150,400 bp（约 46.3% 分箱）。内部评价单独固定 7 个大窗口、20 个小窗口，覆盖 307,200 bp；每个区域另做 20 次同配额随机参考。内部评价已知结构仅统计完整落在评价区的标注。全基因组探索结果包含训练区域，不能冒充泛化成绩。

## 训练结果

'''
    text += '\n'.join(training_table) + '''

## 相同候选预算下的复现结果

每行候选均为 189 个。“稳定新簇配置数”累计该方法四种 K、三个聚类种子的通过次数，不代表不同生物学簇的数量。

'''
    text += '\n'.join(method_table) + '''

## 内部空间评价区的已知召回

下表每方法均为 27 个互不重叠窗口；召回要求覆盖结构中心且覆盖至少一半标注区间。分母较小，尤其 CHID，少数命中即可明显改变百分比。随机参考区间是 20 次随机运行的经验范围，不是经多重检验校正的显著性区间。

'''
    text += '\n'.join(recall_table) + '''

## 聚类和新簇判据

每种方法取 K=4/6/8/12、种子 42/43/44，每次十个初始化，按训练簇内平方误差选择。聚类中心仅拟合训练区窗口，再分配候选；没有按新簇数量选 K 或模型。

重复门槛保持为 r > max(0.5, 同尺度背景相关性的 95% 分位数)，至少 10 个可用背景，每候选最多 20 个。背景无已知注释、不与当前候选相交，背景之间也不相交；允许与其他候选相交，因此不是保证为阴性的样本。对照与候选、所有聚类参数在重载 rep2 前冻结。

原操作性新簇判据保持为簇中没有已知注释重叠成员，至少 5 个空间独立成员通过重复门槛。本轮另要求同方法同 K 下，该簇与其他两个聚类种子中最佳匹配簇的成员 Jaccard 均 ≥0.6。此稳定性只衡量聚类初始化，不等同于跨训练种子或跨生物学重复的类型确认。

普通 AE 与 DAE 的三个训练种子全部报告，不把最好的训练种子作为主结论。`quality_correlations.csv` 给出内部评价区重建残差与平均 O/E、有效比例、零比例的 Spearman 相关，帮助检查质量混杂；它不是生物学验证。

## 结论边界与下一步

本轮最清楚的对照结果是：形态基线 189 个候选中有 82 个通过重复门槛，21 个未注释；普通 AE 与 DAE 的纯残差评分在三个种子下均没有选出通过门槛的窗口。主 DAE 融合配置只有 6 个通过，全部与已知注释重叠。重建残差与内部评价区平均 O/E 的 Spearman 相关约为 −0.58，说明残差明显关联信号特征，不能直接当作结构新颖性的可靠代理。此观察不等于证明所有深度表征无效，但不支持继续把本轮残差评分作为主要候选提取方法。

形态基线在内部评价区召回 OPCID 7/9、CHIN 21/45、CHID 2/5。它提供了较多复现候选，但仍未把未注释模式与已知结构分成满足操作性新簇规则的类群。建议后续聚焦这 21 个互不重叠窗口，比较它们与同尺度已知结构的形态及边界，确定是已知类型的未标注实例、连续形态变化，还是有进一步验证价值的不同模式；这应作为单独的后续新颖性实验。

目标仍然是发现可复现的新结构类群。未注释复现窗口适合作为人工形态审阅、边界核查和独立数据验证的起点；单窗口复现或彼此相似不等于相对已知类型的新颖性。若没有通过规则的新簇，应保留阴性结果，不能通过减少所需成员、调低相关性或选择最有利随机种子来保证发现。

WT 两个重复和基因组区域此前都已被查看。本轮实现了当前训练流程的空间隔离，但仍是探索性方法开发，没有恢复一个全新独立的测试集。文库覆盖限制、标注不完整、暂定坐标、仅扫描近对角线局部窗口均可能影响结果。下一步应优先审阅输出的互不重叠未注释复现窗口；要确认新类型，需要另行制定形态区别的验证方案以及未用于开发的同条件生物学数据。

本轮曾发现初版 QC 汇总了跨分区接触，因此将其产物保留在 `results/task2_v2_superseded_qc/`，修复后全部六次训练与扫描重跑；该目录不用于最终结论。测试覆盖“改变非训练接触不影响训练背景与质量阈值”、对称遮挡不修改目标、按尺度固定预算和空间分区等行为。

## 复现

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/discover_task2_v2.py
.venv/bin/python scripts/check_task2_v2.py
.venv/bin/python scripts/report_task2_v2.py
.venv/bin/python -m unittest discover -s tests -v
```

源码与配置哈希、输入大小与修改时间在 `run_manifest.json` 中记录。已完整训练的模型可在输入不变时复用；输入变化会拒绝混用原目录。
'''
    (ROOT / 'docs/任务二第二轮训练结果.md').write_text(text)
    print('Wrote task2_v2/index.html, summary plots, candidate_new_structures.csv and Chinese report')


if __name__ == '__main__':
    main()
