"""Summarize already-frozen test evaluation, without selecting or tuning models."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from train_task1 import OUT,ROOT,LABELS,confusion_figure
from task1_baselines import f1_score
import matplotlib.pyplot as plt


def main():
    records=json.loads((OUT/'evaluation.json').read_text())
    selection=json.loads((OUT/'selection_before_test.json').read_text())
    table=list(csv.DictReader((OUT/'comparison.csv').open()))
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    x=np.arange(len(table))
    ax.bar(x-.18,[float(r['validation_macro_f1']) for r in table],.36,label='Validation (selection)')
    ax.bar(x+.18,[float(r['test_macro_f1_mean']) for r in table],.36,yerr=[float(r['test_macro_f1_std']) for r in table],capsize=3,label='Test mean +/- seed SD')
    ax.set(xticks=x,xticklabels=[r['model'] for r in table],ylabel='Structure-level macro-F1',ylim=(0,1),title='Frozen comparison: A2 reuses M1');ax.legend()
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,name in zip(axes.flat,selection['validation_means']):
        for seed in (42,43,44):
            history=json.loads((OUT/f'{name}_{seed}_history.json').read_text())
            ax.plot([r['epoch'] for r in history],[r['validation_macro_f1'] for r in history],label=str(seed))
        ax.set(title=name,xlabel='Epoch',ylabel='Validation macro-F1',ylim=(0,1));ax.legend()
    fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
    for r in records:
        np.savetxt(OUT/f'{r["model"]}_{r["seed"]}_confusion.csv',r['test']['confusion_matrix'],fmt='%d',delimiter=',',header=','.join(LABELS),comments='')
    name=selection['selected_cnn'];seed=selection['explanation_seed']
    selected=next(r for r in records if r['model']==name and r['seed']==seed)
    predictions=list(csv.DictReader((OUT/f'{name}_{seed}_predictions.csv').open()))
    manifest={r['structure_id']:r for r in csv.DictReader((ROOT/'data/processed/classification_split.csv').open())}
    groups=np.array([manifest[r['structure_id']]['group_id'] for r in predictions])
    unique=np.unique(groups);y=np.array([int(r['true']) for r in predictions]);p=np.array([int(r['predicted']) for r in predictions])
    rng=np.random.default_rng(42);boot=[];missing=0
    for _ in range(2000):
        indices=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        missing+=len(np.unique(y[indices]))<3
        boot.append(float(f1_score(y[indices],p[indices])))
    uncertainty=dict(model=name,seed=seed,test_groups=len(unique),bootstrap_samples=2000,
                     macro_f1_percentile95=np.quantile(boot,[.025,.975]).tolist(),resamples_missing_a_class=int(missing),
                     note='Spatial-group bootstrap; fixed three-class macro-F1 even in resamples missing a class; exploratory interval, not a biological replicate CI.')
    (OUT/'group_bootstrap.json').write_text(json.dumps(uncertainty,indent=2))
    report=dict(selection=selection,comparison=table,selected_seed42=selected,uncertainty=uncertainty,
                explanations=json.loads((OUT/'explanations.json').read_text()),
                source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'scripts/train_task1.py',ROOT/'scripts/task1_baselines.py',ROOT/'scripts/report_task1.py',ROOT/'requirements-training.txt']})
    (ROOT/'reports/task1_results.json').write_text(json.dumps(report,indent=2))
    lines=['# 任务一：训练、消融与评价结果','',
           '所有模型选择和早停使用验证集；全部训练结束后统一评价固定测试集。以下数值为结构级指标，两个重复先平均概率。', '',
           '| 模型 | 验证 macro-F1 | 测试 macro-F1（均值 ± 种子标准差） | 测试准确率 |',
           '|---|---:|---:|---:|']
    for r in table:lines.append(f"| {r['model']} | {float(r['validation_macro_f1']):.3f} | {float(r['test_macro_f1_mean']):.3f} ± {float(r['test_macro_f1_std']):.3f} | {float(r['test_accuracy_mean']):.3f} |")
    lines+=['',f'验证集三种子均值选出的 CNN 配置：**{name}**。用于解释的种子预先固定为 42，没有选测试表现最好的种子。',
            '',f"该运行的测试准确率为 {selected['test']['accuracy']:.3f}，macro-F1 为 {selected['test']['macro_f1']:.3f}。",'',
            '| 类别 | Precision | Recall | F1 | 测试结构数 |','|---|---:|---:|---:|---:|']
    for label in LABELS:
        r=selected['test']['per_class'][label]
        lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {r['support']} |")
    lines+=['','## 图表和文件','',
            '- [模型比较图](../results/task1/comparison.png)',
            '- [验证学习曲线](../results/task1/learning_curves.png)',
            f'- [所选 CNN 的混淆矩阵](../results/task1/{name}_42_confusion.png)',
            '- [全部数值结果](../results/task1/evaluation.json)',
            '- [对照表 CSV](../results/task1/comparison.csv)',
            '- [显著性图与遮挡数值](../results/task1/explanations.json)',
            '', '## 实验设置与解释边界','',
            'B0=多数类；B1=标注长度/密度逻辑回归（特权信息诊断）；B2=PCA16+逻辑回归；M1=大窗口 CNN；M2=双尺度 CNN；A1=小窗口；A3=深度归一化代替 O/E；A4=无类别权重；A5=无显式掩码通道。A2 与 M1 完全相同，不重复计实验。',
            '', '每种 CNN 运行种子 42/43/44，固定 1e-3 AdamW、batch 32、最多 100 epoch、patience 15、同种子相同样本顺序。所有运行不使用数据增强。CPU、4 线程、确定性算法；共享 CNN 编码器通道为 16/32/64。参数量、最佳轮数、实际耗时见 evaluation.json。',
            '', '基线因 scikit-learn 下载失败，使用本项目 NumPy/SciPy 实现。PCA 使用训练集精确薄 SVD；逻辑回归是带训练类别权重和 L2 正则的三分类 softmax，L-BFGS 求解并检查收敛；截距不惩罚。相关计算已有合成数据与手算指标测试。',
            '', '标准差反映三次训练初始化的波动，不是独立生物学样本的置信区间。测试 CHID 只有 4 个，少错或多错一个就改变 25 个百分点的召回率。空间组 bootstrap 结果见 group_bootstrap.json；不含某类的重采样也用固定三类宏平均，因此区间只作探索性参考。',
            '', '显著性图使用预测类别 logit 对输入信号的绝对梯度，按每类 ID 排序选取首个预测正确/错误例子（如存在）。不是按视觉效果挑选。仅展示 rep1，图中同时注明结构级预测。遮挡以标准化信号 0（训练平均）替换同等数量的结构/背景像素，掩码保持不变；只是局部诊断，不代表因果机制或全面解释验证。',
            '', '原始坐标仍为 provisional。此前看过全基因组探索性统计，因此本测试不是完全未接触过的外部数据。分类模型只适用于以已知结构中心切出的三类窗口，不能直接作为未知结构拒识器。',
            '', '## 复现','', '```bash',
            '.venv/bin/python -m pip install -r requirements-training.txt --extra-index-url https://download.pytorch.org/whl/cpu',
            'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/train_task1.py --threads 4',
            '.venv/bin/python scripts/report_task1.py',
            '.venv/bin/python -m unittest discover -s tests -v','```','',
            '输入和主训练代码的哈希记录于 run_manifest.json，基础算法与报告代码哈希另存 reports/task1_results.json。已完成检查点可复用；输入或主代码改变时拒绝混用原实验目录。模型 .pt 包含架构配置和权重，基线 .pkl 只应加载本项目自行生成的文件。']
    lines += ['','## 完成度与主要不足','', '任务一要求的训练/评价代码、固定权重、留出集指标、混淆矩阵和各类显著性图已齐全。但不满足各类都稳定识别、主要关注结构本身的质量目标。预先固定的 M2/42 中，CHID 仅识别正确 1/4，另有 11 个 CHIN 被预测为 CHID，误报问题突出；三种子结果也有较大波动。后续应优先复核 CHIN/CHID 的嵌套标签与坐标、扩充独立 CHID 样本，并在新预注册评价下研究改进；不能针对已查看的测试集反复调参。']
    lines += ['','## 消融的量化比较','',
              '以下为相对完整双尺度 O/E 模型 M2 的测试 macro-F1 差值，正值代表该变体更高。三次种子远不足以证明统计显著性，且不能依据此表重新选择模型。','',
              '| 变体 | 改动 | 相对 M2 的均值差 |','|---|---|---:|']
    lookup={r['model']:float(r['test_macro_f1_mean']) for r in table}
    for key,change in [('A1','只保留小尺度'),('M1','只保留大尺度（A2）'),('A3','使用深度归一化'),('A4','去掉类别权重'),('A5','去掉显式掩码通道')]:
        lines.append(f'| {key} | {change} | {lookup[key]-lookup["M2"]:+.3f} |')
    explanations=json.loads((OUT/'explanations.json').read_text())
    pairs=[]
    for example in explanations:
        for scale in set(r['scale_bp'] for r in example['occlusion']):
            by_region={r['region']:r for r in example['occlusion'] if r['scale_bp']==scale}
            if by_region['structure']['pixels']>0:
                pairs.append(by_region['structure']['probability_drop']-by_region['background']['probability_drop'])
    if pairs:
        lines += ['',f'等量遮挡的 {len(pairs)} 个示例/尺度比较中，结构遮挡比背景遮挡更降低预测类别概率的有 {sum(v>0 for v in pairs)} 个。这只是所展示样本的诊断结果；不应宣称模型普遍只关注结构。']
    for file in sorted(OUT.glob('explanation_*.png')):lines.append(f'\n- [{file.stem}](../results/task1/{file.name})')
    (ROOT/'docs/任务一实验结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(selected=name,seed42=selected['test'],uncertainty=uncertainty),indent=2))

if __name__=='__main__':main()
