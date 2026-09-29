"""Render primary insufficiency separately from the exploratory calibration supplement."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/microcture-mpl')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discover_task2_v2 import ROOT,read_csv
from report_centering import main as primary_report


def main():
    primary_report()
    cfg=json.loads((ROOT/'configs/centering_sensitivity.json').read_text())
    extra=json.loads((ROOT/'configs/centering_supplement.json').read_text())
    base=ROOT/cfg['output_dir'];out=ROOT/extra['output_dir']
    settings=read_csv(f"{cfg['output_dir']}/variants.csv")
    rows=read_csv(f"{extra['output_dir']}/candidate_variants.csv")
    summary=read_csv(f"{extra['output_dir']}/candidate_summary.csv")
    report=json.loads((out/'report.json').read_text())
    names=[r['entity_id'] for r in summary]
    lookup={(r['entity_id'],r['variant_id'],r['metric']):r['outside_both']=='True' for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(16,9),layout='constrained',sharey=True)
    for ax,anchor in zip(axes,cfg['anchors']):
        vs=[v for v in settings if v['anchor']==anchor]
        matrix=np.array([[int(lookup[name,v['variant_id'],'pca'])+int(lookup[name,v['variant_id'],'shape']) for v in vs] for name in names])
        im=ax.imshow(matrix,vmin=0,vmax=2,cmap='viridis',aspect='auto',interpolation='nearest')
        ax.set(title=anchor,yticks=range(len(names)),yticklabels=names,xticks=range(len(vs)),xticklabels=[f"{int(v['width_bins'])*.1:g}k/{int(v['shift_bins'])*100:+d}/{v['occlusion']}" for v in vs])
        ax.tick_params(axis='x',labelrotation=90,labelsize=7)
    fig.colorbar(im,ax=axes,ticks=[0,1,2],label='Metrics outside range in both repeats',shrink=.5)
    fig.suptitle('Exploratory calibration supplement: 17 known windows; full matched perturbation grid')
    fig.savefig(out/'all_variants.png',dpi=130);plt.close(fig)
    # Replace the uncalibrated primary zero map with an explicit missing-evidence figure.
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');ax.axis('off')
    ax.text(.5,.6,'PRIMARY EXPERIMENT: INSUFFICIENT CALIBRATION',ha='center',fontsize=17)
    ax.text(.5,.4,'8 independent known calibration windows < prespecified minimum 10\nNo anomaly decisions can be made from the primary experiment.\nSee the separately labeled 17-window exploratory supplement.',ha='center',va='center',fontsize=13)
    fig.savefig(base/'all_variants.png',dpi=130);plt.close(fig)
    notice='<p style="background:#fff0cc;padding:16px"><strong>主实验校准不足：8 个窗口少于预设 10 个。CSV 中未通过或零计数表示未作出异常判断，不能当作阴性证据。</strong> <a href="../task2_centering_supplement/index.html">查看单独标记的 17 窗口探索性补充校准</a></p>'
    page=(base/'index.html').read_text()
    if '主实验校准不足：' not in page:(base/'index.html').write_text(notice+page)
    table='<table><tr><th>候选</th><th>原中心通过/27</th><th>重新居中通过/27</th><th>稳定异常</th></tr>'
    for r in summary:table+=f"<tr><td>{r['entity_id']}</td><td>{r['original_outside_variants']}</td><td>{r['boundary_outside_variants']}</td><td>{r['robust_recentered_anomaly']}</td></tr>"
    table+='</table>'
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>探索性补充校准</title><style>body{{max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif;padding:20px}}img{{width:100%}}td,th{{padding:6px;border:1px solid #ccc}}table{{border-collapse:collapse}}</style>
<h1>同规则扰动实验：探索性补充校准</h1><p>主实验只有 8 个独立校准窗口，无法按预设门槛判断。补充组纳入此前已查看的评价区已知窗口，得到 17 个校准窗口；108 个训练参考、21 个候选、54 个变体和全部阈值规则保持不变。该补充组不再具有独立评价集含义。</p>
<p>3 个候选至少一个变体通过双指标双重复规则，其中 2 个在重新居中条件下出现。稳定重居中异常为 0；不存在据此确认的新类型或新簇。</p>
<p><a href="candidate_summary.csv">候选汇总</a> · <a href="candidate_variants.csv">全部变体比较</a> · <a href="isolated_factor_sensitivity.csv">单因素配对敏感性与已知阈值</a> · <a href="thresholds.csv">分变体阈值</a> · <a href="../task2_centering/variant_annotation_context.csv">变换后的已知注释重叠</a> · <a href="../task2_centering/index.html">主实验（校准不足）</a></p>
<p>w64_c10288 的两个中心完全相同，因此它的组间差异来自已知参考一起重新居中及重新校准，不能归因为移动了候选自身。任何单个变体异常都不是稳定新颖性证据。</p><img src="all_variants.png" alt="补充校准全部变体">{table}''')
    path=ROOT/'docs/重新居中与多尺度敏感性实验.md'
    doc=path.read_text()
    intro_end=doc.index('## 在运行前固定的规则')
    intro='''# 重新居中、多尺度、平移与遮挡敏感性实验

21 个候选及已知参考的同规则实验已完成，每个对象包含 54 个变体。

**主实验校准不足，不能作出异常判断。** 固定最大扰动包络且校准窗口互不重叠后，原早停区只有 8 个合格校准窗口，少于预设 10 个。因此主实验 CSV 的零通过计数不是阴性证据。

另行冻结补充规则后，将此前已查看过的评价区已知窗口加入校准，得到 **17 个校准窗口**，训练参考仍为 **108 个**。候选特征、变体、尺度、随机种子、最低校准数量和判据都未改变。该补充组明确属于探索性分析，不再将这些区域作为独立评价集。

补充组中，3 个候选在少数变体下通过两个指标的双重复距离门槛，其中 2 个在重新居中组出现；**跨尺度、平移与遮挡稳定的重新居中异常为 0 个**。这不是不存在新结构的证明，也没有确认新类型或新簇。

- [补充校准完整图表](../results/task2_centering_supplement/index.html)、[候选汇总](../results/task2_centering_supplement/candidate_summary.csv)
- [补充组单因素配对敏感性](../results/task2_centering_supplement/isolated_factor_sensitivity.csv)、[全部变体比较](../results/task2_centering_supplement/candidate_variants.csv)
- [主实验及校准不足说明](../results/task2_centering/index.html)、[每个变体的注释重叠](../results/task2_centering/variant_annotation_context.csv)
- [两组独立代码核验报告](../reports/centering_verification.json)

'''
    doc=intro+doc[intro_end:]
    doc=doc.replace('## 逐候选结果','## 主实验逐候选记录（全部校准不足，未通过不代表阴性）')
    addition='''
## 补充组具体结果与解释

| 候选 | 原中心双指标通过 / 27 | 重新居中双指标通过 / 27 | 解释 |
|---|---:|---:|---|
| w64_c33056 | 0 | 1 | 仅 25.6 kb 一个变体；隔离位置重复差异超限 |
| w64_c28240 | 0 | 2 | 仅 25.6 kb 两个变体，不跨尺度稳定 |
| w64_c10288 | 8 | 0 | 候选两组中心相同，差异来自已知参考与校准同步变化 |

其余 18 个候选没有双指标双重复通过的变体。w64_c10288 的两组候选裁剪相同，不能宣称“重新居中候选后异常消失”；同规则变换会改变已知参考自身的裁剪及距离分布，这正是必须同时处理已知参考的原因。

单因素配对表中，至少出现一次双重复异常敏感性的候选数：重新居中 8 个、平移 14 个、尺寸变化 15 个、遮挡 12 个。它们是大量配置下的探索性计数，不是多重检验校正后的显著性，更不代表新结构数。放大或平移后 12 个候选至少有一个变体纳入已知注释，需要结合上下文表审阅。

补充组只是解决可用参考数量不足以进行描述性校准的问题，并未获得新的生物学重复。后续如继续寻找不同局部模式，应先核查上述少数变体的质量、已知注释上下文与参考选择，不能只保留异常变体后重新命名为新簇。

补充组复现命令：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/supplement_centering.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/report_centering_combined.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/check_centering.py
```
'''
    path.write_text(doc+addition)
    print('Rendered primary insufficiency and separate supplemental calibration report')


if __name__=='__main__':main()
