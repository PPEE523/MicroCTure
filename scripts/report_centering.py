"""Isolate perturbation effects and report annotation context for the fixed grid."""
import json
import numpy as np
from scipy.spatial.distance import cdist
from discover_task2_v2 import ROOT, read_csv
from discover_task2 import write_csv


def main():
    cfg=json.loads((ROOT/'configs/centering_sensitivity.json').read_text())
    out=ROOT/cfg['output_dir']
    report=json.loads((out/'report.json').read_text())
    settings=read_csv(f"{cfg['output_dir']}/variants.csv")
    entities=read_csv(f"{cfg['output_dir']}/entities.csv")
    lookup={(r['anchor'],int(r['width_bins']),int(r['shift_bins']),r['occlusion']):i for i,r in enumerate(settings)}
    with np.load(out/'features.npz') as z:
        xs={name:z[name] for name in ['pca','shape']}
        candidates,calibration,quality=z['candidates'],z['calibration'],z['quality']
    comparisons=[]
    for vi,v in enumerate(settings):
        a,w,s,o=v['anchor'],int(v['width_bins']),int(v['shift_bins']),v['occlusion']
        baselines=[]
        if a=='boundary': baselines.append(('recenter',lookup['original',w,s,o]))
        if s!=0: baselines.append(('translation',lookup[a,w,0,o]))
        if w!=64: baselines.append(('scale',lookup[a,64,s,o]))
        if o!='none': baselines.append(('occlusion',lookup[a,w,s,'none']))
        for factor,base in baselines:
            for metric,distance in [('pca','euclidean'),('shape','cosine')]:
                x=xs[metric]
                values=np.array([[cdist(x[i:i+1,base,rep],x[i:i+1,vi,rep],metric=distance)[0,0] for rep in [0,1]] for i in calibration])
                cutoff=np.quantile(values,cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else np.array([np.nan,np.nan])
                for i in candidates:
                    delta=[float(cdist(x[i:i+1,base,rep],x[i:i+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                    good=bool(quality[i,base].all() and quality[i,vi].all())
                    comparisons.append(dict(entity_id=entities[i]['entity_id'],factor=factor,metric=metric,baseline_variant=settings[base]['variant_id'],target_variant=v['variant_id'],
                                             rep1_change=delta[0],rep2_change=delta[1],known_rep1_q95=float(cutoff[0]),known_rep2_q95=float(cutoff[1]),quality_pass=good,
                                             unusually_sensitive_both=bool(good and np.isfinite(cutoff).all() and np.all(np.array(delta)>cutoff))))
    write_csv(out/'isolated_factor_sensitivity.csv',comparisons)
    known=read_csv('data/processed/structures.csv')
    data=json.loads((ROOT/'configs/data.json').read_text())
    n=(data['chrom_length']+99)//100
    contexts=[]
    for i in candidates:
        e=entities[i]
        for v in settings:
            w,s=int(v['width_bins']),int(v['shift_bins'])
            center=int(e['original_center_bin'] if v['anchor']=='original' else e['boundary_center_bin'])
            start=center+s-w//2
            bins=(start+np.arange(w))%n
            hits=[k['structure_id'] for k in known if np.any((bins>=int(k['start'])//100)&(bins<(int(k['end'])+99)//100))]
            contexts.append(dict(entity_id=e['entity_id'],variant_id=v['variant_id'],start_bp=(start%n)*100,window_length_bp=w*100,known_ids=';'.join(hits),overlaps_known=bool(hits)))
    write_csv(out/'variant_annotation_context.csv',contexts)
    summary=read_csv(f"{cfg['output_dir']}/candidate_summary.csv")
    sensitivity_summary=[]
    for e in [entities[i] for i in candidates]:
        name=e['entity_id']
        row=dict(entity_id=name)
        for factor in ['recenter','translation','scale','occlusion']:
            selected=[r for r in comparisons if r['entity_id']==name and r['factor']==factor]
            # Configuration counts, not independent trials or significance claims.
            row[f'{factor}_sensitive_metric_configurations']=sum(r['unusually_sensitive_both'] for r in selected)
            row[f'{factor}_metric_configurations']=len(selected)
        row['variants_with_known_overlap']=sum(r['entity_id']==name and r['overlaps_known'] for r in contexts)
        sensitivity_summary.append(row)
    write_csv(out/'factor_summary.csv',sensitivity_summary)
    report['isolated_factor_metric_comparisons']=len(comparisons)
    report['candidates_with_new_crop_known_overlap']=sum(r['variants_with_known_overlap']>0 for r in sensitivity_summary)
    (out/'extended_report.json').write_text(json.dumps(report,indent=2))
    table=['| 候选 | 边界可复现 | 原中心双指标通过 / 27 | 重居中双指标通过 / 27 | 全网格质量合格 | 稳定重居中异常 | 含已知注释变体 / 54 |','|---|---|---:|---:|---|---|---:|']
    for r,s in zip(summary,sensitivity_summary):
        table.append(f"| {r['entity_id']} | {r['boundary_reproducible']} | {r['original_outside_variants']} | {r['boundary_outside_variants']} | {r['all_variants_quality_pass']} | {r['robust_recentered_anomaly']} | {s['variants_with_known_overlap']} |")
    text=f'''# 重新居中、多尺度、平移与遮挡敏感性实验
+
+实验已完成。21 个候选与已知参考接受完全相同的 54 个预先固定变体；最终固定参考队列 {report['reference_windows']} 个、互不重叠校准窗口 {report['calibration_windows']} 个。
+
+候选中 {report['reproducible_candidate_boundaries']} 个在本轮统一的 6.4 kb 搜索区内满足局部隔离位置重复要求；{report['candidates_with_any_two_metric_anomaly']} 个至少有一个变体超过两个指标的双重复参考范围；其中 {report['candidates_with_recentered_two_metric_anomaly']} 个发生在重新居中条件。**满足跨尺度与扰动稳定要求的重新居中异常为 {report['robust_recentered_anomalies']} 个。** 这些指标不是新类型确认，也不是新簇数量。
+
+- [全变体图与结果入口](../results/task2_centering/index.html)
+- [候选汇总](../results/task2_centering/candidate_summary.csv)、[全部变体比较](../results/task2_centering/candidate_variants.csv)
+- [单因素配对敏感性](../results/task2_centering/isolated_factor_sensitivity.csv)、[敏感性汇总](../results/task2_centering/factor_summary.csv)
+- [窗口变换后的注释重叠](../results/task2_centering/variant_annotation_context.csv)、[每变体校准阈值](../results/task2_centering/thresholds.csv)
+- [核验报告](../reports/centering_verification.json)
+
+## 在运行前固定的规则
+
+配置保存于 `configs/centering_sensitivity.json`，源码、配置和输入签名在加载矩阵前写入 `run_manifest.json`。候选为上一轮固定的 21 个窗口。没有根据结果改变参数，也不以最大异常变体代表一个候选。
+
+所有对象以原中心为起点，在固定 6.4 kb 窗口中计算两侧各 800 bp 的跨切点/侧内平均 O/E 的 log₂ 比值。rep1 最低点决定重新居中位置；rep2 永远使用这个 rep1 中心裁剪。rep2 自身最低点仅用于评估可复现性：两重复最低比分均小于零，位置相差不超过 500 bp。不能把 rep2 改为自己的最佳中心来人为提高匹配。该最低点是局部隔离候选，不是已确认的结构边界。
+
+每个对象运行原中心和重新居中两组，每组包含 6.4、12.8、25.6 kb 三个尺度，−800、0、+800 bp 三个位移，无遮挡及两个种子（17、29）的 10% 对称随机像素遮挡，共 54 个变体。人工遮挡作为缺失值重新进行有效像素池化，不作为零接触；对应尺寸、种子的掩码在全部对象和两个重复间共享，确保干预一致。这只模拟随机像素缺失，不代表测序深度降低、连续缺口或新的生物学重复。
+
+天然数据质量要求沿用有效比例至少 0.8、零比例不超过 0.8；人工遮挡后不重新套用天然有效比例阈值，避免把实验干预本身当作质量失败。每个候选全部保留，并输出质量标志。已知参考及校准队列要求在全网格中天然质量合格、隔离位置可复现；该条件限制了校准适用范围。
+
+已知参考来自原训练区，校准来自原早停区。每个对象所有裁剪范围的包络必须落在同一空间分区；已知参考与校准都排除候选全部扰动包络，防止共享上下文。校准包络按坐标顺序贪心选择，互不重叠。所有变体使用同一参考与校准队列，避免成员变化造成假敏感性；训练参考之间仍可能重叠。
+
+## 距离、阈值和稳定性
+
+沿用原训练拟合的 8 维形态 PCA 欧氏距离及 16×16 直接形态余弦距离，不重新拟合 PCA。12.8 kb 是新增尺寸，其距离范围通过该尺寸的已知窗口重新校准，不能与其他尺度直接比较原始数值。
+
+每个变体将已知校准窗口与接受同一变换的训练参考比较；每个指标由 rep1 选择最近参考，rep2 比较同一参考。每变体、每指标、每重复分别取已知校准距离的 95% 分位数，至少需要 10 个校准窗口。变体通过要求候选天然质量合格、两个指标都在两个重复中超过相应范围。阈值是经验参照，不是 p 值、FDR，也不保证新颖性；全网格有多次比较，不能把偶然一次通过当作发现。
+
+稳定重新居中异常要求：边界可复现、所有变体质量合格、校准数量足够，且 **三个尺度各至少 80% 的重新居中变体**（每尺度 9 个）同时通过双指标双重复规则。该标准在配置中固定。原中心组完整保留作对照，没有为候选挑选有利尺度或遮挡种子。
+
+## 单因素敏感性如何校准
+
+另外计算对象自身表征的配对变化，每次只改变一个因素：重居中与原中心比较；平移与同组零位移比较；12.8/25.6 kb 与同组 6.4 kb 比较；遮挡与同组无遮挡比较。其余因素保持相同。每个配对在同一已知校准队列上计算变化距离的 95% 分位数，再记录候选是否在两个重复中均超过范围。
+
+这些配对把“对处理本身敏感”与“离所有已知参考较远”分开。异常敏感性不是新类型证据；配置次数也不是独立样本数。`paired_sensitivity.csv` 另保留相对原中心、6.4 kb、零位移、无遮挡基线的总变化，不能把总变化当成单因素效应。
+
+## 逐候选结果
+
+'''.replace('\n+','\n')
    text+='\n'.join(table)
    text+=f'''
+
+放大或移动后，{report['candidates_with_new_crop_known_overlap']} 个候选至少有一个变体纳入已知注释。原窗口未注释不代表所有扩展窗口也未注释；这些重叠已逐项记录，不能把含已知结构的扩展上下文当作新结构。稳定异常标志仅指表征距离稳定异常，仍须结合该重叠表解释。
+
+## 解释限制与复现
+
+这一轮检验的是窗口中心、尺度和扰动能否改变形态判定，并不直接完成聚类。所有 WT 数据此前已经被查看；这是预先固定本轮规则的探索性敏感性分析，不能称为前瞻独立确认。边界、坐标约定和参考注释仍存在既有局限。即使出现稳定异常，也需要独立数据和进一步的结构类别证据。
+
+```bash
+OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/centering_sensitivity.py
+OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/report_centering.py
+OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/check_centering.py
+```
+'''.replace('\n+','\n')
    (ROOT/'docs/重新居中与多尺度敏感性实验.md').write_text(text)
    page=(out/'index.html').read_text()
    addition='<p id="factor-links"><a href="isolated_factor_sensitivity.csv">单因素配对敏感性</a> · <a href="factor_summary.csv">单因素汇总</a> · <a href="variant_annotation_context.csv">变体与已知注释重叠</a></p>'
    if 'id="factor-links"' not in page:
        (out/'index.html').write_text(page+addition)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
