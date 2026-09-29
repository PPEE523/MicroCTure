"""Combine visual observations, coordinate sensitivity and numerical evidence."""
import json
import numpy as np
from discover_task2_v2 import ROOT,read_csv
from discover_task2 import write_csv


def main():
    cfg=json.loads((ROOT/'configs/morphology_audit.json').read_text());out=ROOT/cfg['output_dir']
    observations=json.loads((ROOT/'configs/morphology_observations.json').read_text())
    evidence=read_csv(f"{cfg['output_dir']}/evidence.csv");metrics=read_csv(f"{cfg['output_dir']}/quality_and_shape_metrics.csv")
    known=read_csv('data/processed/structures.csv')
    coordinate_rows=[];review=[]
    for r in evidence:
        start,end=int(r['start_bp']),int(r['end_bp'])
        hits={}
        for scenario,left_delta,right_delta in [('source_as_0_halfopen',0,0),('source_as_1_closed',-1,0),('source_as_0_closed',0,1),('annotation_expand_100bp_sensitivity',-100,100)]:
            ids=[k['structure_id'] for k in known if start<int(k['end'])+right_delta and int(k['start'])+left_delta<end]
            hits[scenario]=ids
            coordinate_rows.append(dict(window_id=r['window_id'],scenario=scenario,known_ids=';'.join(ids),overlaps=bool(ids)))
        ms=[m for m in metrics if m['kind']=='candidate' and m['entity_id']==r['window_id']]
        review.append(dict(**r,visual_description=observations['observations'][r['window_id']],
                           visual_review_scope='montage_and_three_references' if r['window_id'] in observations['focused_reference_review'] else 'montage_plus_computed_reference_comparisons',
                           min_valid_fraction=min(float(m['valid_fraction']) for m in ms),max_zero_fraction=max(float(m['zero_fraction']) for m in ms),
                           coordinate_convention_changes_overlap=bool(hits['source_as_0_halfopen']!=hits['source_as_1_closed'] or hits['source_as_0_halfopen']!=hits['source_as_0_closed']),
                           overlaps_if_annotations_expanded_100bp=bool(hits['annotation_expand_100bp_sensitivity']),
                           morphology_status='replicated_unannotated_pattern_without_confirmed_distinct_type'))
    write_csv(out/'coordinate_sensitivity.csv',coordinate_rows);write_csv(out/'candidate_review_table.csv',review)
    table=['| 候选 | 观察到的形态（描述，不是类型标签） | log-O/E r → 去行列效应后 r | 最近注释间隔 | 三个已知参考 |', '|---|---|---:|---:|---|']
    for r in review:table.append(f"| {r['window_id']} | {r['visual_description']} | {float(r['log_oe_replication']):.3f} → {float(r['residual_replication']):.3f} | {r['annotation_gap_bp']} bp | {r['reference_ids'].replace(';',' / ')} |")
    med_before=float(np.median([float(r['log_oe_replication']) for r in review]));med_after=float(np.median([float(r['residual_replication']) for r in review]))
    flips=sum(r['coordinate_convention_changes_overlap'] for r in review);expanded=sum(r['overlaps_if_annotations_expanded_100bp'] for r in review)
    text=f'''# 候选是否具有不同结构形态：证据核查

**目前没有足够证据确认新的结构形态类别。** 本轮为 21 个候选分别比较了三个同尺度、彼此不重叠且不与候选共享上下文的已知参考，共 63 次参考比较。可复现信号确实存在，但条带、块状富集和离对角线亮斑等主要图像元素也出现在已知参考中；此前三个异常变体候选没有形成稳定的不同形态证据。

这不等于证明候选都属于某个已知类型，更不等于它们都是噪声。最近参考只是相似的局部上下文，标签不能直接转移到候选；已知中心窗口也可能包含相邻结构。

- [全部多参考对照图](../results/task2_morphology_audit/index.html)、[21 个候选总览](../results/task2_morphology_audit/candidate_montage.png)
- [完整逐项核查表](../results/task2_morphology_audit/candidate_review_table.csv)
- [质量与接触集中度](../results/task2_morphology_audit/quality_and_shape_metrics.csv)、[三个参考的双重复距离](../results/task2_morphology_audit/top3_reference_matches.csv)
- [坐标敏感性](../results/task2_morphology_audit/coordinate_sensitivity.csv)、[核验报告](../reports/morphology_audit_verification.json)

## 三个异常变体候选的证据与反证

| 候选 | 支持进一步关注的观察 | 限制新类型解释的证据 |
|---|---|---|
| w64_c33056 | 条带模式跨重复可见，原 O/E 相关 0.890 | 与 CHIN_8、CHIN_19、CHIN_128 的局部上下文具有相似边缘条带；边界差约 2.5 kb；仅一个重新居中变体通过双指标，未跨尺度稳定 |
| w64_c28240 | 局部隔离位置可复现 | 与 OPCID_67、CHIN_8、CHIN_38 共享边缘条带/局部强弱模式；仅两个 25.6 kb 变体异常，其余尺度不支持；去行列效应后相关约 0.384 |
| w64_c10288 | 原中心组有 8/27 个异常变体；局部隔离位置接近 | 与 CHIN_59、OPCID_66、CHIN_8 有相似的边缘富集和中央弱接触；去行列效应后相关约 0.299；原中心就是隔离中心，组间差异来自已知参考和校准同步改变，不能归因于候选被移动 |

已逐图审阅上述三组双重复、多参考对照。此外审阅了两个亮斑窗口：w64_c43040 的离对角线亮斑与 CHIN_164 的局部图像非常相似，不能将亮斑本身视作新的形态元素；w64_c19600 的局部亮斑较集中，但 CHID_12、CHIN_166 和 OPCID_66 上下文同样具有富集成分，且此前敏感性实验未显示稳定异常。w64_c19600 仍适合保留为未注释接触模式实例，尚不能定为新类型。对称矩阵中的两个镜像亮斑不是两个独立结构。

## 质量与行列效应

21 个候选在本轮原尺度的有效比例及零比例均满足已有质量规则；具体值逐重复保留。质量合格并不排除条带偏差或覆盖影响。

对 log1p(O/E) 矩阵做 10 次带缺失掩码的对称行列均值消除，再计算共同有效上三角的重复相关。相关系数中位数从 **{med_before:.3f} 降到 {med_after:.3f}**。这说明共同的行列强弱分量贡献了部分复现性，但不能据此证明技术伪影：真实生物学条带也会被该操作删除。

w64_c38928 去除后仍有约 0.789 的相关，w64_c43040 和 w64_c19600 均约 0.536，说明部分局部模式并非完全由行列效应构成。相反，w256_c34976 从约 0.514 降到 0.101，提示其复现性较依赖行列分量。这里没有为残差相关设置新的通过阈值，也没有借此重选候选。

对前三个异常变体候选生成了独立诊断图，展示原 log-O/E、被移除的行列分量及残差。行列去除不是正式矩阵平衡方法，不能替代原始测序、比对和实验重复层面的偏差检查。

## 注释与坐标核查

重新检查了原 Excel 表 4/5/6 的列名：提供 Start、End（表 5 另有 Center），未声明 0/1-based 及端点包含约定，因此坐标仍为 provisional，未擅自修改。

将注释分别按 0-based 左闭右开、1-based 闭区间转换、0-based 闭区间解释，{flips} 个候选的“与已知注释重叠”状态发生改变。额外将注释两端各扩展 100 bp 作为分箱边界敏感性检查，有 {expanded} 个候选产生重叠；这不是声称真实注释误差为 100 bp。

w256_c44624 距 CHIN_236 仅 40 bp，是必须优先核查注释边缘的窗口；w64_c14000 距 CHIN_80 570 bp，并存在可见缺失条带，也需要谨慎解释。没有重叠或相距较远都不能证明新类型，原注释可能不完整。此前放大/平移实验中 12 个候选至少有一个扩展变体纳入已知注释，此结论仍保留。

## 21 个窗口的描述性形态记录

总览图逐项检查后记录以下可见图像元素；五个重点窗口另外审阅了完整三参考对照。其余参考距离和质量由脚本逐项计算。本表不是盲法专家标注，也不是聚类标签；没有把下列描述当成新的结构类别。

'''
    text+='\n'.join(table)
    text+='''

## 当前判断

最稳妥的交付是“未注释但可复现的局部接触模式清单”，而非“新结构类型清单”。当前反证包括已知参考中存在相似图像元素、少数变体异常缺乏稳定性、部分重复相关依赖行列分量，以及注释边缘与上下文的不确定性。

若继续追求新类型，下一项有价值的工作是独立同条件数据中的位置复现、原作者注释/坐标核实，以及对候选提出明确的机制或功能假设。现有数据上的距离、裁剪和聚类参数已经反复检查，不宜再以得到一个新簇为依据持续调整。所有 WT 数据此前均被查看，本轮没有独立生物学确认。

## 复现

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/audit_morphology.py
.venv/bin/python scripts/report_morphology_audit.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/check_morphology_audit.py
```

新增测试覆盖行列消除的加性模式行为、原输入不变及缺失值隔离；核验重新计算三参考选择、双重复距离与残差相关，检查参考空间独立和 25 张图片完整性。
'''
    (ROOT/'docs/候选不同形态证据核查.md').write_text(text)
    link='<p id="review-table"><a href="candidate_review_table.csv">含描述性形态记录的逐项证据表</a> · <a href="coordinate_sensitivity.csv">坐标约定敏感性</a></p>'
    page=(out/'index.html').read_text()
    if 'id="review-table"' not in page:(out/'index.html').write_text(page+link)
    print(json.dumps(dict(coordinate_overlap_changes=flips,overlaps_with_100bp_annotation_expansion=expanded,median_log_correlation=med_before,median_residual_correlation=med_after),indent=2))


if __name__=='__main__':main()
