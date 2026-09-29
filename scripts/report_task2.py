"""Describe discovery results, matched-budget reference and reproducibility checks."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/microcture-mpl")
import csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from discover_task2 import ROOT,disjoint_select,extract,correlation


def main():
    cfg=json.loads((ROOT/'configs/discovery.json').read_text());out=ROOT/cfg['output_dir']
    report=json.loads((out/'report.json').read_text());candidates=list(csv.DictReader((out/'candidates.csv').open()))
    scan=list(csv.DictReader((out/'scan_rep1.csv').open()));known=list(csv.DictReader((ROOT/'data/processed/structures.csv').open()))
    rows=[]
    for r in scan:rows.append(dict(r,start_bin=int(r['start_bin']),width_bins=int(r['width_bins'])))
    n=46417;occupied=np.zeros(n,bool)
    frozen=json.loads((out/'frozen_before_rep2.json').read_text());assert len(candidates)==len(frozen['selected_indices'])
    for r in candidates:
        i=int(r['window_index']);row=rows[i];ids=(row['start_bin']+np.arange(row['width_bins']))%n
        assert not occupied[ids].any();occupied[ids]=True
        assert i in frozen['selected_indices']
        values=r['replicate_correlation'];threshold=r['control_threshold']
        passed=bool(values and threshold and float(values)>float(threshold) and int(r['matched_controls'])>=cfg['min_controls'])
        assert passed==(r['replicate_supported']=='True')
        controls=frozen['controls'][str(i)];assert disjoint_select(controls,rows,n)==controls
        for j in controls:
            assert not rows[j]['known_ids'];assert rows[j]['width_bins']==row['width_bins']
    for indices in frozen['controls'].values():
        for j in indices:assert not occupied[(rows[j]['start_bin']+np.arange(rows[j]['width_bins']))%n].any()
    audit=[]
    for r in candidates:
        if int(r['matched_controls'])<cfg['min_controls']:
            status='insufficient_controls'
        elif not r['replicate_correlation']:
            status='undefined_correlation'
        elif r['replicate_supported']=='True':
            status='supported_known' if r['known_ids'] else 'supported_unannotated'
        else:
            status='below_predeclared_threshold'
        audit.append(dict(candidate_id=r['candidate_id'],validation_status=status,matched_controls=int(r['matched_controls'])))
    with (out/'validation_status.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    # Supplementary descriptive null: same window counts/scales, nonoverlap, rep1 quality filters.
    counts={w:sum(rows[int(r['window_index'])]['width_bins']==w for r in candidates) for w in cfg['window_bins']}
    rng=np.random.default_rng(2026);null=[]
    for trial in range(100):
        taken=[];used=np.zeros(n,bool)
        for width in sorted(counts,reverse=True):
            pool=[i for i,r in enumerate(rows) if r['eligible']=='True' and r['width_bins']==width]
            count=0
            for i in rng.permutation(pool):
                ids=(rows[i]['start_bin']+np.arange(width))%n
                if used[ids].any():continue
                used[ids]=True;taken.append(i);count+=1
                if count==counts[width]:break
            assert count==counts[width]
        recalled={sid for i in taken for sid in rows[i]['recalled_ids'].split(';') if sid}
        for kind in ('OPCID','CHIN','CHID'):
            labels={r['structure_id'] for r in known if r['type']==kind}
            null.append(dict(trial=trial,type=kind,recall=len(recalled&labels)/len(labels)))
        assert used.sum()==occupied.sum()
    null_summary={kind:dict(mean=float(np.mean([r['recall'] for r in null if r['type']==kind])),
                            percentile95=np.quantile([r['recall'] for r in null if r['type']==kind],[.025,.975]).tolist()) for kind in ('OPCID','CHIN','CHID')}
    (out/'matched_budget_random_reference.json').write_text(json.dumps(dict(seed=2026,trials=100,summary=null_summary,note='Post-discovery descriptive reference; not used to change candidates or thresholds.'),indent=2))
    # Contact-band checks against independently stored upper-triangle Cooler querying.
    from plot_examples import aggregate_local
    import cooler
    data=json.loads((ROOT/'configs/data.json').read_text());bands=[]
    for sample in data['samples']:
        with np.load(ROOT/'data/cache/discovery'/f"{sample['sample_id']}.npz") as z:b={k:z[k].copy() for k in z.files if k!='signature'}
        clr=cooler.Cooler(str(ROOT/sample['path']))
        for start,width in [(-32,64),(0,64),(12345,256)]:
            raw,oe,mask=extract(b,start,width);ids=(start+np.arange(width))%n
            independent=aggregate_local(clr,ids,10)
            offdiag=np.abs(np.arange(width)[:,None]-np.arange(width)[None,:])>=2
            np.testing.assert_allclose(raw[offdiag],independent[offdiag])
        bands.append(b)
    for r in candidates:
        row=rows[int(r['window_index'])];a=extract(bands[0],row['start_bin'],row['width_bins']);b=extract(bands[1],row['start_bin'],row['width_bins'])
        assert np.isclose(correlation(a[1],b[1],a[2]&b[2]),float(r['replicate_correlation']))
    for path in (out/'heatmaps').glob('*.png'):
        with Image.open(path) as im:im.verify()
    assert len(list((out/'heatmaps').glob('*.png')))==len(candidates)
    manifest=json.loads((out/'run_manifest.json').read_text())
    for name,value in manifest['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
    checks=dict(candidate_count=len(candidates),no_overlapping_candidate_bins=True,frozen_candidates_match=True,
                control_rules_verified=True,independent_cooler_band_checks=6,replicate_correlations_recomputed=len(candidates),
                pngs_verified=len(candidates),source_hashes_match=True)
    (ROOT/'reports/task2_verification.json').write_text(json.dumps(checks,indent=2))
    # Readable HTML index links every candidate, without hiding failed validation.
    html=['<!doctype html><meta charset="utf-8"><title>MicroCTure task 2</title><style>body{font:15px sans-serif;margin:32px}td,th{padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Task 2 candidates</h1><p>Coordinates remain provisional. Unannotated does not mean a new structure type.</p><table><tr><th>ID</th><th>Cluster</th><th>Known annotations</th><th>Replicate r</th><th>Controls</th><th>Validation passed</th></tr>']
    for r in candidates:html.append(f'<tr><td><a href="heatmaps/{r["candidate_id"]}.png">{r["candidate_id"]}</a></td><td>{r["cluster"]}</td><td>{r["known_ids"] or "unannotated"}</td><td>{float(r["replicate_correlation"]):.3f}</td><td>{r["matched_controls"]}</td><td>{r["replicate_supported"]}</td></tr>')
    (out/'index.html').write_text('\n'.join(html)+ '</table>')
    passing=[r for r in candidates if r['replicate_supported']=='True']
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,r in zip(axes.flat,passing):
        with Image.open(out/'heatmaps'/f"{r['candidate_id']}.png") as im:ax.imshow(im)
        ax.axis('off')
    fig.suptitle('Six regions supported by the predeclared replicate/control criterion');fig.savefig(out/'replicated_gallery.png',dpi=150);plt.close(fig)
    curves=list(csv.DictReader((out/'recall_curves.csv').open()))
    lines=['# 任务二：候选发现与跨重复验证','',
           '**本轮完成了候选发现、表征、聚类和验证流程，并复现了部分已知结构；没有得到满足预设判据的新结构簇。** 这不代表基因组中不存在新结构。','',
           '## 实际结果','',
           f'- 扫描 {report["scan_windows"]} 个窗口，{report["eligible_windows"]} 个通过质量检查。',
           f'- 合并后保留 {report["independent_candidates"]} 个互不重叠的窗口，覆盖约 {100*report["candidate_genome_coverage"]:.1f}% 的基因组分箱。',
           f'- 其中 {report["known_overlap_candidates"]} 个与已知注释重叠，{report["unannotated_candidates"]} 个没有注释重叠。',
           '- 聚为 6 簇，所有簇均含已知注释成员，未形成无已知成员的新簇。',
           '- 6 个窗口超过预设的重复相关性及匹配背景门槛，均与已知结构重叠。',
           '- 72 个未注释窗口中有 16 个相关系数至少 0.5，但没有同时满足完整验证条件的窗口。',
           '- 49 个 25.6 kb 候选仅能找到 6 个互不重叠的背景窗口，低于预设至少 10 个，因此是验证证据不足，不能当成结构不存在。',
           '', '## 推荐先看','',
           '- [聚类与召回总览](../results/task2/overview.png)',
           '- [六处复现结构总览](../results/task2/replicated_gallery.png)',
           '- [C0028 两个重复热图](../results/task2/heatmaps/C0028.png)：对应 CHIN_165–168 / CHID_17，相关系数约 0.957。',
           '- [全部候选的浏览目录](../results/task2/index.html)',
           '- [全部候选 CSV](../results/task2/candidates.csv)',
           '- [新结构候选 CSV](../results/task2/candidate_new_structures.csv)：本轮只有表头，表示没有通过规则的结果，不是漏写文件。',
           '', '## 技术路线与冻结顺序','',
           '使用 100 bp 聚合计数，6.4 kb / 25.6 kb 窗口，每 1.6 kb 扫描一次。全基因组只构造宽 256 bin 的环状近对角线带，不构造完整稠密矩阵。保留跨原点接触，末端不足 100 bp 的 bin 掩蔽；跨原点物理距离仍存在最多 48 bp 的粗分箱近似。',
           '', '所有候选、簇、匹配对照索引和验证门槛均在读取 rep2 接触数据前写入 frozen_before_rep2.json。没有使用任务一测试成绩或三分类器硬过滤候选，也没有为了产生新结构结果在看到 rep2 后放宽阈值。',
           '', '质量规则固定为有效像素比例至少 0.8、有效上三角零值比例不超过 0.8。低覆盖定义为全基因组非零端点覆盖度中位数的 10%；主对角线及紧邻对角线不参与评分和相关性。',
           '', '每个重复独立计算 O/E，距离期望包含零接触位置。窗口 log1p(O/E) 做有效像素内标准化并截断，减少接触量差异；缺失比例作为第二通道。两个尺度都转换为 64×64，不使用热图颜色或标注框。',
           '', '卷积自编码器采用 8/16/32 通道、16 维瓶颈、对称解码器，以有效像素加权重建误差训练。rep1 中按 200 kb 区块划分，完整位于每第五块的窗口用于早停，其余抽样最多 2000 个训练；最多 40 轮、patience 6。期望背景来自整个 rep1，且早停访问了验证块，因此这不是严格独立的泛化评估。候选扫描同时包含参与过训练的窗口和未参与窗口，重建误差可能受训练覆盖影响。',
           '', '形态基线使用标准化窗口的平滑后能量；融合分数是同尺度内形态能量和自编码器误差百分位的等权平均。另报告仅接触量、仅形态、仅重建误差的召回曲线。按融合分数取合格窗口前 20%，再按分数贪心保留不相交的完整窗口，重复滑窗不会计作独立成员。',
           '', '16 维表征标准化后以 PCA 降至 8 维，固定 K=6 的 k-means 在 rep1 上做十次初始化并选择较低的簇内平方误差；二维 PCA 仅用于展示。簇中心和每个扫描窗口的表征均保存。',
           '', '已知注释重叠/未注释窗口的最近质心诊断在空间早停块上的平衡准确率约 0.645（随机基准 0.5）。这是对表征可用性的初步诊断：标签不完整、同一块已经用于早停，不能视为独立分类成绩。',
           '', '## 已知结构召回与覆盖率','',
           '召回规则：窗口包含已知结构中心，并覆盖其标注区间至少一半。任何少量相交只计“有注释重叠”，不自动计为召回。','',
           '| 方法，前 20% 窗口 | OPCID | CHIN | CHID | 基因组覆盖率 |','|---|---:|---:|---:|---:|']
    for method in ('density','shape','autoencoder','fusion'):
        values=[r for r in curves if r['method']==method and float(r['top_fraction'])==.2]
        by={r['type']:float(r['recall']) for r in values}
        lines.append(f'| {method} | {by["OPCID"]:.1%} | {by["CHIN"]:.1%} | {by["CHID"]:.1%} | {float(values[0]["genome_coverage"]):.1%} |')
    lines+=['','曲线比较在每个尺度内对该方法分数再次排序，融合曲线与直接按融合分数跨尺度取前 20% 的最终候选预算并非完全相同；最终清单的精确结果见下表。高覆盖率本身就提高召回，不能只看召回数字宣称检测有效。','',
            '| 最终非重叠窗口 | OPCID | CHIN | CHID |','|---|---:|---:|---:|',
            '| 实际召回 | '+' | '.join(f'{report["deduplicated_known_recall"][k]:.1%}' for k in ('OPCID','CHIN','CHID'))+' |',
            '| 等数量、等尺度、等覆盖随机参考均值 | '+' | '.join(f'{null_summary[k]["mean"]:.1%}' for k in ('OPCID','CHIN','CHID'))+' |','',
            '本轮 CHIN 召回高于随机参考的均值和该参考的 95% 范围；OPCID 低于随机均值，CHID 位于随机参考的 95% 范围内，不能声称三类检测均有提升。随机参考是在候选冻结后补充的描述性对照：100 次固定种子抽样，匹配两种尺度的窗口数量、质量规则及不重叠约束，不用于重选候选或改验证规则。完整分布见 matched_budget_random_reference.json。',
            '', '## 重复验证与新结构判据','',
            '在两个重复同坐标 O/E 矩阵的共同有效上三角上计算 Pearson 相关。rep1 阶段预选同尺度、无注释、与全部候选不重叠的背景，按 rep1 平均 O/E 和有效比例接近程度排序，每个候选最多取 20 个相互不重叠的背景。匹配是近邻匹配，没有要求密度完全相等。',
            '', '通过门槛为 r > max(0.5, 匹配背景相关性的 95% 分位数)，且至少有 10 个可用对照。不同候选可能复用对照；此门槛不是经过多重检验校正的显著性或错误发现率保证。对照不足一律标作证据不足。',
            '', '操作性“候选新簇”要求：簇内没有任何已知注释重叠成员，且至少 5 个空间独立成员通过重复门槛。本轮六簇全部混有已知成员；固定 K=6 可能合并不同形态，所以零新簇不能排除存在更细的未知亚型。最近已知表征距离仅作连续参考，不能证明新类型。',
            '', '全簇成员数阈值与相关性阈值都是预设约定，不是生物学定律。本轮满足实践中的已知结构复现与完整流程交付，但不能宣称发现新的染色质结构类型。',
            '', '## 输出与复现','',
            'candidates.csv 的 start/end/center 是扫描窗口坐标，window_length_bp 是窗口宽度，不是精确结构边界或经生物学测定的长度。wraps_origin=True 时区间跨染色体原点。原始标注坐标仍为 provisional。',
            '', '文件还包括：autoencoder.pt、ae_history.json、scan_rep1.csv、recall_curves.csv、representation.npz、clusters.csv、controls.csv、189 张统一色标的双重复热图。色标上限仅用 rep1 候选确定。当前仅覆盖近对角线局部结构，不包含全部远距离二维相互作用。',
            '', '```bash',
            'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/discover_task2.py',
            '.venv/bin/python scripts/report_task2.py',
            '.venv/bin/python -m unittest discover -s tests -v','```','',
            '重跑会使用输入签名匹配的近对角线缓存，并重新训练和生成结果；输入或发现代码变化时原输出目录拒绝混用。配置见 configs/discovery.json。下一轮若改方法，应另建版本并明确 rep2 已被查看，不能继续当作完全未见的验证。']
    (ROOT/'docs/任务二实验结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(checks=checks,matched_budget_random_reference=null_summary),indent=2))


if __name__=='__main__':main()
