"""Compare frozen replicated windows with same-scale, spatially separated known references."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import hashlib
import json
import html
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discover_task2_v2 import ROOT, read_csv
from discover_task2 import load_band, extract, tensor, write_csv
from discovery_v2_core import bin_roles, window_role, train_background


def shape_feature(oe, mask):
    x = tensor(oe, mask)
    weights = x[1].reshape(16, 4, 16, 4).sum((1, 3))
    values = (x[0] * x[1]).reshape(16, 4, 16, 4).sum((1, 3))
    return np.divide(values, weights, out=np.zeros_like(values), where=weights > 0).ravel()


def boundary_profile(oe, mask, flank, minimum):
    profile = np.full(len(oe), np.nan)
    for cut in range(flank, len(oe) - flank + 1):
        cross_mask = mask[cut-flank:cut, cut:cut+flank]
        left_mask = np.triu(mask[cut-flank:cut, cut-flank:cut], 2)
        right_mask = np.triu(mask[cut:cut+flank, cut:cut+flank], 2)
        within = np.concatenate([oe[cut-flank:cut, cut-flank:cut][left_mask], oe[cut:cut+flank, cut:cut+flank][right_mask]])
        if cross_mask.sum() >= minimum and len(within) >= minimum:
            cross = oe[cut-flank:cut, cut:cut+flank][cross_mask].mean()
            profile[cut] = np.log2((cross + .01) / (within.mean() + .01))
    return profile


def circular_gap(start, end, known, length):
    return min(max(start - (int(known['end']) + shift), (int(known['start']) + shift) - end, 0)
               for shift in [-length, 0, length])


def main():
    cfg = json.loads((ROOT/'configs/novelty_review.json').read_text())
    source = json.loads((ROOT/cfg['source_config']).read_text())
    data = json.loads((ROOT/'configs/data.json').read_text())
    out = ROOT/cfg['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    sources = ['configs/novelty_review.json', cfg['source_config'], cfg['candidate_source'],
               'scripts/review_novelty.py', 'scripts/discovery_v2_core.py', 'scripts/discover_task2.py',
               'data/processed/structures.csv', 'results/task2_v2/shape_representation.npz', 'results/task2_v2/run_manifest.json']
    manifest = dict(config=cfg, sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    path = out/'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Inputs changed: use a new output directory')
    path.write_text(json.dumps(manifest, indent=2))
    candidates = read_csv(cfg['candidate_source'])
    known = read_csv('data/processed/structures.csv')
    n = (data['chrom_length'] + 99)//100
    roles = bin_roles(n, source)
    bands = [train_background(load_band(s, n, 256), roles) for s in data['samples']]
    with np.load(ROOT/'results/task2_v2/shape_representation.npz') as fit:
        pca_mean, components = fit['pca_mean'], fit['components']
    def describe(start, width):
        matrices = [extract(b, start, width) for b in bands]
        features = np.array([shape_feature(oe, mask) for _, oe, mask in matrices])
        quality = []
        for raw, _, mask in matrices:
            upper = np.triu(mask, 2)
            fraction = mask.sum()/(width*width-3*width+2)
            zero = (raw[upper] == 0).mean() if upper.any() else 1.
            quality.append(bool(fraction >= cfg['min_valid_fraction'] and zero <= cfg['max_zero_fraction']))
        return features, (features-pca_mean)@components.T, quality
    references, ref_features, ref_pca = [], [], []
    candidate_bins = np.zeros(n, bool)
    for candidate in candidates:
        candidate_bins[(int(candidate['start_bp'])//100 + np.arange(int(candidate['window_length_bp'])//100)) % n] = True
    for width in source['window_bins']:
        for k in known:
            start = int(float(k['center'])//100) - width//2
            role = window_role(start, width, roles, source['spatial_block_bins'])
            if role not in [cfg['reference_role'], cfg['calibration_role']]:
                continue
            if candidate_bins[(start + np.arange(width)) % n].any():
                continue
            features, projection, quality = describe(start, width)
            if not all(quality):
                continue
            references.append(dict(structure_id=k['structure_id'], type=k['type'], start_bin=start,
                                   width_bins=width, role=role, annotation_start=int(k['start']), annotation_end=int(k['end'])))
            ref_features.append(features)
            ref_pca.append(projection)
    ref_features, ref_pca = np.array(ref_features), np.array(ref_pca)
    write_csv(out/'known_references.csv', references)
    thresholds, calibration = {}, []
    distances = {'pca': (ref_pca, 'euclidean'), 'shape': (ref_features, 'cosine')}
    for width in source['window_bins']:
        training = [i for i, r in enumerate(references) if r['width_bins'] == width and r['role'] == cfg['reference_role']]
        pool = [i for i, r in enumerate(references) if r['width_bins'] == width and r['role'] == cfg['calibration_role']]
        # Spatially independent calibration windows, fixed coordinate order.
        occupied = np.zeros(n, bool)
        validation = []
        for i in sorted(pool, key=lambda i: (references[i]['start_bin'], references[i]['structure_id'])):
            bins = np.arange(references[i]['start_bin'], references[i]['start_bin']+width)
            if not occupied[bins].any():
                occupied[bins] = True
                validation.append(i)
        if not training or not validation:
            raise ValueError('No same-scale known calibration/reference windows')
        thresholds[str(width)] = dict(training_references=len(training), independent_calibration_windows=len(validation))
        for metric, (features, distance) in distances.items():
            first = cdist(features[validation, 0], features[training, 0], metric=distance)
            match = np.array(training)[first.argmin(1)]
            values = []
            for i, j in zip(validation, match):
                pair = [float(cdist(features[i:i+1, rep], features[j:j+1, rep], metric=distance)[0, 0]) for rep in [0, 1]]
                values.append(pair)
                calibration.append(dict(width_bins=width, metric=metric, calibration_id=references[i]['structure_id'],
                                        reference_id=references[j]['structure_id'], calibration_index=i, reference_index=int(j), rep1_distance=pair[0], rep2_distance=pair[1]))
            threshold = np.quantile(values, cfg['distance_quantile'], axis=0).tolist() if len(validation) >= cfg['min_calibration_references'] else None
            thresholds[str(width)][metric] = threshold
    # Choices are based on rep1; calibration uses previously seen rep2 and is explicitly exploratory.
    write_csv(out/'known_distance_calibration.csv', calibration)
    (out/'calibration.json').write_text(json.dumps(thresholds, indent=2))
    result, plots = [], []
    (out/'comparisons').mkdir(exist_ok=True)
    for candidate in candidates:
        start, width = int(candidate['start_bp'])//100, int(candidate['window_length_bp'])//100
        assert candidate['wraps_origin'] == 'False', 'Add circular display support before extending this fixed review set'
        f, p, quality = describe(start, width)
        training = [i for i, r in enumerate(references) if r['width_bins'] == width and r['role'] == cfg['reference_role']]
        row = dict(window_id=candidate['window_id'], start_bp=start*100, end_bp=(start+width)*100,
                   window_length_bp=width*100, correlation=float(candidate['correlation']),
                   replication_margin=float(candidate['correlation'])-float(candidate['cutoff']), quality_pass=all(quality))
        chosen = {}
        for metric, (ref, distance) in distances.items():
            candidate_feature = p if metric == 'pca' else f
            d1 = cdist(candidate_feature[0:1], ref[training, 0], metric=distance)[0]
            j = training[int(d1.argmin())]
            chosen[metric] = j
            ds = [float(cdist(candidate_feature[rep:rep+1], ref[j:j+1, rep], metric=distance)[0, 0]) for rep in [0, 1]]
            threshold = thresholds[str(width)][metric]
            row.update({f'{metric}_nearest_known': references[j]['structure_id'], f'{metric}_nearest_type': references[j]['type'],
                        f'{metric}_rep1_distance': ds[0], f'{metric}_rep2_distance': ds[1],
                        f'{metric}_rep1_cutoff': threshold[0] if threshold else '', f'{metric}_rep2_cutoff': threshold[1] if threshold else '',
                        f'{metric}_outside_both': all(quality) and threshold is not None and all(d > t for d, t in zip(ds, threshold))})
        row['priority_distinct_shape'] = row['pca_outside_both'] and row['shape_outside_both']
        row['review_status'] = 'outside_known_reference_range_both_metrics' if row['priority_distinct_shape'] else ('insufficient_calibration' if thresholds[str(width)]['pca'] is None else 'no_consistent_two_metric_novelty_evidence')
        nearest = min(known, key=lambda k: circular_gap(start*100, (start+width)*100, k, data['chrom_length']))
        row['nearest_annotation_by_position'] = nearest['structure_id']
        row['annotation_gap_bp'] = circular_gap(start*100, (start+width)*100, nearest, data['chrom_length'])
        assert row['annotation_gap_bp'] > 0, 'Candidate unexpectedly overlaps/touches known annotation'
        profiles, matrices, reference_matrices = [], [], []
        j = chosen['pca']
        for b in bands:
            raw, oe, mask = extract(b, start, width)
            matrices.append((oe, mask))
            profiles.append(boundary_profile(oe, mask, cfg['boundary_flank_bins'], cfg['boundary_min_pairs']))
            _, ro, rm = extract(b, references[j]['start_bin'], width)
            reference_matrices.append((ro, rm))
        peaks = [int(np.nanargmin(v)) if np.isfinite(v).any() else None for v in profiles]
        row['rep1_boundary_bp'] = (start+peaks[0])*100 if peaks[0] is not None else ''
        row['rep2_boundary_bp'] = (start+peaks[1])*100 if peaks[1] is not None else ''
        row['boundary_shift_bp'] = abs(peaks[0]-peaks[1])*100 if None not in peaks else ''
        row['rep2_score_at_rep1_boundary'] = float(profiles[1][peaks[0]]) if peaks[0] is not None else ''
        result.append(row)
        limit = float(np.quantile(np.concatenate([np.log1p(a[m]) for a, m in [matrices[0], reference_matrices[0]]]), .99))
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
        for rep in [0, 1]:
            for col, (oe, mask) in enumerate([matrices[rep], reference_matrices[rep]]):
                origin = start if col == 0 else references[j]['start_bin']
                im = axes[rep, col].imshow(np.where(mask, np.log1p(oe), np.nan), origin='lower', cmap='magma', vmin=0, vmax=limit,
                                           extent=(origin*.1, (origin+width)*.1, origin*.1, (origin+width)*.1))
                title = candidate['window_id'] if col == 0 else references[j]['structure_id']
                axes[rep, col].set(title=f'{title} / rep{rep+1}', xlabel='Position (kb)', ylabel='Position (kb)')
                if col == 1:
                    a, z = references[j]['annotation_start']/1000, references[j]['annotation_end']/1000
                    for edge in [a, z]:
                        if origin*.1 <= edge <= (origin+width)*.1:
                            axes[rep, col].axvline(edge, color='cyan', lw=.6)
                            axes[rep, col].axhline(edge, color='cyan', lw=.6)
            axes[rep, 2].plot((start+np.arange(width))*.1, profiles[rep])
            axes[rep, 2].axhline(0, color='gray', ls=':')
            if peaks[0] is not None:
                axes[rep, 2].axvline((start+peaks[0])*.1, color='orange', ls='--', label='Rep1 minimum')
            axes[rep, 2].set(title=f'Local cross/within O/E / rep{rep+1}', xlabel='Position (kb)', ylabel='log2 ratio')
        fig.colorbar(im, ax=axes[:, :2], label='log1p(O/E)', shrink=.5)
        fig.suptitle(f"Exploratory review: {candidate['window_id']} | distinct-shape priority={row['priority_distinct_shape']}\nSame-scale nearest training reference selected with rep1; cyan = known annotation boundaries", fontsize=11)
        name = f"comparisons/{candidate['window_id']}.png"
        fig.savefig(out/name, dpi=110)
        plt.close(fig)
        plots.append((row, name))
    write_csv(out/'candidate_review.csv', result)
    np.savez_compressed(out/'reference_features.npz', shape=ref_features, pca=ref_pca)
    summary = dict(candidates=len(result), same_scale_known_references=len(references), calibration=thresholds,
                   pca_outside_both=sum(r['pca_outside_both'] for r in result),
                   shape_outside_both=sum(r['shape_outside_both'] for r in result),
                   distinct_shape_priority=sum(r['priority_distinct_shape'] for r in result),
                   insufficient_calibration=sum(r['review_status']=='insufficient_calibration' for r in result),
                   figures=len(plots), interpretation=cfg['interpretation'])
    (out/'report.json').write_text(json.dumps(summary, indent=2))
    gallery = []
    for row, name in sorted(plots, key=lambda item: (-item[0]['priority_distinct_shape'], -item[0]['correlation'])):
        title = f"{row['window_id']} · 形态参考 {row['pca_nearest_known']} · 最近注释间隔 {row['annotation_gap_bp']} bp · {row['review_status']}"
        gallery.append(f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)}</figcaption></figure>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>21 个候选形态核查</title>
+<style>body{{max-width:1400px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif}}img{{max-width:100%}}figure{{margin:24px 0}}</style>
+<h1>21 个未注释复现窗口：已知形态与局部边界对照</h1>
+<p>双重复均超出已知距离范围且两种距离指标一致：{summary['distinct_shape_priority']} 个；校准不足：{summary['insufficient_calibration']} 个。此结果用于审阅排序，不构成新类型或新簇确认。</p>
+<p><a href="candidate_review.csv">逐候选核查表</a> · <a href="calibration.json">同尺度校准</a> · <a href="known_references.csv">已知参考清单</a></p>
+<p>训练区已知中心窗口作为参考；早停区互不重叠已知窗口的最近参考距离 95% 分位数作为描述性阈值，至少需要 10 个校准窗口。最近参考仅按 rep1 选定，rep2 比较同一参考。中心窗口会包含周围背景，不是纯类型模板；参考相似不意味着分类已确认。两种指标为训练拟合形态 PCA 欧氏距离和直接形态余弦距离。</p>
+<p>右列显示 800 bp 两侧邻域的跨边界/侧内 O/E 比值，低点仅为局部隔离候选，不是已验证的结构边界；橙线固定为 rep1 最低点。坐标约定暂定。图间色标可能不同，每图四个热图共用仅由 rep1 确定的色标。</p>'''.replace('\n+', '\n')
    (out/'index.html').write_text(page+'\n'.join(gallery)+'</html>')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
