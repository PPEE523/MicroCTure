"""Render current or archived centering evidence into a separate review directory.

Does not modify source results, read raw Micro-C matrices, or claim historical
source manifests match changed code. Legacy False-with-NaN is displayed as NA.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from centering_status import (assess, as_bool, decision_from_row, summarize_candidate,
                              aggregate, metric_count, display, complete_count, require_new_output)

ROOT = _layout_root


def read_rows(path):
    with _layout_resolve(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_analysis(directory, cfg, settings=None):
    directory = _layout_resolve(directory)
    report = json.loads((directory/'report.json').read_text())
    settings = settings if settings is not None else read_rows(directory/'variants.csv')
    count = int(report['calibration_windows'])
    rows = read_rows(directory/'candidate_variants.csv')
    lookup = {}
    normalized = []
    for row in rows:
        decision = assess([row['rep1_distance'], row['rep2_distance']], [row['rep1_cutoff'], row['rep2_cutoff']],
                          as_bool(row['quality_pass']), count, cfg['min_calibration_windows'])
        value = decision['outside_both']
        out = dict(row, **decision)
        lookup[row['entity_id'], row['variant_id'], row['metric']] = value
        normalized.append(out)
    summaries = []
    for old in read_rows(directory/'candidate_summary.csv'):
        flags = {(vi, metric): lookup[old['entity_id'], v['variant_id'], metric]
                 for vi, v in enumerate(settings) for metric in ['pca', 'shape']}
        summary = dict(old)
        summary.update(summarize_candidate(settings, flags, as_bool(old['boundary_reproducible']),
                      as_bool(old['all_variants_quality_pass']), count >= cfg['min_calibration_windows'],
                      cfg['robust_fraction'], cfg['anchors'], cfg['width_bins']))
        summaries.append(summary)
    report.update(aggregate(summaries))
    report['legacy_source_report'] = str(directory/'report.json')
    report['historical_source_manifest_verified_against_current_code'] = False
    return settings, normalized, summaries, report


def render_grid(path, settings, rows, summaries, anchors, title):
    lookup = {(r['entity_id'], r['variant_id'], r['metric']): r['outside_both'] for r in rows}
    names = [r['entity_id'] for r in summaries]
    fig, axes = plt.subplots(1, len(anchors), figsize=(16, max(4, len(names)*.35)), layout='constrained', squeeze=False, sharey=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#cccccc')
    for ax, anchor in zip(axes.flat, anchors):
        vs = [v for v in settings if v['anchor'] == anchor]
        matrix = np.array([[metric_count(lookup[name, v['variant_id'], 'pca'], lookup[name, v['variant_id'], 'shape']) for v in vs] for name in names])
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=2, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set(title=anchor, yticks=range(len(names)), yticklabels=names, xticks=range(len(vs)),
               xticklabels=[f"{int(v['width_bins'])*.1:g}k/{int(v['shift_bins'])*100:+d}/{v['occlusion']}" for v in vs])
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
        if not np.isfinite(matrix).any():
            ax.text(.5, .5, 'NA: not evaluable', transform=ax.transAxes, ha='center')
    fig.colorbar(image, ax=list(axes.flat), ticks=[0, 1, 2], label='Metrics outside range; gray = NA / not evaluable', shrink=.5)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def paired_factors(directory, cfg, settings):
    """Fixed saved-feature arithmetic only; no raw-data reads or fitting."""
    directory = _layout_resolve(directory)
    if not (directory/'features.npz').exists():
        return []
    entities = read_rows(directory/'entities.csv')
    with np.load(directory/'features.npz') as saved:
        xs = {name: saved[name] for name in ['pca', 'shape']}
        candidates, calibration, quality = saved['candidates'], saved['calibration'], saved['quality']
    lookup = {(v['anchor'], int(v['width_bins']), int(v['shift_bins']), v['occlusion']): vi for vi, v in enumerate(settings)}
    rows = []
    for vi, v in enumerate(settings):
        a, w, s, o = v['anchor'], int(v['width_bins']), int(v['shift_bins']), v['occlusion']
        pairs = []
        if a == 'boundary': pairs.append(('recenter', lookup['original', w, s, o]))
        if s: pairs.append(('translation', lookup[a, w, 0, o]))
        if w != 64: pairs.append(('scale', lookup[a, 64, s, o]))
        if o != 'none': pairs.append(('occlusion', lookup[a, w, s, 'none']))
        for factor, base in pairs:
            for metric, distance in [('pca', 'euclidean'), ('shape', 'cosine')]:
                x = xs[metric]
                values = [[cdist(x[i:i+1, base, rep], x[i:i+1, vi, rep], metric=distance)[0, 0] for rep in [0, 1]] for i in calibration]
                cutoff = np.quantile(values, cfg['distance_quantile'], axis=0) if len(calibration) >= cfg['min_calibration_windows'] else [np.nan, np.nan]
                for i in candidates:
                    delta = [float(cdist(x[i:i+1, base, rep], x[i:i+1, vi, rep], metric=distance)[0, 0]) for rep in [0, 1]]
                    good = bool(quality[i, base].all() and quality[i, vi].all())
                    decision = assess(delta, cutoff, good, len(calibration), cfg['min_calibration_windows'])
                    decision['unusually_sensitive_both'] = decision.pop('outside_both')
                    rows.append(dict(entity_id=entities[i]['entity_id'], factor=factor, metric=metric,
                                     baseline_variant=settings[base]['variant_id'], target_variant=v['variant_id'],
                                     rep1_change=delta[0], rep2_change=delta[1], known_rep1_q95=float(cutoff[0]),
                                     known_rep2_q95=float(cutoff[1]), quality_pass=good, **decision))
    return rows


def write_review(directory, output, cfg, settings=None, title='Matched centering evidence', saved_tables_only=False):
    directory, output = _layout_resolve(directory).resolve(), _layout_resolve(output).resolve()
    if not any(base in output.parents for base in (_layout_resolve(ROOT/'results'), _layout_resolve(ROOT/'reports'))):
        raise ValueError('Review output must stay in this clone/results or reports')
    if directory == output or directory in output.parents:
        raise ValueError('Review output must be separate from the read-only input directory')
    require_new_output(output)
    settings, rows, summaries, report = load_analysis(directory, cfg, settings)
    inputs = [directory/name for name in ['report.json', 'candidate_variants.csv', 'candidate_summary.csv']]
    optional = ['entities.csv', 'variants.csv', 'isolated_factor_sensitivity.csv', 'thresholds.csv']
    if not saved_tables_only:
        optional.append('features.npz')
    inputs += [directory/name for name in optional if (directory/name).exists()]
    sources = [Path(__file__).resolve(), Path(__file__).with_name('centering_status.py')]
    manifest = dict(schema_version=2, input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, config=cfg, settings=settings)
    output.mkdir(parents=True, exist_ok=True)
    marker = output/'review_manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != manifest:
        raise ValueError('Review inputs/code changed; use a new output directory')
    marker.write_text(json.dumps(manifest, indent=2))
    write_rows(output/'candidate_variants.csv', rows)
    write_rows(output/'candidate_summary.csv', summaries)
    factors = []
    if saved_tables_only:
        if (directory/'isolated_factor_sensitivity.csv').exists():
            for old in read_rows(directory/'isolated_factor_sensitivity.csv'):
                decision = assess([old['rep1_change'], old['rep2_change']], [old['known_rep1_q95'], old['known_rep2_q95']],
                                  as_bool(old['quality_pass']), report['calibration_windows'], cfg['min_calibration_windows'])
                decision['unusually_sensitive_both'] = decision.pop('outside_both')
                factors.append(dict(old, **decision))
    else:
        factors = paired_factors(directory, cfg, settings)
    write_rows(output/'isolated_factor_sensitivity.csv', factors)
    report['isolated_factor_metric_comparisons'] = len(factors) if factors else None
    report['sensitive_metric_configurations'] = complete_count(r['unusually_sensitive_both'] for r in factors)
    report['assessed_factor_comparisons'] = sum(r['assessment_status'] == 'assessed' for r in factors)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    render_grid(output/'all_variants.png', settings, rows, summaries, cfg['anchors'], title)
    lines = [f'# {title}', '', f"校准窗口 {report['calibration_windows']}，预设最低 {cfg['min_calibration_windows']}；状态 {report['assessment_status']}。",
             f"完整网格可评估候选 {report['assessed_candidates']}，未评估 {report['not_evaluable_candidates']}。",
             f"出现双指标异常的候选：{display(report['candidates_with_any_two_metric_anomaly'])}；稳定重新居中异常：{display(report['robust_recentered_anomalies'])}。",
             '', 'NA/空判定/null 表示证据不足，不是 False/0。有限且质量合格的阴性仍按 False/0 显示。',
             '所有数据均为已查看 WT 的探索性分析；重复相关和形态距离不能确认新生物学类型。',
             '本次只读取保存结果；历史源码签名不宣称与当前修改后的代码相符。', '',
             '![全部变体；灰色为NA](all_variants.png)', '',
             '| 候选 | 状态 | 原中心异常变体 | 重新居中异常变体 | 稳定异常 |', '|---|---|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['entity_id']} | {r['assessment_status']} | {display(r['original_outside_variants'])} | {display(r['boundary_outside_variants'])} | {display(r['robust_recentered_anomaly'])} |")
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>'+title+'</h1><p>状态 '+report['assessment_status']+'；稳定异常 '+display(report['robust_recentered_anomalies'])+'</p><p>灰色 NA 表示未评估，不是阴性；此为探索性分析，未确认新类型。</p><img src="all_variants.png"><p><a href="analysis.md">完整报告</a> · <a href="candidate_summary.csv">候选表</a></p>')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(_layout_resolve(ROOT/'configs/centering_sensitivity.json')))
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir', default=str(_layout_resolve(ROOT/'results/task2_centering_review_v2')))
    args = parser.parse_args()
    cfg = json.loads(_layout_resolve(args.config).read_text())
    report = write_review(args.input_dir or _layout_resolve(ROOT/cfg['output_dir']), args.output_dir, cfg)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
