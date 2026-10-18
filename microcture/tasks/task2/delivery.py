"""Compact, traceable task-two review exports; no fitting or raw contact access."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

from microcture.layout import PROJECT_ROOT, resolve


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path, sources):
    path = Path(path)
    sources.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def read_json(path, sources):
    path = Path(path)
    sources.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    return json.loads(path.read_text(encoding='utf-8'))


def index_rows(rows, field):
    indexed = {r[field]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate {field} in supplementary evidence')
    return indexed


def candidate_catalogue(disjoint, *, novelty_dir=None, morphology_dir=None, centering_dir=None):
    """Join only identical candidate cohorts; missing or old cohorts stay unassessed."""
    from scripts.task2.centering_status import aggregate, as_bool, normalize_summary

    candidates = index_rows(disjoint, 'window_id')
    sources, supplement, rows = [], {}, []
    for name, directory, filename in (
        ('novelty', novelty_dir, 'candidate_review.csv'),
        ('morphology', morphology_dir, 'candidate_review_table.csv'),
    ):
        path = Path(directory) / filename if directory is not None else None
        if path is None or not path.is_file():
            supplement[name] = dict(status='unavailable', rows={})
            continue
        values = index_rows(read_table(path, sources), 'window_id')
        matched = set(values) == set(candidates)
        if matched:
            for key, value in values.items():
                original = candidates[key]
                width = value.get('window_length_bp', value.get('width_bp'))
                if int(value['start_bp']) != int(original['start_bp']) or int(width) != int(original['window_length_bp']):
                    raise ValueError(f'{name}: candidate coordinates differ for {key}')
        supplement[name] = dict(status='matched_saved_cohort' if matched else 'unavailable_candidate_identity_mismatch',
                                rows=values if matched else {})

    cohorts = {}
    for name in ('primary', 'supplement'):
        directory = Path(centering_dir) / name if centering_dir is not None else None
        files = [directory / f for f in ('report.json', 'candidate_summary.csv', 'entities.csv')] if directory else []
        if not files or not all(p.is_file() for p in files):
            cohorts[name] = dict(status='unavailable', rows={})
            continue
        saved = read_json(files[0], sources)
        summaries = index_rows(read_table(files[1], sources), 'entity_id')
        entities = read_table(files[2], sources)
        originals = index_rows([r for r in entities if r['kind'] == 'candidate'], 'entity_id')
        matched = set(summaries) == set(candidates) == set(originals)
        if matched:
            for key, original in originals.items():
                center = int(key.rsplit('_c', 1)[1])
                if int(original['original_center_bin']) != center:
                    raise ValueError(f'{name}: centering anchor identity differs for {key}')
            summaries = {k: normalize_summary(v, int(saved['calibration_windows']), 10) for k, v in summaries.items()}
        else:
            summaries = {}
        calibration_types = {}
        for entity in entities:
            if as_bool(entity.get('calibration_member')) is True:
                kind = entity.get('type', '') or 'unknown'
                calibration_types[kind] = calibration_types.get(kind, 0) + 1
        if sum(calibration_types.values()) != int(saved['calibration_windows']):
            raise ValueError(f'{name}: calibration membership differs from saved count')
        if matched and int(saved['candidates']) != len(candidates):
            raise ValueError(f'{name}: candidate count differs from saved cohort')
        cohorts[name] = dict(status='matched_saved_cohort' if matched else 'unavailable_candidate_identity_mismatch',
                             rows=summaries, calibration_windows=saved['calibration_windows'],
                             calibration_type_counts=calibration_types,
                             aggregate=aggregate(list(summaries.values())) if matched else None)

    for key, original in candidates.items():
        row = dict(original)
        novelty = supplement['novelty']['rows'].get(key, {})
        morphology = supplement['morphology']['rows'].get(key, {})
        row.update(pca_nearest_known=novelty.get('pca_nearest_known'),
                   pca_nearest_type=novelty.get('pca_nearest_type'),
                   shape_nearest_known=novelty.get('shape_nearest_known'),
                   shape_nearest_type=novelty.get('shape_nearest_type'),
                   pca_outside_both=as_bool(novelty.get('pca_outside_both')),
                   shape_outside_both=as_bool(novelty.get('shape_outside_both')),
                   residual_replication=float(morphology['residual_replication']) if morphology else None,
                   annotation_gap_bp=int(morphology['annotation_gap_bp']) if morphology else None,
                   visual_description=morphology.get('visual_description'),
                   visual_review_scope=morphology.get('visual_review_scope'),
                   confirmed_new_type=False)
        for name in cohorts:
            evidence = cohorts[name]['rows'].get(key, {})
            row[name + '_assessment_status'] = evidence.get('assessment_status', 'not_evaluable')
            row[name + '_assessment_reason'] = evidence.get('assessment_reason') or ('' if evidence else cohorts[name]['status'])
            row[name + '_robust_anomaly'] = evidence.get('robust_recentered_anomaly')
        rows.append(row)
    overview = {name: {k:v for k,v in data.items() if k != 'rows'} for name, data in supplement.items()}
    references = Path(novelty_dir)/'known_references.csv' if novelty_dir is not None else None
    if references is not None and references.is_file():
        reference_rows = read_table(references,sources)
        overview['novelty'].update(reference_records=len(reference_rows),
                                  unique_reference_structures=len({r['structure_id'] for r in reference_rows}))
    overview['centering'] = {name: {k:v for k,v in data.items() if k != 'rows'} for name, data in cohorts.items()}
    return rows, overview, sources


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else v for k,v in row.items()}
                         for row in rows])


def project_sources(sources, root=PROJECT_ROOT):
    unique = {}
    for row in sources:
        path = resolve(Path(row['path']), root=root)
        if not path.is_relative_to(root):
            raise ValueError(f'Review sources must remain inside the project: {path}')
        name = path.relative_to(root).as_posix()
        value = dict(path=name, sha256=row['sha256'], bytes=row['bytes'])
        if name in unique and unique[name] != value:
            raise ValueError(f'Conflicting source records: {name}')
        unique[name] = value
    return [unique[k] for k in sorted(unique)]


def render_html(summary, candidates, *, back_link):
    """Render searchable evidence rows, keeping NA distinct from negative findings."""
    def esc(value):
        return html.escape(str(value), quote=True)

    def show(value):
        if value is None or value == '':
            return 'NA'
        if isinstance(value, bool):
            return '是' if value else '否'
        if isinstance(value, float):
            return f'{value:.3f}' if math.isfinite(value) else 'NA'
        return esc(value)

    rows = []
    for row in candidates:
        cells = [row['window_id'], row['start_bp'], row['window_length_bp'], row.get('correlation'),
                 row.get('cutoff'), row.get('pca_nearest_known'), row.get('pca_outside_both'),
                 row.get('shape_outside_both'), row.get('primary_robust_anomaly'),
                 row.get('supplement_robust_anomaly'), row.get('residual_replication')]
        description = esc(row.get('visual_description') or '无已保存描述')
        reason = esc(row.get('supplement_assessment_reason') or '仅探索性子集评价')
        rows.append('<tr data-q="' + esc(str(row['window_id']) + ' ' + str(row.get('pca_nearest_known'))) + '">' +
                    ''.join('<td>'+show(value)+'</td>' for value in cells) +
                    '<td><details><summary>查看</summary><p>'+description+'</p><p>'+reason+
                    '</p><p>历史图板描述，不是专家类型认证；坐标 provisional。</p></details></td></tr>')
    headings = ['候选', '起点 bp', '窗口 bp', '双重复 r', '对照阈值', 'PCA最近参考', 'PCA异常',
                '形态异常', '主方案稳健异常', '补充稳健异常', '去边际相关', '说明']
    discovery = summary['discovery']
    cluster_count = discovery.get('nonempty_cluster_records', discovery.get('cluster_records', 'NA'))
    passing = discovery.get('passing_cluster_records', discovery.get('all_passing_cluster_configurations', 'NA'))
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MicroCTure · 任务二证据审查</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:#173344;font:15px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:36px 22px}h1{font-size:34px;line-height:1.25}.muted{color:#536c7d}a{color:#086c78}.box{background:white;border:1px solid #d6e1e8;border-radius:12px;padding:24px;margin:20px 0}.cards{display:flex;gap:16px;flex-wrap:wrap}.cards div{flex:1;min-width:180px;background:#e7f0f4;padding:16px;border-radius:8px}.cards b{display:block;font-size:30px}img{max-width:100%;height:auto}input{padding:10px;min-width:300px;border:1px solid #abbecb;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce5eb;vertical-align:top}th{background:#edf3f7;white-space:nowrap}.scroll{overflow:auto}.note{border-left:4px solid #d78029;padding-left:16px}details{min-width:100px}details p{min-width:220px}.links{display:flex;gap:18px;flex-wrap:wrap}
</style><main><p class="muted">Micro-C · 固定预算检索与候选核查</p><h1>任务二：从重复支持到新结构判据</h1><p class="note">保存表的一致性核验通过，不等于确认新类型。WT已被查看，以下均为探索性结果；历史模型权重缺失单独报告。NA表示不可判定。</p>''' + \
        '<p><a href="'+esc(back_link)+'">返回项目概览</a></p><div class="cards"><div>去重候选<b>'+str(len(candidates))+ \
        '</b></div><div>非空簇记录<b>'+esc(cluster_count)+'</b></div><div>通过新簇判据<b>'+esc(passing)+ \
        '</b></div></div><section class="box"><h2>模型行为与筛选门槛</h2><img src="model_diagnostics.png" alt="完整方法的已知检索、评分质量相关与簇门槛诊断"><p class="muted">随机参考范围是20次同预算抽样的最小值到最大值，不是生物学置信区间。重建误差与信号丰度的相关不构成因果解释。簇记录复用窗口，不能作为独立样本。</p></section>' + \
        '<section class="box"><h2>逐候选证据</h2><p>未知与已知采用相同的居中、多尺度、平移和遮挡规则。主方案校准不足保持NA；补充的零稳健异常只适用于可评估子集。PCA最近参考仅描述相似性，不赋予候选生物学类别。</p><label>筛选候选或最近参考 <input id="search" type="search" placeholder="如 w64_c43040 或 CHIN"></label><div class="scroll"><table><thead><tr>'+ \
        ''.join('<th>'+esc(name)+'</th>' for name in headings)+'</tr></thead><tbody>'+''.join(rows)+ \
        '</tbody></table></div></section><section class="box"><h2>下载与来源</h2><p class="links"><a href="review.json">完整审查摘要</a><a href="candidate_evidence.csv">候选证据表</a><a href="retrieval_diagnostics.csv">同预算检索表</a><a href="score_quality.csv">评分与质量相关</a><a href="ARTIFACTS.json">文件SHA索引</a></p><p>本页只使用保存数值，未启动模型训练、原始像素扫描或新的形态真值标注。</p></section></main><script>document.getElementById("search").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.dataset.q.toLowerCase().includes(q));});</script></html>'


def write_snapshot(directory, summary, candidates, diagnostics, *, root=PROJECT_ROOT):
    from .diagnostics import plot_diagnostics

    directory = Path(directory)
    if directory.exists() or any(p.is_symlink() for p in (directory,*directory.parents)):
        raise FileExistsError(f'Preserve existing snapshot: {directory}')
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory/'candidate_evidence.csv', candidates)
    write_csv(directory/'retrieval_diagnostics.csv', diagnostics['test_retrieval_rows'])
    write_csv(directory/'score_quality.csv', diagnostics['score_quality_rows'])
    plot_diagnostics(diagnostics, directory/'model_diagnostics.png')
    (directory/'review.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (directory/'index.html').write_text(render_html(summary,candidates,back_link=os.path.relpath(root/'README.md',directory)),encoding='utf-8')
    manifest = dict(schema_version=1, files=[dict(path=p.name, bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(directory.iterdir()) if p.is_file()],
                    scientific_scope='Descriptive reanalysis of previously viewed data; not independent biological confirmation')
    (directory/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')


def verify_snapshot(directory, *, root=PROJECT_ROOT, require_local_inputs=False):
    directory, root = Path(directory), Path(root)
    manifest = json.loads((directory/'ARTIFACTS.json').read_text())
    indexed = set()
    for row in manifest['files']:
        name = row['path']
        if Path(name).name != name or name in indexed or name == 'ARTIFACTS.json':
            raise ValueError('Unsafe or duplicate snapshot file')
        indexed.add(name)
        path = directory/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Changed task-two review artifact: {name}')
    required = {'review.json','candidate_evidence.csv','retrieval_diagnostics.csv','score_quality.csv','model_diagnostics.png','index.html'}
    if not required.issubset(indexed) or {p.name for p in directory.iterdir()} != indexed|{'ARTIFACTS.json'}:
        raise ValueError('Task-two review artifact inventory is incomplete or contains additions')
    summary = json.loads((directory/'review.json').read_text())
    if summary['training_performed'] or summary['raw_pixels_read'] or summary['independent_confirmation']:
        raise ValueError('Saved evidence review cannot claim training or independent confirmation')
    checked, absent = 0, 0
    for row in summary['sources']:
        name = Path(row['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe review source path')
        path = resolve(name,root=root)
        if not path.is_file():
            absent += 1
            if require_local_inputs:
                raise FileNotFoundError(f'Missing local review source: {name}')
        elif any(p.is_symlink() for p in (path,*path.parents)) or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Changed saved review source: {name}')
        else:
            checked += 1
    for row in summary.get('code_sources',[]):
        name = Path(row['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe review code source path')
        path = root/name
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)) or digest(path) != row['sha256']:
            raise ValueError(f'Changed review implementation: {name}')
    return dict(status='verified', snapshot_files=len(indexed), local_input_hashes_checked=checked,
                local_inputs_absent=absent, training_performed=False, independent_confirmation=False)
