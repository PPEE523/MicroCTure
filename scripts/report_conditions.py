"""Build a Chinese report and navigable task-four gallery from computed results."""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'delta_stpA': 'ΔstpA', 'delta_hns_stpA': 'ΔhnsΔstpA'}


def main():
    report = json.loads((ROOT / 'reports/task34_results.json').read_text())
    with (ROOT / 'results/task4/differential_structures.csv').open() as handle:
        differences = list(csv.DictReader(handle))
    table = ['| 条件（相对 WT） | 类型 | 总数 | 质量合格 | 增强 | 减弱 | 合格结构中位 log₂FC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in report['summary']:
        table.append(f"| {LABELS[r['condition']]} | {r['type']} | {r['total']} | {r['quality_pass']} | {r['increased']} | {r['decreased']} | {r['median_log2_fc']:.3f} |")
    sections = []
    examples = []
    for condition, label in LABELS.items():
        ranked = sorted([r for r in differences if r['condition'] == condition and r['quality_pass'] == 'True'],
                        key=lambda r: abs(float(r['log2_fold_change'])), reverse=True)[:10]
        sections.append(f'<h2>{label}：绝对效应量前 10 个合格结构</h2>')
        for r in ranked:
            filename = f"heatmaps/{condition}_{r['structure_id']}.png"
            title = f"{r['structure_id']} · {r['start']}–{r['end']} bp · log₂FC={float(r['log2_fold_change']):.3f} · 满足变化判据={r['descriptive_change']}"
            sections.append(f'<figure><a href="{filename}"><img loading="lazy" src="{filename}" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)}</figcaption></figure>')
        examples.append(f"- {label}：[效应量最大结构 {ranked[0]['structure_id']}](../results/task4/heatmaps/{condition}_{ranked[0]['structure_id']}.png)。")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>任务四：差异结构定位</title>
<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif}img{max-width:100%}figure{margin:24px 0}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>
<h1>任务四：差异结构定位</h1>
<p>WT、ΔstpA、ΔhnsΔstpA，各两个生物学重复。688 次比较；ΔstpA 有 10 个、ΔhnsΔstpA 有 94 个已知结构满足描述性变化判据。坐标约定暂定。</p>
<p>每个结构使用六样本共同有效的接触对，四种跨重复比值全部超过 1.5 或全部低于 1/1.5，且通过覆盖度检查，才记录变化。没有计算统计显著性。</p>
<p><a href="differential_structures.csv">全部差异定位表</a> · <a href="summary.csv">分类汇总</a> · <a href="structure_signals.csv">逐样本信号</a> · <a href="../task3/index.html">任务三全基因组轨道</a></p>
<img src="effects.png" alt="两种处理的结构效应量分布">
<p>下面按深度归一化信号的绝对 log₂FC 选图，排名靠前不代表满足变化判据。每图六个热图共用色标，显示 log1p(O/E) 形态；O/E 与主分析的信号尺度不同，跨图色标也可能不同。</p>
'''
    (ROOT / 'results/task4/index.html').write_text(page + '\n'.join(sections) + '</html>')
    doc = '''# 任务三、四实验结果

任务三和任务四已完成。三条件共六个样本，生成全基因组 465 张轨道图；针对 344 个已知结构完成 688 次处理与 WT 比较。ΔstpA 有 4 个增强、6 个减弱，ΔhnsΔstpA 有 88 个增强、6 个减弱，均指满足本文预设规则的描述性变化。

## 结果入口

- [任务三：全基因组轨道目录](../results/task3/index.html)，每页 10 kb，最后一页到 4,641,652 bp。
- [任务四：结果与六样本热图](../results/task4/index.html)。
- [全部差异定位表](../results/task4/differential_structures.csv)、[逐样本结构信号](../results/task4/structure_signals.csv)、[分类汇总](../results/task4/summary.csv)。
- [结果 JSON](../reports/task34_results.json)、[独立验证报告](../reports/task34_verification.json)、[样本审计](../reports/conditions_audit.json)。

## 样本与可比性

WT 使用原项目的 37°C rep1/rep2；ΔstpA 使用 GSM8950761、GSM8950762；ΔhnsΔstpA 使用 GSM8950763、GSM8950764。后四个样本从 GEO 下载，基因型与参考组装已根据保存的 [SOFT 元数据](../data/metadata/GSE272159_family.soft) 核对；来源为 [GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)。六份 Cooler 的 bin 坐标哈希完全相同，染色体为 NC_000913.3。下载压缩包通过 gzip 完整性检查，原文件与归档 SHA256 记录于审计报告。

原始分辨率为 10 bp，本阶段聚合到 100 bp。比较使用六样本共同有效 bin：45,499 / 46,417；低覆盖 bin 和末端不足 100 bp 的 bin 排除。各样本的低覆盖阈值为正覆盖 bin 中位数的 10%。所有结构在六样本中使用相同接触对集合，避免各自忽略不同位置后直接比较。

## 任务三：轨道怎样读

每个位置汇总与其相隔 200 bp–10 kb 的有效接触。信号为原始计数除以该样本文库总接触数、乘以一百万，再除以有效接触伙伴数。一个接触分别贡献给它的两个端点。全基因组信号表为 [signals.csv](../results/task3/signals.csv)。

每页从上到下为：三条件重复均值的平方根曲线（淡色虚线为单独重复，灰底为 WT 均值）；三条件平均信号（不取平方根）；基因正负链蓝色框；OPCID、CHIN、CHID 橙色框。基因来自原注释表 Table 1，共 4,516 条。平方根仅用于展示，不用于差异计算。各页纵轴自动缩放，跨页比较应读数值，不能直接比曲线高度。

## 任务四：差异如何判定

结构边界向外对齐到 100 bp bin。只统计结构内部上三角、距离 200 bp–10 kb 的接触对，每个接触只计一次。逐样本信号为有效接触对平均计数 × 10⁶ / 文库总接触数。至少 20 个有效接触对、有效比例至少 80% 才通过质量检查；344 个结构中 327 个合格，17 个不合格仍保留在结果表中，不作为变化结构。

主效应量为 log₂[(处理两重复均值 + 0.01)/(WT 两重复均值 + 0.01)]。两处理重复分别与两 WT 重复比较，共四个比值；全部大于 1.5 判为增强，全部小于 1/1.5 判为减弱，其他为变化较小或重复不一致。最终还须通过质量检查，`descriptive_change=True` 才计入变化数。0.01 是每百万接触归一化信号尺度上的伪计数。没有假定 rep1 与另一条件 rep1 配对，也没有把矩阵像素当作独立生物学重复。

另报告样本自身距离期望校正后的平均 O/E 效应量与方向一致性，用作敏感性参照。它不参与主判据，也不与主信号混为一谈。O/E 热图显示相对距离背景的形态，而主分析反映文库内相对接触强度。

'''
    doc += '\n'.join(table) + '''

双缺失条件下，满足规则的增强主要出现在 CHIN（77 个）和 CHID（11 个）；OPCID 有 6 个减弱。单缺失条件满足规则的变化较少。这是当前数据、归一化和筛选规则下的观察，可用于安排后续验证，不能直接推出蛋白作用机制。

## 局部核查与结论边界

每个处理按合格结构绝对 log₂FC 选取前 10 个，共 20 张六样本对照热图；这些展示对象不保证全部满足变化判据，标题明确标记结果。

'''
    doc += '\n'.join(examples) + '''

每条件只有两个生物学重复，本轮未建立计数离散度模型、未计算 p 值或 FDR，不称为“统计显著”。文库大小归一化只能判断相对变化，不能推断绝对接触增加；共同低覆盖掩码可能排除某条件特有的接触丢失。基因型与样本批次效应不能完全分离。

原始注释的 0/1-based 约定尚未核实，当前保留原数值，暂按 0-based 左闭右开；100 bp 外扩边界也会纳入少量结构外接触。环形染色体末端因分箱近似最多产生 48 bp 的环绕偏差。基因表并未核实为完整参考注释。本阶段分析已知结构的条件变化，不改变任务二“未发现符合预设规则的新结构簇”的结论，也没有重新训练或调整任务一的测试集模型。

独立检查重新计算全部 688 次效应量、四种跨重复判据和汇总，核对 2,064 条结构样本信号、465 页连续覆盖及 486 张 PNG 完整性。复现命令见项目 README。
'''
    (ROOT / 'docs/任务三四实验结果.md').write_text(doc)
    print('Wrote docs/任务三四实验结果.md and results/task4/index.html')


if __name__ == '__main__':
    main()
