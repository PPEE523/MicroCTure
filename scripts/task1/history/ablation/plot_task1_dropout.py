#!/usr/bin/env python3
"""Plot an already completed and audited D2/D5 control; no fit or inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from microcture.task1_formal.protocol import _package_signature, sha256
from scripts.audit_task1_dropout import safe_report

CLASSES = ("OPCID", "CHIN", "CHID")
CONDITIONS = ("I3", "G0", "V0", "P0")
SEEDS, MARKERS = (42, 43, 44), ("o", "s", "^")
VARIANTS, COLORS = ("D2", "D5"), {"D2": "#2879a8", "D5": "#d36a36"}
CLASS_COLORS = ("#4f82a8", "#62a078", "#bd695f")
SHORT = ("I3\ncomplete", "G0\ngeometry", "V0\nvalidity", "P0\nprofile")
DIFFICULT = ("CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_report(report):
    integrity = read_json(report / "integrity.json")
    require(integrity["status"] == "passed" and integrity["verified_D5_fits"] == integrity["verified_D2_reference_fits"] == 60 and
            integrity["verified_complete_OOF"] == 24, "A complete120-fit/24-OOF audit is required")
    run = ROOT / integrity["provenance"]["D5"]["run_dir"]
    summary = read_json(run / "summary.json")
    require(summary["status"] == "complete" and summary["completed_jobs"] == 60 and summary["complete_oof"] == 12,
            "Do not plot partial scores")
    for variant in VARIANTS:
        provenance = integrity["provenance"][variant]
        source = ROOT / provenance["run_dir"]
        for name in ("manifest", "summary"):
            require(sha256(source / f"{name}.json") == provenance[f"{name}_sha256"], "Audited result bytes changed")
    require(_package_signature(ROOT, "task1_dropout_v1", "run_task1_dropout.py") == integrity["source_signature"], "Frozen D5 source changed")
    require(sha256(ROOT / "scripts/audit_task1_dropout.py") == integrity["audit_script_sha256"], "Audit implementation changed; recompute the audit")
    require(sha256(ROOT / "scripts/audit_task1_reevaluation_results.py") == integrity["independent_score_script_sha256"], "Independent CSV scorer changed")
    return {"integrity": integrity, "stats": read_json(report / "error_learning.json")["arms"],
        "changes": read_json(report / "error_learning.json")["changes"], "bootstrap": read_json(report / "group_bootstrap.json"),
        "cases": rows(report / "case_review.csv"), "learning": rows(report / "learning_per_job.csv"), "run": run}


def save(fig, output, name):
    import matplotlib.pyplot as plt
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def seed_points(ax, x, values, color, offset=.035):
    for dx, marker, value in zip((-offset, 0, offset), MARKERS, values):
        ax.scatter(x + dx, value, color=color, marker=marker, s=29, edgecolors="white", linewidths=.4, zorder=5)


def plot_performance(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric, title in ((axes[0, 0], "macro_f1", "A   Macro F1"), (axes[0, 1], "CHID_f1", "B   CHID F1")):
        for vi, variant in enumerate(VARIANTS):
            for ci, condition in enumerate(CONDITIONS):
                values = np.asarray(data["bootstrap"]["arms"][f"{variant}/{condition}"][metric]["seed_values"])
                x = ci + (vi - .5) * .30
                ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), color=COLORS[variant], fmt="_", ms=18, capsize=4, lw=1.5)
                seed_points(ax, x, values, COLORS[variant]);ax.text(x, .96 + vi * .045, f"{values.mean():.3f}", ha="center", fontsize=8, color=COLORS[variant])
        ax.set(xticks=range(4), xticklabels=SHORT, ylim=(0, 1.08), ylabel="Structure-level score", title=title)
        if metric == "macro_f1": ax.axhline(.90, color="#b53232", lw=1, ls="--", alpha=.6)
    for ax, metric, title in ((axes[1, 0], "precision", "C   I3 class precision"), (axes[1, 1], "recall", "D   I3 class recall")):
        for vi, variant in enumerate(VARIANTS):
            scores = data["stats"][f"{variant}/I3"]["scores"]
            for ci, label in enumerate(CLASSES):
                values = np.array([score["per_class"][label][metric] for score in scores]);x = ci + (vi - .5) * .3
                ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), color=COLORS[variant], fmt="_", ms=18, capsize=4, lw=1.5)
                seed_points(ax, x, values, COLORS[variant])
        ax.axhline(.85, color="#b53232", lw=1, ls="--", alpha=.6)
        ax.set(xticks=range(3), xticklabels=("OPCID (68)", "CHIN (250)", "CHID (26)"), ylim=(0, 1.04), ylabel="Structure-level score", title=title)
    for ax in axes.flat: ax.grid(axis="y", color="#ddd", lw=.5)
    handles = [Line2D([0], [0], color=COLORS[v], lw=2, label=f"{v}: global dropout p={'.2' if v=='D2' else '.5'}") for v in VARIANTS]
    handles += [Line2D([0], [0], color="#444", marker=m, lw=0, label=f"seed {s}") for m, s in zip(MARKERS, SEEDS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(.5, .015))
    fig.suptitle("Single-factor global dropout control: completed developmental OOF", fontsize=14)
    fig.text(.5, -.017, "Bars show mean ± optimization-seed sample SD, not biological confidence intervals. D2 is a sealed reference; no D2 refits.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.92, bottom=.12, hspace=.35, wspace=.22);save(fig, output, "performance")


def plot_contrasts(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ri, metric in enumerate(("macro_f1", "CHID_f1")):
        ax = axes[ri, 0]
        for i, condition in enumerate(CONDITIONS):
            row = data["bootstrap"]["pairs"][f"D5/{condition}-D2/{condition}"][metric]
            low, high = row["ci95"];ax.plot([low, high], [i, i], lw=3, color=COLORS["D5"], alpha=.55)
            ax.scatter(row["point_mean"], i, color=COLORS["D5"], marker="D", s=48, zorder=5)
            for j, (marker, value) in enumerate(zip(MARKERS, row["seed_deltas"])):
                ax.scatter(value, i + (j - 1) * .10, marker=marker, color="#444", s=20, zorder=6)
        ax.axvline(0, color="#555", lw=1);ax.set(yticks=range(4), yticklabels=CONDITIONS, xlabel="D5 minus D2", title=f"{'AC'[ri]}   {metric}: dropout effect")
        ax = axes[ri, 1]
        for vi, variant in enumerate(VARIANTS):
            for i, control in enumerate(("G0", "V0", "P0")):
                row = data["bootstrap"]["pairs"][f"{variant}/I3-{variant}/{control}"][metric]
                y = i + (vi - .5) * .22;low, high = row["ci95"]
                ax.plot([low, high], [y, y], lw=3, color=COLORS[variant], alpha=.6)
                ax.scatter(row["point_mean"], y, color=COLORS[variant], marker="D", s=40, label=variant if i == 0 else None)
        ax.axvline(0, color="#555", lw=1);ax.set(yticks=range(3), yticklabels=("I3 minus G0", "I3 minus V0", "I3 minus P0"), xlabel="Complete-input advantage", title=f"{'BD'[ri]}   {metric}: input dependence")
        ax.legend(frameon=False)
        for ax in axes[ri]: ax.grid(axis="x", color="#ddd", lw=.5);ax.invert_yaxis()
    fig.suptitle("Paired seed effects and conditional spatial-group uncertainty", fontsize=14)
    fig.text(.5, .003, "Diamonds: three-seed metric mean. Lines: 2,000 paired whole-group percentile95 intervals, conditional on saved adaptive developmental OOF.", ha="center", fontsize=9)
    fig.text(.5, -.019, "These descriptive intervals do not correct for prior researcher adaptation or multiple contrasts and do not establish biological generalization.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.10, wspace=.38, hspace=.36);save(fig, output, "paired_and_ablation")


def plot_learning(data, output):
    import matplotlib.pyplot as plt
    fields = (("weighted_train_ce", "A   Fixed inner-train CE", False), ("weighted_validation_ce", "B   Fixed inner-validation CE", False),
        ("gradient_head_mean", "C   Head gradient norm", True), ("gradient_encoder_mean", "D   Encoder gradient norm", True),
        ("train_head_effective_fraction", "E   Effective head activations", False), ("validation_centered_logit_variance", "F   Validation between-structure logits variance", True))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    for ax, (field, title, log) in zip(axes.flat, fields):
        for variant in VARIANTS:
            quantiles = np.asarray(data["stats"][f"{variant}/I3"]["trajectory_quantiles_25_50_75"][field], dtype=float)
            x = np.arange(41);mask = np.isfinite(quantiles).all(1)
            if log: mask &= (quantiles > 0).all(1)
            ax.fill_between(x[mask], quantiles[mask, 0], quantiles[mask, 2], color=COLORS[variant], alpha=.14)
            ax.plot(x[mask], quantiles[mask, 1], color=COLORS[variant], lw=2, label=variant)
        if log: ax.set_yscale("log")
        ax.set(title=title, xlabel="Epoch", xlim=(0, 40));ax.grid(color="#ddd", lw=.5, alpha=.6)
        if field == "train_head_effective_fraction": ax.set_ylim(0, 1.02)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("I3 learning trajectories: median and interquartile range across all15 fold/seed fits", fontsize=14)
    counts = "; ".join(f"{v} learning {data['stats'][v+'/I3']['learning']['strict_learning_passes']}/15; validation CE20→40 rising {data['stats'][v+'/I3']['validation_CE_rising_20_to40']}/15" for v in VARIANTS)
    fig.text(.5, .018, counts, ha="center", fontsize=10)
    fig.text(.5, -.008, "All endpoints use epoch40. Epoch0 gradient is unavailable; positive gradients are not proof of biological morphology recognition.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.12, hspace=.33, wspace=.27);save(fig, output, "learning")


def plot_confusion(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.2))
    for vi, variant in enumerate(VARIANTS):
        for si, seed in enumerate(SEEDS):
            cm = np.asarray(data["stats"][f"{variant}/I3"]["predictions"][si]["confusion_matrix"])
            row_fraction = cm / cm.sum(1, keepdims=True)
            ax = axes[vi, si];ax.imshow(row_fraction, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i,j]}\n{row_fraction[i,j]*100:.1f}%", ha="center", va="center", color="white" if row_fraction[i,j] > .58 else "#222", fontsize=10)
            ax.set(xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES, xlabel="Predicted label", ylabel="Annotated label", title=f"{variant} / seed{seed} / n=344")
    fig.suptitle("I3 confusion for every seed: counts and row-normalized percentages", fontsize=14)
    fig.text(.5, -.005, "Each panel contains344 unique annotated structures. Two WT probabilities average before argmax; seeds are shown separately, never pooled as independent cases.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.91, hspace=.38, wspace=.24, bottom=.08);save(fig, output, "confusion")


def natural_id(sid):
    label, number = sid.rsplit("_", 1)
    return CLASSES.index(label), int(number)


def plot_cases(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch, Rectangle
    cases = {v: {r["structure_id"]: r for r in data["cases"] if r["arm"] == f"{v}/I3"} for v in VARIANTS}
    require(all(len(v) == 344 for v in cases.values()), "Every I3 case must be audited")
    chids = sorted([sid for sid, r in cases["D2"].items() if r["type"] == "CHID"], key=natural_id)
    false_positive = set().union(*(set(data["stats"][f"{v}/I3"]["stable_FP_ids"]) for v in VARIANTS))
    other = sorted(false_positive | {"CHIN_51", "OPCID_66"}, key=natural_id)
    ids = chids + other;require(len(chids) == 26, "All26 annotated CHID must be displayed")
    fig, ax = plt.subplots(figsize=(12.7, 2.9 + len(ids) * .34))
    status = {}
    for y, sid in enumerate(ids):
        truth = cases["D2"][sid]["type"]
        ax.text(-.7, y, ("* " if sid in DIFFICULT else "") + sid, ha="right", va="center", fontsize=9)
        for vi, variant in enumerate(VARIANTS):
            row = cases[variant][sid]
            for si, seed in enumerate(SEEDS):
                x = vi * 3.35 + si;pred = row[f"seed_{seed}_prediction"];wrong = pred != truth
                ax.add_patch(Rectangle((x - .47, y - .44), .94, .87, facecolor=to_rgba(CLASS_COLORS[CLASSES.index(pred)], .28), edgecolor="#d0d0d0", lw=.4))
                ax.text(x, y - .10, pred, ha="center", va="center", fontsize=8.3, fontweight="bold" if wrong else "normal")
                ax.text(x, y + .20, f"pCID {float(row[f'seed_{seed}_p_CHID']):.2f}", ha="center", va="center", fontsize=6.7, color="#555")
        counts = [int(cases[v][sid]["correct_seed_count"]) for v in VARIANTS]
        outcome = "stable wrong" if counts[1] == 0 else "all correct" if counts[1] == 3 else "seed dependent"
        status[sid] = {"D2_correct_seeds": counts[0], "D5_correct_seeds": counts[1]}
        ax.text(6.18, y, f"{counts[0]}/3 → {counts[1]}/3  {outcome}", ha="left", va="center", fontsize=8, color="#ab3535" if counts[1] == 0 else "#444")
    for vi, variant in enumerate(VARIANTS):
        for si, seed in enumerate(SEEDS): ax.text(vi * 3.35 + si, -1.0, f"seed{seed}", ha="center", fontsize=9)
        ax.text(vi * 3.35 + 1, -1.65, f"{variant}: dropout p={'.2' if variant=='D2' else '.5'}", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.18, -1., "Correct seeds / D5 status", fontsize=9)
    ax.axhline(25.55, color="#777", lw=.8)
    ax.set(xlim=(-2.35, 9.1), ylim=(len(ids) + .4, -2.2));ax.axis("off")
    fig.suptitle("Every CHID plus stable-CHID-FP union and preregistered historical anchors", fontsize=13, y=.997)
    fig.legend(handles=[Patch(facecolor=to_rgba(c, .28), label=f"Predicted {label}") for c, label in zip(CLASS_COLORS, CLASSES)], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .026))
    fig.text(.5, .010, "Bold labels are wrong relative to annotation; * marks the six historical difficult CHID. pCID is an uncalibrated model score, not biological certainty.", ha="center", fontsize=8)
    fig.subplots_adjust(top=.95, bottom=.07, left=.05, right=.99);save(fig, output, "chid_cases")
    return {"all26_CHID_shown": True, "displayed_ids": ids, "selection": "All CHID; stable CHID false-positive union across D2/D5; fixed CHIN51/OPCID66 historical anchors", "correct_seed_counts": status}


def write_analysis(data, output):
    stats, boot = data["stats"], data["bootstrap"]
    first, second, paired = stats["D2/I3"], stats["D5/I3"], boot["pairs"]["D5/I3-D2/I3"]
    val2, val5 = [stats[f"{v}/I3"]["trajectory_quantiles_25_50_75"]["weighted_validation_ce"][40][1] for v in VARIANTS]
    lines = ["# 单因素 Dropout 结果分析", "", "本轮只把完整 ROI100 M2+/H3 的两个既有 Dropout 同步从0.2改为0.5，新增60次 D5；D2 的60次旧 R1 结果仅只读引用。两组固定空间折、三种子、初始化、批序、类别权重、输入预处理及40轮末轮权重一致。", "", "此数据已被多轮开发使用，以下是开发性复评；没有独立样本和生物专家形态真值，不能据此宣称已可靠识别生物学形态。", "", "## 本轮判定", "",
        f"D5 完整输入 macro-F1 相对D2改变{paired['macro_f1']['point_mean']:+.4f}，CHID-F1改变{paired['CHID_f1']['point_mean']:+.4f}。"
        f"两组均{'通过' if first['development_gate_pass'] and second['development_gate_pass'] else '未同时通过'}预登记开发门槛。"
        f"末轮验证CE中位数为{val2:.5f}→{val5:.5f}，末段反弹{first['validation_CE_rising_20_to40']}/15→{second['validation_CE_rising_20_to40']}/15；"
        f"稳定错误{len(first['stable_wrong_ids'])}→{len(second['stable_wrong_ids'])}，稳定CHID漏检{len(first['stable_FN_ids'])}→{len(second['stable_FN_ids'])}，稳定CHID误报{len(first['stable_FP_ids'])}→{len(second['stable_FP_ids'])}。"]
    if paired['macro_f1']['ci95'][0] <= 0 <= paired['macro_f1']['ci95'][1] and paired['CHID_f1']['ci95'][0] <= 0 <= paired['CHID_f1']['ci95'][1]:
        lines += ["", "macro-F1与CHID-F1的成对条件性95%区间均跨0，有限平均收益不足以证明稳定改善。"]
    if val5 >= val2 and second['validation_CE_rising_20_to40'] >= first['validation_CE_rising_20_to40']:
        lines += ["", "增强Dropout没有改善本轮末轮验证loss和末段反弹，因此“过拟合已被解决”的假说未获支持。"]
    if not second['development_gate_pass'] and len(second['stable_wrong_ids']) >= len(first['stable_wrong_ids']):
        lines += ["", "建议本轮不将p=0.5替换为已验证的默认方案：保留D2作为冻结参考，D5作为完整的受控实验结果。下一步应直接处理残留系统性错误和空间形态证据，不能根据这次平均F1上升宣称结构形态可靠。"]
    lines += ["", "## 完整输入与控制", "", "| 条件 | D2 macro-F1 | D5 macro-F1 | D2 CHID-F1 | D5 CHID-F1 | Δmacro-F1 [条件性95%区间] | ΔCHID-F1 [条件性95%区间] |", "| - | -: | -: | -: | -: | -: | -: |"]
    for c in CONDITIONS:
        arm2, arm5, pair = boot["arms"][f"D2/{c}"], boot["arms"][f"D5/{c}"], boot["pairs"][f"D5/{c}-D2/{c}"]
        def ci(metric):
            r = pair[metric];return f"{r['point_mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        lines.append(f"| {c} | {arm2['macro_f1']['point_mean']:.4f} | {arm5['macro_f1']['point_mean']:.4f} | {arm2['CHID_f1']['point_mean']:.4f} | {arm5['CHID_f1']['point_mean']:.4f} | {ci('macro_f1')} | {ci('CHID_f1')} |")
    lines += ["", "区间对49个空间组做2,000次成对重采样；每次先分别计算各seed指标，再对seed取均值。它只描述已保存OOF的条件性不确定性，未校正既往开发适应或多重比较；种子标准差也不是生物学置信区间。G0保留几何/长度，V0额外保留有效覆盖，P0只保留距离轮廓，因此完整输入高于这些控制才提供额外形态信息的有限证据。", "", "## 持续学习与过拟合", "", "| I3 指标（各15次fit） | D2 | D5 |", "| - | -: | -: |"]
    for title, field in (("持续学习通过", "learning"), ("各训练轮有效梯度", "effective_gradients_all_epochs_count"), ("验证CE20→40反弹", "validation_CE_rising_20_to40"), ("外层单一argmax类别", "outer_single_argmax_class_count"), ("外层常数概率", "outer_constant_probabilities_count")):
        vals = [stats[f"{v}/I3"][field]["strict_learning_passes"] if field == "learning" else stats[f"{v}/I3"][field] for v in VARIANTS]
        lines.append(f"| {title} | {vals[0]}/15 | {vals[1]}/15 |")
    for title, field in (("第40轮训练CE中位数", "weighted_train_ce"), ("第40轮验证CE中位数", "weighted_validation_ce")):
        vals = [stats[f"{v}/I3"]["trajectory_quantiles_25_50_75"][field][40][1] for v in VARIANTS]
        lines.append(f"| {title} | {vals[0]:.5f} | {vals[1]:.5f} |")
    vals = [stats[f"{v}/I3"]["train40_CE_sample_sd"] for v in VARIANTS]
    lines.append(f"| 第40轮训练CE跨fit样本标准差 | {vals[0]:.5f} | {vals[1]:.5f} |")
    for v in VARIANTS:
        arm = stats[f"{v}/I3"]
        lines.append(f"\n{v} 完整输入开发门槛：{'通过' if arm['development_gate_pass'] else '未通过'}。预登记要求每个seed macro-F1≥0.90、各类precision/recall≥0.85且15次持续学习全部通过。")
    lines += ["", "## 最终预测分布与系统性错误", "", "每个seed均包含344个唯一标注结构；空间相邻结构和seed之间并不独立，不能把表格计数相加当作新样本量。", "", "| I3 | seed | macro-F1 | CHID precision / recall / F1 | CHID TP / FP / FN | 预测OPCID / CHIN / CHID数 |", "| - | -: | -: | -: | -: | -: |"]
    for v in VARIANTS:
        for si, seed in enumerate(SEEDS):
            score = stats[f"{v}/I3"]["scores"][si];p = stats[f"{v}/I3"]["predictions"][si];cid = score["per_class"]["CHID"]
            distribution = " / ".join(str(p["predicted_counts"][c]) for c in CLASSES)
            lines.append(f"| {v} | {seed} | {score['macro_f1']:.4f} | {cid['precision']:.4f} / {cid['recall']:.4f} / {cid['f1']:.4f} | {p['CHID_TP']} / {p['CHID_FP']} / {p['CHID_FN']} | {distribution} |")
    lines += ["", "| I3 跨三seed指标 | D2 | D5 |", "| - | -: | -: |"]
    for title, field in (("三seed都错", "stable_wrong_ids"), ("CHID三seed漏检", "stable_FN_ids"), ("非CHID三seed都误报CHID", "stable_FP_ids")):
        lines.append(f"| {title} | {len(stats['D2/I3'][field])} | {len(stats['D5/I3'][field])} |")
    lines += ["", f"不再稳定错的旧案例：{', '.join(data['changes']['removed_stable_errors']) or '无'}。其中三seed全部修复：{', '.join(data['changes']['fully_repaired_all_seeds']) or '无'}。", "", f"新增三seed稳定错误：{', '.join(data['changes']['added_stable_errors']) or '无'}。", "", "六个历史困难CHID均保留，未按结果筛选：", "", "| 案例 | D2 三seed预测 | D5 三seed预测 |", "| - | - | - |"]
    cases = {v: {r["structure_id"]: r for r in data["cases"] if r["arm"] == f"{v}/I3"} for v in VARIANTS}
    for sid in DIFFICULT:
        labels = [" / ".join(cases[v][sid][f"seed_{s}_prediction"] for s in SEEDS) for v in VARIANTS]
        lines.append(f"| {sid} | {labels[0]} | {labels[1]} |")
    lines += ["", "全部26个CHID与两组稳定CHID误报并集见下方病例图；混淆矩阵按seed分开显示，避免将优化种子当作独立生物样本。图中概率只是模型分数，没有作生物学校准。", "", "Dropout实验只检验一个正则化假说。是否缓解过拟合，应结合末轮验证CE、反弹次数、配对OOF差值及新增错误一起判断；单独训练loss升高或某个种子高分不构成通过。后续若采用此改动重新比较骨干，必须统一协议重跑，不能与另一分辨率或分类头的旧分数混合。", "", "## 结果图", ""]
    for name, title, caption in (
            ("performance", "完整输入、消融与逐类表现", "点形状区分seed，误差条为三seed样本标准差，虚线为预登记开发门槛。"),
            ("paired_and_ablation", "配对Dropout收益与输入依赖", "菱形为三seed均值，线段为成对空间组bootstrap条件性95%区间；跨0的区间不能作为稳定提升证据。"),
            ("learning", "持续学习与过拟合轨迹", "曲线为15次fold/seed的中位数，阴影为四分位范围；所有终点固定为第40轮。"),
            ("confusion", "每个seed的混淆矩阵", "每个面板分别包含344个唯一标注结构；格内为计数和按真实类归一化的百分比。"),
            ("chid_cases", "全部CHID与系统性误报", "展示全部26个CHID、D2/D5稳定CHID误报并集及预先固定的历史锚点；粗体为相对标注错误的预测，星号为六个历史困难案例。")):
        lines += [f"### {title}", "", f"![{title}]({name}.png)", "", f"{caption} [下载PDF]({name}.pdf)。", ""]
    lines += ["## 核验与数值文件", "",
        "[完整性与配对审计](integrity.json) · [各条件错误与学习指标](error_learning.json) · [成对空间组bootstrap](group_bootstrap.json)", "",
        "[逐fit学习记录 CSV](learning_per_job.csv) · [逐结构、逐seed预测 CSV](case_review.csv) · [图表来源与SHA核验](plot_summary.json)", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/task1_dropout_review_v1")
    args = parser.parse_args();report = args.report_dir.resolve();data = load_report(report)
    safe_report(report, data["run"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    plot_performance(data, report);plot_contrasts(data, report);plot_learning(data, report);plot_confusion(data, report)
    case_summary = plot_cases(data, report);write_analysis(data, report)
    figure_files = [f"{name}.{ext}" for name in ("performance", "paired_and_ablation", "learning", "confusion", "chid_cases") for ext in ("png", "pdf")]
    result = {"status": "complete", "fit_reference_new": [60, 60], "complete_oof": 24, "n_structures_per_oof": 344,
        "plot_script_sha256": sha256(Path(__file__)), "cases": case_summary,
        "audit_input_sha256": {name: sha256(report / name) for name in ("integrity.json", "error_learning.json", "group_bootstrap.json", "case_review.csv", "learning_per_job.csv")},
        "figure_sha256": {name: sha256(report / name) for name in figure_files},
        "analysis_sha256": sha256(report / "analysis.md"), "scope": "Saved completed adaptive developmental OOF; not independent biological validation"}
    (report / "plot_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": figure_files, "analysis": str(report / "analysis.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
