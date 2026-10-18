#!/usr/bin/env python3
"""Scientific native-detail figures from the completed read-only audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.audit_task1_native_roi import ARMS, SEEDS, CLASSES, PAIRS, require, read_json, safe_report, sha256, json_signature, _guard_run

NAMES = tuple(f"{a}/{c}" for a, c in ARMS)
COLORS = ("#2878b5", "#d65d35", "#8c8c8c", "#7e57a4", "#3e9172", "#b58c26")
LABELS = ("C0 complete", "C1 native detail", "Shared geometry", "Shared validity", "C0 profile", "C1 profile")
MARKERS = ("o", "s", "^")


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_audit_snapshot(run, report):
    """Verify that figures describe the same sealed bytes that were audited."""
    _guard_run(run)
    integrity = read_json(report / "integrity.json")
    require((ROOT / integrity["run_dir"]).resolve() == run.resolve(), "Figures refer to a different scientific run")
    require(integrity["status"] == "passed" and integrity["verified_fit_indices"] == 90 and
            integrity["verified_complete_OOF"] == 18, "Completed native90/18 audit required")
    for filename, digest in integrity["sources"].items():
        require(sha256(run / filename) == digest, "Manifest/summary changed after audit")
    inventory = integrity["scientific_inventory"]
    require(json_signature(inventory) == integrity["inventory_signature"], "Audited inventory signature changed")
    for record in inventory:
        path = run / record["path"]
        require(not path.is_symlink() and path.stat().st_size == record["size_bytes"] and sha256(path) == record["sha256"],
                f"Scientific bytes changed after audit: {path}")
    result = read_json(report / "error_learning.json")
    bootstrap = read_json(report / "group_bootstrap.json")
    for name in NAMES:
        source = result["arms"][name]["scores"]
        require(len(source) == 3, "Three complete endpoint seeds required")
        for metric in ("macro_f1", "CHID_f1"):
            values = [s["macro_f1"] if metric == "macro_f1" else s["per_class"]["CHID"]["f1"] for s in source]
            require(np.allclose(values, bootstrap["arms"][name][metric]["seed_values"], atol=1e-12, rtol=0), "Bootstrap/data seed values differ")
    return integrity, result, bootstrap, csv_rows(report / "case_review.csv")


def save(figure, output, name):
    import matplotlib.pyplot as plt
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=210 if extension == "png" else None, bbox_inches="tight")
    plt.close(figure)


def seed_points(axis, positions, values, color, width=.09):
    for displacement, value, marker in zip((-width, 0, width), values, MARKERS):
        axis.scatter(positions + displacement, value, marker=marker, s=30, color=color, edgecolors="white", linewidths=.4, zorder=4)


def performance(result, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for index, name in enumerate(NAMES):
        values = np.array([s["macro_f1"] for s in result["arms"][name]["scores"]])
        axes[0, 0].errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="_", markersize=21, capsize=4, color=COLORS[index])
        seed_points(axes[0, 0], index, values, COLORS[index])
    axes[0, 0].set(xticks=range(6), xticklabels=("C0\ncomplete", "C1\nnative", "G0\nshared", "V0\nshared", "C0\nprofile", "C1\nprofile"), ylabel="OOF macro F1", title="A   Complete spatial-group OOF")
    axes[0, 0].axhline(.9, color="#ad3434", linestyle="--", linewidth=.8)
    for ci, label in enumerate(CLASSES):
        axis = axes.flat[ci + 1]
        for index, name in enumerate(NAMES):
            position = np.arange(3) + (index - 2.5) * .125
            values = np.array([[s["per_class"][label][metric] for s in result["arms"][name]["scores"]]
                               for metric in ("precision", "recall", "f1")])
            axis.errorbar(position, values.mean(1), yerr=values.std(1, ddof=1), color=COLORS[index], fmt="_", markersize=10, capsize=2)
            for pi, v in zip(position, values):
                seed_points(axis, pi, v, COLORS[index], width=.023)
        axis.set(xticks=range(3), xticklabels=("Precision", "Recall", "F1"), title=f"{'BCD'[ci]}   {label}")
        axis.axhline(.85, color="#ad3434", linestyle="--", linewidth=.8)
    for axis in axes.flat:
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=.2)
    fig.legend(handles=[Line2D([], [], color=c, label=l, lw=2) for c, l in zip(COLORS, LABELS)],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Matched native detail: developmental scores, means ± sample SD across seeds", fontsize=14)
    fig.text(.5, .065, "Shared C0 G0/V0 controls are identical for both alignments. Dashed lines are prospective development gates, not biological certification.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=.18, hspace=.32, wspace=.22)
    save(fig, output, "performance")


def paired_effects(bootstrap, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    names = [f"{first}-{second}" for first, second in PAIRS]
    for axis, metric in zip(axes, ("macro_f1", "CHID_f1")):
        for index, name in enumerate(names):
            value = bootstrap["pairs"][name][metric]
            low, high = np.array(value["ci95"]) * 100
            axis.plot([low, high], [index, index], color="#555555", lw=2)
            axis.scatter(value["point_mean"] * 100, index, color="#222222", marker="|", s=180, zorder=5)
            for offset, change, marker in zip((-.14, 0, .14), value["seed_deltas"], MARKERS):
                axis.scatter(change * 100, index + offset, color=COLORS[1], marker=marker, s=32, zorder=4)
        axis.axvline(0, color="#999999", ls="--", lw=.8)
        axis.set(xlabel="Difference in percentage points", title="Macro F1" if metric == "macro_f1" else "CHID F1",
                 yticks=range(len(names)), yticklabels=names)
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Paired saved-OOF contrasts: 2,000 draws of whole spatial groups", fontsize=14)
    fig.text(.5, .015, "Bars: conditional unadjusted percentile95 intervals; dots: three endpoint seeds. No refitting or correction for adaptive dataset reuse.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, output, "paired_effects")


def errors(result, cases, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(15, 12))
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.8), wspace=.32, hspace=.24)
    for index, name in enumerate(("C0/I3", "C1/I3")):
        axis = fig.add_subplot(grid[0, index])
        cm = np.mean([r["confusion_matrix"] for r in result["arms"][name]["predictions"]], axis=0)
        axis.imshow(cm, cmap="Blues", vmin=0)
        for i, j in np.ndindex(3, 3):
            axis.text(j, i, f"{cm[i,j]:.1f}", ha="center", va="center", color="white" if cm[i,j] > cm.max()*.5 else "black")
        axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                 xlabel="Predicted", ylabel="Annotated", title=f"{name}: mean count per seed")
    axis = fig.add_subplot(grid[0, 2])
    for index, name in enumerate(NAMES):
        total = 0
        for ci, label in enumerate(CLASSES):
            count = np.mean([r["predicted_counts"][label] for r in result["arms"][name]["predictions"]])
            axis.bar(index, count, bottom=total, color=("#508bb8", "#df9c42", "#63a674")[ci])
            total += count
    axis.set(xticks=range(6), xticklabels=NAMES, ylabel="Mean predicted structures per seed", title="Final prediction distribution", ylim=(0, 344))
    axis.tick_params(axis="x", rotation=55)
    axis = fig.add_subplot(grid[1, :2])
    chosen = ("C0/I3", "C1/I3", "C0/P0", "C1/P0")
    by_id = {(r["arm"], r["structure_id"]): r for r in cases}
    ids = sorted([r["structure_id"] for r in cases if r["arm"] == "C0/I3" and r["type"] == "CHID"], key=lambda s: int(s.split("_")[-1]))
    require(len(ids) == 26, "All26 CHID cases must be shown")
    image = np.array([[CLASSES.index(by_id[name, sid][f"seed_{seed}_prediction"]) for name in chosen for seed in SEEDS] for sid in ids])
    axis.imshow(image, interpolation="nearest", cmap=ListedColormap(("#508bb8", "#df9c42", "#63a674")), vmin=0, vmax=2, aspect="auto")
    axis.set(yticks=range(26), yticklabels=ids, xticks=range(12),
             xticklabels=[f"{name}\ns{seed}" for name in chosen for seed in SEEDS], title="Every annotated CHID: final class, all seeds")
    axis.tick_params(labelsize=8)
    for boundary in (2.5, 5.5, 8.5): axis.axvline(boundary, color="white", lw=2)
    axis.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(("#508bb8", "#df9c42", "#63a674"), CLASSES)], loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3, frameon=False)
    axis = fig.add_subplot(grid[1, 2])
    axis.barh(range(6), [len(result["arms"][name]["stable_wrong_ids"]) for name in NAMES], color=COLORS)
    axis.set(yticks=range(6), yticklabels=NAMES, xlabel="Unique structures wrong in every seed", title="Persistent errors, all344 structures")
    axis.invert_yaxis()
    fig.suptitle("Matched native-detail classification errors; two WT probabilities averaged per structure", fontsize=14)
    fig.subplots_adjust(bottom=.12, top=.94)
    save(fig, output, "predictions_and_CHID")


def learning(result, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fields = ("weighted_train_ce", "weighted_validation_ce", "gradient_head_mean", "train_head_effective_fraction", "validation_centered_logit_variance")
    titles = ("Fixed training CE", "Inner-validation CE", "Head gradient norm", "Training head active fraction", "Between-structure centered logits variance")
    for axis, field, title in zip(axes.flat, fields, titles):
        for index, name in enumerate(NAMES):
            values = np.asarray(result["arms"][name]["trajectory_quantiles_25_50_75"][field], dtype=float)
            epochs = np.arange(41)
            axis.plot(epochs, values[:, 1], color=COLORS[index], lw=1.4)
            axis.fill_between(epochs, values[:, 0], values[:, 2], color=COLORS[index], alpha=.09)
        axis.set(xlabel="Epoch", title=title)
        if field in ("gradient_head_mean", "validation_centered_logit_variance"): axis.set_yscale("log")
        if field == "train_head_effective_fraction": axis.set_ylim(0, 1.02)
        axis.grid(alpha=.2)
    axis = axes.flat[-1]
    for index, name in enumerate(NAMES):
        axis.bar(index, result["arms"][name]["learning"]["strict_learning_passes"], color=COLORS[index])
    axis.set(xticks=range(6), xticklabels=NAMES, ylim=(0, 15.8), ylabel="Accepted fits /15", title="Full-trajectory continuous-learning acceptance")
    axis.tick_params(axis="x", rotation=45)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=c, label=l) for c, l in zip(COLORS, LABELS)], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Native-detail optimization diagnostics: median and IQR across15 fold/seed fits", fontsize=14)
    fig.subplots_adjust(bottom=.15, hspace=.35, wspace=.28)
    save(fig, output, "learning")


def write_analysis(run, report, integrity, result, bootstrap):
    complete_delta = bootstrap["pairs"]["C1/I3-C0/I3"]
    macro_delta, chid_delta = complete_delta["macro_f1"], complete_delta["CHID_f1"]
    lines = ["# 原生10bp细节受控实验：完整结果复核", "", f"**完整输入的native细节增量仍未证实。** C1−C0的macro-F1平均差值为{macro_delta['point_mean']*100:+.2f}个百分点，空间组条件95%区间[{macro_delta['ci95'][0]*100:+.2f}, {macro_delta['ci95'][1]*100:+.2f}]；CHID F1为{chid_delta['point_mean']*100:+.2f}个百分点，区间[{chid_delta['ci95'][0]*100:+.2f}, {chid_delta['ci95'][1]*100:+.2f}]。两项区间均跨0，CHID仍有持续系统性错误，不能据此声称已可靠识别三类生物学形态。", "", "本报告只分析已完成的90项训练及18份完整OOF。C0与C1共享E10、物理距离≥200bp支持、validity/mask、原25.6kb上下文及长度；C0先投影成100bp cell均值，C1保留native细节。所有模型仍为M2+/H3、1e-4和40轮末轮。", "",
             "旧R1的E100与距离支持不同，只作历史参照；本轮判断来自C1−C0。全部结果是反复检查过的开发数据，不能作为新盲测或专家生物学真值。", "", "## 性能与共享控制", "", "| 输入 | macro-F1 | CHID precision | CHID recall | CHID F1 | 持续学习通过 |", "|---|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        score = result["arms"][name]["scores"]
        values = [[s["macro_f1"] for s in score]] + [[s["per_class"]["CHID"][m] for s in score] for m in ("precision", "recall", "f1")]
        display = [f"{np.mean(v):.4f}±{np.std(v,ddof=1):.4f}" for v in values]
        lines.append(f"| {name} | " + " | ".join(display) + f" | {result['arms'][name]['learning']['strict_learning_passes']}/15 |")
    lines += ["", "±是三个优化seed的样本SD，不是生物学置信区间。G0只保留几何及长度，V0额外保留有效性，P0保留按模型网格距离的有效性加权信号均值；P0不是无信号基线。", "", "![性能](performance.png)", "", "## 配对效应及空间组不确定性", "", "| 对比 | macro-F1差值（百分点） | CHID F1差值（百分点） |", "|---|---:|---:|"]
    for first, second in PAIRS:
        name = f"{first}-{second}"
        text = []
        for metric in ("macro_f1", "CHID_f1"):
            value = bootstrap["pairs"][name][metric]
            text.append(f"{value['point_mean']*100:+.2f} [{value['ci95'][0]*100:+.2f}, {value['ci95'][1]*100:+.2f}]")
        lines.append(f"| {name} | {' | '.join(text)} |")
    lines += ["", "区间由49个空间组整组配对重采样2,000次产生，同次抽样用于所有输入和seed。它们是保存OOF预测的条件性、未作多重比较校正的描述区间，不含重新训练不确定性或反复调开发数据的选择偏差。", "", "![配对效应](paired_effects.png)", "", "## 稳定错误和最终预测分布", "", "| 输入 | 全seed持续误分类数 | 持续CHID漏检 | 持续CHID误报 |", "|---|---:|---|---|"]
    for name in NAMES:
        item = result["arms"][name]
        lines.append(f"| {name} | {len(item['stable_wrong_ids'])} | {', '.join(item['stable_FN_ids']) or '无'} | {', '.join(item['stable_FP_ids']) or '无'} |")
    changes = result["changes"]
    lines += ["", f"C0→C1新增稳定错误：{', '.join(changes['added_stable_errors']) or '无'}。旧稳定错误在C1全部seed修复：{', '.join(changes['fully_repaired_all_seeds']) or '无'}。从稳定错误名单移出不等于所有seed修复。", "", "![全部26个CHID、混淆和预测分布](predictions_and_CHID.png)", "", "全部344例、六输入的逐seed结果见[case_review.csv](case_review.csv)。", "", "## 持续学习与稳定性", "", "| 输入 | 末轮train CE均值 | 末轮CE跨fit SD | 有效head梯度fit | outer常数概率fit | outer单类argmax fit | 验证CE 20→40回升fit |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item = result["arms"][name]
        learn = item["learning"]
        event = learn["prediction_events"]["outer"]
        lines.append(f"| {name} | {learn['last_train_ce']['mean']:.5f} | {item['train40_CE_sample_sd']:.5f} | {learn['effective_gradients']['head']['effective_cases']}/15 | {event['constant_probability']['final_cases']}/15 | {event['single_class_argmax']['final_cases']}/15 | {item['validation_CE_rising_20_to40']}/15 |")
    passed = [name for name in NAMES if result["arms"][name]["development_gate_pass"]]
    lines += ["", "![学习轨迹](learning.png)", "", "常数概率与单类argmax分别统计；同类argmax不自动等于网络失活。早期、末轮及末10轮事件见[error_learning.json](error_learning.json)，每fit损失和梯度见[learning_per_job.csv](learning_per_job.csv)。", "", "## 证据与结论边界", "", f"预先登记的数值门槛要求每个seed的macro-F1≥0.90、各类precision/recall均≥0.85且15/15通过持续学习。本轮通过输入：{', '.join(passed) or '无'}。这个门槛本身不等同生物学形态可靠。", "", f"已核对{integrity['indexed_scientific_files']}个科学文件、{integrity['indexed_bytes']:,}字节，15组fold/seed的六路初始化、minibatch顺序、类别权重配对一致；五个E10拟合的机会数从inner-train空间几何独立复算。共享C0/G0的15套权重张量、数值轨迹（除耗时）与预测CSV均精确复现旧R1/G0，用作环境切换后的重复性对照；V0支持已改变，不作旧V0精确复现要求。科学文件封存、双WT概率均值、OOF合并及全部数值验收见[integrity.json](integrity.json)。", "", "本轮可以判断保留10bp内部排列是否在同场控制下提供增量，不能由分数或敏感性单独证明已可靠识别生物学形态。原标注、全部实例和失败学习模型保持，不修改标签或筛选容易案例。没有新独立样本和专家复核，本轮仍是开发性证据。", ""]
    (report / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_native_roi_v2")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/task1_native_roi_review_v1")
    args = parser.parse_args()
    report = safe_report(args.report_dir, args.run_dir.resolve())
    integrity, result, bootstrap, cases = verify_audit_snapshot(args.run_dir, report)
    previous = os.environ.get("MPLCONFIGDIR")
    try:
        with tempfile.TemporaryDirectory(prefix="microcture-native-review-mpl-", dir="/tmp") as cache:
            os.environ["MPLCONFIGDIR"] = cache
            import matplotlib
            matplotlib.use("Agg")
            performance(result, report)
            paired_effects(bootstrap, report)
            errors(result, cases, report)
            learning(result, report)
    finally:
        if previous is None: os.environ.pop("MPLCONFIGDIR", None)
        else: os.environ["MPLCONFIGDIR"] = previous
    write_analysis(args.run_dir, report, integrity, result, bootstrap)
    names = ("performance", "paired_effects", "predictions_and_CHID", "learning")
    summary = {"status": "complete", "experiment_signature": integrity["experiment_signature"],
        "plot_script_sha256": sha256(Path(__file__)), "audit_inventory_signature": integrity["inventory_signature"],
        "sources": {name: sha256(report / name) for name in ("integrity.json", "error_learning.json", "group_bootstrap.json", "case_review.csv")},
        "figures": {f"{name}.{ext}": sha256(report / f"{name}.{ext}") for name in names for ext in ("png", "pdf")}}
    (report / "figures_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": 8, "report": str(report / "analysis.md")}))


if __name__ == "__main__":
    main()
