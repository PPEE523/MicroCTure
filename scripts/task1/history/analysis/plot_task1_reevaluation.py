#!/usr/bin/env python3
"""Verified plots and Chinese report of frozen paired checkpoint policies."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_task1_reevaluation_results import (BACKBONES, CLASSES, DIFFICULT, audit,
    csv_rows, read_json, require)
from microcture.task1_formal.protocol import sha256

MODEL_LABELS = {"m2": "M2", "m2plus": "M2+", "cnn_transformer": "CNN-Transformer",
                "pca_lr": "PCA + LR", "length_only": "Length", "mask_only": "Mask", "mask_length": "Mask + length"}
POLICIES = ("final40", "inner_ce")
POLICY_LABELS = ("Final epoch 40", "Minimum inner-val CE")
COLORS = ("#2373a4", "#cc6c3f")


def save(fig, output, name):
    import matplotlib.pyplot as plt
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=175 if suffix == "png" else None, bbox_inches="tight")
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    plt.close(fig)
    return {"name": name, "files": files}


def performance(data, output):
    import matplotlib.pyplot as plt
    a = data["audit"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    metrics = ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")
    labels = ("Macro F1", "CHID precision", "CHID recall", "CHID F1")
    x = np.arange(3)
    for ax, metric, label in zip(axes.flat, metrics, labels):
        for p, (policy, color) in enumerate(zip(POLICIES, COLORS)):
            values = [a["comparison"][f"{policy}/I3/{model}"][metric] for model in BACKBONES]
            means = [v["mean"] for v in values]
            ax.bar(x + (p - .5) * .32, means, width=.32,
                   yerr=[v["seed_sample_sd"] for v in values], capsize=4, color=color, label=POLICY_LABELS[p])
            for xx, yy, value in zip(x + (p - .5) * .32, means, values):
                ax.text(xx, yy + value["seed_sample_sd"] + .017, f"{yy:.3f}", ha="center", fontsize=9)
        for name, color, style in (("pca_lr", "#858585", "--"), ("length_only", "#8d9e77", ":")):
            ax.axhline(a["comparison"][f"baseline/I3/{name}"][metric]["mean"], color=color, ls=style,
                       label=MODEL_LABELS[name] + " (one seed)")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in BACKBONES])
        ax.set_ylim(0, 1.07); ax.set_title(label); ax.grid(axis="y", alpha=.18)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("I3 unified reevaluation: one full-40 fit, two endpoint policies\n"
                 "344 structures per seed; mean ± sample SD of 3 seeds (not a confidence interval)", fontsize=12)
    return save(fig, output, "performance")


def ablations(data, output):
    import matplotlib.pyplot as plt
    a = data["audit"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8), constrained_layout=True)
    for row, policy in enumerate(POLICIES):
        for col, (metric, label) in enumerate((("macro_f1", "Macro F1"), ("CHID_f1", "CHID F1"))):
            ax = axes[row, col]
            values = np.array([[a["comparison"][f"{policy}/{condition}/{model}"][metric]["mean"]
                                for condition in ("I0", "G0", "I3")] for model in BACKBONES])
            im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            for i in range(3):
                for j in range(3): ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center",
                                           color="white" if values[i,j] > .65 else "#222222")
            ax.set_xticks(range(3), ["I0: image", "G0: geometry", "I3: full"])
            ax.set_yticks(range(3), [MODEL_LABELS[m] for m in BACKBONES]); ax.set_title(POLICY_LABELS[row] + ": " + label)
    fig.colorbar(im, ax=axes, fraction=.022, pad=.02)
    fig.suptitle("Same input protocol and folds for every model\nI0 removes target geometry; G0 removes contact signal; I3 combines both", fontsize=12)
    return save(fig, output, "ablations")


def learning(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    a = data["audit"]
    for col, model in enumerate(BACKBONES):
        histories = data["histories"][f"I3/{model}"]
        for key, color, label in (("weighted_train_ce", "#2373a4", "Fixed train CE"),
                                  ("weighted_validation_ce", "#cc6c3f", "Inner-val CE")):
            values = np.array([[row[key] for row in h] for h in histories])
            median = np.median(values, axis=0)
            axes[0, col].plot(range(41), median, color=color, label=label)
            low, high = np.percentile(values, (25, 75), axis=0)
            axes[0, col].fill_between(range(41), low, high, color=color, alpha=.15)
        selected = a["learning"][f"I3/{model}"]["selected_epochs"]
        axes[0, col].set_title(MODEL_LABELS[model] + ": 15 full-fit trajectories")
        axes[0, col].set_xlabel("Training epoch"); axes[0, col].set_ylabel("Class-weighted fixed-view CE")
        axes[0, col].grid(alpha=.15); axes[0, col].legend(fontsize=8)
        bottom = axes[1, col]
        for s, seed in enumerate((42, 43, 44)):
            records = sorted([r for r in selected if r["seed"] == seed], key=lambda r: r["fold"])
            bottom.scatter([r["fold"] + (s - 1)*.11 for r in records], [r["epoch"] for r in records],
                           label=f"Seed {seed}", s=34)
        bottom.axhline(40, color="#777", ls="--", label="Final40")
        bottom.set_xticks(range(5)); bottom.set_ylim(0, 43); bottom.set_xlabel("Frozen spatial fold")
        bottom.set_ylabel("Inner CE selected epoch"); bottom.set_title("Selection uses inner validation only")
        if col == 0: bottom.legend(fontsize=8, loc="lower left")
    fig.suptitle("Learning continues; minimum inner CE changes the recall / precision trade-off\n"
                 "Top: median and interquartile range over 15 fits; neither is a confidence interval", fontsize=12)
    return save(fig, output, "learning_and_selection")


def confusions(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    for row, policy in enumerate(POLICIES):
        for col, model in enumerate(BACKBONES):
            scored = data["audit"]["seed_scores"][f"{policy}/I3/{model}"]
            total = np.sum([np.asarray(s["confusion_matrix"]) for s in scored.values()], axis=0)
            normalized = total / total.sum(axis=1, keepdims=True)
            ax = axes[row, col]
            image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{normalized[i,j]:.1%}\n{total[i,j]} seed-events", ha="center", va="center",
                            fontsize=8, color="white" if normalized[i,j] > .60 else "#222")
            ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
            ax.set_xlabel("Predicted class"); ax.set_ylabel("Annotated class")
            ax.set_title(MODEL_LABELS[model] + " / " + POLICY_LABELS[row], fontsize=10)
    fig.colorbar(image, ax=axes, fraction=.022, pad=.02)
    fig.suptitle("I3 confusion: row-normalized average of three seed evaluations\n"
                 "Counts are 1,032 prediction events from 344 structures; seeds do not add biological samples", fontsize=12)
    return save(fig, output, "confusions")


def chid_cases(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 11), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    sids = [f"CHID_{i}" for i in range(1, 27)]
    keys = [f"{policy}/I3/{model}" for model in BACKBONES for policy in POLICIES]
    correct = np.array([[sum(int(rows[sid]["y_pred"]) == 2 for rows in data["predictions"][key].values())
                         for key in keys] for sid in sids])
    ax = axes[0]
    ax.imshow(correct, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
    for i in range(26):
        for j in range(6): ax.text(j, i, f"{correct[i,j]}/3", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(6), [MODEL_LABELS[model] + "\n" + policy for model in BACKBONES for policy in POLICIES], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(26), [sid + (" *" if sid in DIFFICULT else "") for sid in sids], fontsize=9)
    ax.set_title("All 26 CHID: correct optimization seeds\n* = six historical systematic misses", fontsize=11)
    focus = (*DIFFICULT, "OPCID_66", "CHIN_51", "CHIN_73")
    probabilities = []
    text_values = []
    for sid in focus:
        y = 2 if sid.startswith("CHID") else (0 if sid.startswith("OPCID") else 1)
        means, descriptions = [], []
        for policy in POLICIES:
            rows = list(data["predictions"][f"{policy}/I3/m2plus"].values())
            samples = [r[sid] for r in rows]
            means.append(np.mean([float(r[f"p_{CLASSES[y]}"]) for r in samples]))
            counts = np.bincount([int(r["y_pred"]) for r in samples], minlength=3)
            descriptions.append(f"{means[-1]:.3f}\n" + ", ".join(f"{CLASSES[k]}:{n}" for k,n in enumerate(counts) if n))
        probabilities.append(means); text_values.append(descriptions)
    ax = axes[1]
    image = ax.imshow(probabilities, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(focus)):
        for j in range(2): ax.text(j, i, text_values[i][j], ha="center", va="center", fontsize=8,
                                   color="white" if probabilities[i][j] > .65 else "#222")
    ax.set_xticks(range(2), POLICY_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(focus)), focus)
    ax.set_title("M2+ nine known error anchors\nMean probability of annotated class + seed prediction counts", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=.038, pad=.02)
    fig.suptitle("Endpoint selection partly recovers CHID but does not remove systematic mistakes", fontsize=13)
    return save(fig, output, "CHID_cases")


def format_score(score):
    return f"{score['mean']:.3f}" + (f" ± {score['seed_sample_sd']:.3f}" if score["seed_sample_sd"] is not None else "")


def report(data, output, figures):
    audit_result = data["audit"]
    comparisons = audit_result["comparison"]
    lines = ["# 任务一统一复评：分类头学习稳定，系统性形态错误仍在", "",
             "日期：2026-10-14。复评使用已查看过的 344 个候选区间、原 5 个空间折和 seeds 42/43/44。",
             "最低内层验证 CE 权重与第 40 轮权重来自同一次训练；本报告只读取保存结果，没有再次训练或推理。",
             "", "**结论：最低内层验证 CE 没有解决可靠形态识别。** 它普遍提高 I3 的 CHID 召回，却增加误报；",
             "M2+ 与 CNN-Transformer 的 macro-F1 和 CHID F1 均下降。M2+ 末轮仍是当前综合表现最好的方案，",
             "但 CHID 仅 26 例、稳定漏检与误报并存，因此不能据此声称生物学形态识别可靠。", "",
             "## 完整性与复现", "",
             "- 155/155 次拟合、290/290 个固定端点、58/58 组完整 OOF 全部核验，零缺失、执行失败或失效结果。",
             "- 全部递归 ARTIFACTS 索引、文件大小与 SHA256、来源签名、预处理、固定划分、端点选择和两 WT 概率平均均通过。",
             "- 独立从 CSV 重算混淆矩阵、逐类指标、种子均值与样本标准差，与保存结果一致。",
             "- 155 个末轮/传统基线的外层及内层预测 CSV 精确复现 formal_v1；135 条神经训练数值轨迹也完全一致，仅运行耗时不同。",
             "- 135/135 条完整神经轨迹通过持续学习验收。通过数值验收只说明学习与梯度正常，不是形态识别验收。",
             "", f"递归核验科学文件：{audit_result['checks']['sealed_scientific_files']} 个，{audit_result['checks']['sealed_scientific_bytes']} 字节。",
             "[数值核验记录](audit.json)；[逐结构、逐模型策略变化](case_changes.csv)。", "",
             "## I3 主比较", "", "所有模型使用统一的目标区间掩码、训练集长度标准化；神经模型使用 H3 和 1e-4 学习率。",
             "均值 ± 3 个优化种子的样本标准差；标准差不是总体置信区间。PCA+LR 只有固定 seed 42。", "",
             "| 模型 | 固定端点规则 | macro-F1 | CHID precision | CHID recall | CHID F1 |", "|---|---|---:|---:|---:|---:|"]
    for model in BACKBONES:
        for policy in POLICIES:
            score = comparisons[f"{policy}/I3/{model}"]
            lines.append(f"| {MODEL_LABELS[model]} | {policy} | " + " | ".join(format_score(score[m]) for m in
                         ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")) + " |")
    pca = comparisons["baseline/I3/pca_lr"]
    lines.append("| PCA + LR | baseline | " + " | ".join(format_score(pca[m]) for m in
                 ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")) + " |")
    lines += ["", "![主比较](performance.png)", "[导出 PDF](performance.pdf)", "",
              "## 同种子配对变化", "", "差值均为 inner_ce − final40；两个端点共享训练，不能当成独立重复实验。", "",
              "| 模型 | macro-F1 变化 | CHID F1 变化 | CHID recall 变化 | CHID precision 变化 |", "|---|---:|---:|---:|---:|"]
    for model in BACKBONES:
        score = audit_result["paired_policy_deltas"][f"I3/{model}"]
        lines.append("| " + MODEL_LABELS[model] + " | " + " | ".join(f"{score[m]['mean']*100:+.2f} pp" for m in
                      ("macro_f1", "CHID_f1", "CHID_recall", "CHID_precision")) + " |")
    lines += ["", "M2+ macro-F1 三个种子均下降；CHID F1 的变化分别为 " +
              "、".join(f"{(audit_result['seed_scores']['inner_ce/I3/m2plus'][str(s)]['CHID_f1'] - audit_result['seed_scores']['final40/I3/m2plus'][str(s)]['CHID_f1'])*100:+.2f} pp" for s in (42,43,44)) + "。",
              "选择更早权重不能视为已经找到新默认策略，也不能根据看过的外层指标再挑选每个 fold 的轮数。", "",
              "## 仅几何与去目标输入", "", "| 模型 / 规则 | I0 macro / CHID F1 | G0 macro / CHID F1 | I3 macro / CHID F1 |", "|---|---:|---:|---:|"]
    for model in BACKBONES:
        for policy in POLICIES:
            cells = [f"{comparisons[f'{policy}/{c}/{model}']['macro_f1']['mean']:.3f} / {comparisons[f'{policy}/{c}/{model}']['CHID_f1']['mean']:.3f}" for c in ("I0","G0","I3")]
            lines.append(f"| {MODEL_LABELS[model]} / {policy} | " + " | ".join(cells) + " |")
    lines += ["", "| 传统几何基线 | macro-F1 | CHID F1 |", "|---|---:|---:|"]
    for model in ("length_only", "mask_only", "mask_length"):
        score = comparisons[f"baseline/I3/{model}"]
        lines.append(f"| {MODEL_LABELS[model]} | {score['macro_f1']['mean']:.3f} | {score['CHID_f1']['mean']:.3f} |")
    lines += ["", "I3 依然明显优于仅几何 G0，说明接触图携带额外分类信息；此差值仍不能区分具体形态、强度、背景和技术特征。",
              "仅长度或掩码也能达到非零 CHID F1，后续形态改进需要继续保留 G0 控制，避免把几何捷径当作形态收益。", "",
              "![输入消融](ablations.png)", "[导出 PDF](ablations.pdf)", "",
              "## 持续学习与内层选择", "", "| I3 模型 | 通过 / 15 | 末轮训练 CE 中位数 | 末轮内层 CE 中位数 | 所选轮数范围 / 中位数 |", "|---|---:|---:|---:|---:|"]
    for model in BACKBONES:
        record = audit_result["learning"][f"I3/{model}"]
        epochs = [r["epoch"] for r in record["selected_epochs"]]
        lines.append(f"| {MODEL_LABELS[model]} | {record['accepted']}/15 | {record['final_train_ce_median']:.4f} | {record['final_validation_ce_median']:.4f} | {min(epochs)}–{max(epochs)} / {np.median(epochs):.0f} |")
    lines += ["", "训练 loss 与内层验证 loss 的差距、内层选择较早轮次说明泛化仍有限，",
              "但改选择规则的本轮受控结果已经表明：单靠更早权重不能消除系统性错误。",
              "I3 三骨干的末 10 轮每个 fit 头部/编码器梯度均持续有效；末轮固定训练与验证视图均无单类 argmax。",
              "![学习与选择](learning_and_selection.png)", "[导出 PDF](learning_and_selection.pdf)", "",
              "## 最终预测分布与系统性错误", "", "以下“稳定”指三个优化种子均错；种子不是新的生物样本。", "",
              "| 模型 / 规则 | 三种子全对 | 三种子全错 | 稳定 CHID 漏检 | 稳定 CHID 误报 |", "|---|---:|---:|---|---|"]
    for model in BACKBONES:
        for policy in POLICIES:
            error = audit_result["I3_errors"][f"{policy}/I3/{model}"]
            lines.append(f"| {MODEL_LABELS[model]} / {policy} | {error['correct_all_seeds']} | {len(error['stable_wrong'])} | " +
                         ", ".join(error["stable_CHID_FN"]) + " | " + ", ".join(error["stable_CHID_FP"]) + " |")
    lines += ["", "M2+ 的两个端点共同稳定漏检 CHID_8 与 CHID_11。inner_ce 在 CHID_9/12/15/19 上仅恢复部分种子，",
              "并新增 CHID_14 三种子稳定漏检。因此六个困难案例既不能单独作为调参目标，也没有被内层 CE 规则统一修复。", "",
              "| I3 模型 / 规则 / seed | OPCID 预测数 | CHIN 预测数 | CHID 预测数 | CHID TP / FP / FN |", "|---|---:|---:|---:|---:|"]
    for model in BACKBONES:
        for policy in POLICIES:
            for seed in (42, 43, 44):
                record = audit_result["seed_scores"][f"{policy}/I3/{model}"][str(seed)]
                counts = record["prediction_counts"]
                chid = record["CHID_counts"]
                lines.append(f"| {MODEL_LABELS[model]} / {policy} / {seed} | {counts['OPCID']} | {counts['CHIN']} | {counts['CHID']} | {chid['TP']} / {chid['FP']} / {chid['FN']} |")
    lines += ["", "每行覆盖 344 个结构，不把两个 WT 或三个种子当成独立样本。各 I3 外层端点均存在三类预测；",
              "没有重新退化为常数分类器。CHID 误报和漏检属于分类质量问题，不能通过持续学习验收排除。", "",
              "![混淆矩阵](confusions.png)", "[导出 PDF](confusions.pdf)", "",
              "![全部 CHID 与历史错误锚点](CHID_cases.png)", "[导出 PDF](CHID_cases.pdf)", "",
              "## 下一轮改进的证据边界", "",
              "当前支持继续改进候选区间的空间形态表达、CHIN↔CHID 边界和跨种子稳定性；不支持宣称 Transformer 或更早权重已经解决问题。",
              "后续应保持已冻结基线，逐项改变因素，并同时记录 I3、无目标输入和几何控制及形态扰动结果。",
              "已看过的外层结果仅用于开发诊断，禁止据此编造独立盲测或生物专家复核；原始标签全部保留。",
              "给定候选区间分类不代表自动发现结构。用户的新训练授权不改变这些验收边界。", "",
              "来源：`results/task1_reevaluation_v1/`；experiment signature `" + audit_result["experiment_signature"] + "`；",
              "source signature `" + audit_result["source_signature"] + "`。新报告产品位于独立 reports 目录，原实验未覆盖。", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"experiment_signature": audit_result["experiment_signature"], "source_signature": audit_result["source_signature"],
               "audit_sha256": sha256(output / "audit.json"), "script_sha256": sha256(Path(__file__)),
               "figures": figures, "report_sha256": sha256(output / "analysis.md"),
               "case_changes_sha256": sha256(output / "case_changes.csv"),
               "unit": "344 unique structures per OOF seed; repeated seeds and paired endpoints are not independent samples"}
    (output / "figures_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_reevaluation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_reevaluation_review_v1")
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to((ROOT / "reports").resolve()), "Write derived reports inside reports/")
    data = audit(args.run_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(data["audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    changes = data["case_changes"]
    with (args.output / "case_changes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0])); writer.writeheader(); writer.writerows(changes)
    figures = [function(data, args.output) for function in (performance, ablations, learning, confusions, chid_cases)]
    report(data, args.output, figures)
    print(json.dumps({"status": "complete", "figures": len(figures), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
