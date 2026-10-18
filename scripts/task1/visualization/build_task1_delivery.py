"""Recompute compact task-one OOF evidence and render delivery figures, without fitting."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from check_submission_evidence import _score

ROOT = _layout_root
CLASSES = ("OPCID", "CHIN", "CHID")
MODELS = ("pca_lr", "m2", "m2plus", "cnn_transformer")
NAMES = ("PCA + LR", "M2", "M2+", "CNN-Transformer")
COLORS = ("#8b9aab", "#507da3", "#167d8d", "#b18042")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    base = ROOT / "submission/task1/formal"
    table = pd.read_csv(base / "comparison.csv")
    metadata = json.loads((base / "metadata.json").read_text())
    sources = [base / "comparison.csv", base / "metadata.json"]
    identity, models, complete = None, {}, 0
    for directory in sorted((base / "oof/I3").glob("*/seed_*")):
        frame = pd.read_csv(directory / "predictions.csv").sort_values("structure_id")
        expected = json.loads((directory / "metrics.json").read_text())
        assert len(frame) == 344 and frame.structure_id.nunique() == 344
        current = frame[["structure_id", "y_true", "group_id", "fold"]].reset_index(drop=True)
        if identity is None:
            identity = current
        else:
            pd.testing.assert_frame_equal(identity, current)
        assert frame.group_id.nunique() == 49
        assert (frame.replicate_count == 2).all()
        probabilities = frame[["p_" + c for c in CLASSES]].to_numpy()
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), frame.y_pred)
        for _, row in frame.iterrows():
            wt = json.loads(row.replicate_probabilities_json)
            assert len(wt) == 2
            np.testing.assert_allclose(np.mean([x["probabilities"] for x in wt], axis=0),
                                       row[["p_" + c for c in CLASSES]].to_numpy(float),
                                       atol=1e-7, rtol=0)
        matrix = np.zeros((3, 3), dtype=int)
        np.add.at(matrix, (frame.y_true.to_numpy(int), frame.y_pred.to_numpy(int)), 1)
        score = _score(matrix.tolist())
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            np.testing.assert_allclose(score[key], expected[key], atol=1e-12, rtol=0)
        for name in CLASSES:
            for key in ("precision", "recall", "f1", "support"):
                np.testing.assert_allclose(score["per_class"][name][key],
                                           expected["per_class"][name][key], atol=1e-12, rtol=0)
        score["seed"] = int(directory.name.removeprefix("seed_"))
        score["seed_predictions"] = frame.y_pred.to_numpy(int).tolist()
        score["WT_disagreement_count"] = int(frame.replicate_disagree.sum())
        models.setdefault(directory.parent.name, []).append(score)
        sources.extend([directory / "predictions.csv", directory / "metrics.json"])
        complete += 1
    assert complete == 13
    for model, scores in models.items():
        row = table[(table.condition == "I3") & (table.model == model)].iloc[0]
        np.testing.assert_allclose(np.mean([s["macro_f1"] for s in scores]),
                                   row.macro_f1_mean, atol=1e-12, rtol=0)
    scores = models["m2plus"]
    predictions = np.asarray([x["seed_predictions"] for x in scores])
    truths = identity.y_true.to_numpy(int)
    stable_wrong = (predictions != truths).all(axis=0)
    stable_wrong_ids = identity.loc[stable_wrong, "structure_id"].tolist()
    class_errors = {name: int((stable_wrong & (truths == i)).sum()) for i, name in enumerate(CLASSES)}
    stage_path = ROOT / "submission/task1/autonomous/stage_scores.csv"
    stages = pd.read_csv(stage_path)
    decision_path = ROOT / "submission/task1/autonomous/decision.json"
    decision = json.loads(decision_path.read_text())
    demo_path = ROOT / "submission/demo/task1/expected.json"
    sources.extend([stage_path, decision_path, demo_path, ROOT / "submission/demo/task1/input.npz",
                    ROOT / "submission/task1/controlled/head_aggregate.csv", Path(__file__)])
    summary = {
        "date": "2026-10-15", "new_training": False, "raw_contact_scan": False,
        "status": "classification_comparison_complete_reliability_not_established",
        "oof_files_recomputed": complete, "oof_rows_recomputed": complete * 344,
        "unique_labels": 344, "spatial_groups": 49,
        "class_counts": {c: int((truths == i).sum()) for i, c in enumerate(CLASSES)},
        "unit": "unique intervals after averaging WT probabilities; seed SD is optimization variation",
        "models": models,
        "m2plus_mean_confusion": np.mean([x["confusion_matrix"] for x in scores], axis=0).tolist(),
        "m2plus_all_three_seeds_wrong": stable_wrong_ids,
        "m2plus_stable_errors_by_true_class": class_errors,
        "m2plus_seed_agreement": float((predictions == predictions[0]).all(axis=0).mean()),
        "m2plus_mean_class_f1": {c: float(np.mean([x["per_class"][c]["f1"] for x in scores])) for c in CLASSES},
        "head_control": pd.read_csv(ROOT / "submission/task1/controlled/head_aggregate.csv")[["head", "expected_cases", "strict_pass_cases"]].to_dict("records"),
        "ablation": table[table.model.isin(MODELS[1:])][["condition", "model", "macro_f1_mean", "CHID_f1_mean"]].to_dict("records"),
        "reliability_decision": decision,
        "limitations": ["All outer spatial groups previously inspected; not an independent blind test",
                        "Two WT views do not add unique annotated intervals",
                        "Coordinates provisional; no independent expert morphology truth",
                        "Ablations read saved summaries; thirteen I3 OOF CSVs are independently recomputed",
                        "Later development recipes differ in inputs; they are not a unified model ranking"],
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sources],
    }
    return table, stages, summary


def save_figure(fig, filename):
    fig.savefig(ROOT / "submission/figures" / filename, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_comparison(table, summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(4)
    for k, (metric, color) in enumerate((("macro_f1", "#167d8d"), ("CHID_f1", "#c78845"))):
        rows = [table[(table.condition == "I3") & (table.model == model)].iloc[0] for model in MODELS]
        means = [r[metric + "_mean"] for r in rows]
        sd = [0 if pd.isna(r[metric + "_seed_sample_std"]) else r[metric + "_seed_sample_std"] for r in rows]
        bars = axes[0].bar(x + (k - .5) * .34, means, .32, yerr=sd, capsize=3,
                           color=color, label="Macro-F1" if k == 0 else "CHID F1")
        axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axes[0].set(xticks=x, xticklabels=NAMES, ylim=(0, 1.0), ylabel="Complete spatial OOF F1",
                title="A  Fair I3/H3 comparison")
    axes[0].legend(loc="upper left", ncols=2, frameon=False)
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].grid(axis="y", alpha=.18)
    axes[0].set_axisbelow(True)
    matrix = np.asarray(summary["m2plus_mean_confusion"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    axes[1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{normalized[i,j]:.1%}\n({matrix[i,j]:.2f})", ha="center", va="center",
                         fontsize=11, color="white" if normalized[i, j] > .55 else "#172c41")
    axes[1].set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                xlabel="Predicted class", ylabel="True class", title="B  M2+ mean confusion")
    fig.suptitle("Task 1 | model selection and the minority-class limitation", fontsize=16, fontweight="bold")
    fig.text(.5, .012, "344 unique intervals | 49 spatial groups | 5 folds | 3 seeds (PCA+LR: 1)\n"
             "Bars: seed mean +/- sample SD. Confusion parentheses: mean count per seed, not pooled independent samples.",
             ha="center", fontsize=9, color="#526273")
    fig.tight_layout(rect=(0, .08, 1, .91))
    save_figure(fig, "task1_delivery_comparison.png")


def render_development(table, stages):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(3)
    for j, (condition, label, color) in enumerate((("I0", "I0: signal + validity", "#8b9aab"),
                                                 ("G0", "G0: geometry only", "#c78845"),
                                                 ("I3", "I3: complete", "#167d8d"))):
        rows = [table[(table.condition == condition) & (table.model == model)].iloc[0] for model in MODELS[1:]]
        bars = axes[0].bar(x + (j - 1) * .25, [r.macro_f1_mean for r in rows], .23,
                           color=color, label=label)
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set(xticks=x, xticklabels=NAMES[1:], ylim=(0, 1), ylabel="Complete spatial OOF macro-F1",
                title="A  Same-backbone input controls")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=.18)
    axes[0].set_axisbelow(True)
    labels = ("Original M2+ (R0)", "Full target ROI (R1)", "Matched coarse (C0)", "Native detail (C1)",
              "ROI, dropout 0.5", "Fixed morphology + LR*", "Fixed morphology + SVC*", "Fixed 0.5 LR + 0.5 CNN")
    y = np.arange(len(stages))
    for offset, metric, color, name in ((-.12, "macro_f1", "#167d8d", "Macro-F1"),
                                         (.12, "CHID_f1", "#c78845", "CHID F1")):
        sdcol = "macro_f1_seed_SD" if metric == "macro_f1" else "CHID_f1_seed_SD"
        axes[1].errorbar(stages[metric], y + offset, xerr=stages[sdcol].fillna(0),
                          fmt="o", markersize=5, capsize=3, color=color, label=name)
    axes[1].set(yticks=y, yticklabels=labels, xlim=(.35, .98), xlabel="Complete OOF score",
                title="B  Development history (different recipes)")
    axes[1].invert_yaxis()
    axes[1].axvline(.90, linestyle="--", color="#a54444", alpha=.6)
    axes[1].text(.904, 3.0, "macro-F1 gate 0.90", rotation=90, va="center", fontsize=8, color="#a54444")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].grid(axis="x", alpha=.18)
    fig.suptitle("Task 1 | what helped, what remained unsolved", fontsize=16, fontweight="bold")
    fig.text(.5, .012, "I3-G0 restores both signal and validity; it does not isolate morphology alone.\n"
             "Right panel is development history, not a fair new architecture ranking. *One seed. No recipe passed all reliability gates.",
             ha="center", fontsize=9, color="#526273")
    fig.tight_layout(rect=(0, .08, 1, .92))
    save_figure(fig, "task1_delivery_controls.png")


def render_demo():
    path = ROOT / "submission/demo/task1"
    expected = json.loads((path / "expected.json").read_text())
    with np.load(path / "input.npz", allow_pickle=False) as values:
        images = values["x"].copy()
        ids = values["structure_id"].copy()
    fig, axes = plt.subplots(3, 3, figsize=(11.3, 10), gridspec_kw={"width_ratios": [1, 1, .95]})
    bounds = np.nanpercentile(images[:, :, 0], [1, 99])
    for row, sid in enumerate(expected["selection_ids_class_order"]):
        selected = images[ids == sid]
        mean = selected.mean(axis=0)
        for scale in range(2):
            signal = np.ma.masked_where(mean[scale, 1] <= 0, mean[scale, 0])
            axes[row, scale].imshow(signal, origin="lower", cmap="magma", vmin=bounds[0], vmax=bounds[1])
            mask = mean[scale, 2]
            if mask.min() < .5 < mask.max():
                axes[row, scale].contour(mask, levels=[.5], colors="#54e0d1", linewidths=1)
            axes[row, scale].set_xticks([])
            axes[row, scale].set_yticks([])
            if row == 0:
                axes[row, scale].set_title(("6.4 kb processed input", "25.6 kb processed input")[scale], fontsize=11)
            if scale == 0:
                axes[row, scale].set_ylabel(sid, fontsize=12, fontweight="bold")
        prediction = next(p for p in expected["structure_predictions"] if p["structure_id"] == sid)
        ps = [prediction["p_" + c] for c in CLASSES]
        axes[row, 2].barh(CLASSES[::-1], ps[::-1], color=["#c78845", "#507da3", "#167d8d"])
        axes[row, 2].set_xlim(0, 1.12)
        for y, value in enumerate(ps[::-1]):
            axes[row, 2].text(value + .02, y, f"{value:.3f}", va="center", fontsize=10)
        axes[row, 2].set_title(f"Predicted: {prediction['predicted']}", color="#a54444" if sid.startswith("CHID") else "#172c41", fontsize=11)
        axes[row, 2].grid(axis="x", alpha=.15)
        axes[row, 2].set_axisbelow(True)
    fig.suptitle("Task 1 | fixed demonstration, including the CHID_11 error", fontsize=15, fontweight="bold")
    fig.text(.5, .012, "Per class: first lexicographic ID in frozen fold 0; no filtering for correctness. Two WT inputs are averaged for display.\n"
             "Cyan contour: target mask. Shared color limits: processed signal, not raw contact count. Probabilities: saved CPU reference.\n"
             "These three examples demonstrate the pipeline; full-344 OOF metrics determine performance.",
             ha="center", fontsize=9, color="#526273")
    fig.tight_layout(rect=(0, .085, 1, .94))
    save_figure(fig, "task1_delivery_examples.png")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "font.size": 10})
    table, stages, summary = collect()
    render_comparison(table, summary)
    render_development(table, stages)
    render_demo()
    for model in summary["models"].values():
        for score in model:
            del score["seed_predictions"]
    (ROOT / "submission/task1_delivery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"OOF_files_recomputed": 13, "OOF_rows_recomputed": 4472,
                      "figures": 3, "stable_wrong": summary["m2plus_stable_errors_by_true_class"]}))


if __name__ == "__main__":
    main()
