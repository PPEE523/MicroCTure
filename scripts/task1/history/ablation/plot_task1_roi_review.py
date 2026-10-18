#!/usr/bin/env python3
"""Draw developmental ROI controls from sealed saved results; never fit or infer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

CLASSES = ("OPCID", "CHIN", "CHID")
ARMS = ("R0/I3", "R1/I3", "R1/G0", "R1/V0", "R1/P0")
COLORS = ("#2474a5", "#d06a38", "#59916b", "#8b6da8", "#707070")
LABELS = ("R0: original complete", "R1: ROI complete", "R1: geometry + length",
          "R1: validity + geometry + length", "R1: distance profile only")
SHORT = ("R0\ncomplete", "R1\ncomplete", "R1\ngeometry", "R1\nvalidity", "R1\nprofile")
SEEDS = (42, 43, 44)
SEED_MARKERS = ("o", "s", "^")
PAIRS = ("R1/I3-R0/I3", "R1/I3-R1/G0", "R1/I3-R1/V0", "R1/I3-R1/P0")
PAIR_LABELS = ("ROI minus original", "Complete minus geometry", "Complete minus validity", "Complete minus profile")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def metrics(predictions):
    cm = np.zeros((3, 3), dtype=int)
    for row in predictions:
        truth, prediction = int(row["y_true"]), int(row["y_pred"])
        require(0 <= truth < 3 and 0 <= prediction < 3, "Invalid class index")
        cm[truth, prediction] += 1
    true_positive = np.diag(cm).astype(float)
    precision = np.divide(true_positive, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    recall = np.divide(true_positive, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {"macro_f1": float(f1.mean()), "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "confusion_matrix": cm.tolist()}


def load_results(run_dir):
    from microcture.task1_formal.protocol import json_signature, _package_signature
    summary, manifest = read_json(run_dir / "summary.json"), read_json(run_dir / "manifest.json")
    require(summary["status"] == "complete" and summary["completed_jobs"] == 75 and summary["complete_oof"] == 15,
            "All 75 fits and 15 complete OOFs are required")
    require(not any(summary[name] for name in ("pending_jobs", "invalid_jobs", "execution_failure_jobs", "numerical_failure_jobs")),
            "Failed or missing jobs cannot be omitted")
    require(manifest["experiment_signature"] == json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}),
            "ROI manifest changed")
    require(_package_signature(ROOT, "task1_roi", "run_task1_roi.py") == manifest["source_signature"], "ROI source changed")
    for field in ("experiment_signature", "source_signature", "dataset_signature"):
        require(summary[field] == manifest[field], f"Summary {field} differs")
    config = manifest["config"]
    require(config["seeds"] == list(SEEDS) and config["training"]["epochs"] == 40, "Registered seeds/epochs differ")
    data = {"summary": summary, "manifest": manifest, "predictions": {}, "scores": {}, "histories": {},
            "acceptance": {}, "files": {"manifest.json": sha256(run_dir / "manifest.json"),
                                        "summary.json": sha256(run_dir / "summary.json")}}
    structure_ids = None
    for arm in ARMS:
        alignment, condition = arm.split("/")
        data["predictions"][arm], data["scores"][arm] = {}, {}
        data["histories"][arm], data["acceptance"][arm] = [], []
        for seed in SEEDS:
            source = {}
            for fold in range(5):
                directory = run_dir / "runs" / alignment / condition / "m2plus" / f"seed_{seed}" / f"fold_{fold}"
                artifact = read_json(directory / "ARTIFACTS.json")
                require(artifact["experiment_signature"] == manifest["experiment_signature"], "Job belongs to another experiment")
                require(artifact["signature"] == json_signature(artifact["files"]),
                        "Artifact index signature changed")
                for item in artifact["files"]:
                    path = directory / item["name"]
                    require(path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"],
                            f"Sealed evidence changed: {path}")
                data["files"][str((directory / "ARTIFACTS.json").relative_to(run_dir))] = sha256(directory / "ARTIFACTS.json")
                history = rows(directory / "history.csv")
                require([int(row["epoch"]) for row in history] == list(range(41)), "Incomplete registered trajectory")
                json_history = read_json(directory / "history.json")["epochs"]
                for row, original in zip(history, json_history):
                    for field in ("weighted_train_ce", "weighted_validation_ce"):
                        require(np.isclose(float(row[field]), original[field], atol=1e-12, rtol=0), "CSV trajectory differs from JSON")
                data["histories"][arm].append(history)
                data["acceptance"][arm].append(read_json(directory / "metrics.json")["acceptance"])
                for row in rows(directory / "outer_predictions.csv"):
                    require(row["structure_id"] not in source, "Repeated held-out structure in seed")
                    source[row["structure_id"]] = row
            oof_dir = run_dir / "oof" / alignment / condition / f"seed_{seed}"
            prediction_rows = rows(oof_dir / "predictions.csv")
            by_id = {row["structure_id"]: row for row in prediction_rows}
            require(len(by_id) == len(prediction_rows) == 344 and set(by_id) == set(source), "OOF must contain 344 unique structures")
            if structure_ids is None:
                structure_ids = set(by_id)
            require(set(by_id) == structure_ids, "OOF identities differ between arms")
            for sid, row in by_id.items():
                probability = np.array([float(row[f"p_{label}"]) for label in CLASSES])
                repeat_probability = np.array([record["probabilities"] for record in json.loads(row["replicate_probabilities_json"])])
                require(np.isfinite(probability).all() and np.isclose(probability.sum(), 1, atol=1e-6), "Invalid probability")
                require(repeat_probability.shape == (2, 3) and np.allclose(probability, repeat_probability.mean(0), atol=1e-10, rtol=0),
                        "OOF probability is not the mean of two WT probabilities")
                require(probability.argmax() == int(row["y_pred"]), "OOF argmax differs")
                for field in ("y_true", "y_pred", "group_id", "p_OPCID", "p_CHIN", "p_CHID", "replicate_probabilities_json"):
                    require(str(row[field]) == str(source[sid][field]), "OOF differs from sealed per-job evidence")
            score = metrics(prediction_rows)
            stored = summary["models"][arm]["seeds"][str(seed)]["metrics"]
            require(score["confusion_matrix"] == stored["confusion_matrix"], "Recomputed OOF confusion matrix differs")
            require(np.isclose(score["macro_f1"], stored["macro_f1"], atol=1e-12, rtol=0), "Recomputed macro F1 differs")
            for index, label in enumerate(CLASSES):
                for name in ("precision", "recall", "f1"):
                    require(np.isclose(score[name][index], stored["per_class"][label][name], atol=1e-12, rtol=0), "Recomputed class metric differs")
            data["scores"][arm][seed], data["predictions"][arm][seed] = score, by_id
            data["files"][str((oof_dir / "predictions.csv").relative_to(run_dir))] = sha256(oof_dir / "predictions.csv")
    data["case_metadata"] = {(row["alignment"] + "/" + row["condition"], row["structure_id"]): row
                              for row in rows(run_dir / "case_predictions.csv")}
    require(len(data["case_metadata"]) == 5 * 344, "Derived case registry must cover all 5 arms")
    for arm in ARMS:
        for sid, saved in data["predictions"][arm][SEEDS[0]].items():
            metadata = data["case_metadata"][(arm, sid)]
            require(metadata["type"] == CLASSES[int(saved["y_true"])] and
                    int(metadata["group_id"]) == int(saved["group_id"]) and float(metadata["length_bp"]) > 0,
                    "Case registry identity or geometry differs")
            for seed in SEEDS:
                require(metadata[f"seed_{seed}_prediction"] == CLASSES[int(data["predictions"][arm][seed][sid]["y_pred"])],
                        "Case registry predictions differ from sealed OOF")
    data["files"]["case_predictions.csv"] = sha256(run_dir / "case_predictions.csv")
    return data


def save(fig, output, name):
    import matplotlib.pyplot as plt
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=220 if extension == "png" else None, bbox_inches="tight")
    plt.close(fig)


def sample_points(ax, x, values, color, width=.12):
    for offset, value, marker in zip((-width, 0, width), values, SEED_MARKERS):
        ax.scatter(x + offset, value, color=color, marker=marker, s=31, zorder=4, edgecolors="white", linewidths=.4)


def plot_performance(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    ax = axes[0, 0]
    for i, arm in enumerate(ARMS):
        values = np.array([data["scores"][arm][seed]["macro_f1"] for seed in SEEDS])
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="_", markersize=25, color=COLORS[i], capsize=5, linewidth=2)
        sample_points(ax, i, values, COLORS[i])
        ax.text(i, .945, f"{values.mean():.3f}\n±{values.std(ddof=1):.3f}", ha="center", fontsize=9)
    ax.axhline(.90, color="#ac3030", lw=1, ls="--", alpha=.65)
    ax.set(xticks=range(5), xticklabels=SHORT, ylabel="Structure-level score", title="A   Macro F1", ylim=(.1, 1.03))
    for label, ax, index in zip(CLASSES, (axes[0, 1], axes[1, 0], axes[1, 1]), range(3)):
        for arm_index, arm in enumerate(ARMS):
            offset = (arm_index - 2) * .14
            vals = np.array([[data["scores"][arm][seed][name][index] for seed in SEEDS] for name in ("precision", "recall", "f1")])
            ax.errorbar(np.arange(3) + offset, vals.mean(1), yerr=vals.std(1, ddof=1), color=COLORS[arm_index],
                        marker="_", markersize=12, capsize=3, linewidth=1.2, alpha=.8)
            for metric_index, values in enumerate(vals):
                sample_points(ax, metric_index + offset, values, COLORS[arm_index], width=.026)
        ax.axhline(.85, color="#ac3030", lw=1, ls="--", alpha=.65)
        ax.set(xticks=range(3), xticklabels=("Precision", "Recall", "F1"), ylim=(.1, 1.03), xlim=(-.48, 2.48),
               ylabel="Structure-level score", title=f"{'BCD'[index]}   {label} (n = {68 if index == 0 else 250 if index == 1 else 26})")
    for ax in axes.flat:
        ax.grid(axis="y", color="#dddddd", lw=.6)
    legend = [Line2D([0], [0], color=c, lw=2, label=label) for c, label in zip(COLORS, LABELS)]
    seed_legend = [Line2D([0], [0], color="#333333", marker=m, lw=0, label=f"seed {seed}") for m, seed in zip(SEED_MARKERS, SEEDS)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .048), ncol=3, fontsize=9, frameon=False)
    fig.legend(handles=seed_legend, loc="lower center", bbox_to_anchor=(.5, .014), ncol=3, fontsize=8, frameon=False)
    fig.suptitle("ROI control: developmental performance across all frozen spatial groups", fontsize=15, y=.99)
    fig.text(.5, -.015, "Means ± sample SD across 3 optimization seeds; SD is not a population confidence interval. Dashed lines: preregistered development gates.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.92, bottom=.18, hspace=.32, wspace=.22)
    save(fig, output, "performance")


def bootstrap_interval(bootstrap, comparison, metric, expected_mean=None):
    """Use only a supplied group-bootstrap interval; never synthesize one."""
    if bootstrap is None:
        return None
    records = bootstrap.get("pairs", bootstrap.get("paired", bootstrap.get("paired_deltas", bootstrap.get("comparisons", {}))))
    if isinstance(records, list):
        found = [row for row in records if row.get("comparison", row.get("name")) == comparison]
        if not found:
            return None
        record = found[0]
    elif isinstance(records, dict):
        record = records.get(comparison)
    else:
        return None
    if record is None:
        return None
    if expected_mean is not None and "point_mean" in record:
        require(np.isclose(float(record["point_mean"]), expected_mean, atol=1e-12, rtol=0),
                "Bootstrap point estimate differs from current saved OOF")
    record = record.get(metric, record.get("metrics", {}).get(metric))
    if record is None:
        return None
    interval = record.get("ci95", record.get("percentile_95_ci", record.get("ci")))
    if interval is None and "lower" in record and "upper" in record:
        interval = [record["lower"], record["upper"]]
    if interval is None:
        return None
    if isinstance(interval, dict):
        interval = [interval["lower"], interval["upper"]]
    require(len(interval) == 2 and np.isfinite(interval).all() and interval[0] <= interval[1], "Invalid supplied bootstrap interval")
    return np.asarray(interval, float)


def plot_ablation(data, output, bootstrap):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), gridspec_kw={"width_ratios": [1, 1.3]})
    used_ci = {}
    for row_index, name in enumerate(("macro_f1", "CHID_f1")):
        ax, paired_ax = axes[row_index]
        title = "Macro F1" if name == "macro_f1" else "CHID F1"
        for arm_index, arm in enumerate(ARMS):
            values = np.array([data["scores"][arm][seed]["macro_f1"] if name == "macro_f1" else data["scores"][arm][seed]["f1"][2] for seed in SEEDS])
            ax.bar(arm_index, values.mean(), color=COLORS[arm_index], alpha=.15, width=.65)
            ax.errorbar(arm_index, values.mean(), yerr=values.std(ddof=1), color=COLORS[arm_index], fmt="_", markersize=22, capsize=5, lw=1.7)
            sample_points(ax, arm_index, values, COLORS[arm_index])
            ax.text(arm_index, values.max() + .045, f"{values.mean():.3f}", ha="center", fontsize=9)
        ax.set(xticks=range(5), xticklabels=SHORT, ylim=(0, .99), title=f"{'AC'[row_index]}   Absolute {title}", ylabel="OOF score")
        for pair_index, pair in enumerate(PAIRS):
            left, right = pair.split("-")
            values = np.array([(data["scores"][left][seed]["macro_f1"] - data["scores"][right][seed]["macro_f1"])
                               if name == "macro_f1" else (data["scores"][left][seed]["f1"][2] - data["scores"][right][seed]["f1"][2]) for seed in SEEDS])
            for offset, value, marker in zip((-.13, 0, .13), values, SEED_MARKERS):
                paired_ax.scatter(value * 100, pair_index + offset, marker=marker, color=COLORS[1], s=37, zorder=4)
            paired_ax.scatter(values.mean() * 100, pair_index, marker="|", color="#202020", s=210, zorder=6)
            interval = bootstrap_interval(bootstrap, pair, name, expected_mean=float(values.mean()))
            if interval is not None:
                paired_ax.plot(interval * 100, (pair_index, pair_index), color="#202020", lw=2, zorder=3)
                paired_ax.plot([interval[0] * 100] * 2, (pair_index - .06, pair_index + .06), color="#202020", lw=1)
                paired_ax.plot([interval[1] * 100] * 2, (pair_index - .06, pair_index + .06), color="#202020", lw=1)
                used_ci[f"{pair}:{name}"] = interval.tolist()
            paired_ax.text(.98, pair_index, f"{values.mean() * 100:+.2f} pp", transform=paired_ax.get_yaxis_transform(),
                           ha="right", va="center", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
        paired_ax.axvline(0, color="#aa3333", ls="--", lw=1)
        paired_ax.set(yticks=range(4), yticklabels=PAIR_LABELS, ylim=(3.55, -.55),
                      xlabel="Same-seed paired effect (percentage points)", title=f"{'BD'[row_index]}   {title}: paired effects")
        paired_ax.set_xlim(-20, 58)
        for current in (ax, paired_ax):
            current.grid(axis="y" if current is ax else "x", color="#dddddd", lw=.6)
    fig.suptitle("ROI and signal controls: developmental paired effects", fontsize=15, y=.99)
    fig.legend(handles=[Line2D([0], [0], marker=m, color=COLORS[1], lw=0, label=f"seed {seed}") for m, seed in zip(SEED_MARKERS, SEEDS)]
                      + [Line2D([0], [0], color="#202020", lw=2, label="Seed-mean effect" + (" + 95% group bootstrap CI" if used_ci else " (no CI supplied)"))],
               loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .035))
    fig.text(.5, .002, "Absolute bars: 3-seed mean ± sample SD. P0 signals use offset means; validity, geometry and length remain. P0 is not random shuffling.", ha="center", fontsize=9)
    if used_ci:
        fig.text(.5, -.02, "CI conditions on the existing saved models and sampled spatial groups; it does not cover training-data or independent-sample uncertainty.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.92, bottom=.16, left=.07, right=.97, wspace=.55, hspace=.32)
    save(fig, output, "ablation_and_paired")
    return used_ci


def plot_learning(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 9.3))
    learning_summary = {}
    for index, (ax, arm) in enumerate(zip(axes.flat, ARMS)):
        history = data["histories"][arm]
        train = np.array([[float(row["weighted_train_ce"]) for row in trace] for trace in history])
        val = np.array([[float(row["weighted_validation_ce"]) for row in trace] for trace in history])
        for values, color, style in ((train, "#2474a5", "-"), (val, "#d06a38", "--")):
            lower, median, upper = np.quantile(values, (.25, .5, .75), axis=0)
            ax.plot(range(41), median, color=color, ls=style, lw=2)
            ax.fill_between(range(41), lower, upper, color=color, alpha=.15)
        accepted = sum(bool(record["accepted"]) for record in data["acceptance"][arm])
        min_head_grad = min(float(row["gradient_head_mean"]) for trace in history for row in trace[-10:])
        constants = data["summary"]["models"][arm]["learning"]["prediction_events"]
        constant_events = sum(constants[split]["constant_probability"]["events"] for split in ("train", "validation"))
        epoch40_argmax = sum(constants[split]["single_class_argmax"]["final_cases"] for split in ("train", "validation", "outer"))
        learning_summary[arm] = {"strict_learning_passes": accepted, "jobs": len(history), "terminal_train_ce_median": float(np.median(train[:, -1])),
            "terminal_validation_ce_median": float(np.median(val[:, -1])), "tail_min_head_mean_gradient_norm": min_head_grad,
            "post_epoch0_constant_probability_events_train_plus_validation": constant_events,
            "terminal_single_class_argmax_jobs_across_train_validation_outer": epoch40_argmax,
            "median_train_ce_epoch20": float(np.median(train[:, 20])), "median_validation_ce_epoch20": float(np.median(val[:, 20]))}
        ax.set(xlim=(0, 40), ylim=(0, 1.45), xlabel="Epoch (fixed endpoint = 40)", ylabel="Class-weighted CE",
               title=f"{'ABCDE'[index]}   {arm}: {LABELS[index].split(': ',1)[1]}")
        ax.text(.03, .94, f"Learning passes {accepted}/15\nFinal train/val median: {np.median(train[:, -1]):.3f} / {np.median(val[:, -1]):.3f}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(color="#dddddd", lw=.6)
    ax = axes[1, 2]
    ax.axis("off")
    table_rows = [[arm, f"{record['strict_learning_passes']}/15", f"{record['tail_min_head_mean_gradient_norm']:.3f}",
                   str(record["post_epoch0_constant_probability_events_train_plus_validation"]),
                   str(record["terminal_single_class_argmax_jobs_across_train_validation_outer"])] for arm, record in learning_summary.items()]
    table = ax.table(cellText=table_rows, colLabels=("Arm", "Learning\npasses", "Tail min\nhead grad", "Const. prob.\nevents", "Final single\nclass jobs"),
                     cellLoc="center", loc="upper center", bbox=(0, .32, 1, .58))
    table.auto_set_font_size(False);table.set_fontsize(8.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:cell.set_facecolor("#eceff2")
    ax.set_title("F   Learning diagnostics (all 75 fits)")
    ax.text(0, .24, "Tail: epochs 31–40, minimum batch-mean\nhead gradient norm across all 15 traces.\nConstant events: train + inner validation,\nepochs 1–40; epoch zero excluded.\nFinal single-class: train / validation / outer\njob-view counts summed; probability constancy\nand single-class argmax are distinct checks.",
            transform=ax.transAxes, fontsize=8.6, va="top", linespacing=1.35)
    fig.suptitle("ROI controls: full learning trajectories and collapse checks", fontsize=15, y=.99)
    fig.legend(handles=(Line2D([0], [0], color="#2474a5", lw=2, label="Fixed training view"),
                        Line2D([0], [0], color="#d06a38", lw=2, ls="--", label="Inner validation view")),
               loc="lower center", ncol=2, bbox_to_anchor=(.5, .025), frameon=False)
    fig.text(.5, .003, "Median with interquartile shading across 5 folds × 3 seeds. These are trajectory summaries, not confidence intervals. Outer evaluation occurs only at the fixed endpoint.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.11, hspace=.37, wspace=.24)
    save(fig, output, "learning")
    return learning_summary


def natural_id(sid):
    label, index = sid.rsplit("_", 1)
    return (CLASSES.index(label), int(index))


def plot_cases(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.colors import to_rgba
    keys = ("R0/I3", "R1/I3")
    base = data["predictions"][keys[0]][SEEDS[0]]
    chids = sorted((sid for sid, row in base.items() if int(row["y_true"]) == 2), key=natural_id)
    stable_fn = {key: {sid for sid in chids if all(int(data["predictions"][key][seed][sid]["y_pred"]) != 2 for seed in SEEDS)} for key in keys}
    stable_fp = {key: {sid for sid, row in base.items() if int(row["y_true"]) != 2 and
                      all(int(data["predictions"][key][seed][sid]["y_pred"]) == 2 for seed in SEEDS)} for key in keys}
    anchors = sorted(stable_fp[keys[0]] | stable_fp[keys[1]] | {"OPCID_66", "CHIN_51", "CHIN_73"}, key=natural_id)
    case_ids = chids + anchors
    n = len(case_ids)
    fig, ax = plt.subplots(figsize=(13.8, .45 * n + 2.7))
    class_colors = ("#4a83a9", "#eaa052", "#5a9f72")
    xs = (0, 1, 2, 3.35, 4.35, 5.35)
    row_y = {sid: i + (0 if i < len(chids) else .8) for i, sid in enumerate(case_ids)}
    status = {}
    for sid in case_ids:
        y = row_y[sid]
        for column, (key, seed) in enumerate((key, seed) for key in keys for seed in SEEDS):
            record = data["predictions"][key][seed][sid]
            label = int(record["y_pred"])
            color = to_rgba(class_colors[label], alpha=.25)
            ax.add_patch(Rectangle((xs[column] - .46, y - .43), .92, .86, facecolor=color, edgecolor="white", linewidth=.8))
            ax.text(xs[column], y - .10, CLASSES[label], ha="center", va="center", fontsize=9.1,
                    fontweight="bold" if label != int(record["y_true"]) else "normal")
            probability = float(record["p_CHID"])
            probability_label = f"p(CHID)={probability:.3f}" if probability >= .001 else "p(CHID)<0.001"
            ax.text(xs[column], y + .19, probability_label, ha="center", va="center", fontsize=7.5)
        metadata = data["case_metadata"][(keys[0], sid)]
        ax.text(-.70, y, f"{sid}  ({float(metadata['length_bp']) / 1000:.2f} kb)", ha="right", va="center", fontsize=9)
        if sid in chids:
            before, after = sid in stable_fn[keys[0]], sid in stable_fn[keys[1]]
            outcome = "Stable miss in both" if before and after else "NEW stable miss" if after else "No longer stable miss*" if before else ""
            status[sid] = outcome
        else:
            before, after = sid in stable_fp[keys[0]], sid in stable_fp[keys[1]]
            outcome = "Stable CHID FP in both" if before and after else "NEW stable CHID FP" if after else "No longer stable CHID FP*" if before else ""
            status[sid] = outcome
        ax.text(6.13, y, outcome, ha="left", va="center", fontsize=8.5,
                color="#a43030" if outcome.startswith("NEW") else "#244d75" if outcome.startswith("No longer") else "#444444")
    for index, seed in enumerate(SEEDS):
        for start in (0, 3.35):
            ax.text(start + index, -1.1, f"seed {seed}", ha="center", fontsize=10)
    ax.text(1, -1.72, "R0/I3: original complete input", ha="center", fontsize=12, fontweight="bold")
    ax.text(4.35, -1.72, "R1/I3: ROI complete input", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.13, -1.1, "All-three-seed status", ha="left", fontsize=10)
    ax.axhline(len(chids) - .05, color="#888888", lw=1)
    ax.text(2.7, len(chids) + .12, "True non-CHID: stable-FP union + original anchors", ha="center", va="center", fontsize=8, color="#555555")
    ax.set(xlim=(-2.95, 8.15), ylim=(n + 1, -2.5))
    ax.axis("off")
    fig.suptitle("Every annotated CHID and systematic false positives: developmental OOF predictions", fontsize=14, y=.995)
    fig.legend(handles=[Patch(facecolor=to_rgba(color, .35), label=f"Predicted {label}") for color, label in zip(class_colors, CLASSES)],
               loc="lower center", ncol=3, bbox_to_anchor=(.5, .045), frameon=False, fontsize=10)
    fig.text(.5, .018, "Bold = incorrect predicted class. Probabilities average two WT replicates; they are model scores, not calibrated biological confidence.", ha="center", fontsize=9)
    fig.text(.5, .000, "*No longer stable means at least one seed changed; it does not require every seed to become correct. All 26 CHIDs are shown in numeric order.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.94, bottom=.085, left=.04, right=.99)
    save(fig, output, "chid_cases")
    return {"case_ids": case_ids, "all_26_chid_shown": len(chids) == 26,
            "stable_chid_miss_ids": {key: sorted(ids, key=natural_id) for key, ids in stable_fn.items()},
            "stable_chid_false_positive_ids": {key: sorted(ids, key=natural_id) for key, ids in stable_fp.items()},
            "case_status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_roi_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_roi_review_v1")
    parser.add_argument("--bootstrap", type=Path, help="Optional externally computed group-bootstrap JSON")
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    require(output != run_dir and run_dir not in output.parents and output not in run_dir.parents, "Do not write plots into science results")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    data = load_results(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    bootstrap_path = args.bootstrap or (output / "group_bootstrap.json")
    bootstrap = read_json(bootstrap_path) if bootstrap_path.exists() else None
    plot_performance(data, output)
    intervals = plot_ablation(data, output, bootstrap)
    learning = plot_learning(data, output)
    cases = plot_cases(data, output)
    result = {"execution": "Read-only sealed predictions and histories; no contact data, checkpoint loading, training or inference",
              "experiment_signature": data["manifest"]["experiment_signature"], "source_signature": data["manifest"]["source_signature"],
              "evaluation_scope": data["summary"]["evaluation_scope"], "verified_jobs": 75, "verified_oof": 15,
              "n_structures_per_oof": 344, "seeds": list(SEEDS), "seed_sd_is_not_ci": True,
              "bootstrap_file": str(bootstrap_path.relative_to(ROOT)) if bootstrap is not None and bootstrap_path.is_relative_to(ROOT) else str(bootstrap_path) if bootstrap is not None else None,
              "bootstrap_sha256": sha256(bootstrap_path) if bootstrap is not None else None,
              "group_bootstrap_intervals_used": intervals, "learning": learning, "cases": cases, "sealed_input_sha256": data["files"],
              "figures": [str((output / f"{name}.{extension}").relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output / f"{name}.{extension}")
                          for name in ("performance", "ablation_and_paired", "learning", "chid_cases") for extension in ("png", "pdf")]}
    (output / "plot_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified_jobs": 75, "verified_oof": 15, "figures": result["figures"],
                      "bootstrap_intervals": len(intervals), "stable_fn": cases["stable_chid_miss_ids"],
                      "stable_fp": cases["stable_chid_false_positive_ids"]}, indent=2))


if __name__ == "__main__":
    main()
