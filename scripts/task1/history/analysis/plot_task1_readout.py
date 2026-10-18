#!/usr/bin/env python3
"""Verify saved O0/O1 evidence and draw four report figures, without inference."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
import numpy as np

CLASSES = ("OPCID", "CHIN", "CHID")
KEYS = ("I3/O0", "I3/O1", "G0/O0", "G0/O1")
COLORS = ("#2474a5", "#ce6940", "#609c70", "#9274af")
SCORE_NAMES = ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")
SCORE_LABELS = ("Macro F1", "CHID precision", "CHID recall", "CHID F1")
STABLE_FN = ("CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19")
STABLE_FP = ("OPCID_66", "CHIN_51", "CHIN_73")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def confusion_metrics(y_true, y_pred) -> dict:
    """Compute scores from unique structures, using all three fixed classes."""
    true, pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape or true.ndim != 1 or len(true) == 0:
        raise ValueError("True and predicted labels must be equal nonempty vectors")
    if not np.isin(true, range(3)).all() or not np.isin(pred, range(3)).all():
        raise ValueError("Labels must use the registered three classes")
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (true, pred), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(3), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(3), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {"macro_f1": float(f1.mean()), "CHID_precision": float(precision[2]),
            "CHID_recall": float(recall[2]), "CHID_f1": float(f1[2]),
            "accuracy": float(tp.sum() / len(true)), "balanced_accuracy": float(recall.mean()),
            "confusion_matrix": cm.tolist(), "predicted_counts": cm.sum(axis=0).tolist(),
            "class_support": cm.sum(axis=1).tolist(), "n_structures": len(true)}


def aggregate_complete_seed_scores(scores: dict[int, dict], seeds: list[int]) -> dict:
    if len(seeds) < 2 or set(scores) != set(seeds):
        raise ValueError("All registered seeds are required for a mean and sample SD")
    return {name: {"mean": float(np.mean([scores[seed][name] for seed in seeds])),
                   "seed_sample_sd": float(np.std([scores[seed][name] for seed in seeds], ddof=1))}
            for name in SCORE_NAMES}


def paired_seed_effects(scores: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    requests = (("I3: O1 - O0", "I3/O1", "I3/O0"), ("G0: O1 - O0", "G0/O1", "G0/O0"),
                ("O0: I3 - G0", "I3/O0", "G0/O0"), ("O1: I3 - G0", "I3/O1", "G0/O1"))
    for key in KEYS:
        if set(scores[key]) != set(seeds):
            raise ValueError("Paired effects require every registered seed in all four conditions")
    return {label: {name: {str(seed): scores[left][seed][name] - scores[right][seed][name]
                          for seed in seeds} for name in SCORE_NAMES}
            for label, left, right in requests}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_results(run_dir: Path) -> dict:
    """Use the signed trajectory audit, then independently recompute CSV scores."""
    from microcture.task1_readout.protocol import planned_jobs, job_directory, source_signature
    from microcture.task1_readout.evaluation import _load_job
    from microcture.task1_formal.protocol import json_signature, sha256, _validate_cache_metadata, _package_signature
    from microcture.task1_formal.evaluation import _identity, _fold_ids, _guard_run
    from microcture.task1_formal.runner import assert_resume

    _guard_run(run_dir)
    manifest, summary = read_json(run_dir / "manifest.json"), read_json(run_dir / "summary.json")
    if json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) != manifest["experiment_signature"]:
        raise ValueError("Manifest signature changed")
    if source_signature(ROOT) != manifest["source_signature"]:
        raise ValueError("Recorded readout training source changed")
    for package, entry in (("task1", "run_task1_comparison.py"), ("task1_controlled", "run_task1_controlled.py"),
                           ("task1_formal", "run_task1_formal.py")):
        if _package_signature(ROOT, package, entry) != manifest["dependency_source_signature"][package]:
            raise ValueError(f"Recorded {package} dependency source changed")
    for name in ("experiment_signature", "source_signature", "dataset_signature"):
        if summary[name] != manifest[name]:
            raise ValueError(f"Summary {name} differs from manifest")
    if summary["status"] != "complete" or any(summary[name] for name in
            ("pending_jobs", "invalid_jobs", "numerical_failure_jobs", "execution_failure_jobs")):
        raise ValueError("This four-condition plot requires complete results; failures must remain reported")
    config = manifest["config"]
    metadata, _ = _validate_cache_metadata(ROOT, project_path(config["cache_dir"]), manifest["dataset_signature"])
    identity = _identity(metadata)
    folds = _fold_ids(manifest, identity, run_dir)
    jobs = planned_jobs(config)
    if len(jobs) != 60 or summary["completed_jobs"] != len(jobs) or summary["expected_jobs"] != len(jobs):
        raise ValueError("Registered 60-job coverage changed")
    records, inputs = [], {run_dir / "manifest.json", run_dir / "summary.json"}
    for job in jobs:
        directory = job_directory(run_dir, job)
        if assert_resume(directory, job, manifest) is None:
            raise ValueError(f"Missing sealed job: {directory}")
        record = _load_job(run_dir, job, manifest, identity, folds)
        if record["status"] != "completed":
            raise ValueError("Failed or pending fit cannot be silently omitted")
        rows = csv_rows(directory / "history.csv")
        if [int(row["epoch"]) for row in rows] != list(range(config["training"]["epochs"] + 1)):
            raise ValueError("History CSV does not contain all registered epochs")
        for row, original in zip(rows, record["history"]):
            for field in ("weighted_train_ce", "weighted_validation_ce"):
                if not np.isclose(float(row[field]), original[field], atol=1e-12, rtol=0):
                    raise ValueError("History CSV differs from sealed JSON trajectory")
        record["csv_history"] = rows
        records.append(record)
        inputs.add(directory / "ARTIFACTS.json")
        inputs.update(directory / item["name"] for item in read_json(directory / "ARTIFACTS.json")["files"])
    seeds = sorted(config["seeds"])
    scores, predictions, aggregates, learning = {}, {}, {}, {}
    for key in KEYS:
        condition, variant = key.split("/")
        relevant = [record for record in records if (record["job"]["condition"], record["job"]["model"]) == (condition, variant)]
        scores[key], predictions[key] = {}, {}
        for seed in seeds:
            directory = run_dir / "oof" / condition / variant / f"seed_{seed}"
            status, stored = read_json(directory / "status.json"), read_json(directory / "metrics.json")
            for document in (status, stored):
                if document.get("status") != "complete" or document.get("oof_current") is not True:
                    raise ValueError("Incomplete or stale OOF")
                for field in ("experiment_signature", "source_signature", "dataset_signature"):
                    if document[field] != manifest[field]:
                        raise ValueError(f"OOF {field} changed")
                if (document["condition"], document["model"], document["seed"]) != (condition, variant, seed):
                    raise ValueError("OOF identity changed")
            rows = csv_rows(directory / "predictions.csv")
            by_id = {row["structure_id"]: row for row in rows}
            if len(by_id) != len(rows) or set(by_id) != set(identity):
                raise ValueError("OOF must cover every unique frozen structure once")
            original = {row["structure_id"]: (row, record["job"]["fold"])
                        for record in relevant if record["job"]["seed"] == seed
                        for row in record["predictions"]["rows"]}
            for sid, row in by_id.items():
                source, fold = original[sid]
                if int(row["fold"]) != fold:
                    raise ValueError("OOF fold differs from sealed job")
                for field in ("y_true", "y_pred", "group_id", "replicate_count", "replicate_disagree"):
                    if str(row[field]) != str(source[field]):
                        raise ValueError(f"OOF {field} differs from sealed job")
                probabilities = np.array([float(row[f"p_{name}"]) for name in CLASSES])
                replicate_probs = np.array([item["probabilities"] for item in json.loads(row["replicate_probabilities_json"])])
                if (not np.isfinite(probabilities).all() or not np.allclose(probabilities, replicate_probs.mean(axis=0), atol=1e-10, rtol=0)
                        or not np.isclose(probabilities.sum(), 1, atol=1e-6, rtol=0) or np.any(probabilities < 0) or np.any(probabilities > 1)
                        or probabilities.argmax() != int(row["y_pred"])):
                    raise ValueError("Invalid OOF biological-replicate probability aggregation")
                for field in ("p_OPCID", "p_CHIN", "p_CHID"):
                    if not np.isclose(float(row[field]), float(source[field]), atol=1e-10, rtol=0):
                        raise ValueError("OOF probability differs from sealed job")
                if row["replicate_probabilities_json"] != source["replicate_probabilities_json"]:
                    raise ValueError("OOF replicate evidence differs from sealed job")
            score = confusion_metrics([int(row["y_true"]) for row in rows], [int(row["y_pred"]) for row in rows])
            expected = summary["models"][key]["seeds"][str(seed)]["oof"]
            if score["confusion_matrix"] != expected["confusion_matrix"] or score["confusion_matrix"] != stored["confusion_matrix"]:
                raise ValueError("Recomputed OOF confusion matrix differs from saved metrics")
            for name in SCORE_NAMES:
                location = expected[name] if name == "macro_f1" else expected["per_class"]["CHID"][name.removeprefix("CHID_")]
                if not np.isclose(score[name], location, atol=1e-12, rtol=0):
                    raise ValueError("Recomputed OOF metric differs from saved metrics")
            scores[key][seed], predictions[key][seed] = score, by_id
            inputs.update(directory / filename for filename in ("status.json", "metrics.json", "predictions.csv"))
        aggregates[key] = aggregate_complete_seed_scores(scores[key], seeds)
        comparison = [item for item in summary["comparison"] if (item["condition"], item["model"]) == (condition, variant)]
        if len(comparison) != 1:
            raise ValueError("Comparison row coverage changed")
        for name in SCORE_NAMES:
            for field, suffix in (("mean", "_mean"), ("seed_sample_sd", "_seed_sample_std")):
                if not np.isclose(aggregates[key][name][field], comparison[0][name + suffix], atol=1e-12, rtol=0):
                    raise ValueError("Comparison mean or seed SD differs from CSV recomputation")
        tail = [record["metrics"]["acceptance"]["primary"] for record in relevant]
        learning[key] = {"jobs": len(relevant), "strict_learning_passes": sum(record["metrics"]["acceptance"]["accepted"] for record in relevant),
                         "terminal_constant_train_jobs": sum(float(record["csv_history"][-1]["train_majority_fraction"]) == 1 for record in relevant),
                         "terminal_constant_validation_jobs": sum(float(record["csv_history"][-1]["validation_majority_fraction"]) == 1 for record in relevant),
                         "terminal_constant_outer_jobs": sum(len({row['y_pred'] for row in record["predictions"]["rows"]}) == 1 for record in relevant),
                         "terminal_train_ce_median": float(np.median([record["history"][-1]["weighted_train_ce"] for record in relevant])),
                         "terminal_val_ce_median": float(np.median([record["history"][-1]["weighted_validation_ce"] for record in relevant])),
                         "tail_min_head_nonzero_gradient_fraction": min(item["gradients"]["head"]["nonzero_fraction_min"] for item in tail),
                         "tail_min_head_mean_gradient_norm": min(item["gradients"]["head"]["mean_norm_min"] for item in tail),
                         "tail_min_head_effective_fraction": min(float(row["train_head_effective_fraction"]) for record in relevant for row in record["csv_history"][-10:]),
                         "tail_min_train_centered_logit_variance": min(item["views"]["train"]["centered_logit_variance_min"] for item in tail),
                         "tail_min_validation_centered_logit_variance": min(item["views"]["validation"]["centered_logit_variance_min"] for item in tail),
                         "tail_constant_train_probability_jobs": sum(item["views"]["train"]["probabilities_constant_epochs"] > 0 for item in tail),
                         "tail_constant_validation_probability_jobs": sum(item["views"]["validation"]["probabilities_constant_epochs"] > 0 for item in tail),
                         "terminal_train_head_positive_fraction_median": float(np.median([float(record["csv_history"][-1]["train_head_positive_fraction"]) for record in relevant])),
                         "terminal_train_head_inactive_units_max": max(int(record["csv_history"][-1]["train_head_inactive_units"]) for record in relevant),
                         "median_train_ce_epoch20": float(np.median([record["history"][20]["weighted_train_ce"] for record in relevant])),
                         "median_val_ce_epoch20": float(np.median([record["history"][20]["weighted_validation_ce"] for record in relevant]))}
    verification = {"verified_jobs": len(records), "verified_complete_oof": 12, "n_structures_per_oof": len(identity),
                    "score_unit": "344 unique structures per seed; WT replicate probabilities averaged before argmax",
                    "seed_sample_sd_note": "Optimization variability across three seeds; not a population confidence interval",
                    "execution": "Saved evidence read only; no contact data, checkpoint deserialization, fitting or model inference",
                    "saved_input_sha256": {str(path.relative_to(run_dir)): sha256(path) for path in sorted(inputs)}}
    return {"manifest": manifest, "summary": summary, "scores": scores, "predictions": predictions,
            "aggregates": aggregates, "records": records, "learning": learning, "seeds": seeds,
            "paired_effects": paired_seed_effects(scores, seeds), "verification": verification}


def save_figure(fig, output: Path, name: str) -> dict:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    return {"name": name, "paths": paths}


def plot_performance(data: dict, output: Path) -> dict:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    seeds = data["seeds"]
    for ax, name, title in zip(axes.flat, SCORE_NAMES, SCORE_LABELS):
        for index, (key, color) in enumerate(zip(KEYS, COLORS)):
            values = np.array([data["scores"][key][seed][name] for seed in seeds])
            ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="_", markersize=30, color=color,
                        capsize=7, elinewidth=2)
            for offset, value, seed in zip((-.14, 0, .14), values, seeds):
                ax.scatter(index + offset, value, c=color, s=46, marker={42: "o", 43: "s", 44: "^"}[seed], zorder=3)
            ax.text(index, min(1.015, values.max() + .065), f"{values.mean():.3f} ± {values.std(ddof=1):.3f}", ha="center", fontsize=9)
        ax.set(title=title, ylim=(.15, 1.05), xlim=(-.65,3.65), xticks=range(4), xticklabels=KEYS)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Target spatial readout control: full OOF performance", fontsize=15)
    fig.supxlabel("Each point: one complete 344-structure OOF seed (circle 42, square 43, triangle 44).\nBars: mean ± sample SD of three seeds; no fold averaging, no confidence intervals.", fontsize=9)
    artifact = save_figure(fig, output, "performance")
    plt.close(fig)
    return artifact


def plot_paired(data: dict, output: Path) -> dict:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    labels = list(data["paired_effects"])
    for ax, name, title in zip(axes, ("macro_f1", "CHID_f1"), ("Macro F1 change", "CHID F1 change")):
        ax.axhline(0, color="black", linewidth=1)
        for index, label in enumerate(labels):
            values = np.array([data["paired_effects"][label][name][str(seed)] for seed in data["seeds"]]) * 100
            color = COLORS[index]
            ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="_", markersize=26, capsize=6, color=color)
            for offset, value, seed in zip((-.12, 0, .12), values, data["seeds"]):
                ax.scatter(index + offset, value, color=color, marker={42: "o", 43: "s", 44: "^"}[seed], s=50)
            ax.annotate(f"{values.mean():+.2f} pp", (index, values.max()), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=title, ylabel="Percentage points", xticks=range(4), xticklabels=[label.replace(": ", ":\n") for label in labels])
        ax.margins(y=.2)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Paired effects from matched full OOF seeds", fontsize=15)
    fig.supxlabel("Deltas use complete OOF scores from the same seed, not averages of fold deltas.\nI3 - G0 estimates the value of image signal under this protocol; it does not prove biological morphology recognition.", fontsize=9)
    artifact = save_figure(fig, output, "paired_effects")
    plt.close(fig)
    return artifact


def plot_predictions(data: dict, output: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig = plt.figure(figsize=(15, 14), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 2.35))
    for column, key in enumerate(("I3/O0", "I3/O1")):
        ax = fig.add_subplot(grid[0, column])
        cm = np.mean([data["scores"][key][seed]["confusion_matrix"] for seed in data["seeds"]], axis=0)
        rates = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rates, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{cm[row,col]:.1f}\n{rates[row,col]:.1%}", ha="center", va="center", color="white" if rates[row,col] > .55 else "black")
        ax.set(title=f"{key}: average counts / seed", xticks=range(3), yticks=range(3), xticklabels=CLASSES,
               yticklabels=CLASSES, xlabel="Predicted", ylabel="Annotated")
    ax = fig.add_subplot(grid[0, 2])
    counts = np.array([data["scores"][key][seed]["predicted_counts"] for key in KEYS for seed in data["seeds"]])
    bottom = np.zeros(len(counts))
    class_colors = ("#c2a150", "#6389b0", "#a2609b")
    for index, (name, color) in enumerate(zip(CLASSES, class_colors)):
        ax.bar(range(len(counts)), counts[:, index], bottom=bottom, color=color, label=name)
        bottom += counts[:, index]
    ax.axhline(344, color="black", linewidth=.7)
    ax.set(title="OOF predicted class counts", ylabel="Structures / seed", xticks=range(len(counts)),
           xticklabels=[f"{key}\n{seed}" for key in KEYS for seed in data["seeds"]])
    ax.tick_params(axis="x", labelsize=7, rotation=70)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.14))
    ax = fig.add_subplot(grid[1, :])
    ids = sorted([sid for sid, row in data["predictions"]["I3/O0"][data["seeds"][0]].items() if int(row["y_true"]) == 2],
                 key=lambda sid: int(sid.split("_")[-1])) + list(STABLE_FP)
    columns = [(key, seed) for key in ("I3/O0", "I3/O1") for seed in data["seeds"]]
    values = np.array([[int(data["predictions"][key][seed][sid]["y_pred"]) for key, seed in columns] for sid in ids])
    cmap = ListedColormap(class_colors)
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm((-.5,.5,1.5,2.5),3), aspect="auto")
    for row, sid in enumerate(ids):
        for col, (key, seed) in enumerate(columns):
            probability = float(data["predictions"][key][seed][sid]["p_CHID"])
            ax.text(col, row, f"{CLASSES[values[row,col]]}\np(CHID)={probability:.2f}", ha="center", va="center", fontsize=7, color="white" if values[row,col] == 2 else "black")
    ax.set(xticks=range(6), xticklabels=[f"{key}, seed {seed}" for key, seed in columns], yticks=range(len(ids)),
           yticklabels=[sid + ("  *" if sid in STABLE_FN else "  dagger" if sid in STABLE_FP else "") for sid in ids],
           title="All 26 annotated CHID cases + three previously stable CHID false-positive anchors")
    for text, sid in zip(ax.get_yticklabels(), ids):
        if sid in STABLE_FN or sid in STABLE_FP:
            text.set_color("#a32b2b")
            text.set_fontweight("bold")
    ax.axhline(25.5, color="black", linewidth=2)
    ax.axvline(2.5, color="white", linewidth=3)
    fig.suptitle("Prediction distribution and persistent error cases", fontsize=15)
    fig.supxlabel("* Previous all-seed CHID → CHIN misses. dagger: previous all-seed CHID false positives.\nConfusion counts are averaged across repeated seeds; each seed scores the same 344 structures. Heatmap color = predicted class.", fontsize=9)
    artifact = save_figure(fig, output, "predictions_and_errors")
    plt.close(fig)
    return artifact


def plot_learning(data: dict, output: Path) -> dict:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    settings = (("weighted_train_ce", "Fixed inner-train weighted CE", "linear"),
                ("weighted_validation_ce", "Fixed inner-validation weighted CE", "linear"),
                ("gradient_head_mean", "Head gradient norm / epoch", "log"),
                ("train_head_effective_fraction", "Train head effective activation fraction", "linear"))
    for ax, (field, title, scale) in zip(axes.flat[:4], settings):
        for key, color in zip(KEYS, COLORS):
            condition, variant = key.split("/")
            rows = [record["csv_history"] for record in data["records"] if (record["job"]["condition"], record["job"]["model"]) == (condition, variant)]
            epochs = list(range(41)) if field != "gradient_head_mean" else list(range(1, 41))
            values = np.array([[float(history[epoch][field]) for epoch in epochs] for history in rows])
            low, median, high = np.quantile(values, (.25, .5, .75), axis=0)
            ax.plot(epochs, median, color=color, label=key)
            ax.fill_between(epochs, low, high, color=color, alpha=.10)
        ax.set(title=title, xlabel="Epoch", yscale=scale)
        if field == "train_head_effective_fraction":
            ax.set_ylim(.94, 1.01)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8, ncol=2)
    ax = axes[1,1]
    for index, (key, color) in enumerate(zip(KEYS, COLORS)):
        condition, variant = key.split("/")
        values = [record["history"][-1]["weighted_train_ce"] for record in data["records"]
                  if (record["job"]["condition"], record["job"]["model"]) == (condition, variant)]
        offsets = np.linspace(-.18, .18, len(values))
        ax.scatter(index + offsets, values, color=color, alpha=.8, s=30)
        ax.plot([index-.25,index+.25],[np.median(values)]*2,color="black",linewidth=2)
    ax.set(title="Epoch-40 fixed train CE", xticks=range(4), xticklabels=KEYS, ylabel="One dot / fit")
    ax.grid(axis="y", alpha=.2)
    ax = axes[1,2]
    ax.axis("off")
    rows = [[key, f"{data['learning'][key]['strict_learning_passes']}/15",
             f"{data['learning'][key]['terminal_constant_train_jobs']}/{data['learning'][key]['terminal_constant_validation_jobs']}/{data['learning'][key]['terminal_constant_outer_jobs']}",
             f"{data['learning'][key]['tail_min_head_mean_gradient_norm']:.3g}"] for key in KEYS]
    table = ax.table(cellText=rows, colLabels=("Protocol", "Learn pass", "Const T/V/O", "Tail grad min"), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    ax.set_title("Numerical learning audit", pad=15)
    ax.text(.5,.12,"T/V/O: terminal single-class prediction fits\n(train / validation / outer).\nTail min: minimum epoch-mean head norm, epochs 31–40.\nLearning pass does not establish recognition quality.",ha="center",va="center",transform=ax.transAxes,fontsize=9)
    fig.suptitle("Training trajectories and numerical stability", fontsize=15)
    fig.supxlabel("Trajectory lines: medians over 15 fold × seed fits per protocol; shading: interquartile range.\nThese overlapping training sets are descriptive trajectories, not 15 independent biological replicates. Final epoch stays fixed at 40.", fontsize=9)
    artifact = save_figure(fig, output, "learning")
    plt.close(fig)
    return artifact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/task1_readout_v1")
    parser.add_argument("--output-dir", default="reports/task1_readout_review_v1")
    args = parser.parse_args(argv)
    run_dir, output = project_path(args.run_dir).resolve(), project_path(args.output_dir)
    if output.is_symlink() or any(path.is_symlink() for path in output.parents):
        raise ValueError("Report output must not follow symbolic links")
    output = output.resolve()
    if ROOT / "reports" not in output.parents or output == run_dir or run_dir in output.parents:
        raise ValueError("Use a separate report directory below reports/")
    if any(path in output.parents or path == output for path in (ROOT / "reports/task1_controlled_review_v1",)):
        raise ValueError("Frozen report directories cannot be overwritten")
    print("Verifying sealed trajectories and independently recomputing complete OOF scores...", flush=True)
    data = verify_results(run_dir)
    marker = output / "figures_summary.json"
    if marker.is_file() and read_json(marker).get("experiment_signature") != data["manifest"]["experiment_signature"]:
        raise ValueError("This report directory belongs to a different experiment")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="microcture-readout-figures-", dir="/tmp") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
        figures = [plot_performance(data, output), plot_paired(data, output), plot_predictions(data, output), plot_learning(data, output)]
        from microcture.task1_formal.protocol import sha256
        for name, expected in data["verification"]["saved_input_sha256"].items():
            if sha256(run_dir / name) != expected:
                raise ValueError(f"Scientific input changed while rendering: {name}")
        document = {"status": "complete", "experiment_signature": data["manifest"]["experiment_signature"],
                    "source_signature": data["manifest"]["source_signature"], "dataset_signature": data["manifest"]["dataset_signature"],
                    "figures": figures, "aggregates": data["aggregates"], "per_seed_scores": data["scores"],
                    "paired_effects": data["paired_effects"], "learning": data["learning"], "verification": data["verification"],
                    "plotting_source_sha256": sha256(Path(__file__)), "plotting_environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
                    "saved_inputs_unchanged_after_plotting": True,
                    "limitations": ["已查看空间折上的开发性复评，没有新增样本或专家真值。", "三种种子的SD表示优化变动，不是生物学总体置信区间。", "几何控制仅用于本协议内解释增量，不能证明或否定特定生物学形态机制。", "60次训练均保留；图表使用末轮权重，不做验证集挑轮。"]}
        marker.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "verified_jobs": 60, "complete_oof": 12,
                      "figures": 4, "output": str(output), "scientific_results_modified": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
