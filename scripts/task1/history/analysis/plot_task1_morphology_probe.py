#!/usr/bin/env python3
"""Read sealed fixed-weight sensitivity results; export seed-aware figures."""
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
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.probe_task1_morphology_dependence import signature
from microcture.task1_formal.protocol import sha256
from microcture.task1.evaluation import metrics


def run(probe_dir: Path, output_dir: Path) -> dict:
    probe_dir, output_dir = probe_dir.resolve(), output_dir.resolve()
    if ROOT / "reports" not in output_dir.parents or probe_dir in output_dir.parents:
        raise ValueError("Figures must use a separate reports/ subdirectory")
    summary = json.loads((probe_dir / "summary.json").read_text())
    seal = json.loads((probe_dir / "ARTIFACTS.json").read_text())
    if summary["status"] != "completed" or seal["signature"] != signature(seal["files"]):
        raise ValueError("Probe is incomplete or its seal changed")
    for item in seal["files"]:
        path = probe_dir / item["name"]
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Saved probe artifact changed: {path}")
    with (probe_dir / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    conditions, seeds = ("baseline", "roi", "background"), (42, 43, 44)
    comparisons = summary["comparisons"]
    if summary["models"] != ["m2plus"] or len(comparisons) != 9:
        raise ValueError("This figure requires M2+ with exactly three seeds and three shuffle repeats")
    for comparison in comparisons:
        selected = [row for row in rows if int(row["seed"]) == comparison["seed"] and int(row["repeat"]) == comparison["repeat"]]
        if len(selected) != 344 or len({row["structure_id"] for row in selected}) != 344:
            raise ValueError("Each seed/repeat must have all unique structures")
        for condition in conditions:
            calculated = metrics(np.array([int(row["y_true"]) for row in selected]),
                                 np.array([[float(row[f"{condition}_p_{name}"]) for name in ("OPCID", "CHIN", "CHID")] for row in selected]))
            stored = comparison["full_OOF_includes_unavailable_unchanged_cases"]["metrics"][condition]
            if calculated["confusion_matrix"] != stored["confusion_matrix"]:
                raise ValueError("Saved probe metrics differ from its prediction CSV")
    aggregate = {}
    for condition in conditions:
        per_seed = {}
        for seed in seeds:
            chosen = [row for row in comparisons if row["seed"] == seed]
            per_seed[str(seed)] = {
                "macro_f1": float(np.mean([row["full_OOF_includes_unavailable_unchanged_cases"]["metrics"][condition]["macro_f1"] for row in chosen])),
                "CHID_f1": float(np.mean([row["full_OOF_includes_unavailable_unchanged_cases"]["metrics"][condition]["per_class"]["CHID"]["f1"] for row in chosen])),
            }
        aggregate[condition] = {"per_seed_repeat_mean": per_seed}
        for metric in ("macro_f1", "CHID_f1"):
            values = [per_seed[str(seed)][metric] for seed in seeds]
            aggregate[condition][metric] = {"mean": float(np.mean(values)), "seed_sample_sd": float(np.std(values, ddof=1))}
    critical = ["CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19"]
    case_scores = {}
    for sid in critical:
        selected = [row for row in rows if row["structure_id"] == sid]
        case_scores[sid] = {"all_roi_predictions_remain_CHIN": all(row["roi_prediction"] == "1" for row in selected),
                            "all_background_predictions_remain_CHIN": all(row["background_prediction"] == "1" for row in selected)}
        for condition in conditions:
            case_scores[sid][condition] = [float(np.mean([float(row[f"{condition}_trueclass_probability"]) for row in selected if int(row["seed"]) == seed])) for seed in seeds]
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="microcture-probe-plots-", dir="/tmp") as cache:
        os.environ["MPLBACKEND"], os.environ["MPLCONFIGDIR"] = "Agg", cache
        import matplotlib.pyplot as plt
        colors, labels = ("#555555", "#D55E00", "#0072B2"), ("Unchanged input", "ROI shuffle", "Matched background shuffle")
        with plt.rc_context({"pdf.fonttype": 42, "font.size": 10}):
            fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
            for ax, metric, title in zip(axes[:2], ("macro_f1", "CHID_f1"), ("Complete OOF macro F1", "Complete OOF CHID F1")):
                for x, condition in enumerate(conditions):
                    values = [aggregate[condition]["per_seed_repeat_mean"][str(seed)][metric] for seed in seeds]
                    for offset, value, marker in zip((-.09, 0, .09), values, ("o", "s", "^")):
                        ax.scatter(x + offset, value, color=colors[x], marker=marker, s=40)
                    stat = aggregate[condition][metric]
                    ax.errorbar(x, stat["mean"], yerr=stat["seed_sample_sd"], color="black", fmt="D", capsize=4)
                    ax.text(x, stat["mean"] + stat["seed_sample_sd"] + .035, f"{stat['mean']:.3f}", ha="center")
                ax.set_xticks(range(3), ("Baseline", "ROI", "Background"))
                ax.set_ylim(0, 1)
                ax.set_title(title)
                ax.grid(axis="y", alpha=.2)
                ax.spines[["top", "right"]].set_visible(False)
            ax = axes[2]
            for index, condition in enumerate(conditions):
                means = [np.mean(case_scores[sid][condition]) for sid in critical]
                sds = [np.std(case_scores[sid][condition], ddof=1) for sid in critical]
                ax.errorbar(np.arange(6) + (index - 1) * .12, means, yerr=sds, fmt="o", capsize=3,
                            color=colors[index], markersize=4, label=labels[index])
            ax.set_xticks(range(6), [sid.replace("CHID_", "") for sid in critical])
            ax.set_xlabel("CHID ID (all remain predicted CHIN)")
            ax.set_ylabel("Mean probability of annotated CHID")
            ax.set_ylim(0, .55)
            ax.set_title("Six persistent misses: fixed weights")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8, loc="upper left")
            fig.suptitle("M2+ / I3 / H3: matched spatial-permutation sensitivity", fontsize=14)
            fig.text(.5, .018, "Each point: one endpoint seed, averaged over 3 shuffle repeats; SD across 3 endpoint seeds, not CI.\n"
                     "344 unique structures per complete OOF. Selected pixels matched by model64 offset; geometry/validity/value multisets retained.\n"
                     "Out-of-distribution sensitivity, not biological causality or independent validation.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .15, 1, .94))
            for extension in ("png", "pdf"):
                fig.savefig(output_dir / f"shape_probe.{extension}", dpi=220, facecolor="white")
            plt.close(fig)
    result = {"probe_signature": summary["probe_signature"], "source_summary_sha256": sha256(probe_dir / "summary.json"),
              "aggregation": "Average shuffle repeats within each endpoint seed, then mean/sample SD across 3 endpoint seeds; no expanded sample size",
              "unique_structures": 344, "conditions": aggregate, "six_persistent_misses": case_scores,
              "all_baselines_exactly_reproduced": all(row["max_abs_probability_difference"] == 0 for row in summary["baseline_checks"]),
              "all_comparisons_have_344_effectively_matched_cases": all(row["paired_matched_effects"]["all"]["n_unique_structures"] == 344 for row in comparisons),
              "plot_builder_sha256": sha256(Path(__file__))}
    (output_dir / "shape_probe_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "results/task1_morphology_probe_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/task1_morphology_review_v1")
    args = parser.parse_args()
    result = run(args.probe_dir, args.output_dir)
    print(json.dumps({"status": "complete", "conditions": result["conditions"], "unique_structures": 344}, ensure_ascii=False))
