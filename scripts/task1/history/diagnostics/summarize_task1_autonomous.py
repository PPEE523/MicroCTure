#!/usr/bin/env python3
"""Plot completed developmental stages; this is not a matched model ranking."""
from __future__ import annotations
import csv
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/task1_autonomous_review_v1")
    args = parser.parse_args()
    output = args.report_dir if args.report_dir.is_absolute() else ROOT / args.report_dir
    if any(p.is_symlink() for p in (output, *output.parents)) or ROOT / "reports" not in output.resolve().parents:
        raise ValueError("Use a separate project reports/ directory without symlinks")
    output = output.resolve()
    if output.exists() and any(p.is_file() and not (p.stat().st_mode & 0o222) for p in output.iterdir()):
        raise ValueError("Report is read-only; choose a fresh --report-dir")
    rows = []
    for run, arms, counts in [
        ("task1_roi_v1", [("R0", "R0: original crop"), ("R1", "R1: full ROI")], (75, 15)),
        ("task1_native_roi_v2", [("C0", "C0: matched coarse"), ("C1", "C1: native detail")], (90, 18)),
        ("task1_dropout_v1", [("R1", "D5: full ROI, dropout .5")], (60, 12)),
    ]:
        d = json.loads((ROOT / "results" / run / "summary.json").read_text())
        if d["status"] != "complete" or (d["completed_jobs"], d["complete_oof"]) != counts:
            raise ValueError(f"Incomplete registered study: {run}")
        for arm, label in arms:
            r = next(z for z in d["comparison"] if z["alignment"] == arm and z["condition"] == "I3")
            row = {"label": label, "run_dir": f"results/{run}", "source_signature": d["source_signature"],
                   "experiment_signature": d["experiment_signature"], "seeds": 3,
                   "stable_wrong_across_seeds": r["stable_wrong_count"], "development_gate_pass": r["development_gate_pass"]}
            for metric in ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1"):
                row[metric] = r[f"{metric}_mean"]
                row[f"{metric}_seed_SD"] = r[f"{metric}_seed_sample_std"]
            rows.append(row)
    d = json.loads((ROOT / "results/task1_shape_baseline_v2/summary.json").read_text())
    if d["status"] != "complete" or (d["completed_fits"], d["complete_OOF"]) != (40, 8):
        raise ValueError("Incomplete shape baseline")
    for model, label in [("lr_shape", "R1: fixed455 + LR"), ("svc_shape", "R1: fixed455 + SVC")]:
        r = next(z for z in d["comparison"] if z["model"] == model and z["condition"] == "I3")
        rows.append({"label": label, "run_dir": "results/task1_shape_baseline_v2", "source_signature": d["source_signature"],
                     "experiment_signature": d["experiment_signature"], "seeds": 1, "stable_wrong_across_seeds": "NA: one seed",
                     "development_gate_pass": r["development_gate_pass"],
                     **{k: r[k] for k in ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")},
                     **{f"{k}_seed_SD": "" for k in ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")}})
    d = json.loads((ROOT / "results/task1_fixed_fusion_v1/summary.json").read_text())
    if d["status"] != "complete" or (d["main_OOF"], d["pooled_OOF"]) != (12, 4):
        raise ValueError("Incomplete fixed fusion")
    r = next(z for z in d["comparisons"] if z["condition"] == "I3")
    import numpy as np
    scores = [d["conditions"]["I3"]["seeds"][str(seed)] for seed in (42, 43, 44)]
    fusion = {"label": "R1: fixed half LR + CNN", "run_dir": "results/task1_fixed_fusion_v1", "source_signature": d["source_signature"],
              "experiment_signature": d["experiment_signature"], "seeds": 3,
              "stable_wrong_across_seeds": len(d["conditions"]["I3"]["stable_wrong_ids"]), "development_gate_pass": False}
    for k in ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1"):
        values = [s[k] if k == "macro_f1" else s["per_class"]["CHID"][k.split("_", 1)[1]] for s in scores]
        fusion[k], fusion[f"{k}_seed_SD"] = float(np.mean(values)), float(np.std(values, ddof=1))
    rows.append(fusion)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with tempfile.TemporaryDirectory(prefix="microcture-stage-plot-", dir="/tmp") as tmp:
        os.environ["MPLCONFIGDIR"] = tmp
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(13, 8), layout="constrained"); FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        y = np.arange(len(rows))
        colors = ["#6d6d6d", "#2878b5", "#1b998b", "#e77932", "#7a5195", "#b18d17", "#9a9a9a", "#bd3e64"]
        for ax, metric, title in zip(axes.flat, ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1"), ("Macro F1", "CHID precision", "CHID recall", "CHID F1")):
            for i, row in enumerate(rows):
                sd = row[f"{metric}_seed_SD"]
                ax.errorbar(row[metric], i, xerr=None if sd == "" else sd, fmt="s" if row["seeds"] == 1 else "o", color=colors[i], capsize=3)
                ax.text(.98, i, f"{row[metric]:.3f}", va="center", ha="right", fontsize=9)
            ax.set(xlim=(0, 1.02), yticks=y, yticklabels=[r["label"] for r in rows], title=title)
            ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
            if metric != "CHID_f1":
                ax.axvline(.9 if metric == "macro_f1" else .85, color="#bc4444", ls="--", lw=1)
        fig.suptitle("Completed task1 studies on reused spatial groups: developmental evidence", fontsize=14)
        fig.supxlabel("Different input protocols across stages; no model ranking. Bars: optimization-seed SD, not biological CI.\nSquares: one fixed classical seed. Fusion shares the same LR member. Dashed: individual prospective gates.", fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(output / f"stage_scores.{ext}", dpi=220, bbox_inches="tight")
    print(json.dumps({"status": "complete", "stage_rows": len(rows), "actual_new_fits": 265, "complete_new_model_OOF": 53,
                      "primary_fusion_OOF": 12, "pooled_diagnostic_OOF": 4, "passing_complete_input_rows": sum(bool(r["development_gate_pass"]) for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
