"""Build compact task 5/6 delivery evidence from retained arrays; never train.

Run from the project root with Conda microcture. Outputs are restricted to
existing submission/ and submission/figures/ directories. Saved predictions,
coordinates, historical reports, and checkpoints are never modified.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve, logical as _layout_logical
_layout_bootstrap()
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from review_saved_task56 import review_task5, review_task6


ROOT = _layout_root
BASE = _layout_resolve(ROOT / "data/legacy_experiments/results")
OUT = _layout_resolve(ROOT / "submission")
FIGURES = OUT / "figures"
FAMILIES5 = ("bilinear", "bicubic", "plain_cnn", "residual_cnn")
FAMILIES6 = ("mds", "geometry", "graph")
LABELS = {"bilinear": "Bilinear", "bicubic": "Bicubic", "plain_cnn": "Plain CNN",
          "residual_cnn": "Residual CNN", "mds": "MDS", "geometry": "Geometry", "graph": "Graph decoder"}
COLORS = {"bilinear": "#94a3b8", "bicubic": "#475569", "plain_cnn": "#d97706",
          "residual_cnn": "#0284c7", "mds": "#475569", "geometry": "#d97706", "graph": "#0284c7"}


def read_csv(path):
    with _layout_resolve(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def family(name):
    return name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name


def digest(path):
    value = hashlib.sha256()
    with _layout_resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_once(path, value):
    """Identical reruns are allowed; scientific artifacts cannot be replaced."""
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Existing delivery artifact differs: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(value)


def figure(fig, name):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=160, facecolor="white",
                metadata={"Software": "MicroCTure retained-result delivery"})
    plt.close(fig)
    path = FIGURES / name
    write_once(path, stream.getvalue())
    return str(path.relative_to(ROOT))


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), seed_std_population=float(values.std(ddof=0)),
                n_seeds=len(values))


def summary5(records):
    result = {}
    for name in FAMILIES5:
        result[name] = {}
        for rep in ("1", "2"):
            result[name][f"rep{rep}"] = {}
            for kind in ("all", "OPCID", "CHIN", "CHID"):
                selected = [r for r in records if family(r["model"]) == name
                            and r["replicate"] == rep and r["type"] == kind]
                result[name][f"rep{rep}"][kind] = {
                    key: stats([float(r[key]) for r in selected])
                    for key in ("psnr", "ssim", "shape_correlation", "profile_correlation", "boundary_shift_bp")}
    return result


def summary6(records):
    result = {}
    for name in FAMILIES6:
        result[name] = {}
        for rep in ("1", "2"):
            selected = [r for r in records if family(r["model"]) == name
                        and r["replicate"] == rep and r["partition"] == "test"]
            result[name][f"rep{rep}"] = {key: stats([float(r[key]) for r in selected]) for key in
                ("log_distance_rmse", "contact_log_pearson", "contact_spearman",
                 "distance_adjusted_contact_pearson", "within_distance_pearson")}
    return result


def task5_comparison(summary):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    for axis, metric, label in zip(axes, ("psnr", "ssim", "boundary_shift_bp"),
                                    ("PSNR (dB): higher is better", "SSIM: higher is better",
                                     "Boundary shift (bp): lower is better")):
        for index, name in enumerate(FAMILIES5):
            for rep, offset, hatch in (("rep1", -.18, ""), ("rep2", .18, "///")):
                value = summary[name][rep]["all"][metric]
                axis.bar(index + offset, value["mean"], width=.32, color=COLORS[name],
                         yerr=value["seed_std_population"], capsize=3, hatch=hatch,
                         alpha=1 if rep == "rep1" else .65, edgecolor="white")
        axis.set_xticks(range(4), [LABELS[n] for n in FAMILIES5], rotation=24, ha="right")
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#64748b", label="WT replicate 1"),
                        Patch(facecolor="#64748b", hatch="///", alpha=.65, label="WT replicate 2")],
               loc="upper center", bbox_to_anchor=(.51, .91), ncol=2, frameon=False)
    fig.suptitle("Task 5 | Pixel fidelity improves; boundary localization does not", fontsize=16, fontweight="bold")
    fig.text(.5, .025, "53 test positions per replicate; CNN bars average seeds 42/43/44. Error bars: seed SD (population), not confidence intervals.\n"
             "Synthetic 800-to-200 bp reconstruction; noisy measurements are the reference.", ha="center", fontsize=9, color="#475569")
    fig.subplots_adjust(top=.78, bottom=.25, wspace=.28)
    return figure(fig, "task56_delivery_task5_tradeoff.png")


def paired_task5(records, windows):
    test = {r["structure_id"]: r for r in windows if r["split"] == "test" and r["replicate"] == "1"}
    result = {}
    for metric in ("psnr", "ssim", "boundary_shift_bp"):
        per_position = []
        for structure_id in sorted(test):
            delta = []
            for rep in ("1", "2"):
                selected = [r for r in records if r["structure_id"] == structure_id and r["replicate"] == rep]
                base = next(float(r[metric]) for r in selected if r["model"] == "bicubic")
                neural = np.mean([float(r[metric]) for r in selected if family(r["model"]) == "residual_cnn"])
                delta.append(neural - base)
            per_position.append(dict(structure_id=structure_id, group_id=test[structure_id]["group_id"],
                                     type=test[structure_id]["type"], delta=float(np.mean(delta))))
        values = np.array([r["delta"] for r in per_position])
        result[metric] = dict(mean_delta=float(values.mean()), median_delta=float(np.median(values)),
                              positive=int((values > 1e-10).sum()), negative=int((values < -1e-10).sum()),
                              tied=int((abs(values) <= 1e-10).sum()), n_positions=len(values),
                              per_position=per_position)
    return dict(comparison="Residual CNN minus bicubic; average 3 seeds first, then 2 replicates per position",
                independent_unit_warning="53 positions lie in 7 spatial groups; counts are descriptive, not independent-sample significance tests.",
                spatial_groups=sorted({r["group_id"] for r in test.values()}, key=int), metrics=result)


def task5_examples(windows):
    directory = _layout_resolve(BASE / "task5")
    with np.load(directory / "bicubic_reconstruction.npz", allow_pickle=False) as z:
        bicubic, target, masks = z["prediction"], z["target"], z["mask"]
        indices = z["window_indices"]
    with np.load(directory / "residual_cnn_42_reconstruction.npz", allow_pickle=False) as z:
        residual = z["prediction"]
        np.testing.assert_array_equal(indices, z["window_indices"])
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 9.3))
    selection = []
    for row, kind in enumerate(("OPCID", "CHIN", "CHID")):
        candidates = [(i, windows[int(index)]) for i, index in enumerate(indices)
                      if windows[int(index)]["replicate"] == "1" and windows[int(index)]["type"] == kind]
        candidates.sort(key=lambda item: (int(item[1]["start"]), item[1]["structure_id"]))
        i, meta = candidates[(len(candidates) - 1) // 2]
        selection.append(dict(type=kind, structure_id=meta["structure_id"], start=int(meta["start"]),
                              end=int(meta["end"]), replicate=1, seed=42, total_class_positions=len(candidates)))
        vmax = max(float(np.quantile(target[i][masks[i]], .99)), 1e-6)
        error_limit = max(float(np.quantile(np.abs(residual[i] - target[i])[masks[i]], .99)), 1e-6)
        for col, matrix in enumerate((target[i], bicubic[i], residual[i], residual[i] - target[i])):
            axis = axes[row, col]
            image = np.where(masks[i], matrix, np.nan)
            kwargs = dict(cmap="coolwarm", vmin=-error_limit, vmax=error_limit) if col == 3 else dict(cmap="magma", vmin=0, vmax=vmax)
            axis.imshow(image, origin="lower", interpolation="nearest", **kwargs)
            axis.set_xticks([]); axis.set_yticks([])
            if row == 0:
                axis.set_title(("Measured target (200 bp)", "Bicubic", "Residual CNN (seed 42)", "CNN minus target")[col], fontsize=10)
            if col == 0:
                axis.set_ylabel(f"{meta['structure_id']}\n{meta['start']}–{meta['end']} bp", fontsize=10)
    fig.suptitle("Task 5 | Fixed-rule examples across all three annotated classes", fontsize=16, fontweight="bold")
    fig.text(.5, .02, "Selection: lower median genomic start within each class, WT replicate 1; seed 42 fixed. No performance-based example selection.\n"
             "Each row uses one shared target-derived intensity scale (99th percentile); invalid pixels are blank. Error color scales are row-specific.",
             ha="center", fontsize=9, color="#475569")
    fig.subplots_adjust(top=.91, bottom=.1, left=.13, right=.98, wspace=.06, hspace=.12)
    return figure(fig, "task56_delivery_task5_examples.png"), selection


def task6_comparison(summary):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    metrics = ("log_distance_rmse", "contact_log_pearson", "distance_adjusted_contact_pearson", "within_distance_pearson")
    labels = ("Log-distance RMSE: lower is better", "Global log-contact Pearson: higher is better",
              "Background-subtracted Pearson: higher is better", "Within-separation Pearson: higher is better")
    for axis, metric, label in zip(axes.flat, metrics, labels):
        for i, name in enumerate(FAMILIES6):
            for rep, offset, marker in (("rep1", -.08, "o"), ("rep2", .08, "s")):
                value = summary[name][rep][metric]
                axis.errorbar(i + offset, value["mean"], yerr=value["seed_std_population"], marker=marker,
                              color=COLORS[name], linestyle="none", markersize=8, capsize=4,
                              markerfacecolor=COLORS[name] if rep == "rep1" else "white")
                axis.annotate(f"{value['mean']:.3f}", (i + offset, value["mean"]),
                              xytext=(-9, 11 if rep == "rep1" else -17), textcoords="offset points", fontsize=9)
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_xticks(range(3), [LABELS[n] for n in FAMILIES6])
        axis.set_xlim(-.4, 2.4); axis.margins(y=.35)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Task 6 | Better distance fitting is not better on every contact metric", fontsize=15, fontweight="bold")
    fig.text(.5, .035, "64,089 held-out pairs on the same 929-node chromosome. Filled circles: WT rep1; open squares: WT rep2.\n"
             "Geometry/graph: mean ± population seed SD (42/43/44). Transductive edge holdout; relative coordinates only.",
             ha="center", fontsize=9, color="#475569")
    fig.subplots_adjust(top=.89, bottom=.13, wspace=.3, hspace=.48)
    return figure(fig, "task56_delivery_task6_tradeoff.png")


def task6_embedding():
    directory = _layout_resolve(BASE / "task6")
    with np.load(directory / "evaluation_inputs.npz", allow_pickle=False) as z:
        observed = z["counts_rep1"]
    fig = plt.figure(figsize=(13.8, 8.6))
    length = json.loads((directory / "manifest.json").read_text())["chromosome_length"]
    for i, name in enumerate(("mds", "geometry_42", "graph_42")):
        coordinates = read_csv(directory / f"{name}_coordinates.csv")
        xyz = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in coordinates])
        axis = fig.add_subplot(2, 3, i + 1, projection="3d")
        axis.plot(*xyz.T, color="#64748b", linewidth=.45, alpha=.45)
        axis.scatter(*xyz.T, c=np.arange(len(xyz)) / len(xyz), cmap="turbo", s=3, alpha=.9)
        axis.set_title(LABELS[family(name)] + (" (seed 42)" if name != "mds" else ""), fontweight="bold")
        axis.set_xlabel("x (relative)", fontsize=8); axis.set_ylabel("y (relative)", fontsize=8)
        axis.set_zlabel("z (relative)", fontsize=8)
        axis.tick_params(labelsize=7); axis.set_box_aspect((1, 1, 1))
    vmax = float(np.quantile(np.log1p(observed)[np.triu_indices(len(observed), 2)], .995))
    for i, (name, title) in enumerate(((None, "Observed WT rep1 contacts"), ("mds", "MDS predicted contacts"),
                                      ("graph_42", "Graph decoder predicted contacts"))):
        if name is None:
            matrix = observed
        else:
            with np.load(directory / f"{name}_maps.npz", allow_pickle=False) as z:
                matrix = z["predicted_contacts"]
        axis = fig.add_subplot(2, 3, i + 4)
        display = np.log1p(matrix).copy(); np.fill_diagonal(display, np.nan)
        image = axis.imshow(display, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                            extent=(0, length / 1e6, 0, length / 1e6), interpolation="nearest")
        axis.set_title(title, fontsize=10); axis.set_xlabel("Genomic position (Mb)", fontsize=9)
        axis.set_ylabel("Genomic position (Mb)", fontsize=9)
        fig.colorbar(image, ax=axis, fraction=.045, pad=.02, label="log(1 + contact)")
    fig.suptitle("Task 6 | Relative 3D embeddings and reconstructed contact maps", fontsize=16, fontweight="bold")
    fig.text(.5, .025, "Coordinate colors follow genomic order; displayed orientations have no biological meaning. Seed 42 is fixed for both learned models.\n"
             "Contact maps share one scale fixed from observed rep1. These embeddings do not establish a unique physical single-cell structure.",
             ha="center", fontsize=9, color="#475569")
    fig.subplots_adjust(top=.92, bottom=.115, wspace=.4, hspace=.24)
    return figure(fig, "task56_delivery_task6_embeddings.png")


def main():
    if not OUT.is_dir() or not FIGURES.is_dir():
        raise ValueError("Use the existing project submission directories")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    checked5, _ = review_task5(_layout_resolve(BASE / "task5"))
    checked6, _ = review_task6(_layout_resolve(BASE / "task6"))
    source_files = []
    for task in ("task5", "task6"):
        for path in sorted((_layout_resolve(BASE / task)).iterdir()):
            if path.is_file() and path.suffix in (".npz", ".csv", ".json"):
                source_files.append(dict(path=str(_layout_logical(path)), bytes=path.stat().st_size, sha256=digest(path)))
    rows5 = read_csv(_layout_resolve(BASE / "task5/metrics.csv"))
    windows = read_csv(_layout_resolve(BASE / "task5/windows.csv"))
    values5 = summary5(read_csv(_layout_resolve(BASE / "task5/summary.csv")))
    values6 = summary6(read_csv(_layout_resolve(BASE / "task6/metrics.csv")))
    paired = paired_task5(rows5, windows)
    figure5 = task5_comparison(values5)
    examples, selected = task5_examples(windows)
    figure6 = task6_comparison(values6)
    embedding = task6_embedding()
    stability = []
    for name in ("geometry", "graph"):
        distances = {}
        for seed in (42, 43, 44):
            rows = read_csv(_layout_resolve(BASE / f"task6/{name}_{seed}_coordinates.csv"))
            distances[seed] = pdist(np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows]))
        for a, b in ((42, 43), (42, 44), (43, 44)):
            stability.append(dict(family=name, seed_a=a, seed_b=b,
                                  distance_correlation=float(np.corrcoef(distances[a], distances[b])[0, 1])))
    missing = [str(_layout_logical(_layout_resolve(BASE / task / f"{name}_{seed}.pt")))
               for task, names in (("task5", ("residual_cnn", "plain_cnn")), ("task6", ("geometry", "graph")))
               for name in names for seed in (42, 43, 44)
               if not (_layout_resolve(BASE / task / f"{name}_{seed}.pt")).is_file()]
    result = dict(schema_version=1, status="retained_results_recomputed_and_visualized",
                  interpretation="Development evidence; completion of numerical experiments is distinct from biological target achievement.",
                  training_performed=False, raw_preprocessing_performed=False, checkpoint_restoration_performed=False,
                  task5=dict(verification=checked5, models=values5, paired_descriptive_comparison=paired,
                             examples_selection_rule="WT rep1, lower median genomic start in each class; fixed seed42",
                             examples=selected, figures=[figure5, examples],
                             status="Synthetic 800-to-200 bp benchmark completed; structural-boundary improvement unsupported",
                             budget=json.loads((_layout_resolve(BASE / "task5/manifest.json")).read_text())["config"],
                             limitations=["53 positions in 7 spatial groups, CHID only4; replicates and seeds are not new positions.",
                                          "Reference is noisy measured contact matrix, not latent high-resolution truth.",
                                          "Original coordinate_status is provisional; boundary metric is local profile-minimum shift, not expert truth.",
                                          "No real low-depth to independently measured high-depth validation; no new training in delivery."]),
                  task6=dict(verification=checked6, models=values6, seed_stability_recomputed=stability,
                             figures=[figure6, embedding], status="Transductive relative 3D embedding completed; physical conformation validation unsupported",
                             budget=json.loads((_layout_resolve(BASE / "task6/manifest.json")).read_text())["config"],
                             limitations=["Same chromosome nodes shared across train/validation/test; edge holdout is not independent biological sampling.",
                                          "Contact-to-distance exponent1/3 and pseudocount0.5 are modeling assumptions.",
                                          "No calibrated spatial units or unique single-cell conformation; schematic domain labels do not validate geometry.",
                                          "No independent imaging or biological perturbation validation."]),
                  missing_checkpoints=missing, source_files=source_files,
                  figures_sha256={str(path.relative_to(ROOT)): digest(path) for path in
                      (_layout_resolve(ROOT / figure5), _layout_resolve(ROOT / examples), _layout_resolve(ROOT / figure6), _layout_resolve(ROOT / embedding))})
    output = OUT / "task56_delivery_summary.json"
    write_once(output, (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), task5_rows=checked5["metric_rows"],
                          task6_rows=checked6["metric_rows"], missing_checkpoints=len(missing),
                          figures=[figure5, examples, figure6, embedding],
                          training_performed=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
