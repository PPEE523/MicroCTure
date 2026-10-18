"""Audit retained task 3/4 numbers and build two compact delivery figures.

This reads saved small NPZ/CSV tables only. It neither scans raw contacts nor
fits models, and never rewrites the historical task directories. Run from the
project root with the microcture Conda environment.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = _layout_root
TYPES = ("OPCID", "CHIN", "CHID")
CONDITIONS = ("WT_37C", "delta_stpA", "delta_hns_stpA")
LABELS = ("WT", r"$\Delta stpA$", r"$\Delta hns\Delta stpA$")
COLORS = ("#3266a5", "#d38630", "#a74978")


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def boolean(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Invalid stored boolean: {value!r}")
    return value == "True"


def file_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def contrast(wt: list[float], treated: list[float], pseudo: float,
             fold: float) -> tuple[float, float, float, str]:
    wt_array = np.asarray(wt, dtype=float)
    treated_array = np.asarray(treated, dtype=float)
    cross = np.log2((treated_array[:, None] + pseudo) / (wt_array[None, :] + pseudo))
    threshold = np.log2(fold)
    direction = ("increased" if cross.min() > threshold else
                 "decreased" if cross.max() < -threshold else "inconsistent_or_small")
    effect = float(np.log2((treated_array.mean() + pseudo) / (wt_array.mean() + pseudo)))
    return effect, float(cross.min()), float(cross.max()), direction


def inspect(config: dict, tracks: dict, pages: list[dict], values: list[dict],
            effects: list[dict], saved_summary: list[dict], structures: list[dict],
            genes: list[dict]) -> dict:
    signal = tracks["signals"]
    common = tracks["common_valid"]
    positions = tracks["positions"]
    assert signal.shape == (6, len(common))
    assert np.array_equal(positions, np.arange(len(common)) * config["resolution_bp"])
    assert np.all(np.isfinite(signal[:, common]))
    assert np.allclose(tracks["condition_means"], signal.reshape(3, 2, -1).mean(axis=1), equal_nan=True)
    assert len(pages) == 465 and int(pages[0]["start"]) == 0
    assert all(int(a["end"]) == int(b["start"]) for a, b in zip(pages, pages[1:]))
    assert int(pages[-1]["end"]) == config["chrom_length"]
    assert len(structures) == 344 and len({r["structure_id"] for r in structures}) == 344
    lookup = {(r["structure_id"], r["condition"], int(r["replicate"])): r for r in values}
    assert len(lookup) == len(values) == 344 * 6
    assert len(effects) == 344 * 2
    max_error = 0.0
    for row in effects:
        sid, condition = row["structure_id"], row["condition"]
        wt = [lookup[(sid, "WT_37C", j)] for j in (1, 2)]
        treated = [lookup[(sid, condition, j)] for j in (1, 2)]
        base = contrast([float(r["signal"]) for r in wt], [float(r["signal"]) for r in treated],
                        config["pseudocount_per_million"], config["fold_change_threshold"])
        oe = contrast([float(r["mean_oe"]) for r in wt], [float(r["mean_oe"]) for r in treated],
                      0.01, config["fold_change_threshold"])
        for key, calculated in zip(("log2_fold_change", "min_cross_repeat_log2_fc", "max_cross_repeat_log2_fc"), base[:3]):
            error = abs(float(row[key]) - calculated)
            assert error < 1e-12, (sid, condition, key, error)
            max_error = max(max_error, error)
        assert abs(float(row["oe_log2_fold_change"]) - oe[0]) < 1e-12
        assert row["direction"] == base[3] and row["oe_direction"] == oe[3]
        quality = (int(row["valid_pairs"]) >= config["structure_min_pairs"]
                   and float(row["valid_fraction"]) >= config["structure_min_valid_fraction"])
        assert boolean(row["quality_pass"]) == quality
        assert boolean(row["descriptive_change"]) == (quality and base[3] != "inconsistent_or_small")
        assert boolean(row["oe_direction_agrees"]) == bool(np.sign(base[0]) == np.sign(oe[0]))
    summary = []
    for condition in CONDITIONS[1:]:
        for kind in TYPES:
            subset = [r for r in effects if r["condition"] == condition and r["type"] == kind]
            good = [r for r in subset if boolean(r["quality_pass"])]
            record = {"condition": condition, "type": kind, "total": len(subset), "quality_pass": len(good),
                      "increased": sum(boolean(r["descriptive_change"]) and r["direction"] == "increased" for r in subset),
                      "decreased": sum(boolean(r["descriptive_change"]) and r["direction"] == "decreased" for r in subset),
                      "median_log2_fc": float(np.median([float(r["log2_fold_change"]) for r in good]))}
            saved = next(r for r in saved_summary if r["condition"] == condition and r["type"] == kind)
            for key in ("total", "quality_pass", "increased", "decreased"):
                assert record[key] == int(saved[key])
            assert abs(record["median_log2_fc"] - float(saved["median_log2_fc"])) < 1e-12
            summary.append(record)
    valid_intervals = {r["structure_id"] for r in effects if boolean(r["quality_pass"])}
    changed = [r for r in effects if boolean(r["descriptive_change"])]
    disagreement = [r for r in changed if r["oe_direction"] != r["direction"]]
    return {
        "task3": {"status": "numeric_tracks_and_compact_visualization_verified",
                  "samples": 6, "conditions": 3, "resolution_bp": config["resolution_bp"],
                  "total_bins": len(common), "common_valid_bins": int(common.sum()),
                  "common_valid_percent": float(common.mean() * 100), "contiguous_page_index": len(pages),
                  "genome_length_bp": config["chrom_length"], "last_page": [int(pages[-1]["start"]), int(pages[-1]["end"])],
                  "gene_records": len(genes), "author_intervals": len(structures),
                  "repeat_pearson": {c: float(np.corrcoef(signal[i, common], signal[i+1, common])[0, 1])
                                     for c, i in zip(CONDITIONS, (0, 2, 4))},
                  "complete_legacy_figure_set_present": False,
                  "complete_figure_set_regenerated_this_run": False},
        "task4": {"status": "descriptive_comparisons_independently_recomputed",
                  "structure_sample_values": len(values), "condition_comparisons": len(effects),
                  "eligible_intervals": len(valid_intervals), "excluded_intervals": len(structures)-len(valid_intervals),
                  "eligible_comparison_rows": sum(boolean(r["quality_pass"]) for r in effects),
                  "max_main_effect_recomputation_error": max_error, "summary": summary,
                  "descriptive_changes_total": len(changed),
                  "oe_same_effect_sign": sum(boolean(r["oe_direction_agrees"]) for r in changed),
                  "oe_same_four_ratio_direction": len(changed)-len(disagreement),
                  "oe_threshold_exceptions": [{k: r[k] for k in ("structure_id", "condition", "start", "end", "log2_fold_change", "oe_log2_fold_change", "oe_direction")} for r in disagreement]},
        "rule": {"fold_change": config["fold_change_threshold"], "four_cross_repeat_contrasts_required": True,
                 "biological_replicates_per_condition": 2, "minimum_valid_fraction": config["structure_min_valid_fraction"],
                 "minimum_valid_pairs": config["structure_min_pairs"]},
        "limitations": ["No raw contact scan, preprocessing or model training was performed.",
                        "The 465-page index is not 465 currently shipped images; this run creates one compact task-3 overview.",
                        "Only two biological replicates per condition; four cross-repeat ratios are not four independent replicates.",
                        "Effects are relative, descriptive contact changes; no p-values, FDR, causal or morphology certification.",
                        "Gene coordinates remain provisional; shared masks may omit condition-specific loss.",
                        "Track repeat correlation is reproducibility of a contact summary, not classification accuracy."]}


def plot_tracks(plt, Rectangle, config, tracks, structures, genes, audit, output):
    # Predeclared round-number locus: never select a window by effect size.
    lo, hi = 1_050_000, 1_100_000
    signal, pos = tracks["signals"], tracks["positions"]
    common = tracks["common_valid"]
    masked = np.where(common[None, :], signal, np.nan)
    means = masked.reshape(3, 2, -1).mean(axis=1)
    fig, axes = plt.subplots(6, 1, figsize=(13, 12.5), gridspec_kw={"height_ratios": [2.2, 1.6, 1.6, 1.6, .8, 1.1]})
    fig.subplots_adjust(top=.91, bottom=.08, hspace=.40, left=.095, right=.97)
    ax = axes[0]
    for j in range(3):
        # Same 10-kb display aggregation for every condition; full 100-bp data retained.
        block_x, block_y = [], []
        for start in range(0, len(pos), 100):
            values = means[j, start:start+100]
            block_x.append(pos[start:start+100].mean()/1e6)
            block_y.append(np.sqrt(np.nanmean(values)) if np.isfinite(values).any() else np.nan)
        correlation = audit["task3"]["repeat_pearson"][CONDITIONS[j]]
        ax.plot(block_x, block_y, color=COLORS[j], lw=1.15, label=f"{LABELS[j]} (repeat r={correlation:.4f})")
    ax.axvspan(lo/1e6, hi/1e6, color="#333333", alpha=.15)
    ax.set(xlim=(0, config["chrom_length"]/1e6), ylabel="sqrt(relative contact signal)",
           title="A | Genome position (Mb), 10-kb display bins; gray band = fixed locus below")
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    selected = (pos >= lo) & (pos < hi)
    ymax = float(np.nanmax(np.sqrt(masked[:, selected]))) * 1.12
    for j, ax in enumerate(axes[1:4]):
        for k in (2*j, 2*j+1):
            ax.plot(pos[selected]/1000, np.sqrt(masked[k, selected]), lw=.65, alpha=.4, color=COLORS[j])
        ax.plot(pos[selected]/1000, np.sqrt(means[j, selected]), lw=1.15, color=COLORS[j])
        ax.set(xlim=(lo/1000, hi/1000), ylim=(0, ymax), ylabel=LABELS[j])
        if j == 0:
            ax.set_title("B | Fixed 1,050-1,100 kb locus, 100-bp bins; thin = repeats, thick = condition mean", fontsize=10)
        ax.tick_params(labelbottom=False)
    gene_ax = axes[4]
    for row in genes:
        start, end = int(row["start"]), int(row["end"])
        if start < hi and end > lo:
            lane = 0 if row["strand"] == "+" else 1
            left, right = max(start, lo)/1000, min(end, hi)/1000
            gene_ax.add_patch(Rectangle((left, lane-.25), right-left, .5, facecolor="#8299b0", edgecolor="white", linewidth=.4))
    gene_ax.set(xlim=(lo/1000, hi/1000), ylim=(1.6, -.6), yticks=(0, 1), yticklabels=("+", "-"), ylabel="Genes")
    gene_ax.tick_params(labelbottom=False)
    struct_ax = axes[5]
    displayed = []
    for row in structures:
        start, end = int(row["start"]), int(row["end"])
        if start < hi and end > lo:
            displayed.append(row["structure_id"])
            lane = TYPES.index(row["type"])
            left, right = max(start, lo)/1000, min(end, hi)/1000
            struct_ax.add_patch(Rectangle((left, lane-.25), right-left, .5, facecolor=COLORS[lane], alpha=.5, edgecolor="white", linewidth=.5))
    struct_ax.set(xlim=(lo/1000, hi/1000), ylim=(2.6, -.6), yticks=(0, 1, 2), yticklabels=TYPES,
                  xlabel="Genome position (kb)", ylabel="Author\nintervals")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e8ec", linewidth=.5)
    fig.suptitle("Task 3 | Multi-condition contact tracks", x=.095, ha="left", fontsize=19, fontweight="bold")
    fig.text(.095, .935, f"6 samples | 46,417 bins | 45,499 common-valid bins (98.02%) | 465-page coordinate index", fontsize=10, color="#4e5968")
    fig.text(.095, .025, "Relative signal: contacts per million / valid pair, distances 200 bp-10 kb.\nProvisional gene/structure coordinates; overlapping author intervals retained. Locus fixed before reading effects.", fontsize=9, color="#4e5968")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    return {"fixed_locus_bp": [lo, hi], "selection_rule": "Fixed round-number locus, not selected by differential effect or classification success",
            "overlapping_author_intervals": displayed, "genome_display_bin_bp": 10_000, "local_display_bin_bp": 100}


def plot_effects(plt, config, effects, audit, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [1.6, 1]})
    fig.subplots_adjust(top=.86, bottom=.13, left=.09, right=.97, wspace=.24, hspace=.34)
    threshold = np.log2(config["fold_change_threshold"])
    all_good = [float(r["log2_fold_change"]) for r in effects if boolean(r["quality_pass"])]
    lower, upper = min(all_good)-.12, max(all_good)+.55
    rng = np.random.default_rng(42)  # visualization jitter only
    for col, condition in enumerate(CONDITIONS[1:]):
        ax = axes[0, col]
        subsets = [[r for r in effects if r["condition"] == condition and r["type"] == kind and boolean(r["quality_pass"])] for kind in TYPES]
        data = [[float(r["log2_fold_change"]) for r in subset] for subset in subsets]
        ax.boxplot(data, positions=np.arange(3), widths=.40, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#edf0f5", "edgecolor": "#7b8794"}, medianprops={"color": "#222222", "linewidth": 1.8},
                   whiskerprops={"color": "#7b8794"}, capprops={"color": "#7b8794"})
        for i, subset in enumerate(subsets):
            jitter = rng.uniform(-.18, .18, len(subset))
            for j, row in enumerate(subset):
                changed = boolean(row["descriptive_change"])
                color = "#cb5548" if changed and row["direction"] == "increased" else "#3266a5" if changed else "#a7adb5"
                ax.scatter(i+jitter[j], float(row["log2_fold_change"]), s=17 if changed else 9, color=color, alpha=.8 if changed else .5, zorder=3)
            median = float(np.median(data[i]))
            ax.text(i, upper-.13, f"n={len(subset)}\nmedian {median:+.3f}", ha="center", va="top", fontsize=8)
        ax.axhline(0, color="#777777", lw=.8)
        for y in (-threshold, threshold):
            ax.axhline(y, color="#969da7", ls="--", lw=.8)
        ax.set(xticks=np.arange(3), xticklabels=TYPES, ylabel="Relative-contact log2 fold change", ylim=(lower, upper), title=f"{LABELS[col+1]} / WT")
        count_ax = axes[1, col]
        summary = [r for r in audit["task4"]["summary"] if r["condition"] == condition]
        inc = [r["increased"] for r in summary]
        dec = [r["decreased"] for r in summary]
        count_ax.bar(np.arange(3)-.18, inc, width=.34, color="#cb5548", label="Increased")
        count_ax.bar(np.arange(3)+.18, dec, width=.34, color="#3266a5", label="Decreased")
        for i in range(3):
            count_ax.text(i-.18, inc[i]+1.3, str(inc[i]), ha="center", fontsize=10)
            count_ax.text(i+.18, dec[i]+1.3, str(dec[i]), ha="center", fontsize=10)
        count_ax.set(xticks=np.arange(3), xticklabels=[f"{kind}\neligible {r['quality_pass']}" for kind, r in zip(TYPES, summary)],
                     ylabel="Intervals meeting the full rule", ylim=(0, 90), title=f"{sum(inc)+sum(dec)} / 327 eligible intervals changed")
        count_ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e8ec", linewidth=.5, zorder=0)
    fig.suptitle("Task 4 | Replicate-aware descriptive condition effects", x=.09, ha="left", fontsize=18, fontweight="bold")
    fig.text(.09, .90, "344 author intervals | 327 eligible, 17 excluded | 688 comparisons independently recomputed", fontsize=10, color="#4e5968")
    fig.text(.09, .045,
             "Each dot = one eligible author interval; colors mark the full four-ratio rule, gray = below/inconsistent.\n"
             "Dashed lines show mean-effect +/- log2(1.5) for orientation; the actual rule requires ALL four cross-repeat ratios.\n"
             "Two biological replicates per condition; descriptive relative changes only, without p-values, FDR or causal claims.", fontsize=9, color="#4e5968")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=_layout_resolve(ROOT / "submission"))
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT) or not (out / "figures").is_dir():
        raise ValueError("Use an existing project-internal output directory with a figures child")
    source = _layout_resolve(ROOT / "data/legacy_experiments/results")
    paths = {"config": _layout_resolve(ROOT / "configs/conditions.json"), "tracks": source / "task3/signals.npz",
             "pages": source / "task3/pages.csv", "values": source / "task4/structure_signals.csv",
             "effects": source / "task4/differential_structures.csv", "summary": source / "task4/summary.csv",
             "structures": _layout_resolve(ROOT / "data/processed/structures.csv"), "genes": _layout_resolve(ROOT / "data/processed/genes.csv")}
    paths = {key: _layout_resolve(path) for key, path in paths.items()}
    config = json.loads(paths["config"].read_text())
    with np.load(paths["tracks"]) as saved:
        tracks = {key: saved[key].copy() for key in saved.files}
    pages, values, effects, saved_summary, structures, genes = [rows(paths[key]) for key in ("pages", "values", "effects", "summary", "structures", "genes")]
    audit = inspect(config, tracks, pages, values, effects, saved_summary, structures, genes)
    with tempfile.TemporaryDirectory(prefix="microcture-delivery-task34-") as temporary:
        import os
        os.environ["MPLCONFIGDIR"] = temporary
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "normal"})
        track_path = out / "figures/task34_delivery_tracks.png"
        effect_path = out / "figures/task34_delivery_effects.png"
        audit["task3"]["display"] = plot_tracks(plt, Rectangle, config, tracks, structures, genes, audit, track_path)
        plot_effects(plt, config, effects, audit, effect_path)
    audit["status"] = "passed"
    audit["execution"] = {"raw_pixels_scanned": False, "models_fitted": 0, "figures_created": 2,
                          "recomputed_from_saved_tables_only": True}
    audit["inputs"] = {key: file_record(path) for key, path in paths.items()}
    audit["source"] = file_record(Path(__file__).resolve())
    audit["outputs"] = [file_record(track_path), file_record(effect_path)]
    summary_path = out / "task34_delivery_summary.json"
    summary_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "comparisons": len(effects),
                      "figures": [str(track_path.relative_to(ROOT)), str(effect_path.relative_to(ROOT))],
                      "summary": str(summary_path.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
