"""Read and plot saved discovery diagnostics without fitting or reading contacts.

Consistency of saved tables is distinct from reproducing missing model weights.
The random-reference range below is descriptive, never a confidence interval.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path


TYPES = ("OPCID", "CHIN", "CHID")
QUALITY_METRICS = ("mean_oe", "valid_fraction", "zero_fraction")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, description):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer for {description}: {value!r}") from error
    _require(str(number) == str(value), f"Invalid integer for {description}: {value!r}")
    return number


def _finite(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid number for {description}: {value!r}") from error
    _require(math.isfinite(number), f"Nonfinite {description}")
    return number


def _boolean(value, description):
    if value in (True, "True", "true"):
        return True
    if value in (False, "False", "false"):
        return False
    raise ValueError(f"Invalid boolean for {description}: {value!r}")


def load_diagnostics(experiment_dir):
    """Validate five saved inputs and return JSON/CSV-ready diagnostic records.

    No checkpoint is loaded. No statistics are refitted. In particular, an
    internally consistent correlation table does not verify its original pixels.
    The complete method, class, domain and random-reference grids are required.
    """
    directory = Path(experiment_dir).resolve()
    sources = []

    def read(name, *, csv_file=False):
        path = directory / name
        payload = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload)})
        decoded = payload.decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(decoded))) if csv_file else json.loads(decoded)

    report = read("report.json")
    cfg = report["config"]
    seeds = [_integer(seed, "training seed") for seed in cfg["seeds"]]
    _require(len(seeds) == len(set(seeds)) and bool(seeds), "Training seeds are duplicate or empty")
    model_names = [f"{family}_{seed}" for family in ("ae", "dae") for seed in seeds]
    methods = ["shape", "density"]
    for family in ("ae", "dae"):
        for seed in seeds:
            methods.extend((f"{family}_{seed}", f"{family}_fusion_{seed}"))
    random_count = _integer(cfg["random_references"], "random references")
    _require(random_count > 0, "At least one random reference is required")
    random_names = [f"random_{seed}" for seed in range(random_count)]
    if "methods" in report:
        _require([row["method"] for row in report["methods"]] == methods,
                 "Report method grid differs from configured methods")
    budgets = {}
    for domain, key in (("full", "full_candidate_quotas"), ("test", "test_candidate_quotas")):
        quotas = [(_integer(width, "quota width"), _integer(count, "quota count"))
                  for width, count in cfg[key].items()]
        _require(bool(quotas) and all(width > 0 and count > 0 for width, count in quotas),
                 "Candidate quotas must contain positive widths and counts")
        budgets[domain] = {"windows": sum(count for _, count in quotas),
                           "covered_bins": sum(width * count for width, count in quotas)}

    recall_rows = read("known_recall.csv", csv_file=True)
    expected = {(domain, method, kind) for domain in budgets
                for method in methods + random_names for kind in TYPES}
    recalls, denominators = {}, {}
    for row in recall_rows:
        key = (row["domain"], row["method"], row["type"])
        _require(key in expected and key not in recalls, f"Unexpected or duplicate recall record: {key}")
        domain, _, kind = key
        values = {field: _integer(row[field], f"{key}/{field}")
                  for field in ("recalled", "total", "windows", "covered_bins")}
        _require(0 <= values["recalled"] <= values["total"], f"Invalid recall counts: {key}")
        _require(all(values[field] == budgets[domain][field] for field in budgets[domain]),
                 f"Candidate budget differs from configured scale quotas: {key}")
        denominator_key = (domain, kind)
        if denominator_key in denominators:
            _require(denominators[denominator_key] == values["total"],
                     f"Recall denominator differs across methods/random references: {denominator_key}")
        denominators[denominator_key] = values["total"]
        if values["total"]:
            recall = _finite(row["recall"], f"{key}/recall")
            _require(math.isclose(recall, values["recalled"] / values["total"], rel_tol=1e-10, abs_tol=1e-12),
                     f"Recall does not equal recalled/total: {key}")
        else:
            _require(row["recall"] in ("", "None", "null"), f"Zero denominator must retain NA: {key}")
            recall = None
        recalls[key] = {"domain": domain, "method": row["method"], "type": kind, **values, "recall": recall}
    _require(set(recalls) == expected, "Recall method/class/domain/random grid is incomplete")
    if "test_known_counts" in report:
        _require(all(denominators[("test", kind)] == report["test_known_counts"][kind] for kind in TYPES),
                 "Test recall denominator differs from report")
    test_retrieval = []
    for method in methods:
        for kind in TYPES:
            row = dict(recalls[("test", method, kind)])
            random_rows = [recalls[("test", name, kind)] for name in random_names]
            row["random_references"] = random_count
            row["random_range_interpretation"] = "observed minimum/maximum; not a confidence interval"
            for field in ("recall", "recalled"):
                values = [other[field] for other in random_rows]
                for suffix, value in (("mean", math.fsum(values) / len(values) if row["total"] else None),
                                      ("min", min(values) if row["total"] else None),
                                      ("max", max(values) if row["total"] else None)):
                    row[f"random_{field}_{suffix}"] = value
            test_retrieval.append(row)

    training_rows = read("training_summary.csv", csv_file=True)
    trainings = {}
    for row in training_rows:
        name = row["model"]
        _require(name in model_names and name not in trainings, f"Unexpected or duplicate trained model: {name}")
        family, seed = name.rsplit("_", 1)
        _require(row["family"] == family and _integer(row["seed"], name) == int(seed),
                 f"Training identity differs: {name}")
        epochs = _integer(row["epochs"], name)
        best_epoch = _integer(row["best_epoch"], name)
        _require(1 <= best_epoch <= epochs <= cfg["ae_max_epochs"], f"Invalid training epoch budget: {name}")
        for field in ("train_windows", "validation_windows"):
            _require(_integer(row[field], f"{name}/{field}") > 0, f"Empty training partition: {name}")
        for field in ("validation_loss", "test_loss"):
            _require(_finite(row[field], f"{name}/{field}") >= 0, f"Negative loss: {name}")
        trainings[name] = row
    _require(set(trainings) == set(model_names), "Training summary model grid is incomplete")

    quality_rows = read("quality_correlations.csv", csv_file=True)
    quality = {}
    for row in quality_rows:
        key = (row["model"], row["quality_metric"])
        _require(key[0] in model_names and key[1] in QUALITY_METRICS and key not in quality,
                 f"Unexpected or duplicate score-quality record: {key}")
        correlation = _finite(row["test_spearman"], str(key))
        _require(-1 <= correlation <= 1, f"Spearman correlation outside [-1,1]: {key}")
        quality[key] = {"model": key[0], "quality_metric": key[1], "test_spearman": correlation}
    _require(set(quality) == {(name, metric) for name in model_names for metric in QUALITY_METRICS},
             "Score-quality model/metric grid is incomplete")

    cluster_rows = read("clusters.csv", csv_file=True)
    cluster_grid = {(method, int(k), int(seed)) for method in methods
                    for k in cfg["clusters"] for seed in cfg["clustering_seeds"]}
    gates, seen, runs = [], set(), set()
    minimum_members = int(cfg["min_new_cluster_members"])
    minimum_jaccard = _finite(cfg["cluster_min_jaccard"], "minimum Jaccard")
    for row in cluster_rows:
        record = {field: _integer(row[field], f"cluster/{field}")
                  for field in ("k", "seed", "cluster", "members", "annotated_members", "supported_members")}
        run = (row["method"], record["k"], record["seed"])
        identity = (*run, record["cluster"])
        _require(run in cluster_grid and identity not in seen, f"Unexpected or duplicate cluster record: {identity}")
        seen.add(identity)
        runs.add(run)
        _require(0 <= record["cluster"] < record["k"] and record["members"] > 0,
                 f"Invalid nonempty cluster: {identity}")
        _require(all(0 <= record[field] <= record["members"] for field in ("annotated_members", "supported_members")),
                 f"Invalid cluster member counts: {identity}")
        jaccard = _finite(row["min_seed_jaccard"], f"{identity}/Jaccard")
        _require(0 <= jaccard <= 1, f"Jaccard outside [0,1]: {identity}")
        unannotated = record["annotated_members"] == 0
        member_gate = record["supported_members"] >= minimum_members
        stability_gate = jaccard >= minimum_jaccard
        _require(_boolean(row["passes_original_rule"], str(identity)) == (unannotated and member_gate),
                 f"Saved cluster member rule differs: {identity}")
        _require(_boolean(row["stable_candidate_new_cluster"], str(identity)) == (unannotated and member_gate and stability_gate),
                 f"Saved stable cluster rule differs: {identity}")
        gates.append({"method": row["method"], **record, "min_seed_jaccard": jaccard,
                      "unannotated": unannotated, "passes_member_gate": member_gate,
                      "passes_stability_gate": stability_gate,
                      "stable_candidate_new_cluster": unannotated and member_gate and stability_gate})
    _require(runs == cluster_grid, "Cluster method/K/seed grid is incomplete")
    return {"test_retrieval_rows": test_retrieval,
            "score_quality_rows": [quality[(name, metric)] for name in model_names for metric in QUALITY_METRICS],
            "cluster_gate_rows": gates, "sources": sources, "methods": methods,
            "model_names": model_names, "types": list(TYPES), "budgets": budgets,
            "rules": {"minimum_supported_members": minimum_members, "minimum_seed_jaccard": minimum_jaccard},
            "training_performed": False, "raw_contact_pixels_read": False,
            "independent_external_confirmation": False,
            "limitations": ["Previously inspected WT regions: all diagnostics remain exploratory.",
                            "Random-reference minimum/maximum is not a confidence interval or a significance test.",
                            "Known-structure recall is retrieval diagnostic, not evidence of a new biological type.",
                            "Saved-table consistency does not recover missing checkpoints or validate original pixels.",
                            "Cluster records reuse windows and are not independent observations."]}


def plot_diagnostics(data, figurepath):
    """Write an English-labelled PNG of the complete saved diagnostic grids."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figurepath = Path(figurepath)
    _require(figurepath.suffix.lower() == ".png", "Diagnostic figure must use a PNG path")
    methods, types = data["methods"], data["types"]
    colors = ("#2b5797", "#1c886b", "#d37d25")
    fig, (retrieval, quality, clusters) = plt.subplots(
        1, 3, figsize=(18, 9), gridspec_kw={"width_ratios": (1.55, 1.05, 1.35)})
    lookups = {(row["method"], row["type"]): row for row in data["test_retrieval_rows"]}
    for offset, (kind, color) in enumerate(zip(types, colors)):
        y = np.arange(len(methods)) + (offset - 1) * .22
        values = [lookups[(method, kind)]["recall"] for method in methods]
        denominator = lookups[(methods[0], kind)]["total"]
        retrieval.scatter(values, y, color=color, s=26, label=f"{kind} (n={denominator})", zorder=3)
        row = lookups[(methods[0], kind)]
        if row["random_recall_mean"] is not None:
            center, low, high = (row[f"random_recall_{suffix}"] for suffix in ("mean", "min", "max"))
            retrieval.errorbar(center, len(methods) + offset + .5,
                              xerr=[[center - low], [high - center]], fmt="o", color=color,
                              capsize=4, linewidth=2, markersize=5)
    retrieval.set_yticks(list(range(len(methods))) + [len(methods) + i + .5 for i in range(3)],
                        [method.replace("_", " ") for method in methods] + [f"Random {kind}" for kind in types])
    retrieval.invert_yaxis()
    retrieval.set(xlim=(-.03, 1.03), xlabel="Known-structure recall in previously viewed test regions")
    retrieval.axhline(len(methods) - .4, color="#999999", linewidth=.8)
    retrieval.grid(axis="x", alpha=.2)
    retrieval.set_axisbelow(True)
    retrieval.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.12),
                     ncol=3, borderaxespad=0, handletextpad=.25, columnspacing=.6)
    test_budget = data["budgets"]["test"]
    retrieval.set_title(f"A  All methods; same {test_budget['windows']}-window budget\nRandom marks: mean and observed range, not CI", loc="left", fontsize=11, fontweight="bold", pad=14)

    qlookup = {(row["model"], row["quality_metric"]): row["test_spearman"] for row in data["score_quality_rows"]}
    values = np.array([[qlookup[(name, metric)] for metric in QUALITY_METRICS] for name in data["model_names"]])
    im = quality.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    quality.set_xticks(range(3), ["Mean O/E", "Valid pixels", "Zero pixels"], rotation=25, ha="right", fontsize=9)
    quality.set_yticks(range(len(data["model_names"])), [name.replace("_", " ") for name in data["model_names"]], fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            quality.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=10,
                         color="white" if abs(values[i, j]) > .65 else "#173344")
    fig.colorbar(im, ax=quality, shrink=.55, pad=.03, label="Test Spearman correlation")
    quality.set_title("B  Reconstruction score and quality\nAssociation does not establish mechanism", loc="left", fontsize=11, fontweight="bold", pad=14)

    for unannotated, color, label, alpha, marker in ((False, "#8ca2ba", "Has annotated members", .25, "o"),
                                                  (True, "#d37d25", "No annotated members", .95, "D")):
        rows = [row for row in data["cluster_gate_rows"] if row["unannotated"] == unannotated]
        clusters.scatter([row["supported_members"] for row in rows], [row["min_seed_jaccard"] for row in rows],
                         color=color, label=f"{label} ({len(rows):,})", s=32 if unannotated else 15,
                         alpha=alpha, marker=marker, edgecolors="none")
    member_cutoff = data["rules"]["minimum_supported_members"]
    jaccard_cutoff = data["rules"]["minimum_seed_jaccard"]
    clusters.axvline(member_cutoff, color="#1c886b", linestyle="--", linewidth=1.2)
    clusters.axhline(jaccard_cutoff, color="#1c886b", linestyle="--", linewidth=1.2)
    clusters.set(xscale="symlog", ylim=(-.03, 1.06), xlabel="Replicate-supported members (symlog scale)",
                 ylabel="Minimum best-match Jaccard across seeds")
    clusters.legend(loc="upper left", bbox_to_anchor=(0, -.12), frameon=False, fontsize=8, borderaxespad=0)
    clusters.grid(alpha=.15)
    passing = sum(row["stable_candidate_new_cluster"] for row in data["cluster_gate_rows"])
    clusters.set_title(f"C  {len(data['cluster_gate_rows']):,} reused cluster records; {passing} pass\nNeed no annotations and both dashed gates", loc="left", fontsize=11, fontweight="bold", pad=14)
    fig.suptitle("Task 2 | Fair retrieval and model behaviour diagnostics", x=.03, ha="left", fontsize=16, fontweight="bold")
    fig.text(.03, .02, "Saved tables only; no training or genome rescan. Previously viewed WT data: exploratory, not independent biological confirmation.", fontsize=10, color="#444444")
    fig.subplots_adjust(left=.095, right=.98, top=.86, bottom=.15, wspace=.58)
    figurepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(figurepath, dpi=180, facecolor="white")
    finally:
        plt.close(fig)
    return figurepath
