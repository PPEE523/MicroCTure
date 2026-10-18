"""Reconstruct V2 discovery decisions using only small saved CSV/JSON files.

This audits bookkeeping and arithmetic, not pixels, fitted embeddings or model
weights. Missing replicate evidence remains unknown; it is never a negative.
All validation uses explicit exceptions so it remains active with ``python -O``.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bool(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    if value in (None, "", "NA", "null", "None"):
        return None
    raise ValueError(f"Invalid boolean: {value!r}")


def _number(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc
    return result if math.isfinite(result) else None


def _equal_number(saved, expected, message):
    saved = _number(saved)
    _require((saved is None and expected is None) or
             (saved is not None and expected is not None and
              math.isclose(saved, expected, rel_tol=1e-9, abs_tol=1e-10)), message)


def _quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _unique(rows, field, label, convert=lambda x: x):
    result = {}
    for row in rows:
        key = convert(row[field])
        _require(key not in result, f"Duplicate {label}: {key}")
        result[key] = row
    return result


def _ari(a, b):
    """Adjusted Rand index from saved labels; no feature/PCA access."""
    pairs = lambda value: value * (value - 1) / 2
    n = len(a)
    if n < 2:
        return 1.0
    observed = sum(pairs(v) for v in Counter(zip(a, b)).values())
    first = sum(pairs(v) for v in Counter(a).values())
    second = sum(pairs(v) for v in Counter(b).values())
    expected = first * second / pairs(n)
    maximum = (first + second) / 2
    return (observed - expected) / (maximum - expected) if maximum != expected else 1.0


def audit_discovery(directory, genome_bins=46417):
    """Validate retained V2 tables and return JSON-compatible reconstructed evidence.

    ``directory`` must contain the saved scan, frozen selection/labels, replicate
    correlations, candidate and cluster tables, report, manifest and histories.
    A missing *file* is an error; a missing correlation value is explicitly NA.
    No original file is changed. Checkpoint availability is reported separately
    and missing historical weights do not invalidate the saved-table audit.
    """
    directory = Path(directory).resolve()
    _require(isinstance(genome_bins, int) and genome_bins > 0, "genome_bins must be positive")
    sources, checks = [], []

    def read(name, *, table=False):
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), f"Missing or symlinked saved input: {path}")
        data = path.read_bytes()
        sources.append(dict(name=name, path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        text = data.decode("utf-8-sig")
        return list(csv.DictReader(text.splitlines())) if table else json.loads(text)

    def passed(name, **detail):
        checks.append(dict(name=name, status="passed", **detail))

    report = read("report.json")
    manifest = read("run_manifest.json")
    config = report["config"]
    _require(manifest["config"] == config, "Report/manifest config mismatch")
    _require(0 <= config["control_quantile"] <= 1, "Invalid control quantile")
    _require(config["min_controls"] > 0, "min_controls must be positive")
    resolution = int(config["resolution_bp"])
    scan = read("scan_rep1.csv", table=True)
    _unique(scan, "window_id", "scan window ID")
    for row in scan:
        for name in ("start_bin", "width_bins", "center_bin", "split"):
            row[name] = int(row[name])
        _require(0 < row["width_bins"] <= genome_bins, "Invalid window width")
        row["eligible"] = _bool(row["eligible"])
        _require(row["eligible"] is not None, "Missing QC decision in scan")
        _require(row["split"] in (-1, 0, 1, 2), "Invalid spatial split")
        for name in ("valid_fraction", "zero_fraction", "mean_oe"):
            row[name] = _number(row[name])
            _require(row[name] is not None, f"Missing scan metric {name}")
        expected_qc = row["valid_fraction"] >= config["min_valid_fraction"] and row["zero_fraction"] <= config["max_zero_fraction"]
        _require(row["eligible"] == expected_qc, "Scan eligibility disagrees with fixed QC thresholds")
    _require(scan, "Empty scan table")
    passed("scan_ids_and_quality", windows=len(scan))
    frozen = read("frozen_before_rep2_reload.json")
    frozen_sha = sources[-1]["sha256"]
    _require(report["frozen_sha256"] == frozen_sha, "Frozen selection/labels hash mismatch")
    selected = frozen["selected"]
    expected_methods = {"shape", "density"} | {
        f"{family}{fusion}_{seed}" for family in ("ae", "dae")
        for fusion in ("", "_fusion") for seed in config["seeds"]}
    _require(set(selected) == expected_methods, "Frozen ranking methods disagree with configured seeds/families")
    _require(set(frozen["test_selected"]) == set(selected), "Missing/extra test method")

    def bins(index):
        row = scan[index]
        return {(row["start_bin"] + j) % genome_bins for j in range(row["width_bins"])}

    def indices(values, label):
        _require(all(type(i) is int and 0 <= i < len(scan) for i in values), f"Invalid index in {label}")
        _require(len(set(values)) == len(values), f"Duplicate index in {label}")

    def disjoint(values, label, focal=None):
        indices(values, label)
        occupied = bins(focal) if focal is not None else set()
        for i in values:
            here = bins(i)
            _require(not here & occupied, f"Overlapping circular windows in {label}")
            occupied.update(here)

    def shortlist(values, quotas, label, test=False):
        disjoint(values, label)
        expected = {int(k): int(v) for k, v in quotas.items()}
        actual = dict(Counter(scan[i]["width_bins"] for i in values))
        _require(actual == expected, f"Candidate budget/quota mismatch in {label}: {actual} != {expected}")
        _require(all(scan[i]["eligible"] for i in values), f"Ineligible window in {label}")
        _require(not test or all(scan[i]["split"] == 2 for i in values), f"Non-test window in {label}")

    for method, values in selected.items():
        shortlist(values, config["full_candidate_quotas"], f"selected/{method}")
        shortlist(frozen["test_selected"][method], config["test_candidate_quotas"], f"test_selected/{method}", True)
    for domain, quotas in (("full", config["full_candidate_quotas"]), ("test", config["test_candidate_quotas"])):
        references = frozen["random_selected"][domain]
        _require(len(references) == config["random_references"], "Random-reference count mismatch")
        for seed, values in enumerate(references):
            shortlist(values, quotas, f"random/{domain}/{seed}", domain == "test")
    all_selected = sorted({i for values in selected.values() for i in values})
    _require(set(frozen["controls"]) == {str(i) for i in all_selected}, "Control-set identity mismatch")
    for i in all_selected:
        controls = frozen["controls"][str(i)]
        disjoint(controls, f"controls/{i}", focal=i)
        _require(len(controls) <= config["controls_per_candidate"], "Control budget exceeded")
        _require(all(scan[j]["eligible"] and not scan[j]["known_ids"] and
                     scan[j]["width_bins"] == scan[i]["width_bins"] for j in controls),
                 f"Unmatched control (annotation/scale/QC) for {i}")
    passed("fixed_budgets_circular_disjointness_and_controls", methods=len(selected), unique_candidates=len(all_selected))

    correlations = _unique(read("replicate_correlations.csv", table=True), "window_index", "replicate correlation index", int)
    used = set(all_selected) | {j for values in frozen["controls"].values() for j in values}
    _require(set(correlations) == used, "Missing/extra replicate correlation record")
    correlations = {i: _number(row["correlation"]) for i, row in correlations.items()}
    _require(all(v is None or -1.0000000001 <= v <= 1.0000000001 for v in correlations.values()), "Correlation outside [-1, 1]")
    validations = _unique(read("candidate_validation.csv", table=True), "window_index", "candidate index", int)
    _require(set(validations) == set(all_selected), "Candidate validation identity mismatch")
    support, candidate_rows = {}, []

    def compare_identity(saved, i):
        row = scan[i]
        expected = dict(window_id=row["window_id"], start_bp=(row["start_bin"] % genome_bins) * resolution,
                        window_length_bp=row["width_bins"] * resolution, known_ids=row["known_ids"], split=row["split"])
        for name, value in expected.items():
            actual = int(saved[name]) if isinstance(value, int) else saved[name]
            _require(actual == value, f"Candidate {i} identity/coordinate mismatch: {name}")
        wraps = row["start_bin"] < 0 or row["start_bin"] + row["width_bins"] > genome_bins
        _require(_bool(saved["wraps_origin"]) == wraps, f"Candidate {i} circular-coordinate flag mismatch")
        return expected, wraps

    for i in all_selected:
        saved = validations[i]
        expected, wraps = compare_identity(saved, i)
        values = [correlations[j] for j in frozen["controls"][str(i)] if correlations[j] is not None]
        cutoff = max(config["replicate_min_correlation"], _quantile(values, config["control_quantile"])) if len(values) >= config["min_controls"] else None
        corr = correlations[i]
        reason = "insufficient_controls" if cutoff is None else "missing_replicate_correlation" if corr is None else ""
        outcome = None if reason else corr > cutoff
        _equal_number(saved["correlation"], corr, f"Candidate {i} correlation mismatch")
        _equal_number(saved["cutoff"], cutoff, f"Candidate {i} control threshold mismatch")
        _require(int(saved["matched_controls"]) == len(values), f"Candidate {i} valid-control count mismatch")
        stored = _bool(saved["replicate_supported"])
        _require(stored == outcome or (outcome is None and stored is False), f"Candidate {i} replication support mismatch")
        if "replicate_status" in saved:
            expected_status = "not_evaluable" if outcome is None else "supported" if outcome else "not_supported"
            _require(saved["replicate_status"] == expected_status, f"Candidate {i} replication status mismatch")
        support[i] = outcome
        candidate_rows.append(dict(window_index=i, **expected, wraps_origin=wraps, correlation=corr,
                                   matched_controls=len(values), cutoff=cutoff,
                                   replication_margin=corr-cutoff if outcome is not None else None,
                                   replicate_supported=outcome, assessment_status="not_evaluable" if reason else "assessed",
                                   assessment_reason=reason, saved_replicate_supported=stored,
                                   coordinate_status=saved.get("coordinate_status", "unspecified")))
    passed("replicate_control_quantiles_and_support", candidates=len(all_selected),
           not_evaluable=sum(value is None for value in support.values()))

    seeds = config["clustering_seeds"]
    _require(len(set(seeds)) == len(seeds) and len(seeds) >= 2, "At least two distinct cluster seeds are required")
    expected_runs = {f"{method}:{k}:{seed}" for method in selected for k in config["clusters"] for seed in seeds}
    _require(set(frozen["clustering"]) == expected_runs, "Missing/extra frozen clustering run")
    partitions = {}
    for key, run in frozen["clustering"].items():
        method, k, seed = key.split(":")
        labels = run["labels"]
        _require(len(labels) == len(selected[method]), f"Cluster labels/member count mismatch: {key}")
        _require(all(type(label) is int and 0 <= label < int(k) for label in labels), f"Invalid cluster label: {key}")
        partitions[key] = {label: {i for i, value in enumerate(labels) if value == label} for label in set(labels)}
    saved_clusters = read("clusters.csv", table=True)
    saved_clusters_by_key = {}
    for row in saved_clusters:
        key = (row["method"], int(row["k"]), int(row["seed"]), int(row["cluster"]))
        _require(key not in saved_clusters_by_key, "Duplicate cluster record")
        saved_clusters_by_key[key] = row
    expected_cluster_keys = {(method, int(k), int(seed), label) for key, groups in partitions.items()
                             for method, k, seed in [key.split(":")] for label in groups}
    _require(set(saved_clusters_by_key) == expected_cluster_keys, "Missing/extra nonempty cluster record")
    cluster_rows = []
    for key in sorted(expected_cluster_keys):
        method, k, seed, label = key
        saved = saved_clusters_by_key[key]
        positions = partitions[f"{method}:{k}:{seed}"][label]
        members = [selected[method][j] for j in sorted(positions)]
        annotated = sum(bool(scan[i]["known_ids"]) for i in members)
        supported = sum(support[i] is True for i in members)
        missing = sum(support[i] is None for i in members)
        jaccard = min(max(len(positions & other) / len(positions | other)
                          for other in partitions[f"{method}:{k}:{sd}"].values()) for sd in seeds if sd != seed)
        support_gate = None if supported < config["min_new_cluster_members"] <= supported + missing else supported >= config["min_new_cluster_members"]
        operational = False if annotated else support_gate
        uncertain = operational is None
        stable = False if jaccard < config["cluster_min_jaccard"] or operational is False else operational
        for name, value in (("members", len(members)), ("annotated_members", annotated), ("supported_members", supported)):
            _require(int(saved[name]) == value, f"Cluster {key} {name} mismatch")
        _equal_number(saved["min_seed_jaccard"], jaccard, f"Cluster {key} Jaccard mismatch")
        for field, outcome in (("passes_original_rule", operational), ("stable_candidate_new_cluster", stable)):
            stored = _bool(saved[field])
            _require(stored == outcome or (outcome is None and stored is False), f"Cluster {key} {field} mismatch")
        if "not_evaluable_members" in saved:
            _require(int(saved["not_evaluable_members"]) == missing, f"Cluster {key} NA-member count mismatch")
        if "cluster_status" in saved:
            expected_status = "not_evaluable" if stable is None else "passes_rule" if stable else "does_not_pass_rule"
            _require(saved["cluster_status"] == expected_status, f"Cluster {key} decision status mismatch")
        reasons = []
        if annotated:
            reasons.append("contains_annotated_members")
        if supported < config["min_new_cluster_members"]:
            reasons.append("insufficient_assessed_support" if uncertain else "fewer_than_minimum_supported_members")
        if jaccard < config["cluster_min_jaccard"]:
            reasons.append("unstable_across_cluster_seeds")
        cluster_rows.append(dict(method=method, k=k, seed=seed, cluster=label, members=len(members),
                                 annotated_members=annotated, supported_members=supported, not_evaluable_members=missing,
                                 min_seed_jaccard=jaccard, passes_original_rule=operational,
                                 stable_candidate_new_cluster=stable,
                                 gate_unannotated=annotated == 0, gate_supported_members=support_gate,
                                 gate_stable_across_seeds=jaccard >= config["cluster_min_jaccard"],
                                 assessment_status="not_evaluable" if stable is None else "assessed",
                                 gate_reasons=";".join(reasons), member_window_indices=members))
    passed("all_cluster_members_support_jaccard_and_rules", runs=len(expected_runs), clusters=len(cluster_rows))
    if (directory / "cluster_stability.csv").exists():
        ari_rows = read("cluster_stability.csv", table=True)
        ari_keys = set()
        for row in ari_rows:
            key = (row["method"], int(row["k"]))
            _require(key not in ari_keys, "Duplicate ARI summary")
            ari_keys.add(key)
            values = [_ari(frozen["clustering"][f"{key[0]}:{key[1]}:{a}"]["labels"],
                           frozen["clustering"][f"{key[0]}:{key[1]}:{b}"]["labels"]) for a, b in itertools.combinations(seeds, 2)]
            _equal_number(row["mean_pairwise_ari"], sum(values)/len(values), f"ARI mismatch: {key}")
        _require(ari_keys == {(method, k) for method in selected for k in config["clusters"]}, "Missing/extra ARI summary")
        passed("pairwise_adjusted_rand_index")

    method_rows = []
    saved_methods = _unique(read("method_summary.csv", table=True), "method", "method summary")
    report_methods = _unique(report["methods"], "method", "report method")
    _require(set(saved_methods) == set(report_methods) == set(selected), "Method summary identity mismatch")
    for method, ids in selected.items():
        groups = [row for row in cluster_rows if row["method"] == method]
        values = dict(candidates=len(ids), covered_bins=sum(scan[i]["width_bins"] for i in ids),
                      unannotated=sum(not scan[i]["known_ids"] for i in ids),
                      replicate_supported=sum(support[i] is True for i in ids),
                      unannotated_supported=sum(support[i] is True and not scan[i]["known_ids"] for i in ids),
                      passing_cluster_configurations=sum(row["passes_original_rule"] is True for row in groups),
                      stable_cluster_configurations=sum(row["stable_candidate_new_cluster"] is True for row in groups))
        for name, value in values.items():
            _require(int(saved_methods[method][name]) == int(report_methods[method][name]) == value,
                     f"Method {method} summary mismatch: {name}")
        unavailable_candidates = sum(support[i] is None for i in ids)
        unavailable_clusters = sum(row["stable_candidate_new_cluster"] is None for row in groups)
        for field, expected in (("not_evaluable_candidates", unavailable_candidates), ("not_evaluable_cluster_configurations", unavailable_clusters)):
            for recorded in (saved_methods[method], report_methods[method]):
                if field in recorded:
                    _require(int(recorded[field]) == expected, f"Method {method} {field} mismatch")
        method_rows.append(dict(method=method, **values, not_evaluable_candidates=unavailable_candidates,
                                not_evaluable_cluster_configurations=unavailable_clusters))
    passed("method_summaries")

    union = [i for i in all_selected if support[i] is True and not scan[i]["known_ids"]]
    disjoint_ids, occupied = [], set()
    for i in sorted(union, key=lambda i: (-correlations[i], i)):
        here = bins(i)
        if not here & occupied:
            disjoint_ids.append(i)
            occupied.update(here)
    for name, expected in (("unannotated_supported_union.csv", union), ("unannotated_supported_disjoint.csv", disjoint_ids)):
        rows = read(name, table=True)
        _require([int(row["window_index"]) for row in rows] == expected, f"Candidate union/greedy disjoint identity/order mismatch: {name}")
        for row in rows:
            i = int(row["window_index"])
            compare_identity(row, i)
            for field in ("correlation", "cutoff"):
                _equal_number(row[field], _number(validations[i][field]), f"Candidate subset {field} mismatch")
            _require(int(row["matched_controls"]) == int(validations[i]["matched_controls"]) and
                     _bool(row["replicate_supported"]) is True, "Candidate subset support mismatch")
    for row in candidate_rows:
        i = row["window_index"]
        row["in_unannotated_supported_union"] = i in union
        row["in_disjoint_candidates"] = i in disjoint_ids
        row["selected_by_methods"] = ";".join(method for method, ids in selected.items() if i in ids)
    passed("cross_method_union_and_fixed_greedy_disjoint_subset", union=len(union), disjoint=len(disjoint_ids))

    training = read("training_summary.csv", table=True)
    training_by_name = _unique(training, "model", "training model")
    report_training = _unique(report["training"], "model", "report training model")
    models = {f"{family}_{seed}" for family in ("ae", "dae") for seed in config["seeds"]}
    _require(set(training_by_name) == set(report_training) == models, "Training model/seed set mismatch")
    histories = []
    partitions_count = {name: sum(row["eligible"] and row["split"] == role for row in scan)
                        for role, name in enumerate(("train", "validation", "test"))}
    for name, saved in training_by_name.items():
        history = read(f"{name}_history.json")
        _require([r["epoch"] for r in history] == list(range(1, len(history)+1)), f"Nonconsecutive training epochs: {name}")
        _require(0 < len(history) <= config["ae_max_epochs"], f"Invalid epoch budget: {name}")
        best, best_epoch, stale = float("inf"), None, 0
        for epoch in history:
            losses = [_number(epoch[key]) for key in ("train_loss", "validation_loss")]
            _require(all(loss is not None and loss >= 0 for loss in losses), f"Invalid training loss: {name}")
            if losses[1] < best - 1e-6:
                best, best_epoch = losses[1], epoch["epoch"]
                stale = 0
            else:
                stale += 1
            if "ae_patience" in config and stale >= config["ae_patience"]:
                _require(epoch["epoch"] == len(history), f"Training continued after patience budget: {name}")
        if "ae_patience" in config and len(history) < config["ae_max_epochs"]:
            _require(stale >= config["ae_patience"], f"Truncated training history without early stopping: {name}")
        family, seed = name.split("_")
        expected = dict(family=family, seed=int(seed), epochs=len(history), best_epoch=best_epoch,
                        train_windows=min(partitions_count["train"], config["ae_max_train_windows"]),
                        validation_windows=partitions_count["validation"], validation_loss=best)
        for field, value in expected.items():
            for recorded in (saved[field], report_training[name][field]):
                if isinstance(value, str):
                    _require(recorded == value, f"Training {name} {field} mismatch")
                else:
                    _equal_number(recorded, value, f"Training {name} {field} mismatch")
        _equal_number(saved["test_loss"], _number(report_training[name]["test_loss"]), f"Training {name} saved test-loss mismatch")
        _require(_number(saved["test_loss"]) is not None and float(saved["test_loss"]) >= 0, f"Missing/invalid saved test loss: {name}")
        histories.append(dict(model=name, epochs=len(history), best_epoch=best_epoch, validation_loss=best))
    passed("training_summary_and_checkpoint_selection_history", models=len(models))

    clean = [r for r in cluster_rows if r["annotated_members"] == 0]
    passing = [r for r in cluster_rows if r["stable_candidate_new_cluster"] is True]
    primary = [r for r in cluster_rows if r["method"] == config["primary_method"] and r["k"] == config["primary_k"] and r["seed"] == config["primary_cluster_seed"]]
    def endpoint(rows):
        return None if any(row["stable_candidate_new_cluster"] is None for row in rows) else sum(row["stable_candidate_new_cluster"] is True for row in rows)
    numeric_summary = dict(scan_windows=len(scan), eligible_windows=sum(r["eligible"] for r in scan),
                           unique_selected_windows=len(all_selected), unannotated_supported_union=len(union),
                           unannotated_supported_disjoint=len(disjoint_ids), cluster_runs=len(expected_runs),
                           primary_candidate_new_clusters=endpoint(primary), all_passing_cluster_configurations=endpoint(cluster_rows))
    for name, expected in numeric_summary.items():
        # Older reports cast missing evidence to False. Keep their bytes, but
        # normalize the current complete-cohort endpoint back to null/NA.
        legacy_zero = expected is None and report[name] == 0 and name in ("primary_candidate_new_clusters", "all_passing_cluster_configurations")
        _require(report[name] == expected or legacy_zero, f"Report summary mismatch: {name}")
    for field, expected in (("confirmed_passing_cluster_configurations", len(passing)),
                            ("not_evaluable_cluster_configurations", sum(r["stable_candidate_new_cluster"] is None for r in cluster_rows))):
        if field in report:
            _require(report[field] == expected, f"Report summary mismatch: {field}")
    _require(report["partition_windows"] == partitions_count, "Report spatial partition counts mismatch")
    passed("report_summary_and_frozen_hash")
    missing_checkpoints = sorted(name + ".pt" for name in models if not (directory / (name + ".pt")).is_file())
    not_evaluable = sum(value is None for value in support.values())
    summary = dict(**numeric_summary, partition_windows=partitions_count, methods=len(selected),
                   candidate_budget_per_method=sum(config["full_candidate_quotas"].values()),
                   candidate_validation_records=len(candidate_rows), replicate_supported_windows=sum(value is True for value in support.values()),
                   not_evaluable_candidates=not_evaluable, nonempty_cluster_records=len(cluster_rows),
                   unannotated_cluster_records=len(clean),
                   unannotated_with_five_supported_members=sum(r["supported_members"] >= config["min_new_cluster_members"] for r in clean),
                   unannotated_with_minimum_supported_members=sum(r["supported_members"] >= config["min_new_cluster_members"] for r in clean),
                   max_supported_members_in_unannotated_cluster=max((r["supported_members"] for r in clean), default=None),
                   max_jaccard_in_unannotated_cluster=max((r["min_seed_jaccard"] for r in clean), default=None),
                   stable_cluster_records=sum(r["min_seed_jaccard"] >= config["cluster_min_jaccard"] for r in cluster_rows),
                   supported_cluster_records=sum(r["supported_members"] >= config["min_new_cluster_members"] for r in cluster_rows),
                   stable_unannotated_cluster_records=sum(r["min_seed_jaccard"] >= config["cluster_min_jaccard"] for r in clean),
                   unique_control_windows=len({i for values in frozen["controls"].values() for i in values}),
                   candidate_control_pairs=sum(map(len, frozen["controls"].values())),
                   candidate_assessment_status="not_evaluable" if not_evaluable else "assessed",
                   confirmed_passing_cluster_configurations=len(passing),
                   not_evaluable_cluster_configurations=sum(r["stable_candidate_new_cluster"] is None for r in cluster_rows),
                   threshold_rules={key: config[key] for key in ("replicate_min_correlation", "control_quantile", "min_controls", "min_new_cluster_members", "cluster_min_jaccard")})
    return dict(schema_version=1, action="saved_discovery_arithmetic_audit", summary=summary,
                candidate_rows=candidate_rows, disjoint_candidate_rows=[next(row for row in candidate_rows if row["window_index"] == i) for i in disjoint_ids],
                cluster_rows=cluster_rows, method_rows=method_rows, checks=checks,
                reproducibility=dict(saved_table_audit="passed", model_checkpoint_status="unavailable_missing_checkpoints" if missing_checkpoints else "available_not_loaded",
                                     missing_checkpoints=missing_checkpoints, expected_checkpoints=len(models), training_histories=histories,
                                     historical_source_signatures_match_current_source="not_asserted",
                                     model_weights_loaded=False, ranking_scores_recomputed=False, clustering_centers_refit=False,
                                     test_losses_recomputed=False, raw_contact_pixels_read=False, training_performed=False,
                                     independent_external_confirmation=False), sources=sources)
