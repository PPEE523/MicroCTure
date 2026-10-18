"""Verify the compact coursework package without its original WSL workspace.

Uses only the Python standard library. Optional --source-root additionally
compares copied evidence to the read-only scientific workspace. This never
loads model weights, reads raw matrices, fits a model or rewrites evidence.
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
import math
import statistics
from itertools import product
from pathlib import Path, PurePosixPath


ROOT = _layout_root
CLASS_NAMES = ("OPCID", "CHIN", "CHID")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or "\\" in relative or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe package path: {relative!r}")
    target = _layout_resolve(root.joinpath(*path.parts), root=root)
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError(f"Symlinks are not supported in the evidence package: {relative}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes root: {relative}")
    return target


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _equal_float(actual, expected, description: str) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=1e-7, rel_tol=0):
        raise ValueError(f"Numeric mismatch: {description}")


def _score(confusion: list[list[int]]) -> dict:
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                           "support": support}
    count = sum(sum(row) for row in confusion)
    return {"n_structures": count, "confusion_matrix": confusion, "per_class": per_class,
            "accuracy": sum(confusion[i][i] for i in range(3)) / count,
            "macro_f1": statistics.mean(row["f1"] for row in per_class.values()),
            "balanced_accuracy": statistics.mean(row["recall"] for row in per_class.values())}


def verify_scientific_snapshot(submission: Path, specification: dict) -> dict:
    """Recompute complete OOF numbers from the bundled, structure-level CSVs."""
    base = safe_path(submission, specification["formal_directory"])
    manifest = read_json(base / "manifest.json")
    if signature({key: value for key, value in manifest.items() if key != "experiment_signature"}) != manifest["experiment_signature"]:
        raise ValueError("Formal experiment manifest has an invalid canonical signature")
    metadata = read_json(base / "metadata.json")
    if metadata["signature"] != manifest["dataset_signature"]:
        raise ValueError("Bundled metadata uses a different dataset signature")
    structures = {row["structure_id"]: row for row in metadata["structures"]}
    if len(structures) != len(metadata["structures"]) or len(structures) != specification["unique_structures"]:
        raise ValueError("Bundled structure count or uniqueness differs")
    groups = dict(zip((row["structure_id"] for row in metadata["structures"]), metadata["group_id"]))
    annotations = read_csv(safe_path(submission, specification["annotations"]))
    if len(annotations) != len(structures) or {row["structure_id"] for row in annotations} != set(structures):
        raise ValueError("Bundled annotations differ from the scientific metadata")
    for row in annotations:
        original = structures[row["structure_id"]]
        if any(str(row[key]) != str(original[key]) for key in ("type", "start", "end")):
            raise ValueError("Annotation class/interval differs from the frozen metadata")
    owners = {}
    for number, partitions in manifest["fold_ids"].items():
        sets = [set(partitions[key]) for key in ("train_ids", "validation_ids", "test_ids")]
        if set.union(*sets) != set(structures) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Frozen folds do not partition unique structures")
        group_sets = [{groups[sid] for sid in subset} for subset in sets]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Spatial group leakage in the bundled folds")
        fold = read_json(base / "folds" / f"fold_{number}.json")
        if signature({key: value for key, value in fold.items() if key != "signature"}) != fold["signature"]:
            raise ValueError("Bundled fold wrapper signature differs")
        if signature(fold["frozen_fold"]) != manifest["fold_signatures"][number]:
            raise ValueError("Bundled frozen fold signature differs")
        preprocessing = fold["preprocessing"]
        if signature({key: value for key, value in preprocessing.items() if key != "signature"}) != preprocessing["signature"]:
            raise ValueError("Bundled preprocessing signature differs")
        if preprocessing["train_ids"] != partitions["train_ids"] or preprocessing["length_scaler"]["fitted_unique_structures"] != len(partitions["train_ids"]):
            raise ValueError("Preprocessing is not bound to the frozen inner training set")
        for sid in partitions["test_ids"]:
            if sid in owners:
                raise ValueError("A structure occurs in multiple outer folds")
            owners[sid] = int(number)
    if set(owners) != set(structures):
        raise ValueError("Outer folds do not cover all bundled structures")
    summary = read_json(base / "summary.json")
    if summary["status"] != "complete" or summary["completed_jobs"] != specification["historical_complete_fits"]:
        raise ValueError("Frozen formal experiment is not complete")
    summary_rows = {(row["condition"], row["model"]): row for row in summary["comparison"]}
    comparisons = {(row["condition"], row["model"]): row for row in read_csv(base / "comparison.csv")}
    oof_count = 0
    for model, seeds in specification["oof_models"].items():
        scores = []
        for seed in seeds:
            directory = base / "oof" / "I3" / model / f"seed_{seed}"
            stored, state = read_json(directory / "metrics.json"), read_json(directory / "status.json")
            for record in (stored, state):
                if record["status"] != "complete" or record["oof_current"] is not True:
                    raise ValueError("Bundled OOF is not marked as current and complete")
                for key in ("experiment_signature", "source_signature", "dataset_signature"):
                    if record[key] != manifest[key]:
                        raise ValueError(f"OOF {key} differs from its source protocol")
            rows = read_csv(directory / "predictions.csv")
            if len(rows) != len(structures) or {row["structure_id"] for row in rows} != set(structures):
                raise ValueError("OOF must predict every unique structure exactly once")
            confusion = [[0] * 3 for _ in range(3)]
            for row in rows:
                sid = row["structure_id"]
                y, predicted = int(row["y_true"]), int(row["y_pred"])
                probabilities = [float(row[f"p_{name}"]) for name in CLASS_NAMES]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities) or not math.isclose(sum(probabilities), 1., rel_tol=0, abs_tol=1e-6):
                    raise ValueError("Invalid saved probability vector")
                if y != CLASS_NAMES.index(structures[sid]["type"]) or int(row["group_id"]) != groups[sid] or int(row["fold"]) != owners[sid]:
                    raise ValueError("OOF label/group/fold differs from the frozen identity")
                if predicted != max(range(3), key=probabilities.__getitem__):
                    raise ValueError("OOF argmax differs from its probability vector")
                replicates = json.loads(row["replicate_probabilities_json"])
                if len(replicates) != int(row["replicate_count"]) or len(replicates) != specification["biological_views_per_structure"]:
                    raise ValueError("Biological replicate count differs")
                for index in range(3):
                    _equal_float(statistics.mean(rep["probabilities"][index] for rep in replicates), probabilities[index], "average WT probabilities before scoring")
                confusion[y][predicted] += 1
            calculated = _score(confusion)
            if stored["confusion_matrix"] != calculated["confusion_matrix"]:
                raise ValueError("OOF confusion matrix differs from its predictions")
            for key in ("accuracy", "macro_f1", "balanced_accuracy"):
                _equal_float(stored[key], calculated[key], f"{model}/seed{seed}/{key}")
            for name in CLASS_NAMES:
                for key in ("precision", "recall", "f1", "support"):
                    _equal_float(stored["per_class"][name][key], calculated["per_class"][name][key], f"{model}/seed{seed}/{name}/{key}")
            scores.append(calculated)
            oof_count += 1
        table = summary_rows[("I3", model)]
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            mean = statistics.mean(score[metric] for score in scores)
            _equal_float(table[f"{metric}_mean"], mean, f"{model}/{metric}/mean")
            _equal_float(comparisons[("I3", model)][f"{metric}_mean"], mean, f"comparison.csv/{model}/{metric}")
            deviation = table[f"{metric}_seed_sample_std"]
            if len(scores) > 1:
                _equal_float(deviation, statistics.stdev(score[metric] for score in scores), f"{model}/{metric}/seed SD")
            elif deviation is not None:
                raise ValueError("Single deterministic baseline cannot claim stochastic seed SD")
    controlled = read_csv(safe_path(submission, specification["controlled_aggregate"]))
    observed = {row["head"]: (int(row["strict_pass_cases"]), int(row["expected_cases"])) for row in controlled}
    expected = {name: (count, 6) for name, count in specification["controlled_passes"].items()}
    if observed != expected:
        raise ValueError("Controlled H0-H3 pass table differs from the frozen decision")
    decision = read_json(safe_path(submission, specification["autonomous_decision"]))
    if decision["biological_morphology_reliability_established"] is not False or decision["predeclared_numeric_gate_passed"] is not False or decision["default_change_adopted"] is not False:
        raise ValueError("Autonomous development must retain its unmet reliability decision")
    return {"unique_structures": len(structures), "CHID_structures": sum(row["type"] == "CHID" for row in structures.values()),
            "complete_bundled_oof": oof_count, "formal_historical_fits": summary["completed_jobs"],
            "controlled_passes": {name: count for name, (count, _) in observed.items()},
            "reliability_established": False, "raw_data_read": False, "training_launched": False}


def verify_demo_metadata(submission: Path, repository: Path, specification: dict) -> dict:
    reference = read_json(safe_path(submission, specification["expected"]))
    if reference["training_performed_for_demo"] is not False:
        raise ValueError("Portable demo must retain its no-new-training provenance")
    bindings = ((safe_path(submission, specification["input"]), reference["input_npz_sha256"]),
                (safe_path(repository, reference["checkpoint_relative_path"]), reference["checkpoint_sha256"]),
                (safe_path(repository, reference["fold_protocol_relative_path"]), reference["fold_protocol_sha256"]))
    if any(sha256(path) != expected for path, expected in bindings):
        raise ValueError("Demo input/checkpoint/fold provenance differs")
    fold = read_json(bindings[2][0])
    metadata = read_json(safe_path(submission, "task1/formal/metadata.json"))
    labels = {row["structure_id"]: row["type"] for row in metadata["structures"]}
    selected = [min(sid for sid in fold["frozen_fold"]["test_ids"] if labels[sid] == name) for name in CLASS_NAMES]
    if reference["selection_ids_class_order"] != selected:
        raise ValueError("Demo selection differs from the predetermined class-first rule")
    if reference["dataset_signature"] != metadata["signature"] or reference["preprocessing_signature"] != fold["preprocessing"]["signature"]:
        raise ValueError("Demo dataset/preprocessing provenance differs")
    replicate_ids, probabilities = reference["replicate_structure_ids"], reference["replicate_probabilities"]
    if len(replicate_ids) != len(probabilities) or len(replicate_ids) != 6:
        raise ValueError("Demo must contain three fixed structures, each with two WT views")
    by_id = {row["structure_id"]: row for row in reference["structure_predictions"]}
    if set(by_id) != set(selected):
        raise ValueError("Demo prediction identities differ")
    incorrect = []
    for sid in selected:
        vectors = [vector for identity, vector in zip(replicate_ids, probabilities) if identity == sid]
        if len(vectors) != 2:
            raise ValueError("Demo replicate counts differ")
        averages = [statistics.mean(vector[index] for vector in vectors) for index in range(3)]
        for index, name in enumerate(CLASS_NAMES):
            _equal_float(by_id[sid][f"p_{name}"], averages[index], "demo WT averaging")
        predicted = CLASS_NAMES[max(range(3), key=averages.__getitem__)]
        if by_id[sid]["predicted"] != predicted:
            raise ValueError("Demo saved class differs from its probabilities")
        if predicted != labels[sid]:
            incorrect.append(sid)
    return {"fixed_structures": selected, "retained_incorrect_examples": incorrect,
            "metadata_and_bindings_verified": True, "model_inference_performed_by_this_check": False,
            "performance_evidence": False}


def _csv_decision(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    raise ValueError(f"Invalid centering nullable decision: {value!r}")


def _centering_value(actual, expected, description):
    if expected is None:
        if actual not in ("", None):
            raise ValueError(f"Centering NA must not become a negative: {description}")
    elif isinstance(expected, bool):
        if _csv_decision(actual) is not expected:
            raise ValueError(f"Centering decision differs: {description}")
    else:
        _equal_float(actual, expected, f"centering/{description}")


def verify_centering_snapshot(submission: Path, specification: dict) -> dict:
    """Independently audit saved 2-replicate decisions using only stdlib tables.

    This checks the recorded distance/threshold decisions and missingness, not
    raw contacts, threshold fitting, or the biological truth of any candidate.
    """
    base = safe_path(submission, specification["directory"])
    review = read_json(base / "review.json")
    config = review["config"]
    if (config["anchors"] != ["original", "boundary"] or config["width_bins"] != [64, 128, 256]
            or config["shift_bins"] != [-8, 0, 8]
            or [row["name"] for row in config["occlusions"]] != ["none", "drop10_s17", "drop10_s29"]
            or config["min_calibration_windows"] != 10 or config["robust_fraction"] != .8
            or config["distance_quantile"] != .95 or config["min_valid_fraction"] != .8
            or config["max_zero_fraction"] != .8):
        raise ValueError("Centering fixed grid or decision rule differs")
    if (review["training_performed"] is not False or review["raw_pixels_read"] is not False
            or review["historical_source_hash_match_claimed"] is not False
            or specification["independent_confirmation"] is not False
            or specification["previously_inspected_data"] is not True
            or "no independent biological confirmation" not in review["interpretation"]):
        raise ValueError("Centering review cannot become independent confirmation")
    variants = {f"{anchor}_w{width}_s{shift}_{occlusion}": (anchor, width, shift, occlusion)
                for anchor, width, shift, occlusion in product(config["anchors"], config["width_bins"],
                                                             config["shift_bins"], [x["name"] for x in config["occlusions"]])}
    metric_keys = set(product(variants, ("pca", "shape")))
    identities = specification["candidate_ids"]
    if len(identities) != len(set(identities)) or len(identities) != specification["candidates"]:
        raise ValueError("Centering candidate identity specification differs")
    result, first_rows = {}, None
    for name in ("primary", "supplement"):
        directory = base / name
        report = read_json(directory / "report.json")
        if review["results"][name] != report or report["historical_source_manifest_verified_against_current_code"] is not False:
            raise ValueError("Centering retained review/report provenance differs")
        count = specification["calibration_windows"][name]
        cohort = review["cohorts"][name]
        if cohort["exploratory"] is not True or cohort["calibration_windows"] != count or report["calibration_windows"] != count:
            raise ValueError("Centering calibration provenance differs")
        if name == "supplement" and "not independent evaluation" not in report["reason"]:
            raise ValueError("Centering supplement cannot become independent confirmation")
        threshold_rows = read_csv(directory / "thresholds.csv")
        thresholds = {(r["variant_id"], r["metric"]): r for r in threshold_rows}
        if len(thresholds) != len(threshold_rows) or set(thresholds) != metric_keys:
            raise ValueError("Centering threshold grid is duplicate or incomplete")
        provenance = [r for r in review["input_files"] if r["path"] == cohort["path"] + "/thresholds.csv"]
        if (len(provenance) != 1 or provenance[0]["sha256"] != sha256(directory / "thresholds.csv")
                or provenance[0]["size_bytes"] != (directory / "thresholds.csv").stat().st_size):
            raise ValueError("Centering historical threshold provenance differs")
        for threshold in thresholds.values():
            cutoffs = [float(threshold[f"rep{i}_cutoff"]) for i in (1, 2)]
            if int(threshold["calibration_windows"]) != count or (all(math.isfinite(v) for v in cutoffs) != (count >= 10)):
                raise ValueError("Centering calibration count/cutoff availability differs")
        summaries = read_csv(directory / "candidate_summary.csv")
        if len(summaries) != len(identities) or {r["entity_id"] for r in summaries} != set(identities):
            raise ValueError("Centering candidate summaries are missing or duplicated")
        rows = read_csv(directory / "candidate_variants.csv")
        lookup = {(r["entity_id"], r["variant_id"], r["metric"]): r for r in rows}
        expected_keys = {(sid, variant, metric) for sid in identities for variant, metric in metric_keys}
        if len(rows) != len(lookup) or set(lookup) != expected_keys:
            raise ValueError("Centering candidate variant grid is duplicate or incomplete")
        if first_rows is None:
            first_rows = lookup
        else:
            for key, row in lookup.items():
                if any(row[field] != first_rows[key][field] for field in ("entity_index", "nearest_reference", "reference_index",
                                                                         "rep1_distance", "rep2_distance", "quality_pass")):
                    raise ValueError("Centering supplement changed the retained candidate evidence")
        decisions = {}
        for key, row in lookup.items():
            sid, variant, metric = key
            if int(row["entity_index"]) != identities.index(sid):
                raise ValueError("Centering candidate identity/index differs")
            if name == "primary" and (row["anchor"], int(row["width_bins"]), int(row["shift_bins"]), row["occlusion"]) != variants[variant]:
                raise ValueError("Centering variant metadata differs from its identity")
            cutoffs = [float(row[f"rep{i}_cutoff"]) for i in (1, 2)]
            for i, cutoff in enumerate(cutoffs, 1):
                reference = float(thresholds[variant, metric][f"rep{i}_cutoff"])
                if not ((math.isnan(cutoff) and math.isnan(reference)) or cutoff == reference):
                    raise ValueError("Centering variant cutoff differs from retained threshold")
            distances = [float(row[f"rep{i}_distance"]) for i in (1, 2)]
            quality = _csv_decision(row["quality_pass"])
            if quality is None:
                raise ValueError("Centering quality flag is missing")
            reason = ("insufficient_calibration" if count < 10 else
                      "nonfinite_threshold" if not all(map(math.isfinite, cutoffs)) else
                      "quality_failed" if not quality else
                      "nonfinite_distance" if not all(map(math.isfinite, distances)) else "")
            decision = None if reason else all(d > c for d, c in zip(distances, cutoffs))
            if row["assessment_status"] != ("not_evaluable" if reason else "assessed") or row["assessment_reason"] != reason:
                raise ValueError("Centering variant evaluability/reason differs")
            _centering_value(row["outside_both"], decision, "/".join(key))
            decisions[key] = decision
        assessed = 0
        for summary in summaries:
            sid = summary["entity_id"]
            if _csv_decision(summary["boundary_reproducible"]) is None:
                raise ValueError("Centering boundary reproducibility flag is missing")
            joint = {variant: (None if any(decisions[sid, variant, m] is None for m in ("pca", "shape"))
                               else all(decisions[sid, variant, m] for m in ("pca", "shape"))) for variant in variants}
            fractions = []
            for anchor in config["anchors"]:
                values = [joint[v] for v, setting in variants.items() if setting[0] == anchor]
                total = sum(values) if all(v is not None for v in values) else None
                for field, value in (("assessed_variants", sum(v is not None for v in values)),
                                     ("outside_variants", total), ("outside_fraction", total / len(values) if total is not None else None)):
                    _centering_value(summary[f"{anchor}_{field}"], value, f"{sid}/{anchor}_{field}")
            for width in config["width_bins"]:
                values = [joint[v] for v, setting in variants.items() if setting[:2] == ("boundary", width)]
                fraction = sum(values) / len(values) if all(v is not None for v in values) else None
                _centering_value(summary[f"boundary_w{width}_outside_fraction"], fraction, f"{sid}/width{width}")
                fractions.append(fraction)
            all_quality = all(_csv_decision(r["quality_pass"]) for k, r in lookup.items() if k[0] == sid)
            _centering_value(summary["all_variants_quality_pass"], all_quality, f"{sid}/quality")
            _centering_value(summary["calibration_sufficient"], count >= 10, f"{sid}/calibration")
            reason = ("insufficient_calibration" if count < 10 else "quality_failed" if not all_quality
                      else "incomplete_grid" if any(v is None for v in joint.values()) else "")
            if summary["assessment_status"] != ("not_evaluable" if reason else "assessed") or summary["assessment_reason"] != reason:
                raise ValueError("Centering candidate evaluability/reason differs")
            robust = None if reason else bool(_csv_decision(summary["boundary_reproducible"]) and min(fractions) >= .8)
            _centering_value(summary["robust_recentered_anomaly"], robust, f"{sid}/robust")
            assessed += not bool(reason)
        if assessed == len(summaries):
            raise ValueError("Retained centering cohorts must preserve their incomplete evidence")
        if (report["candidates"] != len(identities) or cohort["candidates"] != len(identities)
                or report["candidate_metric_comparisons"] != len(rows)
                or report.get("variants_per_candidate", report.get("variants")) != len(variants)
                or report["assessed_candidates"] != assessed or report["not_evaluable_candidates"] != len(identities) - assessed
                or report["assessment_status"] != "not_evaluable"):
            raise ValueError("Centering cohort coverage/assessment totals differ")
        for field in ("candidates_with_any_two_metric_anomaly", "candidates_with_recentered_two_metric_anomaly",
                      "robust_recentered_anomalies", "sensitive_metric_configurations"):
            if report[field] is not None:
                raise ValueError("Centering incomplete cohort aggregate must remain null")
        result[name] = {"candidates": len(identities), "variants_per_candidate": len(variants),
                        "metric_rows": len(rows), "calibration_windows": count,
                        "assessed_candidates": assessed, "not_evaluable_candidates": len(identities) - assessed,
                        "whole_cohort_anomaly_counts": None}
    return {"cohorts": result, "independent_confirmation": False, "training_launched": False, "raw_data_read": False}


def verify_delivery_extension(submission: Path, repository: Path, core_names: set[str]) -> set[str]:
    """Check the separately indexed descriptive report without editing frozen evidence."""
    path = safe_path(submission, "results_delivery_manifest.json")
    if not path.exists():
        return set()
    extension = read_json(path)
    version = extension.get("schema_version")
    reports = extension.get("reports") if version == 2 else [extension.get("report")]
    expected_reports = [f"docs/TASK{number}.md" for number in range(1, 7)]
    if (version not in (1, 2) or extension.get("tasks") != 6
            or extension.get("training_performed") is not False
            or extension.get("raw_contact_scan_performed") is not False
            or (version == 1 and reports != ["docs/RESULTS_DELIVERY.md"])
            or (version == 2 and reports != expected_reports)):
        raise ValueError("Supplemental delivery index format or evidence boundaries differ")
    records = extension.get("files", [])
    names = [row["path"] for row in records]
    if not records or len(names) != len(set(names)) or any(name not in names for name in reports):
        raise ValueError("Supplemental delivery requires a unique file list including all task reports")
    extra = {"results_delivery_manifest.json"}
    for row in records:
        target = safe_path(repository, row["path"])
        if not target.is_file() or target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
            raise ValueError(f"Supplemental delivery differs: {row['path']}")
        if target.is_relative_to(submission):
            relative = target.relative_to(submission).as_posix()
            if relative in core_names or relative == "results_delivery_manifest.json":
                raise ValueError("Supplemental delivery cannot replace the core evidence index")
            extra.add(relative)
    return extra


def verify_submission(submission_root: Path | str, *, repository_root: Path | str | None = None,
                      source_root: Path | str | None = None) -> dict:
    submission = Path(submission_root).absolute()
    repository = Path(repository_root).absolute() if repository_root is not None else submission.parent
    if any(path.is_symlink() for path in submission.rglob("*")):
        raise ValueError("Symlinks are not supported inside a portable evidence package")
    manifest = read_json(safe_path(submission, "manifest.json"))
    if manifest.get("format_version") != 1 or signature({key: value for key, value in manifest.items() if key != "content_signature"}) != manifest.get("content_signature"):
        raise ValueError("Submission manifest format/signature differs")
    boundaries = manifest["scientific_boundaries"]
    if any(boundaries.get(key) is not False for key in ("independent_blind_test", "new_expert_truth_available", "independent_new_samples_available", "reliable_three_class_morphology_established")):
        raise ValueError("Submission must preserve the known scientific evidence boundaries")
    files = manifest["files"]
    names = [record["path"] for record in files]
    if not files or len(names) != len(set(names)):
        raise ValueError("Submission manifest requires a unique nonempty file list")
    total, source_checked = 0, 0
    for record in files:
        path = safe_path(submission, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Packaged evidence differs: {record['path']}")
        total += record["bytes"]
        if source_root is not None and record["source_kind"] == "copied_scientific_evidence":
            original = safe_path(Path(source_root).absolute(), record["source_path"])
            if not original.is_file() or original.stat().st_size != record["bytes"] or sha256(original) != record["sha256"]:
                raise ValueError(f"Read-only source evidence differs: {record['source_path']}")
            source_checked += 1
    actual = {path.relative_to(submission).as_posix() for path in submission.rglob("*") if path.is_file()}
    extension = verify_delivery_extension(submission, repository, set(names))
    expected_names = set(names) | {"manifest.json"} | extension
    if actual != expected_names:
        raise ValueError(f"Unindexed or missing submission files: {sorted(actual ^ expected_names)}")
    if total != manifest["total_bytes"]:
        raise ValueError("Submission byte total differs from manifest")
    legacy_count = 0
    for record in manifest.get("retained_legacy_files", []):
        path = safe_path(repository, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Preserved legacy showcase differs: {record['path']}")
        legacy_count += 1
    scientific = verify_scientific_snapshot(submission, manifest["scientific_checks"]) if manifest.get("scientific_checks") else None
    demo = verify_demo_metadata(submission, repository, manifest["demo_checks"]) if manifest.get("demo_checks") else None
    centering = verify_centering_snapshot(submission, manifest["centering_checks"]) if manifest.get("centering_checks") else None
    return {"status": "passed", "mode": "portable package verification" if source_root is None else "package and original-source verification",
            "packaged_files": len(files), "total_bytes": total, "retained_legacy_files": legacy_count,
            "original_source_files_checked": source_checked, "original_workspace_required": False,
            "scientific_checks": scientific, "demo_checks": demo, "centering_checks": centering}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, default=ROOT / "submission")
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, help="Optional read-only original scientific workspace; omitted for portable verification")
    arguments = parser.parse_args()
    try:
        result = verify_submission(arguments.submission_root, repository_root=arguments.repository_root, source_root=arguments.source_root)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
