#!/usr/bin/env python3
"""Cross-stage errors for every candidate; no model training/inference or selection.

D5 is explicitly excluded by default. --include-dropout refuses every incomplete
60-fit/12-OOF run before loading its scientific predictions.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from microcture.task1_formal.protocol import json_signature, sha256, _package_signature
from microcture.task1_formal.runner import assert_resume, verify_artifact_index
from microcture.task1_formal.evaluation import _guard_run, _identity, _fold_ids, _csv
from scripts.audit_task1_readout_results import scores, compare_scores

SEEDS = (42, 43, 44)
CLASSES = ("OPCID", "CHIN", "CHID")


def require(value, message):
    if not value: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def observed_status(correct):
    """A single fixed traditional seed never becomes cross-seed evidence."""
    if len(correct) == 1:
        return "single_seed_correct" if correct[0] else "single_seed_wrong"
    return "all_seed_correct" if all(correct) else "all_seed_wrong" if not any(correct) else "mixed"


def specs(include_d5=False):
    result = [
        {"name": "M2plus_R0", "family": "roi", "run": "task1_roi_v1", "alignment": "R0", "seeds": SEEDS},
        {"name": "M2plus_R1", "family": "roi", "run": "task1_roi_v1", "alignment": "R1", "seeds": SEEDS},
        {"name": "M2plus_C0", "family": "native", "run": "task1_native_roi_v2", "alignment": "C0", "seeds": SEEDS},
        {"name": "M2plus_C1", "family": "native", "run": "task1_native_roi_v2", "alignment": "C1", "seeds": SEEDS},
        {"name": "shape_LR", "family": "shape", "run": "task1_shape_baseline_v2", "model": "lr_shape", "seeds": (42,)},
        {"name": "shape_SVC", "family": "shape", "run": "task1_shape_baseline_v2", "model": "svc_shape", "seeds": (42,)},
        {"name": "fixed_fusion_main", "family": "fusion", "run": "task1_fixed_fusion_v1", "seeds": SEEDS},
    ]
    if include_d5:
        result.append({"name": "M2plus_D5", "family": "dropout", "run": "task1_dropout_v1", "alignment": "R1", "seeds": SEEDS})
    return result


def require_complete(summary, family):
    require(summary["status"] == "complete", f"Incomplete {family} predictions are not reviewed")
    if family in ("roi", "native", "dropout"):
        fit_count, oof_count = {"roi": (75, 15), "native": (90, 18), "dropout": (60, 12)}[family]
        require(summary["completed_jobs"] == fit_count and summary["complete_oof"] == oof_count
                and not any(summary[k] for k in ("pending_jobs", "invalid_jobs", "execution_failure_jobs", "numerical_failure_jobs")), "Scientific run incomplete/invalid")
    elif family == "shape":
        require(summary["completed_fits"] == 40 and summary["complete_OOF"] == 8 and
                not any(summary[k] for k in ("pending_fits", "invalid_fits", "execution_failure_fits", "numerical_failure_fits")), "Shape run incomplete/invalid")
    else:
        require(summary["derived_fold_jobs"] == 80 and summary["main_OOF"] == 12 and summary["pooled_OOF"] == 4, "Fusion inventory incomplete")


def fold_path(run, specification, seed, fold):
    family = specification["family"]
    if family in ("roi", "native", "dropout"):
        return run / "runs" / specification["alignment"] / "I3/m2plus" / f"seed_{seed}" / f"fold_{fold}"
    if family == "shape":
        return run / "runs/I3" / specification["model"] / f"seed_{seed}" / f"fold_{fold}"
    return run / "runs/main/I3" / f"seed_{seed}" / f"fold_{fold}"


def oof_path(run, specification, seed):
    family = specification["family"]
    if family in ("roi", "native", "dropout"):
        return run / "oof" / specification["alignment"] / "I3" / f"seed_{seed}" / "predictions.csv"
    if family == "shape":
        return run / "oof/I3" / specification["model"] / f"seed_{seed}" / "predictions.csv"
    return run / "oof/main/I3" / f"seed_{seed}" / "predictions.csv"


def verify_source(run, manifest, family):
    require(manifest["experiment_signature"] == json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}), "Manifest signature differs")
    if family == "fusion":
        require(sha256(ROOT / "scripts/evaluate_task1_fixed_fusion.py") == manifest["source_signature"], "Fixed fusion source changed")
    else:
        package, entry = {"roi": ("task1_roi", "run_task1_roi.py"), "native": ("task1_native_roi_v2", "run_task1_native_roi_v2.py"),
                          "shape": ("task1_shape_baseline_v2", "run_task1_shape_baseline_v2.py"), "dropout": ("task1_dropout_v1", "run_task1_dropout.py")}[family]
        require(_package_signature(ROOT, package, entry) == manifest["source_signature"], "Saved training source changed")
    require(manifest["dataset_signature"] == read_json(ROOT / "data/cache/task1_comparison_v1/metadata.json")["signature"], "Cross-stage dataset differs")


def summarize(output, include_d5=False):
    # No D5 path is opened unless the caller explicitly adds that completed stage.
    methods = specs(include_d5)
    metadata = read_json(ROOT / "data/cache/task1_comparison_v1/metadata.json")
    cached_folds = read_json(ROOT / "data/cache/task1_comparison_v1/folds.json")
    require(metadata["signature"] == json_signature({"metadata": {k: v for k, v in metadata.items() if k != "signature"},
                                                    "folds": cached_folds}), "Canonical source annotation/fold metadata signature differs")
    identity = _identity(metadata)
    require(len(identity) == 344 and len(set(metadata["group_id"])) == 49, "Expected all344/49 groups")
    reference_manifest = read_json(ROOT / "results/task1_roi_v1/manifest.json")
    folds = _fold_ids(reference_manifest, identity, ROOT / "results/task1_roi_v1")
    fold_owner = {sid: number for number, fold in folds.items() for sid in fold["test_ids"]}
    by_method, source_inventory, method_scores = {}, [], {}
    checked_runs = set()
    for specification in methods:
        name, family = specification["name"], specification["family"]
        run = ROOT / "results" / specification["run"]
        _guard_run(run)
        summary = read_json(run / "summary.json")
        require_complete(summary, family)  # Before accessing any fold score.
        manifest = read_json(run / "manifest.json")
        verify_source(run, manifest, family)
        if family != "fusion":
            require(manifest["fold_ids"] == reference_manifest["fold_ids"], "Cross-method outer/inner folds differ")
        if run not in checked_runs:
            source_inventory.extend({"path": str((run / filename).relative_to(ROOT)), "sha256": sha256(run / filename)} for filename in ("manifest.json", "summary.json"))
            checked_runs.add(run)
        by_method[name], method_scores[name] = {}, {}
        for seed in specification["seeds"]:
            merged = []
            for number, frozen in folds.items():
                directory = fold_path(run, specification, seed, number)
                value = read_json(directory / "metrics.json")
                job = value["job"]
                require(job["fold"] == number and job["seed"] == seed and job["condition"] == "I3", "Scientific job identity differs")
                if family == "fusion":
                    require(job["mode"] == "main" and value["learned_parameters"] == 0, "Pooled fusion cannot replace main endpoint seeds")
                    verify_artifact_index(directory, job, manifest)
                    recorded_score = value["metrics"]
                else:
                    require(value["status"] == "completed", "A completed case summary cannot omit failed model fits")
                    assert_resume(directory, job, manifest)
                    recorded_score = value["outer_metrics"]
                source_inventory.append({"path": str((directory / "ARTIFACTS.json").relative_to(ROOT)), "sha256": sha256(directory / "ARTIFACTS.json")})
                source = rows(directory / "outer_predictions.csv")
                score = scores(source)  # Independent WT averaging and argmax checks.
                compare_scores(recorded_score, score, f"{name}/fold{number}")
                require(len(source) == len(frozen["test_ids"]) and {r["structure_id"] for r in source} == set(frozen["test_ids"]), "Fold structure inventory differs")
                for row in source:
                    sid = row["structure_id"]
                    require((int(row["y_true"]), int(row["group_id"])) == (identity[sid]["y"], identity[sid]["group_id"]), "Truth/spatial group differs")
                    merged.append({**row, "fold": str(number)})
            merged.sort(key=lambda r: r["structure_id"])
            require(len(merged) == 344 and len({r["structure_id"] for r in merged}) == 344, "Every method needs all344 unique structures")
            require(merged == rows(oof_path(run, specification, seed)), "Derived OOF differs from sealed per-fold predictions")
            by_method[name][seed] = {r["structure_id"]: r for r in merged}
            method_scores[name][str(seed)] = scores(merged)
    # Independent main fusion lineage at each WT before its mean is taken.
    for seed in SEEDS:
        for sid in identity:
            def wt(method, endpoint):
                return {int(r["replicate"]): np.asarray(r["probabilities"], dtype=float)
                        for r in json.loads(by_method[method][endpoint][sid]["replicate_probabilities_json"])}
            lr, cnn, actual = wt("shape_LR", 42), wt("M2plus_R1", seed), wt("fixed_fusion_main", seed)
            require(all(np.allclose(actual[rep], .5 * lr[rep] + .5 * cnn[rep], atol=1e-12, rtol=0) for rep in (1, 2)), "Fixed fusion does not match registered WT-level half/half parents")
    cases, summaries = [], {}
    for sid, reference in sorted(identity.items()):
        case = {"structure_id": sid, "type": CLASSES[reference["y"]], "group_id": reference["group_id"],
                "outer_fold": fold_owner[sid], "length_bp": reference["length_bp"]}
        all_observed_wrong, all_multiseed_wrong, multiseed_count = True, True, 0
        for specification in methods:
            name, seeds = specification["name"], specification["seeds"]
            source = [by_method[name][seed][sid] for seed in seeds]
            predicted = [int(r["y_pred"]) for r in source]
            correct = [p == reference["y"] for p in predicted]
            state = observed_status(correct)
            all_observed_wrong &= not any(correct)
            if len(seeds) > 1:
                all_multiseed_wrong &= not any(correct)
                multiseed_count += 1
            wt_records = [{int(z["replicate"]): z for z in json.loads(r["replicate_probabilities_json"])} for r in source]
            case.update({f"{name}_status": state, f"{name}_observed_seed_count": len(seeds),
                f"{name}_correct_seed_count": sum(correct),
                f"{name}_WT_disagreement_seed_count": sum(int(r["replicate_disagree"]) for r in source),
                f"{name}_WT1_wrong_seed_count": sum(int(r[1]["predicted"]) != reference["y"] for r in wt_records),
                f"{name}_WT2_wrong_seed_count": sum(int(r[2]["predicted"]) != reference["y"] for r in wt_records),
                **{f"{name}_seed_{s}_prediction": CLASSES[p] for s, p in zip(seeds, predicted)},
                **{f"{name}_seed_{s}_CHID_score": float(r["p_CHID"]) for s, r in zip(seeds, source)}})
        case["all_multiseed_methods_wrong"] = bool(all_multiseed_wrong)
        case["all_evaluated_methods_wrong_observed"] = bool(all_observed_wrong)
        case["multiseed_methods_compared"] = multiseed_count
        case["CHID_missed_by_all_multiseed_methods"] = reference["y"] == 2 and all_multiseed_wrong
        case["CHID_missed_by_all_methods_observed"] = reference["y"] == 2 and all_observed_wrong
        cases.append(case)
    for specification in methods:
        name = specification["name"]
        status_counts = {status: sum(r[f"{name}_status"] == status for r in cases) for status in
                         (("single_seed_correct", "single_seed_wrong") if len(specification["seeds"]) == 1 else ("all_seed_correct", "all_seed_wrong", "mixed"))}
        summaries[name] = {"source": specification, "state_counts": status_counts,
            "score_semantics": "SVC uncalibrated softmax(decision) ranking score" if name == "shape_SVC" else "Saved softmax/probability score; calibration not established",
            "complete_OOF_scores": method_scores[name],
            "CHID_missed_in_every_observed_seed": [r["structure_id"] for r in cases if r["type"] == "CHID" and r[f"{name}_correct_seed_count"] == 0],
            "CHID_stability_claim": "not_estimated_single_fixed_seed" if len(specification["seeds"]) == 1 else "three_optimization_seeds_on_same_developmental_cases",
            "WT_disagreement_seed_events": sum(r[f"{name}_WT_disagreement_seed_count"] for r in cases),
            "structures_with_any_WT_disagreement": sum(r[f"{name}_WT_disagreement_seed_count"] > 0 for r in cases)}
    payload = {"status": "complete_registered_stages", "D5_included": include_d5,
        "pending_stages": [] if include_d5 else [{"method": "M2plus_D5", "status": "not_included_pending_complete60_12", "partial_results_read": False}],
        "unique_structures_per_method_seed": 344, "spatial_groups": 49, "n_methods": len(methods),
        "complete_method_OOF": sum(len(m["seeds"]) for m in methods), "methods": summaries,
        "cross_method_persistent_CHID_misses_multiseed_only": [r["structure_id"] for r in cases if r["CHID_missed_by_all_multiseed_methods"]],
        "cross_method_CHID_misses_all_observed_including_single_seed": [r["structure_id"] for r in cases if r["CHID_missed_by_all_methods_observed"]],
        "source_inventory": source_inventory, "source_inventory_signature": json_signature(source_inventory),
        "script_sha256": sha256(Path(__file__)), "dataset_signature": metadata["signature"],
        "limits": ["All methods share original344 candidates, labels, outer folds and WT identities; inputs/model families differ and stages were adaptive.",
                   "Seeds are optimizer repeats on the same structures, not biological replicates. LR and SVC have only fixed seed42.",
                   "Main fusion shares CNN and LR parents, so methods are dependent and cross-method agreement is not independent biological confirmation.",
                   "Every candidate is retained; persistent errors do not authorize relabeling or discarding ambiguous cases."]}
    require((ROOT / "reports") in output.resolve().parents and not any(p.is_symlink() for p in (output, *output.parents)), "Reports must remain inside project reports without symlinks")
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "case_summary.csv", cases)
    (output / "case_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_autonomous_review_v1")
    parser.add_argument("--include-dropout", "--include-d5", dest="include_d5", action="store_true",
                        help="Add completed D5 only after all60 fits/12 OOF pass the complete-run gate")
    args = parser.parse_args()
    result = summarize(args.output, args.include_d5)
    print(json.dumps({k: result[k] for k in ("status", "D5_included", "n_methods", "complete_method_OOF", "cross_method_persistent_CHID_misses_multiseed_only", "cross_method_CHID_misses_all_observed_including_single_seed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
