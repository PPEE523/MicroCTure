"""Saved full-fit trajectories and preregistered endpoint OOF evaluation.

No contact arrays, checkpoint deserialization, inference or fitting. Completed
constant or learning-criterion-failing models remain in their fixed-policy
scores. A failed fit or incomplete seed is never silently repaired or scored.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from microcture.task1.evaluation import CLASS_NAMES, _read_prediction_csv, metrics
from microcture.task1_controlled.diagnostics import learning_acceptance
from microcture.task1_formal.evaluation import (_guard_run, _read_json, _identity,
    _fold_ids, _same_score, _history, _describe, _seed_agreement, _error_rows,
    _neural_summary, _baseline_outer_summary, _events, _write_derived_json, _csv, _FAILED,
    _project_path)
from .protocol import (endpoint_directory, endpoint_policies, job_directory,
                       json_signature, planned_jobs, prepare_manifest)


def _require(value: bool, description: str) -> None:
    if not value:
        raise ValueError(description)


def _near(first, second, description: str) -> None:
    _require(np.isclose(first, second, atol=1e-10, rtol=0), description)


def _load_endpoint(run_dir: Path, job: dict, policy: str, manifest: dict,
                   common: dict, history: list[dict], identity: dict, folds: dict) -> dict:
    from .selection import endpoint_selection_records
    from .protocol import sha256
    directory = endpoint_directory(run_dir, job, policy)
    value, marker = _read_json(directory / "metrics.json"), _read_json(directory / "endpoint.json")
    _require(value.get("status") == "completed" and value.get("job") == marker.get("job") == job,
             "Endpoint completion/job differs")
    _require(value.get("policy") == marker.get("policy") == policy, "Endpoint policy differs")
    for key in ("experiment_signature", "source_signature", "dataset_signature"):
        _require(value.get(key) == manifest[key], f"Endpoint {key} differs")
    _require(value.get("preprocessing_signature") == common["preprocessing_signature"], "Endpoint preprocessing differs")
    selection_sig = manifest["selection_signature"]
    _require(value.get("selection_signature") == marker.get("selection_signature") == selection_sig,
             "Endpoint selection rule signature differs")
    if job["kind"] == "neural":
        choices = endpoint_selection_records(history, manifest["config"])
        selected = choices[policy]["selected_epoch"]
        _require(value.get("selected_epoch") == marker.get("selected_epoch") == selected,
                 "Endpoint epoch differs from preregistered inner-only selection")
        row = history[selected]
        for view in ("train", "validation"):
            _require(value.get(f"{view}_diagnostics") == row[f"{view}_diagnostics"],
                     f"Endpoint {view} diagnostics differ from selected epoch")
            _near(value[f"weighted_{view}_ce"], row[f"weighted_{view}_ce"], f"Selected {view} CE differs")
        checkpoint_name = "weights.pt"
    else:
        _require(value.get("selected_epoch") is None and marker.get("selected_epoch") is None,
                 "Classical baseline must have one fixed endpoint and no neural epoch selection")
        checkpoint_name = "model.joblib"
    _require(marker.get("checkpoint") == checkpoint_name
             and marker.get("checkpoint_sha256") == value.get("checkpoint_sha256") == sha256(directory / checkpoint_name),
             "Endpoint checkpoint identity/bytes differ")
    _require(marker.get("outer_images_constructed") is False and marker.get("training_and_selection_complete") is True
             and value.get("outer_fit_or_selection") is False and value.get("outer_constructed_after_fixed_endpoint") is True,
             "Endpoint selection must precede outer construction")
    predictions = _read_prediction_csv(directory / "outer_predictions.csv")
    validation = _read_prediction_csv(directory / "validation_predictions.csv")
    for view, prediction in (("test", predictions), ("validation", validation)):
        _require(set(prediction["structure_id"]) == set(folds[job["fold"]][f"{view}_ids"]),
                 "Endpoint CSV IDs differ from frozen split")
        for sid, label, group in zip(prediction["structure_id"], prediction["y"], prediction["group_id"]):
            _require((int(label), int(group)) == (identity[str(sid)]["y"], identity[str(sid)]["group_id"]),
                     "Endpoint CSV identity/label/group differs")
    _same_score(value["outer_metrics"], predictions["metrics"], "Endpoint outer metrics")
    if job["kind"] == "neural":
        _same_score(value["outer_diagnostics"]["metrics"], predictions["metrics"], "Endpoint outer diagnostics metrics")
        _same_score(value["validation_diagnostics"]["metrics"], validation["metrics"], "Endpoint validation diagnostics metrics")
        probability = predictions["probabilities"]
        counts = dict(zip(CLASS_NAMES, np.bincount(probability.argmax(axis=1), minlength=3).tolist()))
        diagnostic = value["outer_diagnostics"]
        _require(diagnostic["prediction_counts"] == counts
                 and diagnostic["num_predicted_classes"] == sum(n > 0 for n in counts.values()), "Endpoint predicted-class counts differ")
        _near(diagnostic["majority_prediction_fraction"], max(counts.values()) / len(probability), "Endpoint majority fraction differs")
        _near(diagnostic["probabilities"]["max_coordinate_range"], np.ptp(probability, axis=0).max(), "Endpoint probability range differs")
        _near(diagnostic["probability_between_structure_var"], np.var(probability, axis=0).mean(), "Endpoint probability variance differs")
    else:
        _same_score(value["validation_metrics"], validation["metrics"], "Baseline validation metrics")
    return {"status": "completed", "policy": policy, "metrics": value, "predictions": predictions,
            "directory": str(directory.relative_to(run_dir))}


def _load_fit(run_dir: Path, job: dict, manifest: dict, identity: dict, folds: dict) -> dict:
    from .runner import assert_resume, failed_staging_directory
    directory = job_directory(run_dir, job)
    _require(not failed_staging_directory(run_dir, job).exists(), "Unsealed failed staging is retained; no retry/scoring")
    value = assert_resume(directory, job, manifest)
    if value is None:
        return {"job": job, "status": "pending", "endpoints": {}}
    _require(value.get("preprocessing_signature"), "Fit preprocessing signature is required")
    status = value["status"]
    _require(status in ("completed", "numerical_failure", "execution_failure"), "Unknown fit execution status")
    record = {"job": job, "status": status, "metrics": value, "endpoints": {}}
    history = []
    if job["kind"] == "neural":
        try:
            history = _history(directory / "history.json")
        except (ValueError, OSError) as error:
            if status == "completed":
                raise
            record["history_read_error"] = str(error)
        record["history"] = history
        if status == "completed":
            _require([row["epoch"] for row in history] == list(range(41)) and value.get("epochs") == 40,
                     "Completed fit must preserve epoch zero plus all forty training epochs")
            _near(value["initial_train_ce"], history[0]["weighted_train_ce"], "Initial fit CE differs")
            _near(value["final_train_ce"], history[-1]["weighted_train_ce"], "Full-fit final CE differs")
            _require(value["acceptance"] == learning_acceptance(history, manifest["config"]),
                     "Shared full-forty learning acceptance differs")
            for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256"):
                _require(bool(value.get(field)), f"Missing fit {field}")
            for view in ("train", "validation"):
                _require(value[f"{view}_diagnostics"] == history[-1][f"{view}_diagnostics"], "Full-fit diagnostics differ from epoch forty")
    if status != "completed":
        return record
    protocol = _read_json(directory / "protocol.json")
    shared = _read_json(run_dir / "folds" / f"fold_{job['fold']}.json")
    preprocessing = protocol["preprocessing"]
    _require(preprocessing == shared["preprocessing"]
             and shared["frozen_fold"]["train_ids"] == folds[job["fold"]]["train_ids"]
             and shared["frozen_fold"]["validation_ids"] == folds[job["fold"]]["validation_ids"]
             and shared["frozen_fold"]["test_ids"] == folds[job["fold"]]["test_ids"],
             "Fit/shared preprocessing or split differs")
    _require(json_signature(shared["frozen_fold"]) == manifest["fold_signatures"][str(job["fold"])]
             and preprocessing["frozen_fold_signature"] == manifest["fold_signatures"][str(job["fold"])]
             and preprocessing["signal_preprocessing"] == shared["frozen_fold"]["preprocessing"],
             "Shared frozen fold or inner signal fitting statistics changed")
    _require(json_signature({k: v for k, v in preprocessing.items() if k != "signature"}) == preprocessing["signature"]
             and preprocessing["signature"] == value["preprocessing_signature"], "Fitted transform signature differs")
    _require(preprocessing["outer_images_used"] is False and preprocessing["signal_statistics_refitted"] is False
             and preprocessing["splits_refitted"] is False
             and preprocessing["train_ids"] == folds[job["fold"]]["train_ids"]
             and preprocessing["validation_ids"] == folds[job["fold"]]["validation_ids"], "Preprocessing inner-only scope differs")
    lengths = np.asarray([np.log1p(identity[sid]["length_bp"] / 100) for sid in folds[job["fold"]]["train_ids"]])
    length_scaler = preprocessing["length_scaler"]
    _near(length_scaler["mean"], float(lengths.mean()), "Length mean differs from unique inner training lengths")
    _near(length_scaler["raw_std"], float(lengths.std(ddof=0)), "Length population SD differs from unique inner training lengths")
    _near(length_scaler["std"], float(lengths.std(ddof=0)) if lengths.std(ddof=0) >= 1e-6 else 1., "Length scaler used SD differs")
    _require(length_scaler["fitted_unique_structures"] == len(lengths)
             and length_scaler["fit_scope"] == "frozen inner-training structures only", "Length scaler fitting scope differs")
    _require(protocol["training"] == manifest["config"]["training"]
             and protocol["baseline"] == manifest["config"]["baseline"]
             and protocol["acceptance"] == manifest["config"]["acceptance"], "Fit frozen settings differ")
    if job["kind"] == "neural":
        class_counts = np.bincount([identity[sid]["y"] for sid in folds[job["fold"]]["train_ids"]], minlength=3)
        expected_weights = class_counts.sum() / (3 * class_counts)
        _require(np.allclose(value["class_weights"], expected_weights, atol=1e-12, rtol=0), "Class weights differ from inner training IDs")
        initialization = _read_json(directory / "initialization.json")
        _require(initialization["job"] == job and initialization["seed"] == job["seed"], "Initialization job/seed differs")
        for field in ("initial_state_sha256", "common_initial_state_sha256"):
            _require(initialization[field] == value[field], "Initialization fingerprint differs")
        for row in history:
            for view in ("train", "validation"):
                diagnostic = row[f"{view}_diagnostics"]
                counts = np.bincount([identity[sid]["y"] for sid in folds[job["fold"]][f"{view}_ids"]], minlength=3)
                support = diagnostic["loss_support"]
                _require(support["examples_count_by_class"] == (counts * 2).tolist()
                         and diagnostic["n_structures"] == int(counts.sum())
                         and diagnostic["n_replicate_examples"] == int(counts.sum() * 2), "History fixed-view support differs")
                nll = np.asarray(support["unweighted_nll_sum_by_class"], dtype=float)
                _near(row[f"weighted_{view}_ce"], float((nll * expected_weights).sum() / (counts * 2 * expected_weights).sum()),
                      "History fixed weighted CE differs from saved NLL and class weights")
    fixed = _read_json(directory / "endpoints_fixed.json")
    _require(fixed["job"] == job and fixed["selection_signature"] == manifest["selection_signature"]
             and fixed["outer_images_constructed"] is False and fixed["training_and_selection_complete"] is True
             and value["endpoints_fixed_before_outer"] is True, "All endpoint fixation must precede outer construction")
    for field, split in (("train_structures", "train"), ("validation_structures", "validation"), ("test_structures", "test")):
        _require(value[field] == len(folds[job["fold"]][f"{split}_ids"]), "Fit split support differs")
    _require(np.isfinite(value["runtime_seconds"]) and value["runtime_seconds"] >= 0, "Invalid fit runtime")
    for policy in endpoint_policies(job):
        record["endpoints"][policy] = _load_endpoint(run_dir, job, policy, manifest, value, history, identity, folds)
    _require(set(fixed["checkpoint_sha256_by_policy"]) == set(endpoint_policies(job))
             and set(fixed["selected_epoch_by_policy"]) == set(endpoint_policies(job)), "All registered endpoint hashes/epochs are required")
    for policy, endpoint in record["endpoints"].items():
        marker = _read_json(endpoint_directory(run_dir, job, policy) / "endpoint.json")
        _require(fixed["checkpoint_sha256_by_policy"][policy] == endpoint["metrics"]["checkpoint_sha256"]
                 and fixed["selected_epoch_by_policy"][policy] == endpoint["metrics"]["selected_epoch"]
                 and marker["fixed_endpoint_hashes"] == fixed["checkpoint_sha256_by_policy"]
                 and marker["both_neural_endpoints_fixed_before_outer"] is (job["kind"] == "neural"),
                 "Endpoint paired fixation marker differs from all registered hashes/epochs")
    return record


def _invalidate_pairs(records: dict, errors: list[dict]) -> None:
    by_fold, by_seed_backbone = defaultdict(list), defaultdict(list)
    for record in records.values():
        if record["status"] != "completed":
            continue
        job = record["job"]
        by_fold[job["fold"]].append(record)
        if job["kind"] == "neural":
            by_seed_backbone[(job["model"], job["seed"], job["fold"])].append(record)
    for records_by_key, fields in ((by_fold, ("preprocessing_signature",)),
                                   (by_seed_backbone, ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights"))):
        for key, items in records_by_key.items():
            for field in fields:
                if len({json.dumps(item["metrics"].get(field), sort_keys=True) for item in items}) > 1:
                    reason = f"Paired/shared {field} differs for {key}"
                    errors.append({"reason": reason})
                    for item in items:
                        item.update(status="invalid", reason=reason)


def _write_incomplete_oof(destination: Path, provenance: dict, status: str, reasons: dict) -> None:
    _write_derived_json(destination / "status.json", {**provenance, "status": status, **reasons,
        "oof_current": False, "previous_numeric_artifacts_retained": (destination / "metrics.json").exists() or (destination / "predictions.csv").exists(),
        "interpretation": "Previous numeric files are retained historical evidence and are not valid current OOF."})


def summarize_reevaluation(run_dir: Path | str, metadata: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    _require(not (run_dir / "FROZEN.json").exists(), "Frozen reevaluation output is read-only")
    _guard_run(run_dir)
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    _require(run_dir == _project_path(manifest["config"]["run_dir"]).resolve(), "Read the experiment at its original registered output path")
    _require(json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) == manifest["experiment_signature"],
             "Reevaluation manifest signature differs")
    current_manifest, current_metadata = prepare_manifest(_project_path("."), manifest["config"], manifest["requested_device"])
    _require(current_manifest == manifest, "Current source/config/dependencies/environment differ from trained run")
    metadata = current_metadata if metadata is None else metadata
    _require(metadata.get("signature") == manifest["dataset_signature"], "Dataset signature differs")
    identity, errors, records = _identity(metadata), [], {}
    folds = _fold_ids(manifest, identity, run_dir)
    jobs = planned_jobs(manifest["config"])
    from .runner import unpublished_tmp_staging
    pending_staging = unpublished_tmp_staging(_project_path("."), run_dir)
    for job in jobs:
        key = (job["condition"], job["model"], job["seed"], job["fold"])
        try:
            _require(key not in pending_staging, "Unpublished scientific staging is preserved")
            record = _load_fit(run_dir, job, manifest, identity, folds)
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            record = {"job": job, "status": "invalid", "reason": str(error), "endpoints": {}}
            errors.append({"job": job, "reason": str(error)})
        records[key] = record
    _invalidate_pairs(records, errors)
    fit_rows, endpoint_rows, grouped, fit_learning_groups = [], [], defaultdict(list), defaultdict(list)
    for record in records.values():
        job, value = record["job"], record.get("metrics", {})
        if job["kind"] == "neural":
            fit_learning_groups[(job["condition"], job["model"])].append(record)
        fit_rows.append({"condition": job["condition"], "model": job["model"], "seed": job["seed"], "fold": job["fold"],
            "kind": job["kind"], "status": record["status"], "reason": record.get("reason"),
            "runtime_seconds": value.get("runtime_seconds"), "final_train_ce": value.get("final_train_ce"),
            "shared_full40_learning_pass": (value.get("acceptance") or {}).get("accepted")})
        for policy in endpoint_policies(job):
            endpoint = record.get("endpoints", {}).get(policy, {})
            status = "completed" if record["status"] == "completed" and endpoint.get("status") == "completed" else record["status"]
            endpoint_record = {**record, "status": status, "policy": policy}
            if status == "completed":
                # Frozen summary arithmetic gets the shared forty-epoch fit
                # acceptance and the particular endpoint's outer diagnostics.
                endpoint_record.update(metrics={**value, **endpoint["metrics"],
                    "acceptance": value.get("acceptance"), "final_train_ce": value.get("final_train_ce"),
                    **({f"{view}_diagnostics": value[f"{view}_diagnostics"] for view in ("train", "validation")}
                       if job["kind"] == "neural" else {})},
                    predictions=endpoint["predictions"])
            grouped[(policy, job["condition"], job["model"])].append(endpoint_record)
            endpoint_value = endpoint.get("metrics", {})
            endpoint_rows.append({"policy": policy, "condition": job["condition"], "model": job["model"],
                "seed": job["seed"], "fold": job["fold"], "status": status, "selected_epoch": endpoint_value.get("selected_epoch"),
                "weighted_train_ce": endpoint_value.get("weighted_train_ce"), "weighted_validation_ce": endpoint_value.get("weighted_validation_ce"),
                "outer_macro_f1": endpoint.get("predictions", {}).get("metrics", {}).get("macro_f1"),
                "outer_prediction_counts": endpoint_value.get("outer_diagnostics", {}).get("prediction_counts")})
    shared_learning = {f"{condition}/{model}": _neural_summary(items, manifest["config"])
                       for (condition, model), items in fit_learning_groups.items()}
    for value in shared_learning.values():
        value["prediction_events"].pop("outer", None)
        value["unit"] = "One full-forty fit per fold/seed; used by both policies and counted once."
    comparison, model_details, complete_oof, error_rows = [], {}, {}, []
    for (policy, condition, model), items in grouped.items():
        seeds = sorted({item["job"]["seed"] for item in items})
        seed_detail, seed_oof = {}, {}
        for seed in seeds:
            seed_records = [item for item in items if item["job"]["seed"] == seed]
            destination = run_dir / "oof" / policy / condition / model / f"seed_{seed}"
            provenance = {"policy": policy, "condition": condition, "model": model, "seed": seed,
                **{key: manifest[key] for key in ("experiment_signature", "source_signature", "dataset_signature", "selection_signature")},
                "expected_structures": len(identity), "expected_folds": list(range(5))}
            problems = {name: [item["job"]["fold"] for item in seed_records if item["status"] == value]
                        for name, value in (("pending_folds", "pending"), ("numerical_failure_folds", "numerical_failure"),
                                            ("execution_failure_folds", "execution_failure"), ("invalid_folds", "invalid"))}
            if any(problems.values()):
                status = "invalid" if problems["invalid_folds"] else "failed" if problems["numerical_failure_folds"] or problems["execution_failure_folds"] else "pending"
                seed_detail[str(seed)] = {"status": status, **problems}
                _write_incomplete_oof(destination, provenance, status, problems)
                continue
            prediction_rows = [{**row, "fold": item["job"]["fold"]} for item in seed_records for row in item["predictions"]["rows"]]
            ids = [row["structure_id"] for row in prediction_rows]
            if len(ids) != len(set(ids)) or set(ids) != set(identity):
                reason = "Five complete folds do not cover exactly every unique structure once"
                errors.append({"policy": policy, "condition": condition, "model": model, "seed": seed, "reason": reason})
                seed_detail[str(seed)] = {"status": "invalid", "reason": reason}
                _write_incomplete_oof(destination, provenance, "invalid", {"reason": reason})
                continue
            prediction_rows.sort(key=lambda row: row["structure_id"])
            labels = np.asarray([int(row["y_true"]) for row in prediction_rows])
            probabilities = np.asarray([[float(row[f"p_{name}"]) for name in CLASS_NAMES] for row in prediction_rows])
            score = metrics(labels, probabilities)
            cm = score["confusion_matrix"]
            score["CHID_counts"] = {"TP": cm[2][2], "FP": cm[0][2] + cm[1][2], "FN": cm[2][0] + cm[2][1], "support": sum(cm[2])}
            result = {"metrics": score, "rows": prediction_rows, "by_id": {row["structure_id"]: row for row in prediction_rows}}
            complete_oof[(policy, condition, model, seed)], seed_oof[seed] = result, result
            seed_detail[str(seed)] = {"status": "complete", "oof": score}
            _write_derived_json(destination / "metrics.json", {**score, **provenance, "status": "complete", "oof_current": True,
                "score_unit": "unique annotated structure after two WT probability averaging"})
            _csv(destination / "predictions.csv", prediction_rows)
            _write_derived_json(destination / "status.json", {**provenance, "status": "complete", "oof_current": True, "n_structures": len(ids)})
            error_rows.extend([{**row, "policy": policy} for row in _error_rows(prediction_rows, identity, model, condition, seed)])
        complete = len(seed_oof) == len(seeds)
        scores = [seed_oof[seed]["metrics"] for seed in seeds] if complete else []
        agreement = _seed_agreement(seeds, seed_oof, identity)
        row = {"policy": policy, "condition": condition, "model": model, "status": "complete" if complete else "pending",
            "expected_fits": len(items), "completed_endpoints": sum(item["status"] == "completed" for item in items),
            "complete_seeds": len(seed_oof), "expected_seeds": len(seeds),
            "seed_unanimous_rate": agreement["unanimous_rate"] if agreement else None}
        for name in ("macro_f1", "accuracy", "balanced_accuracy"):
            values = [score[name] for score in scores]
            row[f"{name}_mean"] = float(np.mean(values)) if values else None
            row[f"{name}_seed_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        for name in CLASS_NAMES:
            for measure in ("precision", "recall", "f1"):
                values = [score["per_class"][name][measure] for score in scores]
                row[f"{name}_{measure}_mean"] = float(np.mean(values)) if values else None
                row[f"{name}_{measure}_seed_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        detail = {"seeds": seed_detail, "seed_prediction_agreement": agreement}
        if items[0]["job"]["kind"] == "neural":
            detail["shared_full40_learning_reference"] = f"{condition}/{model}"
            detail["selected_endpoint_events"] = {}
            completed_items = [item for item in items if item["status"] == "completed"]
            for view in ("train", "validation", "outer"):
                observed = [_events(item["endpoints"][policy]["metrics"][f"{view}_diagnostics"], manifest["config"]["acceptance"])
                            for item in completed_items]
                detail["selected_endpoint_events"][view] = {"observed_endpoints": len(observed),
                    "measurement": "One diagnostic at the chosen epoch/endpoint per completed fit; not full-forty terminal diagnostics.",
                    "events": {name: sum(value[name] for value in observed) for name in observed[0]} if observed else {}}
        else:
            detail["outer_events"] = _baseline_outer_summary(items, manifest["config"])
        comparison.append(row)
        model_details[f"{policy}/{condition}/{model}"] = detail
    paired_rows, paired_summaries = [], {}
    for condition in ("I3", "I0", "G0"):
        for model in manifest["config"]["backbones"]:
            rows_for_comparison = []
            for seed in (42, 43, 44):
                first = complete_oof.get(("inner_ce", condition, model, seed))
                second = complete_oof.get(("final40", condition, model, seed))
                if first is None or second is None:
                    continue
                a, b = first["metrics"], second["metrics"]
                rows_for_comparison.append({"condition": condition, "model": model, "seed": seed,
                    "comparison": "inner_ce-final40", "macro_f1_delta": a["macro_f1"] - b["macro_f1"],
                    "CHID_f1_delta": a["per_class"]["CHID"]["f1"] - b["per_class"]["CHID"]["f1"],
                    "balanced_accuracy_delta": a["balanced_accuracy"] - b["balanced_accuracy"]})
            complete = len(rows_for_comparison) == 3
            paired_summaries[f"{condition}/{model}"] = {"status": "complete" if complete else "pending", "complete_seed_pairs": len(rows_for_comparison),
                "expected_seed_pairs": 3, "deltas": rows_for_comparison,
                **{f"{name}_mean": float(np.mean([r[name] for r in rows_for_comparison])) if complete else None
                   for name in ("macro_f1_delta", "CHID_f1_delta", "balanced_accuracy_delta")},
                **{f"{name}_seed_sample_std": float(np.std([r[name] for r in rows_for_comparison], ddof=1)) if complete else None
                   for name in ("macro_f1_delta", "CHID_f1_delta", "balanced_accuracy_delta")},
                "unit": "two policies from the same full-forty fit and same frozen OOF structures; no independent endpoint samples"}
            paired_rows.extend(rows_for_comparison)
    # Input and architecture contrasts use the same complete344 OOF unit
    # under the same policy and seed; never averages of nonlinear fold F1s.
    contrasts = []
    for policy in ("final40", "inner_ce"):
        for model in manifest["config"]["backbones"]:
            for ablation in ("I0", "G0"):
                contrasts.append((f"{policy}/{model}/I3-{ablation}", policy, "I3", model, policy, ablation, model))
        for condition in ("I3", "I0", "G0"):
            contrasts.append((f"{policy}/{condition}/cnn_transformer-m2plus", policy, condition, "cnn_transformer", policy, condition, "m2plus"))
    for name, first_policy, first_condition, first_model, second_policy, second_condition, second_model in contrasts:
        delta_rows = []
        for seed in (42, 43, 44):
            first = complete_oof.get((first_policy, first_condition, first_model, seed))
            second = complete_oof.get((second_policy, second_condition, second_model, seed))
            if first is None or second is None:
                continue
            a, b = first["metrics"], second["metrics"]
            delta_rows.append({"comparison": name, "policy": first_policy, "first_condition": first_condition,
                "second_condition": second_condition, "first_model": first_model, "second_model": second_model, "seed": seed,
                "macro_f1_delta": a["macro_f1"] - b["macro_f1"],
                "CHID_f1_delta": a["per_class"]["CHID"]["f1"] - b["per_class"]["CHID"]["f1"],
                "balanced_accuracy_delta": a["balanced_accuracy"] - b["balanced_accuracy"]})
        complete = len(delta_rows) == 3
        paired_summaries[name] = {"status": "complete" if complete else "pending", "complete_seed_pairs": len(delta_rows),
            "expected_seed_pairs": 3, "deltas": delta_rows,
            **{f"{measure}_mean": float(np.mean([r[measure] for r in delta_rows])) if complete else None
               for measure in ("macro_f1_delta", "CHID_f1_delta", "balanced_accuracy_delta")},
            **{f"{measure}_seed_sample_std": float(np.std([r[measure] for r in delta_rows], ddof=1)) if complete else None
               for measure in ("macro_f1_delta", "CHID_f1_delta", "balanced_accuracy_delta")},
            "unit": "Paired same-seed complete344 OOF under identical checkpoint policy; developmental differences, no population CI."}
        paired_rows.extend(delta_rows)
    fit_counts = {status: sum(record["status"] == status for record in records.values())
                  for status in ("completed", "pending", "numerical_failure", "execution_failure", "invalid")}
    endpoint_counts = {status: sum(row["status"] == status for row in endpoint_rows)
                       for status in ("completed", "pending", "numerical_failure", "execution_failure", "invalid")}
    main_keys = {(job["condition"], job["model"], job["seed"], job["fold"])
                 for job in planned_jobs(manifest["config"], "main")}
    main_records = [record for key, record in records.items() if key in main_keys]
    main_endpoints = [row for row in endpoint_rows if (row["condition"], row["model"], row["seed"], row["fold"]) in main_keys]
    main_fit_counts = {status: sum(record["status"] == status for record in main_records) for status in fit_counts}
    main_endpoint_counts = {status: sum(row["status"] == status for row in main_endpoints) for status in endpoint_counts}
    main_oof_count = sum(condition == "I3" and (model in manifest["config"]["backbones"] or model == "pca_lr")
                         for policy, condition, model, seed in complete_oof)
    main_complete = main_fit_counts["completed"] == 50 and main_endpoint_counts["completed"] == 95 and main_oof_count == 19
    main_status = "invalid" if main_fit_counts["invalid"] else "failed" if main_fit_counts["numerical_failure"] or main_fit_counts["execution_failure"] else "complete" if main_complete else "pending"
    summary = {"status": "invalid" if errors else "complete" if fit_counts["completed"] == len(jobs) else "pending",
        **{key: manifest[key] for key in ("protocol", "experiment_signature", "source_signature", "dataset_signature", "selection_signature")},
        "expected_fits": len(jobs), "completed_fits": fit_counts["completed"], "pending_fits": fit_counts["pending"],
        "numerical_failure_fits": fit_counts["numerical_failure"], "execution_failure_fits": fit_counts["execution_failure"], "invalid_fits": fit_counts["invalid"],
        "expected_endpoints": sum(len(endpoint_policies(job)) for job in jobs), "completed_endpoints": endpoint_counts["completed"],
        "endpoint_status_counts": endpoint_counts, "expected_complete_oof": 58, "complete_oof": len(complete_oof),
        "n_structures": len(identity), "class_support": {name: sum(r["type"] == name for r in identity.values()) for name in CLASS_NAMES},
        "comparison": comparison, "models": model_details, "shared_full40_learning": shared_learning,
        "paired_policy_comparisons": paired_summaries,
        "fit_jobs": fit_rows, "endpoint_jobs": endpoint_rows, "errors": errors,
        "total_fit_runtime_seconds": sum(float(r["metrics"].get("runtime_seconds", 0)) for r in records.values() if r["status"] == "completed"),
        "main_status": main_status, "main_expected_fits": 50, "main_completed_fits": main_fit_counts["completed"],
        "main_fit_status_counts": main_fit_counts, "main_expected_endpoints": 95,
        "main_completed_endpoints": main_endpoint_counts["completed"], "main_endpoint_status_counts": main_endpoint_counts,
        "main_expected_complete_oof": 19, "main_complete_oof": main_oof_count,
        "interpretation": "Developmental spatial CV on inspected data. Full344 OOF per seed, then mean/sample SD; seed SD is optimization variation, not biological CI.",
        "constant_classifier_policy": "Every completed endpoint stays in its preregistered score, including one-class predictions and failed learning acceptance; no fallback or outer selection.",
        "shared_training_unit": "Each neural fit executes all40 once; its two fixed policies are paired outcomes, not independent training runs; no refit."}
    _write_derived_json(run_dir / "summary.json", summary)
    _csv(run_dir / "comparison.csv", comparison)
    _csv(run_dir / "fit_review.csv", fit_rows)
    _csv(run_dir / "endpoint_review.csv", endpoint_rows)
    _csv(run_dir / "paired_policy_deltas.csv", paired_rows)
    _csv(run_dir / "prediction_errors.csv", error_rows)
    return summary
