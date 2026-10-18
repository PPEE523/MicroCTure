"""Frozen R1 arithmetic with one global dropout change; evidence is immutable."""
from __future__ import annotations
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from microcture.task1.data import load_dataset
from microcture.task1.evaluation import CLASS_NAMES, _read_prediction_csv, metrics, save_prediction_csv
from microcture.task1.training import resolve_device, run_lock, score_samples, write_json
from microcture.task1_controlled.diagnostics import learning_acceptance
from microcture.task1_formal.data import apply_condition as formal_condition
from microcture.task1_formal.evaluation import (_guard_run, _identity, _fold_ids, _history, _same_score,
    _neural_summary, _seed_agreement, _csv, _write_derived_json)
from microcture.task1_formal.protocol import _package_signature, json_signature, project_path, sha256
from microcture.task1_formal.runner import assert_resume, frozen_fold, outer_neural_endpoint, write_artifact_index, guard_output as formal_guard
from microcture.task1_readout.protocol import prepare_manifest as readout_manifest
from microcture.task1_readout.runner import fit_diagnostic, restore_endpoint
from microcture.task1_readout.models import model_spec
from microcture.task1_reevaluation.runner import publish_fold
from microcture.task1_roi.data import prepare_roi_inner, prepare_roi_outer
from microcture.task1_roi import experiment as reference_roi
from microcture.task1_controlled.runner import state_fingerprint
from microcture.task1_readout.models import build_model as build_readout_model
from torch.utils.data import DataLoader, TensorDataset
import hashlib

PROTOCOL = "task1_roi_global_dropout_control_v1"
REFERENCE_SOURCE = "a1ed97051ee9d24ba0198bed07fc2884ed56d3ca629f3ed40744f8d2eede6c94"
REFERENCE_EXPERIMENT = "46ff1c0565255fc8ef31070ae9649a3b594460e40ac7054b6d5b3ccf62c21f4e"
ALIGNMENTS, CONDITIONS, SEEDS, FOLDS = ("R1",) , ("I3", "G0", "V0", "P0"), (42, 43, 44), tuple(range(5))
FROZEN_SOURCES = {"task1_formal": "2de4e177d4557773bf61183c090a1a67db002325cb999a7d24d3cfa410013a8c",
    "task1_readout": "414d1c39d477bf4f120ad66ddc66e44ed68a1f00c8d81b9c3d495aa5a9cf6a0a",
    "task1_reevaluation": "59f6c0d88638effabaee5ba3929378d8d00e2b17cd0fd9b0dfb17d8bd5d7c787"}
STAGING_PREFIX, STAGING_OWNER = "microcture-dropout-fit-", "microcture.task1_dropout_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def reference_manifest(root, config):
    path = project_path(root, config["reference_roi_run_dir"]) / "manifest.json"
    require(config["reference_roi_run_dir"] == "results/task1_roi_v1", "Use the frozen R1 dropout0.2 reference")
    value = json.loads(path.read_text())
    require(value["source_signature"] == REFERENCE_SOURCE and value["experiment_signature"] == REFERENCE_EXPERIMENT
            and json_signature({k:v for k,v in value.items() if k != "experiment_signature"}) == REFERENCE_EXPERIMENT,
            "Frozen R1 manifest changed")
    return value


def validate_protocol(config):
    require(config.get("protocol") == PROTOCOL, "Unknown dropout protocol")
    reference = reference_manifest(Path(__file__).resolve().parents[2], config)
    expected = copy.deepcopy(reference["config"])
    expected.update(protocol=PROTOCOL, run_dir=config["run_dir"], alignments=["R1"], reference_roi_run_dir="results/task1_roi_v1")
    expected["models"]["dropout"] = expected["training"]["dropout"] = .5
    require(config == expected, "Only global dropout0.2->0.5 is preregistered; preserve all other R1 settings")
    require(isinstance(config["run_dir"],str) and bool(config["run_dir"].strip()), "Invalid output directory")
    return config


def planned_jobs(config: dict, alignment="all", condition=None, fold=None, seed=None) -> list[dict]:
    validate_protocol(config)
    require(alignment == "all" or alignment in ALIGNMENTS, "Choose all/R1")
    require(condition is None or condition in CONDITIONS, "Unknown condition")
    require(fold is None or type(fold) is int and fold in FOLDS, "Unknown fold")
    require(seed is None or type(seed) is int and seed in SEEDS, "Unknown seed")
    return [{"stage": "dropout", "dropout_variant": "D5", "kind": "neural", "model": "m2plus", "backbone": "m2plus", "variant": "O0",
        "alignment": a, "condition": c, "input_variant": c, "name": f"D5_{c}", "head": "H3", "fold": f,
        "seed": s, "learning_rate": config["training"]["learning_rate"]}
        for a in ALIGNMENTS for c in CONDITIONS for s in SEEDS for f in FOLDS
        if (alignment == "all" or a == alignment) and (condition is None or c == condition)
        and (fold is None or f == fold) and (seed is None or s == seed)]


def job_directory(run_dir: Path, job: dict) -> Path:
    require(job.get("alignment") in ALIGNMENTS and job.get("condition") in CONDITIONS and
            (job["alignment"] != "R0" or job["condition"] == "I3") and job.get("model") == "m2plus" and
            type(job.get("seed")) is int and job["seed"] in SEEDS and type(job.get("fold")) is int and job["fold"] in FOLDS,
            "Unsafe/unregistered ROI job")
    return Path(run_dir) / "runs" / job["alignment"] / job["condition"] / "m2plus" / f"seed_{job['seed']}" / f"fold_{job['fold']}"


def guard_output(root: Path, run_dir: Path) -> Path:
    result = formal_guard(root, run_dir)
    for name in ("task1_formal_v1", "task1_readout_v1", "task1_reevaluation_v1", "task1_morphology_probe_v1", "task1_roi_v1", "task1_native_roi_v1", "task1_native_roi_v2"):
        old = (root / "results" / name).resolve()
        require(result != old and old not in result.parents and result not in old.parents, "Cannot overwrite prior evidence")
    if (result / "manifest.json").exists():
        require(json.loads((result / "manifest.json").read_text()).get("protocol") == PROTOCOL, "Cannot reuse another protocol")
    return result


def _reference_job(config, job):
    return reference_roi.planned_jobs(config, alignment="R1", condition=job["condition"], fold=job["fold"], seed=job["seed"])[0]


def assert_paired_fit(config, job, result, preprocessing=None):
    reference = reference_manifest(Path(__file__).resolve().parents[2], config)
    old_job = _reference_job(reference["config"], job)
    directory = reference_roi.job_directory(Path(__file__).resolve().parents[2] / config["reference_roi_run_dir"], old_job)
    old = assert_resume(directory, old_job, reference)
    require(old is not None and old["status"] == "completed", "Reference paired D2 fit must be sealed and completed")
    for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights"):
        require(result[field] == old[field], f"D5/D2 paired {field} differs")
    if preprocessing is not None:
        require(preprocessing == json.loads((directory / "protocol.json").read_text())["preprocessing"], "D5 changed frozen R1 input protocol")
    return {"reference_job":old_job,"reference_directory":str(directory),
            "reference_ARTIFACTS_sha256":sha256(directory/"ARTIFACTS.json"), "initialization_minibatch_class_weights_preprocessing_exact":True}


def dropout_modules(model):
    return {name: module.p for name,module in model.named_modules() if isinstance(module,torch.nn.modules.dropout._DropoutNd)}


def assert_reference_preprocessing(root, config, fold_number, preprocessing):
    path = root / config["reference_roi_run_dir"] / "folds" / "R1" / f"fold_{fold_number}.json"
    require(json.loads(path.read_text())["preprocessing"] == preprocessing,
            "D5 preprocessing differs from frozen D2 before training")


def prepare_manifest(root, config, requested_device="auto"):
    root=root.resolve();validate_protocol(config);guard_output(root,project_path(root,config["run_dir"]))
    reference=reference_manifest(root,config)
    require(_package_signature(root,"task1_roi","run_task1_roi.py") == REFERENCE_SOURCE, "Frozen R1 source changed")
    # Old validators receive exactly their saved 0.2 configuration. Never pass
    # the new 0.5 config into a frozen family validator.
    current,metadata=reference_roi.prepare_manifest(root,reference["config"],reference["requested_device"])
    require(current == reference, "Reference R1 source/data/environment changed")
    indices={};initial={};reference_dir=project_path(root,config["reference_roi_run_dir"])
    reference_jobs=reference_roi.planned_jobs(reference["config"],alignment="R1")
    require(len(reference_jobs) == 60, "Exactly60 complete D2 reference fits required")
    for old_job in reference_jobs:
        directory=reference_roi.job_directory(reference_dir,old_job)
        result=assert_resume(directory,old_job,reference)
        require(result is not None and result["status"] == "completed", "Missing/failed D2 reference cannot be replaced")
        indices[str(directory.relative_to(reference_dir))]=sha256(directory/"ARTIFACTS.json")
        key=str(old_job["seed"])
        fingerprints=(result["initial_state_sha256"], result["common_initial_state_sha256"])
        require(key not in initial or initial[key] == fingerprints, "D2 common initialization differs across folds/conditions")
        initial[key]=fingerprints
    for seed in SEEDS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=build_readout_model("O0",config["models"])
            require((state_fingerprint(model),state_fingerprint(model,common_only=True)) == initial[str(seed)], "D5 initialization differs from D2")
            modules=dropout_modules(model)
            require(len(modules)==2 and set(modules.values())=={.5}, "Every actual M2+ Dropout must equal0.5")
            require(sum(p.numel() for p in model.parameters()) == 110437, "Dropout changed parameter count")
    value={"protocol":PROTOCOL,"config":copy.deepcopy(config),"source_signature":_package_signature(root,"task1_dropout_v1","run_task1_dropout.py"),
        "dependency_source_signature":{**reference["dependency_source_signature"],"task1_roi":REFERENCE_SOURCE},
        **{k:reference[k] for k in ("dataset_signature","fold_ids","fold_signatures","label_order","environment_record","failed_evidence_sha256","controlled_evidence_sha256")},
        "requested_device":requested_device,"n_structures":len(metadata["structures"]),"job_count":60,"planned_jobs":planned_jobs(config),
        "single_factor":"global Dropout p0.2->0.5 at all two existing FFN/head sites; R1 input and all other training settings unchanged",
        "selection_protocol":"Fixed40 final; no refit/earlystop/checkpoint search; fixed before outer",
        "evaluation_scope":"Previously inspected spatial groups; prospective dropout hypothesis development only, not independent biology certification",
        "D2_reference":{"run_dir":config["reference_roi_run_dir"],"manifest_sha256":sha256(reference_dir/"manifest.json"),
            "source_signature":REFERENCE_SOURCE,"experiment_signature":REFERENCE_EXPERIMENT,"fits":60,"newly_trained":False,
            "artifact_index_sha256":indices,"initial_state_by_seed":{k:list(v) for k,v in initial.items()},
            "summary_sha256":sha256(reference_dir/"summary.json")},
        "actual_dropout_sites":modules}
    value["experiment_signature"]=json_signature(value)
    return value,metadata


def apply_condition(samples: dict, condition: str) -> dict:
    require(condition in CONDITIONS, "Unknown ROI condition")
    if condition in ("I3", "G0"):
        return formal_condition(samples, condition)
    out = {**samples, "x": samples["x"].copy()}
    if condition == "V0":
        out["x"][:, :, 0] = 0
    else:
        offset = np.abs(np.arange(64)[:, None] - np.arange(64)[None, :])
        for d in range(64):
            mask = offset == d
            signal, valid = samples["x"][:, :, 0, mask], samples["x"][:, :, 1, mask]
            amount = valid.sum(axis=-1)
            mean = np.divide((signal * valid).sum(axis=-1), amount, out=np.zeros_like(amount), where=amount > 0)
            out["x"][:, :, 0, mask] = np.where(valid > 0, mean[..., None], 0)
    return out


def staging_identity(job):
    return (job["alignment"], job["condition"], job["seed"], job["fold"])


def staging_context(output: Path, root: Path, run_dir: Path, job: dict):
    write_json(output / "STAGING_CONTEXT.json", {"owner": STAGING_OWNER, "project_root": str(root.resolve()),
        "run_dir": str(run_dir.resolve()), "job": job, "state": "unpublished scientific staging; never automatically retry"})


def unpublished_staging(root: Path, run_dir: Path) -> dict:
    found = {}
    for p in Path("/tmp").glob(STAGING_PREFIX + "*"):
        if p.is_symlink() or not p.is_dir() or (p / "STAGING_CONTEXT.json").is_symlink(): continue
        try:
            value = json.loads((p / "STAGING_CONTEXT.json").read_text())
            if value.get("owner") == STAGING_OWNER and value.get("project_root") == str(root.resolve()) and value.get("run_dir") == str(run_dir.resolve()):
                job_directory(run_dir, value["job"]); found[staging_identity(value["job"])] = p
        except (OSError, ValueError, KeyError, TypeError): pass
    return found


def failed_staging(run_dir, job):
    return run_dir / "failed_staging" / job_directory(run_dir, job).relative_to(run_dir / "runs")


def retain_staging(output, run_dir, job, error, root):
    destination = failed_staging(run_dir, job)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "Prior failed staging must be retained")
        shutil.move(str(output), str(destination)); return destination
    except (OSError, KeyboardInterrupt, ValueError):
        try:
            marker = output / "STAGING_CONTEXT.json"
            value = json.loads(marker.read_text()) if marker.exists() else {}
            if value.get("owner") != STAGING_OWNER or value.get("job") != job:
                staging_context(output, root, run_dir, job)
        except (OSError, KeyboardInterrupt, ValueError, TypeError) as marker_error:
            print(f"Double storage fault prevents a valid retry-blocking marker: {marker_error}; manually retain this exact staging path", file=sys.stderr, flush=True)
        print(f"Cannot publish failure; scientific files remain at {output}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return output


def fit_one(job, train, validation, preprocessing, dataset, fold, config, manifest, output, device):
    write_json(output / "protocol.json", {"job": job, "training": config["training"], "acceptance": config["acceptance"],
        "preprocessing": preprocessing, "selection": manifest["selection_protocol"], "evaluation_scope": manifest["evaluation_scope"]})
    started, result, model, outer_used, stage = time.perf_counter(), {}, None, False, "inner_fit"
    facts = {"train": train.get("roi_resampling", []), "validation": validation.get("roi_resampling", []), "outer": None}
    write_json(output / "roi_resampling.json", facts)
    try:
        result = fit_diagnostic(job, apply_condition(train, job["condition"]), apply_condition(validation, job["condition"]),
            config, config["models"], device, output)
        pairing = assert_paired_fit(config, job, result, preprocessing)
        write_json(output / "D2_pairing.json", pairing)
        checkpoint = output / "final.pt"; endpoint_hash = sha256(checkpoint)
        require(result["status"] == "completed", "Inner fit incomplete")
        write_json(output / "endpoint.json", {"job": job, "checkpoint": "final.pt", "checkpoint_sha256": endpoint_hash,
            "outer_images_constructed": False, "training_and_selection_complete": True})
        stage = "outer_evaluation"
        outer = prepare_roi_outer(dataset, fold, preprocessing, job["alignment"]); outer_used = True
        facts["outer"] = outer.get("roi_resampling", []); write_json(output / "roi_resampling.json", facts)
        model = restore_endpoint(job, config, checkpoint, device); outer = apply_condition(outer, job["condition"])
        probabilities, diagnostic = outer_neural_endpoint(model, outer, config, device)
        require(sha256(checkpoint) == endpoint_hash, "Fixed checkpoint changed during outer evaluation")
        save_prediction_csv(output / "outer_predictions.csv", outer, probabilities)
        result.update(outer_diagnostics=diagnostic, outer_metrics=score_samples(outer, probabilities), checkpoint_sha256=endpoint_hash,
            outer_images_used=True, outer_fit_or_selection=False, outer_constructed_after_fixed_endpoint=True)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="execution_failure" if isinstance(error, (OSError, KeyboardInterrupt, ValueError)) else "numerical_failure",
            error=str(error) or type(error).__name__, error_type=type(error).__name__, failure_stage=stage,
            run_stop_requested=True, outer_images_used=outer_used, outer_fit_or_selection=False)
        write_json(output / "failure.json", {**result, "job": job})
    finally:
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    result.update(job=job, **{k: manifest[k] for k in ("experiment_signature", "source_signature", "dataset_signature")},
        preprocessing_signature=preprocessing["signature"], runtime_seconds=time.perf_counter()-started,
        **{f"{s}_structures": len(fold[f"{s}_ids"]) for s in ("train", "validation", "test")})
    write_json(output / "metrics.json", result); write_artifact_index(output, job, manifest)
    return result


def run_jobs(config, root, run_dir, requested_device="auto", **filters):
    root = root.resolve(); run_dir = guard_output(root, run_dir)
    config = {**config, "run_dir": str(run_dir.relative_to(root))}; jobs = planned_jobs(config, **filters)
    require(bool(jobs), "No registered jobs match filters")
    manifest, metadata = prepare_manifest(root, config, requested_device); device = resolve_device(requested_device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); torch.set_num_threads(config["training"]["cpu_threads"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir), threadpool_limits(limits=config["training"]["cpu_threads"]):
        mp = run_dir / "manifest.json"
        if mp.exists(): require(json.loads(mp.read_text()) == manifest, "Source/config/environment changed; use a new run directory")
        else: write_json(mp, manifest)
        pending, tmp = [], unpublished_staging(root, run_dir)
        for job in jobs:
            require(staging_identity(job) not in tmp and not failed_staging(run_dir, job).exists(), "Retained staging forbids automatic retry")
            old = assert_resume(job_directory(run_dir, job), job, manifest)
            if old is None: pending.append(job)
            else: print(f"Skip saved {old['status']}: {job['name']} fold{job['fold']} seed{job['seed']}", flush=True)
        if pending:
            dataset = load_dataset(project_path(root, config["cache_dir"]))
            require(dataset["signature"] == manifest["dataset_signature"], "Loaded cache changed")
            by_fold = defaultdict(list)
            for job in pending: by_fold[(job["fold"], job["alignment"])].append(job)
            for (f, a), items in sorted(by_fold.items()):
                frozen = frozen_fold(root, config, manifest, f); train, validation, pp = prepare_roi_inner(dataset, frozen, a)
                assert_reference_preprocessing(root, config, f, pp)
                saved = {"frozen_fold": frozen, "preprocessing": pp}; saved["signature"] = json_signature(saved)
                fp = run_dir / "folds" / a / f"fold_{f}.json"
                if fp.exists(): require(json.loads(fp.read_text()) == saved, "Saved preprocessing changed")
                else: write_json(fp, saved)
                for job in items:
                    destination = job_directory(run_dir, job); destination.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Start {job['name']} fold={f} seed={job['seed']}", flush=True)
                    output = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir="/tmp")); published = False
                    try:
                        staging_context(output, root, run_dir, job)
                        result = fit_one(job, train, validation, pp, dataset, frozen, config, manifest, output, device)
                        publish_fold(output, destination); published = True
                        if result["status"] != "completed": raise RuntimeError(f"Failed scientific job retained at {destination}; no retry")
                    except BaseException as error:
                        retained = destination if published else retain_staging(output, run_dir, job, error, root)
                        raise RuntimeError(f"Run stopped after {type(error).__name__}; evidence retained at {retained}") from error
                    finally:
                        if published and output.exists(): shutil.rmtree(output)
                del train, validation
        return summarize(run_dir, root, metadata)


def check_protocol(root, config, run_dir, requested_device="auto"):
    root = root.resolve(); run_dir = guard_output(root, run_dir); config = {**config, "run_dir": str(run_dir.relative_to(root))}
    manifest, metadata = prepare_manifest(root, config, requested_device)
    return {"status": "checked", "training_launched": False, "real_images_preprocessed": False, "n_structures": len(metadata["structures"]),
        "job_count": len(planned_jobs(config)), "D2_reference_fits":60, "all_dropout_modules":manifest["actual_dropout_sites"], "run_dir": str(run_dir), "model": model_spec("O0", config["models"]),
        **{k: manifest[k] for k in ("source_signature", "experiment_signature", "dependency_source_signature", "evaluation_scope")}}


def check_replicate_rows(rows):
    for row in rows:
        records = json.loads(row["replicate_probabilities_json"])
        require(int(row["replicate_count"]) == len(records) == 2 and {str(r["replicate"]) for r in records} == {"1", "2"}, "Two WT identities differ")
        p = np.asarray([r["probabilities"] for r in records], dtype=float)
        require(p.shape == (2, 3) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) and
                np.allclose(p.sum(1), 1, atol=1e-6, rtol=0), "Invalid WT probabilities")
        predicted = p.argmax(1)
        require([r["predicted"] for r in records] == predicted.tolist() and int(row["replicate_disagree"]) == int(predicted[0] != predicted[1]), "WT predictions/disagreement differ")
        require(np.allclose(p.mean(0), [float(row[f"p_{c}"]) for c in CLASS_NAMES], atol=1e-10, rtol=0), "WT mean differs from aggregate")


def _load_saved(run_dir, job, manifest, identity, folds):
    directory = job_directory(run_dir, job); value = assert_resume(directory, job, manifest)
    record = {"job": job, "status": "pending", "directory": str(directory)}
    if value is None: return record
    record.update(status=value["status"], metrics=value)
    try: record["history"] = _history(directory / "history.json")
    except (ValueError, OSError):
        if value["status"] == "completed": raise
        record["history"] = []
    if value["status"] != "completed": return record
    history = record["history"]; require(len(history) == 41 and value["epochs"] == 40, "Missing full40 trajectory")
    require(value["acceptance"] == learning_acceptance(history, manifest["config"]), "Learning acceptance differs")
    for view in ("train", "validation"):
        require(value[f"{view}_diagnostics"] == history[-1][f"{view}_diagnostics"], "Terminal diagnostics differ")
    pp = json.loads((directory / "protocol.json").read_text())["preprocessing"]
    pairing = assert_paired_fit(manifest["config"], job, value, pp)
    require(json.loads((directory / "D2_pairing.json").read_text()) == pairing, "Saved D2 pairing evidence differs")
    require(pp["alignment"] == job["alignment"] and pp["signature"] == json_signature({k: v for k, v in pp.items() if k != "signature"}), "Preprocessing content differs")
    saved_fold = json.loads((run_dir / "folds" / job["alignment"] / f"fold_{job['fold']}.json").read_text())
    require(saved_fold["signature"] == json_signature({k:v for k,v in saved_fold.items() if k != "signature"}) and
            saved_fold["preprocessing"] == pp and json_signature(saved_fold["frozen_fold"]) == manifest["fold_signatures"][str(job["fold"])], "Shared fold preprocessing differs")
    endpoint = json.loads((directory / "endpoint.json").read_text())
    require(endpoint["job"] == job and endpoint["checkpoint"] == "final.pt" and endpoint["checkpoint_sha256"] == value["checkpoint_sha256"] and
        endpoint["training_and_selection_complete"] and endpoint["outer_images_constructed"] is False and
        value["outer_constructed_after_fixed_endpoint"] and value["outer_fit_or_selection"] is False, "Outer boundary differs")
    predictions = _read_prediction_csv(directory / "outer_predictions.csv")
    check_replicate_rows(predictions["rows"])
    require(set(predictions["structure_id"]) == set(folds[job["fold"]]["test_ids"]), "Outer IDs differ")
    for sid, label, group in zip(predictions["structure_id"], predictions["y"], predictions["group_id"]):
        require((int(label), int(group)) == (identity[str(sid)]["y"], identity[str(sid)]["group_id"]), "Outer identity differs")
    _same_score(value["outer_metrics"], predictions["metrics"], "outer_metrics")
    _same_score(value["outer_diagnostics"]["metrics"], predictions["metrics"], "outer_diagnostics")
    validation = _read_prediction_csv(directory / "validation_predictions.csv"); check_replicate_rows(validation["rows"])
    require(set(validation["structure_id"]) == set(folds[job["fold"]]["validation_ids"]), "Validation IDs differ")
    _same_score(value["validation_diagnostics"]["metrics"], validation["metrics"], "validation_metrics")
    record["predictions"] = predictions
    return record


def summarize(run_dir: Path, root: Path, metadata=None):
    run_dir = guard_output(root, run_dir); _guard_run(run_dir); require(not (run_dir / "FROZEN.json").exists(), "Frozen run is read-only")
    manifest = json.loads((run_dir / "manifest.json").read_text()); config = manifest["config"]
    current, current_metadata = prepare_manifest(root, config, manifest["requested_device"])
    require(current == manifest, "Saved source/config/environment differs; do not overwrite derived summaries")
    if metadata is None: metadata = current_metadata
    identity = _identity(metadata); folds = _fold_ids(manifest, identity, run_dir)
    records, errors, tmp = [], [], unpublished_staging(root, run_dir)
    for job in planned_jobs(config):
        try:
            require(staging_identity(job) not in tmp and not failed_staging(run_dir, job).exists(), "Unpublished staging retained")
            record = _load_saved(run_dir, job, manifest, identity, folds)
        except (ValueError, KeyError, TypeError, OSError) as error:
            record = {"job": job, "status": "invalid", "reason": str(error)}; errors.append({"job": job, "reason": str(error)})
        records.append(record)
    paired = defaultdict(list)
    for record in records:
        if record["status"] == "completed": paired[(record["job"]["fold"], record["job"]["seed"])].append(record)
    for key, items in paired.items():
        for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights"):
            if len({json.dumps(r["metrics"][field], sort_keys=True) for r in items}) != 1:
                errors.append({"seed_fold": key, "reason": f"Paired {field} differs"})
                for r in items: r["status"] = "invalid"
    comparisons, details, complete_oof, cases, review = [], {}, {}, [], []
    grouped = defaultdict(list)
    for r in records:
        j = r["job"]; grouped[(j["alignment"], j["condition"])].append(r); v=r.get("metrics", {})
        review.append({**j, "status": r["status"], "reason": r.get("reason"), "runtime_seconds": v.get("runtime_seconds"),
            "learning_pass": v.get("acceptance", {}).get("accepted"), "last_train_ce": v.get("final_train_ce"),
            "outer_prediction_counts": v.get("outer_diagnostics", {}).get("prediction_counts")})
    for (alignment, condition), items in grouped.items():
        oof, seed_details = {}, {}
        for seed in SEEDS:
            sr = [r for r in items if r["job"]["seed"] == seed]; dest = run_dir / "oof" / alignment / condition / f"seed_{seed}"
            status = "complete" if all(r["status"] == "completed" for r in sr) else "incomplete"
            seed_details[str(seed)] = {"status": status, "fit_status_counts": dict(Counter(r["status"] for r in sr))}
            _write_derived_json(dest / "status.json", {**seed_details[str(seed)], "oof_current": status == "complete",
                "source_signature": manifest["source_signature"], "experiment_signature": manifest["experiment_signature"]})
            if status != "complete": continue
            rows = sorted([{**row, "fold": r["job"]["fold"]} for r in sr for row in r["predictions"]["rows"]], key=lambda x: x["structure_id"])
            require(len(rows) == len(identity) and {r["structure_id"] for r in rows} == set(identity), "OOF incomplete/repeated")
            y=np.asarray([int(r["y_true"]) for r in rows]); probabilities=np.asarray([[float(r[f"p_{c}"]) for c in CLASS_NAMES] for r in rows])
            score=metrics(y, probabilities); oof[seed]={"metrics": score, "rows": rows, "by_id": {r["structure_id"]: r for r in rows}}
            complete_oof[(alignment,condition,seed)]=oof[seed]; seed_details[str(seed)]["metrics"]=score
            seed_details[str(seed)]["predicted_counts"] = dict(zip(CLASS_NAMES,np.bincount(probabilities.argmax(1),minlength=3).tolist()))
            seed_details[str(seed)]["development_gate_pass"] = score["macro_f1"] >= .9 and all(score["per_class"][c][m]>=.85 for c in CLASS_NAMES for m in ("precision","recall"))
            _write_derived_json(dest / "metrics.json", {**score, "experiment_signature":manifest["experiment_signature"], "oof_current":True}); _csv(dest / "predictions.csv",rows)
        complete=len(oof)==3; scores=[oof[s]["metrics"] for s in SEEDS] if complete else []
        row={"alignment":alignment,"condition":condition,"status":"complete" if complete else "pending","complete_seeds":len(oof)}
        for name,values in [(m,[s[m] for s in scores]) for m in ("macro_f1","accuracy","balanced_accuracy")]+[(f"{c}_{m}",[s["per_class"][c][m] for s in scores]) for c in CLASS_NAMES for m in ("precision","recall","f1")]:
            row[f"{name}_mean"]=float(np.mean(values)) if values else None;row[f"{name}_seed_sample_std"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        learning=_neural_summary(items,config); agreement=_seed_agreement(list(SEEDS),oof,identity)
        stable=[]
        if complete:
            for sid in sorted(identity):
                preds=[int(oof[s]["by_id"][sid]["y_pred"]) for s in SEEDS]; truth=identity[sid]["y"];wrong=[p!=truth for p in preds]
                cases.append({"alignment":alignment,"condition":condition,"structure_id":sid,"type":CLASS_NAMES[truth],"group_id":identity[sid]["group_id"],"length_bp":identity[sid]["length_bp"],
                    "stable_wrong":all(wrong),"any_wrong":any(wrong),"stable_correct":not any(wrong),"seed_unanimous":len(set(preds))==1,
                    **{f"seed_{s}_prediction":CLASS_NAMES[p] for s,p in zip(SEEDS,preds)}})
                if all(wrong): stable.append(sid)
        row.update(strict_learning_passes=learning["strict_learning_passes"],stable_wrong_count=len(stable) if complete else None,
            development_gate_pass=complete and all(seed_details[str(s)]["development_gate_pass"] for s in SEEDS) and learning["strict_learning_passes"]==15)
        row["dropout_variant"]="D5"
        comparisons.append(row);details[f"{alignment}/{condition}"]={"seeds":seed_details,"learning":learning,"seed_agreement":agreement,"stable_wrong_ids":stable if complete else None}
    deltas=[]
    for aa,ac,ba,bc in [("R1","I3","R1",c) for c in ("G0","V0","P0")]:
        for s in SEEDS:
            a,b=complete_oof.get((aa,ac,s)),complete_oof.get((ba,bc,s))
            if a is not None and b is not None:
                deltas.append({"comparison":f"{aa}/{ac}-{ba}/{bc}","seed":s,"macro_f1_delta":a["metrics"]["macro_f1"]-b["metrics"]["macro_f1"],"CHID_f1_delta":a["metrics"]["per_class"]["CHID"]["f1"]-b["metrics"]["per_class"]["CHID"]["f1"]})
    counts=Counter(r["status"] for r in records)
    summary={"status":"invalid" if errors else "complete" if counts["completed"]==60 else "failed" if counts["execution_failure"]+counts["numerical_failure"] else "pending",
        "protocol":PROTOCOL,**{k:manifest[k] for k in ("experiment_signature","source_signature","dataset_signature","evaluation_scope")},
        "expected_jobs":60,"completed_jobs":counts["completed"],"pending_jobs":counts["pending"],"invalid_jobs":counts["invalid"],
        "execution_failure_jobs":counts["execution_failure"],"numerical_failure_jobs":counts["numerical_failure"],"complete_oof":len(complete_oof),"expected_oof":12,
        "comparison":comparisons,"models":details,"paired_deltas":deltas,"errors":errors,"development_gates":config["development_gates"],
        "score_unit":"344 unique annotated structures; mean replicate probability before scores; seed SD is not biological CI"}
    reference=reference_manifest(root,config)
    reference_summary_path=root/config["reference_roi_run_dir"]/"summary.json"
    require(sha256(reference_summary_path)==manifest["D2_reference"]["summary_sha256"],"D2 summary bytes changed")
    old_summary=json.loads(reference_summary_path.read_text())
    require(old_summary["status"] == "complete" and old_summary["experiment_signature"] == REFERENCE_EXPERIMENT, "D2 summary is incomplete/stale")
    reference_comparison=[{**row,"dropout_variant":"D2","newly_trained":False} for row in old_summary["comparison"] if row["alignment"]=="R1"]
    require(len(reference_comparison)==4,"D2 reference condition coverage differs")
    summary["D2_reference_comparison"]=reference_comparison
    summary["D2_reference_manifest"]=manifest["D2_reference"]
    policy_deltas=[]
    for condition in CONDITIONS:
        key=f"R1/{condition}"
        for seed in SEEDS:
            new=details.get(key,{}).get("seeds",{}).get(str(seed),{})
            old=old_summary["models"][key]["seeds"][str(seed)]
            if new.get("status") == "complete":
                a,b=new["metrics"],old["metrics"]
                policy_deltas.append({"condition":condition,"seed":seed,"comparison":"D5-D2",
                    "macro_f1_delta":a["macro_f1"]-b["macro_f1"],
                    "CHID_f1_delta":a["per_class"]["CHID"]["f1"]-b["per_class"]["CHID"]["f1"]})
    summary["D5_minus_D2_paired_seed_deltas"]=policy_deltas
    summary["reference_oof_not_duplicated"]=True
    _write_derived_json(run_dir / "summary.json",summary);_csv(run_dir / "comparison.csv",comparisons);_csv(run_dir / "job_review.csv",review)
    _csv(run_dir / "D5_minus_D2.csv",policy_deltas)
    _csv(run_dir / "paired_deltas.csv",deltas);_csv(run_dir / "case_predictions.csv",cases)
    return summary
