"""User-run formal CV; outer tensors exist only after a fixed inner endpoint."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import torch

from microcture.task1.data import load_dataset
from microcture.task1.evaluation import save_prediction_csv
from microcture.task1.training import publish_fold, resolve_device, run_lock, score_samples, write_json
from microcture.task1_controlled.diagnostics import diagnose_model
from microcture.task1_controlled.evidence import json_signature, sha256
from microcture.task1_controlled.models import build_controlled_model, model_spec
from microcture.task1_controlled.runner import fit_diagnostic
from .baselines import fit_baseline, predict_baseline
from .data import apply_condition, prepare_inner, prepare_outer
from .protocol import job_directory, planned_jobs, prepare_manifest


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def guard_output(root: Path, run_dir: Path) -> Path:
    """Never create output in a data/source/frozen tree or through a symlink."""
    root = root.resolve()
    path = project_path(root, run_dir)
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symbolic-link result paths are forbidden: {ancestor}")
    path = path.resolve()
    results = root / "results"
    if path == results or results not in path.parents:
        raise ValueError("Formal output must be a separate directory below the project results/")
    for marker in results.rglob("FROZEN.json"):
        protected = marker.parent
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError(f"Formal output overlaps frozen evidence: {protected}")
    if path.exists():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"Symbolic-link artifacts inside a formal run are forbidden: {child}")
    return path


def write_artifact_index(output: Path, job: dict, manifest: dict) -> None:
    records = [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(output.iterdir()) if path.is_file() and path.name != "ARTIFACTS.json"]
    write_json(output / "ARTIFACTS.json", {"job": job, "experiment_signature": manifest["experiment_signature"],
                                          "files": records, "signature": json_signature(records)})


def verify_artifact_index(directory: Path, job: dict, manifest: dict) -> None:
    index = json.loads((directory / "ARTIFACTS.json").read_text(encoding="utf-8"))
    if index["job"] != job or index["experiment_signature"] != manifest["experiment_signature"] or index["signature"] != json_signature(index["files"]):
        raise ValueError(f"Saved artifact index differs from the job/protocol: {directory}")
    names = set()
    for record in index["files"]:
        name = record["name"]
        if Path(name).name != name or name in names or name == "ARTIFACTS.json":
            raise ValueError(f"Invalid/duplicate indexed artifact: {directory}")
        names.add(name)
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Saved artifact changed or missing: {path}")
    if {path.name for path in directory.iterdir()} != names | {"ARTIFACTS.json"}:
        raise ValueError(f"Saved artifact inventory changed: {directory}")


def frozen_fold(root: Path, config: dict, manifest: dict, number: int) -> dict:
    path = project_path(root, config["frozen_run_dir"]) / "folds" / f"fold_{number}.json"
    fold = json.loads(path.read_text(encoding="utf-8"))
    if json_signature(fold) != manifest["fold_signatures"][str(number)]:
        raise ValueError(f"Frozen fold metadata changed: {path}")
    return fold


def restore_endpoint(job: dict, config: dict, checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if (saved["job"] != job or saved["epoch"] != config["training"]["epochs"]
            or saved["model_config"] != config["models"] or saved["head_version"] != "H3"):
        raise ValueError("Checkpoint is not the preregistered final endpoint")
    model = build_controlled_model(job["backbone"], config["models"], "H3").to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def outer_neural_endpoint(model, samples: dict, config: dict, device: torch.device) -> tuple[np.ndarray, dict]:
    """Reuse numerical probes in one final outer forward pass, never for selection.

    The controlled diagnostic's arithmetic is reused here in a new formal
    context, after checkpoint fixation. Capturing probabilities from the same
    forward calls avoids running a second outer prediction for the CSV.
    Outer diagnostics never enter the inner-only learning acceptance.
    """
    outputs = []

    def capture(module, inputs, logits):
        outputs.append(logits.detach().softmax(dim=1).cpu().numpy().copy())

    handle = model.register_forward_hook(capture)
    try:
        diagnostic = diagnose_model(model, samples, device, config["training"]["batch_size"],
                                    epsilon=config["acceptance"]["activation_epsilon"])
    finally:
        handle.remove()
    probabilities = np.concatenate(outputs)
    diagnostic["evaluation_scope"] = "outer test at the fixed final checkpoint; no fitting or selection"
    return probabilities, diagnostic


def fit_one(job: dict, train: dict, validation: dict, preprocessing: dict, dataset: dict,
            fold: dict, config: dict, manifest: dict, output: Path, device: torch.device) -> dict:
    """Fit inner-only, seal the endpoint, then and only then materialize outer."""
    write_json(output / "protocol.json", {
        "job": job, "training": config["training"], "baseline": config["baseline"],
        "acceptance": config["acceptance"], "preprocessing": preprocessing,
        "selection": "fixed final40 neural checkpoint / fixed baseline parameters; no outer selection, no refit",
        "evaluation_scope": "developmental spatial CV on already seen outer partitions; candidate-interval classification",
    })
    started = time.perf_counter()
    result = None
    model = None
    outer_constructed = False
    stage = "inner_fit"
    try:
        if job["kind"] == "neural":
            selected_train = apply_condition(train, job["condition"])
            selected_validation = apply_condition(validation, job["condition"])
            result = fit_diagnostic(job, selected_train, selected_validation, config, config["models"], device, output)
            checkpoint = output / "final.pt"
        else:
            model, result = fit_baseline(job["model"], train, validation, config, output)
            result = {**result, "job": job}
            checkpoint = output / "model.joblib"
        if result["status"] != "completed" or not checkpoint.is_file():
            raise ValueError("Inner fitting did not produce a completed fixed endpoint")
        # Establish and hash the endpoint before any outer input construction.
        endpoint_hash = sha256(checkpoint)
        write_json(output / "endpoint.json", {"job": job, "checkpoint": checkpoint.name,
                                               "checkpoint_sha256": endpoint_hash,
                                               "outer_images_constructed": False,
                                               "training_and_selection_complete": True})
        stage = "outer_evaluation"
        outer = prepare_outer(dataset, fold, preprocessing)
        outer_constructed = True
        if job["kind"] == "neural":
            model = restore_endpoint(job, config, checkpoint, device)
            outer = apply_condition(outer, job["condition"])
            probabilities, diagnostic = outer_neural_endpoint(model, outer, config, device)
            result["outer_diagnostics"] = diagnostic
        else:
            probabilities = predict_baseline(model, job["model"], outer)
        if sha256(checkpoint) != endpoint_hash:
            raise ValueError("Fixed endpoint changed during outer evaluation")
        save_prediction_csv(output / "outer_predictions.csv", outer, probabilities)
        result.update({"outer_metrics": score_samples(outer, probabilities),
                       "outer_images_used": True, "outer_fit_or_selection": False,
                       "outer_constructed_after_fixed_endpoint": True,
                       "checkpoint_sha256": endpoint_hash})
    except (RuntimeError, ValueError, ConvergenceWarning) as error:
        failure_status = "execution_failure" if isinstance(error, ValueError) else "numerical_failure"
        result = {**(result or {}), "status": failure_status, "job": job,
                  "error": str(error), "failure_stage": stage,
                  "outer_images_used": outer_constructed, "outer_fit_or_selection": False}
        initialization = output / "initialization.json"
        if initialization.exists():
            init = json.loads(initialization.read_text(encoding="utf-8"))
            result.update({key: init[key] for key in ("initial_state_sha256", "common_initial_state_sha256")})
        # Publish partial numerical traces as scientific evidence, never retry in place.
        write_json(output / "failure.json", result)
        print(f"Failure retained: {job['model']} {job['condition']} fold{job['fold']} seed{job['seed']}: {error}", flush=True)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result.update({"job": job, "experiment_signature": manifest["experiment_signature"],
                   "source_signature": manifest["source_signature"],
                   "dataset_signature": manifest["dataset_signature"],
                   "preprocessing_signature": preprocessing["signature"],
                   "runtime_seconds": time.perf_counter() - started,
                   "train_structures": len(fold["train_ids"]),
                   "validation_structures": len(fold["validation_ids"]),
                   "test_structures": len(fold["test_ids"])})
    write_json(output / "metrics.json", result)
    write_artifact_index(output, job, manifest)
    return result


def assert_resume(directory: Path, job: dict, manifest: dict) -> dict | None:
    if not directory.exists():
        return None
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"Existing unfinished publication is preserved; use a fresh run directory: {directory}")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("experiment_signature", "source_signature", "dataset_signature"):
        if result.get(key) != manifest[key]:
            raise ValueError(f"Existing job has changed source/data/config: {directory}")
    if result.get("job") != job or result.get("status") not in ("completed", "numerical_failure", "execution_failure"):
        raise ValueError(f"Existing job metadata is inconsistent: {directory}")
    verify_artifact_index(directory, job, manifest)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol["job"] != job or protocol["preprocessing"]["signature"] != result["preprocessing_signature"]:
        raise ValueError(f"Existing preprocessing protocol is inconsistent: {directory}")
    if result["status"] == "completed":
        endpoint = directory / ("final.pt" if job["kind"] == "neural" else "model.joblib")
        if not endpoint.is_file() or sha256(endpoint) != result["checkpoint_sha256"]:
            raise ValueError(f"Saved endpoint changed: {directory}")
        if not (directory / "outer_predictions.csv").is_file():
            raise ValueError(f"Missing completed outer prediction: {directory}")
        for name in ("validation_predictions.csv", "endpoint.json"):
            if not (directory / name).is_file():
                raise ValueError(f"Missing completed scientific evidence: {directory / name}")
        if job["kind"] == "neural":
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))["epochs"]
            if [row["epoch"] for row in history] != list(range(manifest["config"]["training"]["epochs"] + 1)):
                raise ValueError(f"Completed trajectory is not the full endpoint: {directory}")
            if history[-1]["weighted_train_ce"] != result["final_train_ce"]:
                raise ValueError(f"Completed loss differs from the trajectory: {directory}")
    return result


def run_jobs(config: dict, root: Path, run_dir: Path, requested_device: str = "auto", *,
             stage: str = "all", model: str = "all", condition: str | None = None,
             fold: int | None = None, seed: int | None = None) -> dict:
    """This function performs real fits and must be invoked by the user locally."""
    run_dir = guard_output(root, run_dir)
    config = {**config, "run_dir": str(run_dir.relative_to(root.resolve()))}
    jobs = planned_jobs(config, stage, model, condition, fold, seed)
    if not jobs:
        raise ValueError("No jobs match the requested stage/model/condition/fold/seed")
    manifest, metadata = prepare_manifest(root, config, requested_device)
    device = resolve_device(requested_device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["training"]["cpu_threads"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir), threadpool_limits(limits=config["training"]["cpu_threads"]):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Source/config/environment changed; preserve this run and use a new --run-dir")
        else:
            write_json(manifest_path, manifest)
        pending = []
        for job in jobs:
            existing = assert_resume(job_directory(run_dir, job), job, manifest)
            if existing is None:
                pending.append(job)
            else:
                print(f"Skip saved {existing['status']}: {job['condition']}/{job['model']}/fold{job['fold']}/seed{job['seed']}", flush=True)
        if pending:
            dataset = load_dataset(project_path(root, config["cache_dir"]))
            if dataset["signature"] != manifest["dataset_signature"]:
                raise ValueError("Loaded band cache differs from the frozen metadata")
            grouped = defaultdict(list)
            for job in pending:
                grouped[job["fold"]].append(job)
            for number, fold_jobs in sorted(grouped.items()):
                frozen = frozen_fold(root, config, manifest, number)
                train, validation, preprocessing = prepare_inner(dataset, frozen)
                saved_fold = {"frozen_fold": frozen, "preprocessing": preprocessing}
                saved_fold["signature"] = json_signature(saved_fold)
                fold_path = run_dir / "folds" / f"fold_{number}.json"
                if fold_path.exists() and json.loads(fold_path.read_text(encoding="utf-8")) != saved_fold:
                    raise ValueError(f"Shared preprocessing differs from a previous job: {fold_path}")
                write_json(fold_path, saved_fold)
                for job in fold_jobs:
                    destination = job_directory(run_dir, job)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Start {job['condition']}/{job['model']} fold={job['fold']} seed={job['seed']}", flush=True)
                    with tempfile.TemporaryDirectory(prefix="microcture-formal-fit-", dir="/tmp") as temporary:
                        output = Path(temporary)
                        fit_one(job, train, validation, preprocessing, dataset, frozen, config, manifest, output, device)
                        publish_fold(output, destination)
                del train, validation
            del dataset
        from .evaluation import summarize_formal
        return summarize_formal(run_dir, metadata)


def check_protocol(root: Path, config: dict, run_dir: Path, requested_device: str = "auto") -> dict:
    run_dir = guard_output(root, run_dir)
    config = {**config, "run_dir": str(run_dir.relative_to(root.resolve()))}
    manifest, metadata = prepare_manifest(root, config, requested_device)
    specs = {name: model_spec(name, config["models"], "H3") for name in config["backbones"]}
    return {"status": "checked", "training_launched": False, "real_images_preprocessed": False,
            "source_signature": manifest["source_signature"], "experiment_signature": manifest["experiment_signature"],
            "run_dir": str(run_dir), "n_structures": len(metadata["structures"]),
            "job_counts": {stage: len(planned_jobs(config, stage)) for stage in ("main", "ablations", "baselines", "all")},
            "models": specs, "training": config["training"], "baseline": config["baseline"],
            "evaluation_scope": "developmental spatial CV; given-candidate interval classification"}
