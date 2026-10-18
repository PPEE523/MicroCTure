"""Task 1 formal comparison: shared neural protocol and fixed traditional baselines.

Contains the actual baseline pipelines, fold execution, strict resumption and
scientific artifact publication. Input ablations reuse these exact operations."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from microcture.layout import resolve

from .data import load_dataset, prepare_inner, prepare_outer
from .evaluation import grouped_predictions, metrics, save_prediction_csv
from .training import publish_fold, reject_retained_overlap, resolve_device, run_lock, score_samples, write_json
from .ablation import apply_condition, diagnose_model, fit_diagnostic
from .models import build_controlled_model, model_spec
from .protocol import fold_record, job_directory, json_signature, planned_jobs, prepare_manifest, sha256


# Fixed inner-train-only conventional models.
BASELINES = ("length_only", "mask_only", "mask_length", "pca_lr")
IMAGE_FEATURE_COUNT = 2 * 3 * 64 * 64
MASK_FEATURE_COUNT = 2 * 64 * 64


def baseline_features(samples: dict, name: str) -> np.ndarray:
    """Select geometry or flattened full I3 image plus the length column.

    Geometry baselines do not inspect signal or validity values. This matters
    when auditing that they cannot use either contact or missingness patterns.
    """
    if name not in BASELINES:
        raise ValueError(f"Unknown formal baseline: {name}; choose {BASELINES}")
    n = len(samples["y"])
    if samples["x"].shape != (n, 2, 3, 64, 64) or samples["length"].shape != (n, 1):
        raise ValueError("Unexpected formal baseline input layout")
    if name == "length_only":
        features = samples["length"].copy()
    elif name == "pca_lr":
        features = np.concatenate((samples["x"].reshape(n, -1), samples["length"]), axis=1)
    else:
        features = samples["x"][:, :, 2].reshape(n, -1).copy()
        if name == "mask_length":
            features = np.concatenate((features, samples["length"]), axis=1)
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite selected {name} features")
    return np.ascontiguousarray(features)


def _settings(config: dict) -> dict:
    baseline = config["baseline"]
    expected = {"pca_components": 16, "C": 1.0, "max_iter": 3000, "tol": 1e-5, "seed": 42}
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise ValueError(f"Formal baseline.{key} must remain preregistered at {value}")
    if config["training"].get("augmentation") != "reverse":
        raise ValueError("Formal baselines require the frozen reverse augmentation")
    return baseline


def _build_pipeline(name: str, settings: dict) -> Pipeline:
    steps = []
    if name == "pca_lr":
        # Only image columns enter PCA. The existing inner-training normalized
        # length remains an explicit feature and cannot be lost in projection.
        image = Pipeline([
            ("image_scaler", StandardScaler()),
            ("pca", PCA(n_components=settings["pca_components"], svd_solver="randomized",
                        random_state=settings["seed"])),
        ])
        features = ColumnTransformer([
            ("image_pca", image, slice(0, IMAGE_FEATURE_COUNT)),
            ("target_length", "passthrough", [IMAGE_FEATURE_COUNT]),
        ], remainder="drop", sparse_threshold=0)
        steps.append(("features", features))
    steps.extend([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=settings["C"], solver="lbfgs", class_weight="balanced",
                                  max_iter=settings["max_iter"], tol=settings["tol"],
                                  random_state=settings["seed"])),
    ])
    return Pipeline(steps)


def predict_baseline(estimator: Pipeline, name: str, samples: dict) -> np.ndarray:
    """Predict with the fitted pipeline without modifying its fitted state."""
    if not np.array_equal(estimator.classes_, [0, 1, 2]):
        raise ValueError("Unexpected formal baseline class ordering")
    probabilities = np.asarray(estimator.predict_proba(baseline_features(samples, name)), dtype=np.float64)
    # The shared evaluator checks shape, finite bounds and probability sums.
    grouped_predictions(samples, probabilities)
    return probabilities


def fit_baseline(name: str, train: dict, validation: dict, config: dict,
                 output: Path) -> tuple[Pipeline, dict]:
    """Fit one fixed baseline and retain validation evidence, without outer data."""
    if name not in BASELINES:
        raise ValueError(f"Unknown formal baseline: {name}; choose {BASELINES}")
    settings = _settings(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.joblib").exists() or (output / "validation_predictions.csv").exists():
        raise FileExistsError("Baseline output already contains evidence; use a fresh job directory")
    y = np.asarray(train["y"])
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not np.array_equal(np.unique(y), [0, 1, 2]):
        raise ValueError("Every formal baseline training subset must contain fixed classes 0, 1, 2")
    augmented = {**train,
                 "x": np.concatenate((train["x"], train["x"][..., ::-1, ::-1]), axis=0),
                 "length": np.concatenate((train["length"], train["length"]), axis=0),
                 "y": np.concatenate((y, y), axis=0)}
    features = baseline_features(augmented, name)
    if name == "pca_lr" and settings["pca_components"] > min(features.shape[0], IMAGE_FEATURE_COUNT):
        raise ValueError("Inner-training examples are insufficient for the fixed 16-component PCA")
    pipeline = _build_pipeline(name, settings)
    with warnings.catch_warnings(), threadpool_limits(limits=config["training"].get("cpu_threads", 4)):
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features, augmented["y"])
    probabilities = predict_baseline(pipeline, name, validation)
    grouped = save_prediction_csv(output / "validation_predictions.csv", validation, probabilities)
    joblib.dump(pipeline, output / "model.joblib")
    feature_spec = {
        "name": name, "input_condition": "I3", "raw_feature_count": int(features.shape[1]),
        "signal_features_used": name == "pca_lr", "validity_features_used": name == "pca_lr",
        "mask_features_used": name != "length_only", "length_features_used": name != "mask_only",
        "image_layout": "scale,channel(signal/valid_fraction/target_interval_fraction),row,column",
        "fit_scope": "inner-training replicate examples and their fixed reverse augmentation only",
        "length_scaler_scope": "unique inner-training structures only, already fitted by prepare_inner",
        "pipeline": "StandardScaler -> balanced LogisticRegression",
    }
    if name == "pca_lr":
        feature_spec.update({
            "image_features_before_pca": IMAGE_FEATURE_COUNT,
            "pca_components": settings["pca_components"],
            "features_after_pca_and_length": settings["pca_components"] + 1,
            "length_bypasses_pca": True,
            "pipeline": "image: StandardScaler -> PCA(16); concatenate length; StandardScaler -> balanced LogisticRegression",
            "pca_solver": "randomized", "pca_random_state": settings["seed"],
        })
    result = {
        "status": "completed", "validation_metrics": {**metrics(grouped["y"], grouped["probabilities"]),
                                                       "replicate_disagreement": grouped["replicate_disagreement"]},
        "feature_spec": feature_spec,
        "selection": {"rule": "fixed preregistered parameters; no hyperparameter search",
                      "C": settings["C"], "seed": settings["seed"], "max_iter": settings["max_iter"],
                      "tol": settings["tol"], "solver": "lbfgs", "class_weight": "balanced",
                      "refit_on_train_plus_validation": False, "outer_used_for_selection": False},
        "outer_images_used": False, "acceptance": None,
        "training_examples_before_augmentation": int(len(y)),
        "training_examples_after_augmentation": int(len(augmented["y"])),
    }
    return pipeline, result


# Formal execution, checkpoint fixing and outer evaluation.
def project_path(root: Path, value: str | Path) -> Path:
    return resolve(root, value)


def guard_output(root: Path, run_dir: Path) -> Path:
    """Use task/experiment/version outputs and preserve every retained result tree."""
    root = root.resolve()
    path = project_path(root, run_dir)
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symbolic-link result paths are forbidden: {ancestor}")
    path = path.resolve()
    results = root / "results"
    if not path.is_relative_to(results):
        raise ValueError("Formal output must be results/task1/<experiment>/<version>")
    parts = path.relative_to(results).parts
    if len(parts) < 3 or parts[0] != "task1":
        raise ValueError("Formal output must be results/task1/<experiment>/<version>")
    reject_retained_overlap(root, path, kinds=("result",))
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
    fold, _ = fold_record(project_path(root, config["frozen_run_dir"]), number)
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
            dataset = load_dataset(project_path(root, config["cache_dir"]),
                                   source_root=project_path(root, config["source_root"]) if config.get("source_root") else None,
                                   data_root=project_path(root, config["data_root"]) if config.get("data_root") else None)
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
    cache = project_path(root, config["cache_dir"])
    cache_files = [cache / "metadata.json", cache / "folds.json"] + [cache / sample["band_file"] for sample in metadata["samples"]]
    missing = [str(path) for path in cache_files if not path.is_file()]
    return {"status": "checked", "training_launched": False, "real_images_preprocessed": False,
            "cache_status": "not_prepared" if missing else "present_inventory_only",
            "training_ready_by_inventory": not bool(missing),
            "cache_check_scope": "file inventory only; band content and source hashes are checked when training loads the cache",
            "missing_cache_files": missing,
            "readiness_action": "Run scripts/task1/prepare_task1.py --data-root DATA_ROOT or pass --cache EXISTING_CACHE" if missing else "Use the frozen transforms corresponding to this count cache",
            "source_signature": manifest["source_signature"], "experiment_signature": manifest["experiment_signature"],
            "run_dir": str(run_dir), "n_structures": len(metadata["structures"]),
            "job_counts": {stage: len(planned_jobs(config, stage)) for stage in ("main", "ablations", "baselines", "all")},
            "models": specs, "training": config["training"], "baseline": config["baseline"],
            "evaluation_scope": "developmental spatial CV; given-candidate interval classification"}
