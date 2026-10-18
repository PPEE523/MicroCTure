"""User-run paired input and head experiments, restricted to frozen inner folds."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from microcture.task1.data import load_dataset
from microcture.task1.evaluation import save_prediction_csv
from microcture.task1.training import (environment_record, publish_fold, resolve_device, run_lock,
                                      score_samples, seed_everything, source_signature, write_json)
from .data import INPUT_VARIANTS, apply_input_variant, prepare_controlled_fold, target_masks_for_interval
from .diagnostics import diagnose_model, learning_acceptance
from .evidence import json_signature, sha256, verify_evidence
from .models import HEAD_VERSIONS, build_controlled_model, model_spec


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_protocol(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    training = config["training"]
    if training["optimizer"] != "AdamW" or training["loss"] != "class_weighted_cross_entropy":
        raise ValueError("Keep the original AdamW and weighted CE throughout these controls")
    if training["early_stopping"] or training["gradient_clipping"]:
        raise ValueError("Fixed-duration diagnostic runs do not use early stopping or gradient clipping")
    for key in ("batch_size", "epochs", "cpu_threads"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["epochs"] < 2 * config["acceptance"]["window"]:
        raise ValueError("At least two acceptance windows are required for sustained-learning checks")
    if training["augmentation"] != "reverse" or training["dropout"] != 0.2:
        raise ValueError("Preserve the original reversal augmentation and dropout=0.2")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 < config["head_stage"]["H3_learning_rate"] < training["learning_rate"]:
        raise ValueError("H3 must have a lower learning rate than H0-H2")
    if config["head_stage"]["input_variant"] != "I3":
        raise ValueError("Head comparisons use the preregistered I3 target-mask-plus-length input")
    if config["input_stage"]["backbone"] != "m2" or config["input_stage"]["head"] != "H0":
        raise ValueError("Input controls use the fixed M2/H0 reference, independently of head repairs")
    if config["input_stage"]["variants"] != list(INPUT_VARIANTS):
        raise ValueError("Input stage must include the entire I0-I3 factorial")
    if config["head_stage"]["versions"] != list(HEAD_VERSIONS):
        raise ValueError("Head stage must include H0-H3")
    if config["input_stage"]["geometry_baselines"] != ["length_only", "mask_only", "mask_length"]:
        raise ValueError("Retain all three preregistered geometry-only baselines")
    if config["input_stage"]["baseline_C"] != 1.0:
        raise ValueError("Geometry baselines use fixed C=1 without a validation search")
    if type(config["input_stage"]["baseline_max_iter"]) is not int or config["input_stage"]["baseline_max_iter"] < 1:
        raise ValueError("baseline_max_iter must be a positive integer")
    for settings in [config["input_stage"], *config["head_stage"]["cases"].values()]:
        if type(settings["fold"]) is not int or settings["fold"] not in range(5):
            raise ValueError("Use an existing frozen fold number 0-4")
        if type(settings["seed"]) is not int or settings["seed"] < 0:
            raise ValueError("Use a nonnegative integer paired seed")
    if not all(name.isidentifier() and name != "all" for name in config["head_stage"]["cases"]):
        raise ValueError("Historical case names must be safe identifiers")
    if not all(settings["backbone"] in ("m2plus", "cnn_transformer")
               for settings in config["head_stage"]["cases"].values()):
        raise ValueError("Head controls stress the previously failed M2+ and Transformer cells")
    if config["acceptance"]["CHID_F1_threshold"] is not None:
        raise ValueError("This stage verifies learning, without a CHID F1 acceptance threshold")
    learning_acceptance([], config)  # Validate every numeric acceptance setting.
    return config


def controlled_source_signature(root: Path) -> str:
    paths = sorted((root / "microcture/task1_controlled").glob("*.py"))
    paths.append(root / "scripts/run_task1_controlled.py")
    return json_signature({str(path.relative_to(root)): sha256(path) for path in paths})


def target_scope_preflight(metadata: dict) -> dict:
    groups = {}
    for row, center in zip(metadata["structures"], metadata["center_bin"]):
        groups.setdefault(int(center), []).append(row)
    conflicts = []
    config = metadata["config"]
    for center, rows in groups.items():
        if len({row["type"] for row in rows}) < 2:
            continue
        signatures = []
        for row in rows:
            mask = target_masks_for_interval(float(row["start"]), float(row["end"]), center,
                                            n_bins=metadata["n_bins"], chrom_length=metadata["chrom_length"],
                                            resolution_bp=config["resolution_bp"], window_bins=config["window_bins"],
                                            output_size=config["output_size"])
            signatures.append(hashlib.sha256(mask.tobytes()).hexdigest())
        resolved = all(signatures[i] != signatures[j] for i in range(len(rows)) for j in range(i)
                       if rows[i]["type"] != rows[j]["type"])
        conflicts.append({"center_bin": center, "structures": [row["structure_id"] for row in rows],
                          "target_mask_sha256": signatures, "resolved_by_target_mask": resolved})
    return {"historical_center_only_conflicts": conflicts,
            "all_historical_conflicts_resolved": all(row["resolved_by_target_mask"] for row in conflicts),
            "image_tensors_read": False, "inference_scope": "classification given a candidate interval, not genome discovery",
            "mask_length_caveat": "A target interval mask itself encodes length. Geometry-only baselines diagnose this cue."}


def augmented_arrays(samples: dict, augmentation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if augmentation != "reverse":
        raise ValueError("Use the common frozen reversal augmentation")
    return (np.concatenate((samples["x"], samples["x"][..., ::-1, ::-1]), axis=0),
            np.concatenate((samples["length"], samples["length"]), axis=0),
            np.concatenate((samples["y"], samples["y"]), axis=0))


def state_fingerprint(model: nn.Module, *, common_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if common_only and name.startswith("head.normalization."):
            continue
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _gradient_norm(parameters) -> float:
    sums = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(sums).sum().sqrt()) if sums else 0.0


def _gradient_summary(records: list[dict], epsilon: float = 1e-10) -> dict:
    result = {}
    for name in ("total", "encoder", "head"):
        values = np.asarray([row[name] for row in records], dtype=np.float64)
        result[name] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
                        "finite": bool(np.isfinite(values).all()),
                        "nonzero_fraction": float(np.mean(values > epsilon)), "batches": len(values)}
    return result


def weighted_ce(diagnostic: dict, weights: np.ndarray) -> float:
    support = diagnostic["loss_support"]
    return float(np.dot(weights, support["unweighted_nll_sum_by_class"]) /
                 np.dot(weights, support["examples_count_by_class"]))


def flat_history_row(row: dict) -> dict:
    result = {"epoch": row["epoch"], "weighted_train_ce": row["weighted_train_ce"],
              "optimized_training_loss": row.get("optimized_training_loss"),
              "weighted_validation_ce": row["weighted_validation_ce"], "epoch_seconds": row.get("epoch_seconds", 0)}
    for split, key in (("train", "train_diagnostics"), ("validation", "validation_diagnostics")):
        diag = row[key]
        result.update({f"{split}_chid_f1": diag["metrics"]["per_class"]["CHID"]["f1"],
                       f"{split}_macro_f1": diag["metrics"]["macro_f1"],
                       f"{split}_logit_variance": diag["logit_between_structure_var"],
                       f"{split}_centered_logit_variance": diag["centered_logit_between_structure_var"],
                       f"{split}_probability_variance": diag["probability_between_structure_var"],
                       f"{split}_majority_fraction": diag["majority_prediction_fraction"],
                       f"{split}_head_positive_fraction": diag["head"]["positive_fraction"],
                       f"{split}_head_effective_fraction": diag["head"]["effective_fraction"],
                       f"{split}_head_inactive_units": diag["head"]["inactive_units"]})
    for group in ("total", "encoder", "head"):
        for statistic in ("mean", "min", "max", "nonzero_fraction"):
            result[f"gradient_{group}_{statistic}"] = (row["gradient_norms"][group][statistic]
                                                       if row["gradient_norms"] is not None else None)
    return result


def predict(model: nn.Module, samples: dict, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(samples["y"]), batch_size):
            x = torch.from_numpy(samples["x"][start:start + batch_size]).to(device)
            length = torch.from_numpy(samples["length"][start:start + batch_size]).to(device)
            outputs.append(model(x, length).softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def fit_diagnostic(job: dict, train: dict, validation: dict, config: dict, model_config: dict,
                   device: torch.device, output: Path) -> dict:
    settings = config["training"]
    seed_everything(job["seed"], settings["deterministic_algorithms"])
    model = build_controlled_model(job["backbone"], model_config, job["head"]).to(device)
    initial_state = state_fingerprint(model)
    common_state = state_fingerprint(model, common_only=True)
    write_json(output / "initialization.json", {"initial_state_sha256": initial_state,
                                               "common_initial_state_sha256": common_state,
                                               "seed": job["seed"], "job": job})
    x, length, y = augmented_arrays(train, settings["augmentation"])
    counts = np.bincount(train["y"], minlength=3)
    if np.any(counts == 0):
        raise ValueError("Every frozen diagnostic training fold needs three classes")
    weights = len(train["y"]) / (3 * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(job["seed"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(length), torch.from_numpy(y),
                                    torch.arange(len(y))), batch_size=settings["batch_size"], shuffle=True,
                        generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    order_digest = hashlib.sha256()
    history = []
    epsilon = config["acceptance"]["activation_epsilon"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(settings["epochs"] + 1):
            started = time.perf_counter()
            gradients = []
            optimized_loss, optimized_weight = 0.0, 0.0
            if epoch:
                model.train()
                for inputs, lengths, labels, indices in loader:
                    order_digest.update(indices.numpy().astype("<i8").tobytes())
                    inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_function(model(inputs, lengths), labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss at epoch {epoch}; no optimizer update for this batch")
                    loss.backward()
                    norms = {"total": _gradient_norm(model.parameters()),
                             "encoder": _gradient_norm(model.encoder.parameters()),
                             "head": _gradient_norm(model.head.parameters())}
                    if not all(np.isfinite(value) for value in norms.values()):
                        raise RuntimeError(f"Nonfinite gradient at epoch {epoch}; no optimizer update for this batch")
                    gradients.append(norms)
                    batch_weight = float(weight_tensor[labels].sum())
                    optimized_loss += float(loss.detach()) * batch_weight
                    optimized_weight += batch_weight
                    optimizer.step()
            train_diag = diagnose_model(model, train, device, settings["batch_size"], epsilon=epsilon)
            validation_diag = diagnose_model(model, validation, device, settings["batch_size"], epsilon=epsilon)
            row = {"epoch": epoch, "weighted_train_ce": weighted_ce(train_diag, weights),
                   "weighted_validation_ce": weighted_ce(validation_diag, weights),
                   "optimized_training_loss": optimized_loss / optimized_weight if epoch else None,
                   "train_diagnostics": train_diag, "validation_diagnostics": validation_diag,
                   "gradient_norms": _gradient_summary(gradients, config["acceptance"]["gradient_norm_epsilon"]) if epoch else None,
                   "epoch_seconds": time.perf_counter() - started}
            history.append(row)
            write_json(output / "history.json", {"epochs": history})
            flat = flat_history_row(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            stream.flush()
            print(f"  {job['name']} epoch={epoch:02d} fixed_train_CE={row['weighted_train_ce']:.4f} "
                  f"val_CHID_F1={validation_diag['metrics']['per_class']['CHID']['f1']:.4f} "
                  f"head_active={train_diag['head']['effective_fraction']:.3f} "
                  f"val_logit_var={validation_diag['centered_logit_between_structure_var']:.3g}", flush=True)
    acceptance = learning_acceptance(history, config)
    # Last epoch is the primary endpoint: no best-F1 checkpoint selection.
    torch.save({"state_dict": model.state_dict(), "backbone": job["backbone"], "head_version": job["head"],
                "model_config": model_config, "job": job, "epoch": settings["epochs"]}, output / "final.pt")
    save_prediction_csv(output / "validation_predictions.csv", validation,
                        predict(model, validation, device, settings["batch_size"]))
    result = {"status": "completed", "job": job, "acceptance": acceptance,
              "initial_state_sha256": initial_state, "common_initial_state_sha256": common_state,
              "minibatch_order_sha256": order_digest.hexdigest(), "class_weights": weights.tolist(),
              "parameter_spec": model_spec(job["backbone"], model_config, job["head"]),
              "initial_train_ce": history[0]["weighted_train_ce"], "final_train_ce": history[-1]["weighted_train_ce"],
              "train_diagnostics": history[-1]["train_diagnostics"],
              "validation_diagnostics": history[-1]["validation_diagnostics"],
              "epochs": settings["epochs"], "outer_images_used": False,
              "checkpoint_selection": "final epoch only; no selection by CHID F1 or outer results"}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def geometry_features(samples: dict, name: str) -> np.ndarray:
    if name == "length_only":
        return samples["length"]
    mask = samples["x"][:, :, 2].reshape(len(samples["y"]), -1)
    if name == "mask_only":
        return mask
    if name == "mask_length":
        return np.concatenate((mask, samples["length"]), axis=1)
    raise ValueError(f"Unknown geometry-only baseline: {name}")


def fit_geometry(job: dict, train: dict, validation: dict, config: dict, output: Path) -> dict:
    x, length, y = augmented_arrays(train, config["training"]["augmentation"])
    augmented = {**train, "x": x, "length": length, "y": y}
    features = geometry_features(augmented, job["name"])
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(C=config["input_stage"]["baseline_C"], solver="lbfgs",
                                                   class_weight="balanced", random_state=job["seed"],
                                                   max_iter=config["input_stage"]["baseline_max_iter"]))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features, y)
    if not np.array_equal(pipeline[-1].classes_, [0, 1, 2]):
        raise ValueError("Unexpected class ordering")
    probabilities = pipeline.predict_proba(geometry_features(validation, job["name"]))
    save_prediction_csv(output / "validation_predictions.csv", validation, probabilities)
    joblib.dump(pipeline, output / "model.joblib")
    return {"status": "completed", "job": job, "validation_metrics": score_samples(validation, probabilities),
            "feature_count": features.shape[1], "signal_or_validity_features_used": False,
            "selection": "fixed C=1; training-only feature scaler; no validation hyperparameter selection",
            "outer_images_used": False, "acceptance": None}


def planned_jobs(config: dict, stage: str, case_name: str | None = None) -> list[dict]:
    base_lr = config["training"]["learning_rate"]
    if stage == "input":
        settings = config["input_stage"]
        common = {"stage": stage, "fold": settings["fold"], "seed": settings["seed"], "head": "H0",
                  "backbone": "m2", "learning_rate": base_lr}
        return [{**common, "name": name, "input_variant": name, "kind": "neural"} for name in settings["variants"]] + [
            {**common, "name": name, "input_variant": "I3", "kind": "geometry"} for name in settings["geometry_baselines"]]
    if case_name not in config["head_stage"]["cases"]:
        raise ValueError("Choose a preregistered historical failed model/fold/seed case")
    common = {"stage": stage, "case": case_name, **config["head_stage"]["cases"][case_name],
              "input_variant": "I3", "kind": "neural"}
    return [{**common, "name": version, "head": version,
             "learning_rate": config["head_stage"]["H3_learning_rate"] if version == "H3" else base_lr}
            for version in config["head_stage"]["versions"]]


def job_directory(run_dir: Path, job: dict) -> Path:
    return (run_dir / "input" / job["name"] if job["stage"] == "input"
            else run_dir / "heads" / job["case"] / job["name"])


def prepare_manifest(config: dict, root: Path, run_dir: Path, device: str) -> tuple[dict, dict]:
    evidence_dir = project_path(root, config["evidence_dir"])
    verified = verify_evidence(root, evidence_dir)
    freeze = json.loads((evidence_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    if project_path(root, config["cache_dir"]) != project_path(root, freeze["cache_dir"]) or \
            project_path(root, config["frozen_run_dir"]) != project_path(root, freeze["run_dir"]):
        raise ValueError("Controlled cache and frozen run must be the exact paths in the evidence manifest")
    if source_signature(root) != freeze["source_signature"]:
        raise ValueError("Active original task1 source changed relative to the frozen exact snapshot")
    metadata = json.loads((project_path(root, config["cache_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    if metadata["signature"] != verified["dataset_signature"]:
        raise ValueError("Controlled cache has a different frozen dataset signature")
    original_config = json.loads((root / freeze["run_dir"] / "manifest.json").read_text(encoding="utf-8"))["config"]
    for key in ("batch_size", "learning_rate", "weight_decay", "deterministic_algorithms"):
        if config["training"][key] != original_config["training"][key]:
            raise ValueError(f"Keep original training.{key} fixed; only H3 lowers learning rate")
    if config["training"]["dropout"] != original_config["models"]["dropout"]:
        raise ValueError("Preserve the original dropout in all controls")
    for name, case in config["head_stage"]["cases"].items():
        path = root / freeze["run_dir"] / "runs" / case["backbone"] / f"seed_{case['seed']}" / f"fold_{case['fold']}" / "metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        predicted = np.asarray(record["validation"]["confusion_matrix"]).sum(axis=0)
        if predicted.max() / predicted.sum() < 0.95:
            raise ValueError(f"Case {name} is not one of the historical concentrated/failed validation cells")
    scope_check = target_scope_preflight(metadata)
    if not scope_check["all_historical_conflicts_resolved"]:
        raise ValueError("Target masks still leave contradictory target inputs")
    for protected in (project_path(root, config["frozen_run_dir"]), evidence_dir, root / "data", root / "micro-c数据"):
        if run_dir == protected or protected in run_dir.parents:
            raise ValueError("Controlled results must not overwrite any frozen evidence or data directory")
    manifest = {"protocol": config["protocol"], "config": config, "source_signature": controlled_source_signature(root),
                "frozen_evidence_sha256": verified["evidence_manifest_sha256"],
                "dataset_signature": verified["dataset_signature"], "original_run_signature": verified["run_signature"],
                "environment": environment_record(), "requested_device": device, "target_scope_preflight": scope_check,
                "evaluation_scope": "development diagnostics on frozen inner training/validation; no new outer-fold predictions",
                "head_input_selection": "I3 fixed a priori, never selected by maximum validation F1"}
    manifest["experiment_signature"] = json_signature(manifest)
    return manifest, metadata


def run_stage(config: dict, root: Path, stage: str, case_name: str | None, run_dir: Path,
              requested_device: str = "auto") -> dict:
    manifest, _ = prepare_manifest(config, root, run_dir, requested_device)
    if stage == "heads":
        required = planned_jobs(config, "input")
        for job in required:
            path = job_directory(run_dir, job) / "metrics.json"
            if not path.exists():
                raise ValueError("Complete the I0-I3 and geometry-only input controls before running head controls")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "completed" or record.get("experiment_signature") != manifest["experiment_signature"] or record.get("job") != job:
                raise ValueError("Input controls are incomplete or use a different controlled protocol")
    jobs = planned_jobs(config, stage, case_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(requested_device)
    torch.set_num_threads(config["training"]["cpu_threads"])
    with run_lock(run_dir), threadpool_limits(limits=config["training"]["cpu_threads"]):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Controlled code/config/environment changed; preserve evidence and use a new --run-dir")
        else:
            write_json(manifest_path, manifest)
        pending = []
        for job in jobs:
            destination = job_directory(run_dir, job)
            if (destination / "metrics.json").exists():
                stored = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                if stored["experiment_signature"] != manifest["experiment_signature"] or stored.get("job") != job:
                    raise ValueError("Existing result has a different controlled signature")
                if stored["status"] == "completed":
                    print(f"Skip completed {stage}/{job['name']}", flush=True)
                else:
                    print(f"Retain failed evidence for {stage}/{job['name']}; retry with a new protocol/run directory", flush=True)
            elif destination.exists():
                raise ValueError(f"Unfinished result directory: {destination}; retain it and select a fresh run")
            else:
                pending.append(job)
        if pending:
            dataset = load_dataset(project_path(root, config["cache_dir"]))
            frozen_fold = json.loads((project_path(root, config["frozen_run_dir"]) / "folds" /
                                      f"fold_{jobs[0]['fold']}.json").read_text(encoding="utf-8"))
            train, validation, preprocessing = prepare_controlled_fold(dataset, frozen_fold)
            original_config = json.loads((project_path(root, config["frozen_run_dir"]) / "manifest.json").read_text())["config"]
            for job in pending:
                destination = job_directory(run_dir, job)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="microcture-controlled-fit-", dir="/tmp") as temporary:
                    output = Path(temporary)
                    write_json(output / "protocol.json", {"job": job, "preprocessing": preprocessing,
                                                          "training": config["training"], "acceptance": config["acceptance"]})
                    selected_train = apply_input_variant(train, job["input_variant"])
                    selected_validation = apply_input_variant(validation, job["input_variant"])
                    started = time.perf_counter()
                    print(f"Start {stage}/{job['name']} fold={job['fold']} seed={job['seed']} lr={job['learning_rate']:g}", flush=True)
                    try:
                        if job["kind"] == "geometry":
                            result = fit_geometry(job, selected_train, selected_validation, config, output)
                        else:
                            result = fit_diagnostic(job, selected_train, selected_validation, config, original_config["models"], device, output)
                    except (RuntimeError, ValueError, ConvergenceWarning) as exc:
                        # Partial traces are failed-experiment evidence, rather
                        # than one-use files to discard after a numerical failure.
                        result = {"status": "numerical_failure", "job": job, "error": str(exc),
                                  "acceptance": {"accepted": False}, "outer_images_used": False}
                        if (output / "initialization.json").exists():
                            initialization = json.loads((output / "initialization.json").read_text(encoding="utf-8"))
                            result.update({key: initialization[key] for key in ("initial_state_sha256", "common_initial_state_sha256")})
                        write_json(output / "failure.json", result)
                        print(f"Failure retained for {job['name']}: {exc}", flush=True)
                    result.update({"experiment_signature": manifest["experiment_signature"],
                                   "preprocessing_signature": preprocessing["signature"],
                                   "runtime_seconds": time.perf_counter() - started,
                                   "train_structures": len(frozen_fold["train_ids"]),
                                   "validation_structures": len(frozen_fold["validation_ids"])})
                    write_json(output / "metrics.json", result)
                    publish_fold(output, destination)
        return summarize_controlled(run_dir)


def summarize_controlled(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    planned = planned_jobs(config, "input")
    for case in config["head_stage"]["cases"]:
        planned += planned_jobs(config, "heads", case)
    rows, pending, groups = [], [], {}
    for job in planned:
        path = job_directory(run_dir, job) / "metrics.json"
        if not path.exists():
            pending.append({"stage": job["stage"], "case": job.get("case"), "name": job["name"]})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["experiment_signature"] != manifest["experiment_signature"] or result.get("job") != job:
            raise ValueError("Mixed controlled protocols in summary")
        diag = result.get("validation_diagnostics", {})
        metrics = diag.get("metrics", result.get("validation_metrics", {}))
        rows.append({"stage": job["stage"], "case": job.get("case", "input_reference"), "name": job["name"],
                     "input": job["input_variant"], "head": job["head"], "learning_rate": job["learning_rate"],
                     "status": result["status"], "accepted": (result.get("acceptance") or {}).get("accepted"),
                     "initial_train_ce": result.get("initial_train_ce"), "final_train_ce": result.get("final_train_ce"),
                     "validation_CHID_F1": metrics.get("per_class", {}).get("CHID", {}).get("f1"),
                     "validation_macro_F1": metrics.get("macro_f1"),
                     "validation_centered_logit_variance": diag.get("centered_logit_between_structure_var"),
                     "validation_head_effective_fraction": diag.get("head", {}).get("effective_fraction"),
                     "runtime_seconds": result["runtime_seconds"]})
        if job["kind"] == "neural" and result["status"] == "completed":
            key = job.get("case", "input_reference")
            groups.setdefault(key, []).append((job, result, path.parent))
    pairing_checks = {}
    for key, entries in groups.items():
        common = {result["common_initial_state_sha256"] for _, result, _ in entries}
        order = {result["minibatch_order_sha256"] for _, result, _ in entries}
        preprocessing = {result["preprocessing_signature"] for _, result, _ in entries}
        class_weights = {tuple(result["class_weights"]) for _, result, _ in entries}
        initial = {job["name"]: result["initial_state_sha256"] for job, result, _ in entries}
        pairing_checks[key] = {"common_initial_weights_match": len(common) == 1,
                               "minibatch_order_match": len(order) == 1,
                               "preprocessing_match": len(preprocessing) == 1, "class_weights_match": len(class_weights) == 1,
                               "completed_versions": len(entries), "comparison_ready": len(entries) == 4,
                               "H0_H1_full_initial_weights_match": initial["H0"] == initial["H1"] if {"H0", "H1"} <= initial.keys() else None,
                               "H2_H3_full_initial_weights_match": initial["H2"] == initial["H3"] if {"H2", "H3"} <= initial.keys() else None}
        if len(common) != 1 or len(order) != 1 or len(preprocessing) != 1 or len(class_weights) != 1 or \
                pairing_checks[key]["H0_H1_full_initial_weights_match"] is False or pairing_checks[key]["H2_H3_full_initial_weights_match"] is False:
            raise ValueError(f"Paired initialization/minibatches differ for {key}")
    summary = {"execution_status": "complete" if not pending else "partial", "comparison": rows,
               "pending": pending, "pairing_checks": pairing_checks,
               "acceptance_scope": "sustained learning diagnostics, not high F1 and not a new blind test",
               "experiment_signature": manifest["experiment_signature"]}
    write_json(run_dir / "summary.json", summary)
    if rows:
        with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_traces(run_dir, groups)
    return summary


def plot_traces(run_dir: Path, groups: dict) -> None:
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="microcture-controlled-mpl-", dir="/tmp") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            for key, entries in groups.items():
                figure = Figure(figsize=(12, 9), layout="constrained")
                FigureCanvasAgg(figure)
                axes = figure.subplots(3, 2).ravel()
                fields = [("weighted_train_ce", "Fixed train weighted CE"),
                          ("validation_chid_f1", "Inner validation CHID F1 (record only)"),
                          ("train_head_effective_fraction", "Train head effective activation fraction"),
                          ("validation_centered_logit_variance", "Between-structure centered logit variance"),
                          ("gradient_encoder_mean", "CNN encoder gradient norm, before step"),
                          ("gradient_head_mean", "Classification head gradient norm, before step")]
                for job, _, directory in entries:
                    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
                        history = list(csv.DictReader(stream))
                    for axis, (field, title) in zip(axes, fields):
                        points = [(int(row["epoch"]), float(row[field])) for row in history if row[field]]
                        axis.plot([p[0] for p in points], [p[1] for p in points], label=job["name"])
                        axis.set_title(title)
                        axis.set_xlabel("Epoch")
                        axis.grid(alpha=0.2)
                axes[0].legend()
                figure.suptitle(key + ": paired, frozen inner-fold controls")
                figure.savefig(run_dir / f"traces_{key}.png", dpi=150)
                figure.clear()
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
