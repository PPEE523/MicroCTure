"""USER RUN: paired O0/O1 controls; frozen endpoint before outer materialization.

The diagnostic training loop below is copied from the frozen controlled
runner with only the model factory, model specification and checkpoint
variant changed. Shared arithmetic/augmentation helpers stay frozen imports.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits
from microcture.task1.data import load_dataset
from microcture.task1.evaluation import save_prediction_csv
from microcture.task1.training import (publish_fold, resolve_device, run_lock, score_samples,
                                      seed_everything, write_json)
from microcture.task1_controlled.diagnostics import diagnose_model, learning_acceptance
from microcture.task1_controlled.runner import (augmented_arrays, state_fingerprint,
    _gradient_norm, _gradient_summary, weighted_ce, flat_history_row, predict)
from microcture.task1_formal.data import apply_condition, prepare_inner, prepare_outer
from microcture.task1_formal.runner import (assert_resume, write_artifact_index, frozen_fold,
    outer_neural_endpoint, guard_output as formal_guard_output)
from .models import build_model, model_spec
from .protocol import (job_directory, planned_jobs, prepare_manifest, json_signature,
                       project_path, sha256)


class RunStopped(RuntimeError):
    """An interrupted or I/O-failed job was retained; do not start another fit."""


STAGING_PREFIX = "microcture-readout-fit-"
STAGING_CONTEXT = "STAGING_CONTEXT.json"


def write_staging_context(output: Path, root: Path, run_dir: Path, job: dict) -> None:
    """Make /tmp evidence discoverable before any numerical work begins."""
    write_json(output / STAGING_CONTEXT, {
        "owner": "microcture.task1_readout", "project_root": str(root.resolve()),
        "run_dir": str(run_dir.resolve()), "job": job,
        "state": "unpublished staging; do not retry this run/job while it exists"})


def unpublished_tmp_staging(root: Path, run_dir: Path) -> dict[tuple, list[Path]]:
    """Inspect only this runner's prefixed directories and exact owner paths."""
    found = defaultdict(list)
    for directory in Path("/tmp").glob(STAGING_PREFIX + "*"):
        if directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / STAGING_CONTEXT
        if marker.is_symlink() or not marker.is_file():
            continue
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            if (value.get("owner") != "microcture.task1_readout"
                    or value.get("project_root") != str(root.resolve())
                    or value.get("run_dir") != str(run_dir.resolve())):
                continue
            job = value["job"]
            identity = (job["condition"], job["model"], job["seed"], job["fold"])
            # Resolve and validate the job before trusting a /tmp marker.
            job_directory(run_dir, job)
            found[identity].append(directory)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(found)


def failed_staging_directory(run_dir: Path, job: dict) -> Path:
    return run_dir / "failed_staging" / job_directory(run_dir, job).relative_to(run_dir / "runs")


def preserve_failed_staging(output: Path, run_dir: Path, job: dict, error: BaseException,
                            root: Path) -> Path:
    """Retain unsealed numeric files if even failure publication cannot finish.

    If both persistent storage and the move fail, keep this invocation's /tmp
    directory and print its exact location. It is never cleaned on this path.
    """
    destination = failed_staging_directory(run_dir, job)
    if not output.exists():
        return job_directory(run_dir, job)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Previous failed staging must be preserved: {destination}")
        shutil.move(str(output), str(destination))
        print(f"Unsealed scientific staging retained at {destination}; use a fresh run directory.", file=sys.stderr, flush=True)
        return destination
    except (OSError, KeyboardInterrupt) as preserve_error:
        try:
            # This marker normally already exists from before fit. Do not
            # overwrite it during an I/O fault and risk a partial JSON write.
            marker = output / STAGING_CONTEXT
            valid_marker = False
            try:
                context = json.loads(marker.read_text(encoding="utf-8"))
                valid_marker = (context.get("owner") == "microcture.task1_readout"
                                and context.get("project_root") == str(root.resolve())
                                and context.get("run_dir") == str(run_dir.resolve())
                                and context.get("job") == job)
            except (OSError, ValueError, TypeError):
                pass
            if not valid_marker:
                write_staging_context(output, root, run_dir, job)
        except (OSError, KeyboardInterrupt) as marker_error:
            print(f"Cannot write /tmp fallback marker ({type(marker_error).__name__}: {marker_error}); "
                  "automatic retry prevention cannot be persisted under this double storage fault.",
                  file=sys.stderr, flush=True)
        print(f"Cannot persist failed staging ({type(preserve_error).__name__}: {preserve_error}); "
              f"numeric files remain at {output}. Original error: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return output


def guard_output(root: Path, run_dir: Path, formal_dir: Path | None = None) -> Path:
    result = formal_guard_output(root, run_dir)
    protected = (formal_dir or root / 'results/task1_formal_v1').resolve()
    if result == protected or protected in result.parents or result in protected.parents:
        raise ValueError('Readout run must not overlap the original formal result directory')
    if result.exists() and any(result.glob('runs/*/m2plus')):
        raise ValueError('Readout output cannot reuse a formal experiment directory')
    return result


def fit_diagnostic(job: dict, train: dict, validation: dict, config: dict, model_config: dict,
                   device: torch.device, output: Path) -> dict:
    settings = config["training"]
    seed_everything(job["seed"], settings["deterministic_algorithms"])
    model = build_model(job["variant"], model_config).to(device)
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
                "model_config": model_config, "readout_variant": job["variant"], "job": job, "epoch": settings["epochs"]}, output / "final.pt")
    save_prediction_csv(output / "validation_predictions.csv", validation,
                        predict(model, validation, device, settings["batch_size"]))
    result = {"status": "completed", "job": job, "acceptance": acceptance,
              "initial_state_sha256": initial_state, "common_initial_state_sha256": common_state,
              "minibatch_order_sha256": order_digest.hexdigest(), "class_weights": weights.tolist(),
              "parameter_spec": model_spec(job["variant"], model_config),
              "initial_train_ce": history[0]["weighted_train_ce"], "final_train_ce": history[-1]["weighted_train_ce"],
              "train_diagnostics": history[-1]["train_diagnostics"],
              "validation_diagnostics": history[-1]["validation_diagnostics"],
              "epochs": settings["epochs"], "outer_images_used": False,
              "checkpoint_selection": "final epoch only; no selection by CHID F1 or outer results"}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def restore_endpoint(job: dict, config: dict, checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if (saved["job"] != job or saved["epoch"] != config["training"]["epochs"]
            or saved["model_config"] != config["models"] or saved["head_version"] != "H3"
            or saved["readout_variant"] != job["variant"] or saved["backbone"] != "m2plus"):
        raise ValueError("Checkpoint is not this registered readout final endpoint")
    model = build_model(job["variant"], config["models"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def fit_one(job: dict, train: dict, validation: dict, preprocessing: dict, dataset: dict,
            fold: dict, config: dict, manifest: dict, output: Path, device: torch.device) -> dict:
    write_json(output / "protocol.json", {
        "job": job, "training": config["training"], "gate_beta": config["gate_beta"],
        "acceptance": config["acceptance"], "preprocessing": preprocessing,
        "selection": "fixed final40 neural checkpoint; no outer selection/refit",
        "evaluation_scope": manifest["evaluation_scope"]})
    started, result, model = time.perf_counter(), None, None
    stage, outer_constructed = "inner_fit", False
    try:
        result = fit_diagnostic(job, apply_condition(train, job["condition"]),
                                apply_condition(validation, job["condition"]), config,
                                config["models"], device, output)
        checkpoint = output / "final.pt"
        if result["status"] != "completed" or not checkpoint.is_file():
            raise ValueError("Inner fit did not produce a completed fixed endpoint")
        endpoint_hash = sha256(checkpoint)
        write_json(output / "endpoint.json", {"job": job, "checkpoint": checkpoint.name,
                   "checkpoint_sha256": endpoint_hash, "outer_images_constructed": False,
                   "training_and_selection_complete": True})
        stage = "outer_evaluation"
        outer = prepare_outer(dataset, fold, preprocessing)
        outer_constructed = True
        model = restore_endpoint(job, config, checkpoint, device)
        outer = apply_condition(outer, job["condition"])
        probabilities, diagnostic = outer_neural_endpoint(model, outer, config, device)
        if sha256(checkpoint) != endpoint_hash:
            raise ValueError("Fixed endpoint changed during outer evaluation")
        save_prediction_csv(output / "outer_predictions.csv", outer, probabilities)
        result.update(outer_diagnostics=diagnostic, outer_metrics=score_samples(outer, probabilities),
                      outer_images_used=True, outer_fit_or_selection=False,
                      outer_constructed_after_fixed_endpoint=True, checkpoint_sha256=endpoint_hash)
    except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as error:
        result = {**(result or {}), "status": "execution_failure" if isinstance(error, (ValueError, OSError, KeyboardInterrupt)) else "numerical_failure",
                  "job": job, "error": str(error) or type(error).__name__, "error_type": type(error).__name__,
                  "run_stop_requested": isinstance(error, (OSError, KeyboardInterrupt)), "failure_stage": stage,
                  "outer_images_used": outer_constructed, "outer_fit_or_selection": False}
        initialization = output / "initialization.json"
        if initialization.exists():
            init = json.loads(initialization.read_text(encoding="utf-8"))
            result.update({key: init[key] for key in ("initial_state_sha256", "common_initial_state_sha256")})
        write_json(output / "failure.json", result)
        print(f"Failure retained: {job['variant']}/{job['condition']} fold{job['fold']} seed{job['seed']}: {error}", flush=True)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result.update(job=job, experiment_signature=manifest["experiment_signature"],
                  source_signature=manifest["source_signature"], dataset_signature=manifest["dataset_signature"],
                  preprocessing_signature=preprocessing["signature"], runtime_seconds=time.perf_counter()-started,
                  train_structures=len(fold["train_ids"]), validation_structures=len(fold["validation_ids"]),
                  test_structures=len(fold["test_ids"]))
    write_json(output / "metrics.json", result)
    write_artifact_index(output, job, manifest)
    return result


def run_jobs(config: dict, root: Path, run_dir: Path, requested_device: str = "auto", *,
             variant: str = "all", condition: str | None = None,
             fold: int | None = None, seed: int | None = None) -> dict:
    """Real fits; the user invokes this locally after reviewing the control."""
    run_dir = guard_output(root, run_dir, project_path(root, config["formal_run_dir"]))
    config = {**config, "run_dir": str(run_dir.relative_to(root.resolve()))}
    jobs = planned_jobs(config, variant, condition, fold, seed)
    if not jobs:
        raise ValueError("No registered jobs match the requested filters")
    manifest, metadata = prepare_manifest(root, config, requested_device)
    device = resolve_device(requested_device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["training"]["cpu_threads"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir), threadpool_limits(limits=config["training"]["cpu_threads"]):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Source/config/environment changed; preserve this run and use a fresh --run-dir")
        else:
            write_json(manifest_path, manifest)
        pending = []
        tmp_staging = unpublished_tmp_staging(root, run_dir)
        for job in jobs:
            identity = (job["condition"], job["model"], job["seed"], job["fold"])
            if identity in tmp_staging:
                raise ValueError(f"Unpublished /tmp scientific staging is preserved; no automatic retry: {tmp_staging[identity]}")
            if failed_staging_directory(run_dir, job).exists():
                raise ValueError(f"Unsealed failed staging is preserved; no automatic retry: {failed_staging_directory(run_dir, job)}")
            existing = assert_resume(job_directory(run_dir, job), job, manifest)
            if existing is None:
                pending.append(job)
            else:
                print(f"Skip saved {existing['status']}: {job['condition']}/{job['variant']}/fold{job['fold']}/seed{job['seed']}", flush=True)
        if pending:
            dataset = load_dataset(project_path(root, config["cache_dir"]))
            if dataset["signature"] != manifest["dataset_signature"]:
                raise ValueError("Loaded count cache differs from frozen metadata")
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
                    raise ValueError("Saved shared preprocessing changed")
                write_json(fold_path, saved_fold)
                for job in fold_jobs:
                    destination = job_directory(run_dir, job)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Start {job['condition']}/{job['variant']} fold={job['fold']} seed={job['seed']}", flush=True)
                    output = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir="/tmp"))
                    published = False
                    try:
                        write_staging_context(output, root, run_dir, job)
                        result = fit_one(job, train, validation, preprocessing, dataset, frozen, config, manifest, output, device)
                        publish_fold(output, destination)
                        published = True
                        if result.get("run_stop_requested"):
                            raise RunStopped(f"Stopped after {result.get('error_type')}; failed job and trajectory retained at {destination}")
                    except RunStopped:
                        raise
                    except BaseException as error:
                        retained = preserve_failed_staging(output, run_dir, job, error, root)
                        raise RunStopped(f"Stopped after {type(error).__name__}; unsealed scientific files retained at {retained}. "
                                         "No automatic retry; inspect evidence and use a fresh run directory.") from error
                    finally:
                        if published and output.exists():
                            shutil.rmtree(output)
                del train, validation
            del dataset
        from .evaluation import summarize_readout
        return summarize_readout(run_dir, metadata)


def check_protocol(root: Path, config: dict, run_dir: Path, requested_device: str = "auto") -> dict:
    run_dir = guard_output(root, run_dir, project_path(root, config["formal_run_dir"]))
    config = {**config, "run_dir": str(run_dir.relative_to(root.resolve()))}
    manifest, metadata = prepare_manifest(root, config, requested_device)
    return {"status": "checked", "training_launched": False, "real_images_preprocessed": False,
            "source_signature": manifest["source_signature"], "experiment_signature": manifest["experiment_signature"],
            "run_dir": str(run_dir), "n_structures": len(metadata["structures"]), "job_count": len(planned_jobs(config)),
            "models": {name: model_spec(name, config["models"]) for name in config["variants"]},
            "training": config["training"], "gate_beta": config["gate_beta"],
            "formal_reference": manifest["formal_reference"], "evaluation_scope": manifest["evaluation_scope"]}


