"""USER RUN: unchanged forty-epoch fits with two inner-fixed endpoints.

The optimization/augmentation/diagnostic loop preserves the frozen controlled
runner's numerical order. Its only new operation during that loop is saving
the actual weights on strict weighted inner-validation CE improvement. This
does not draw random numbers or inspect an outer image. Both weights and
selection records are fixed before the single outer materialization.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import errno
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
from microcture.task1.training import resolve_device, run_lock, score_samples, seed_everything, write_json
from microcture.task1_controlled.diagnostics import diagnose_model, learning_acceptance
from microcture.task1_controlled.models import build_controlled_model, model_spec
from microcture.task1_controlled.runner import (
    augmented_arrays, state_fingerprint, _gradient_norm, _gradient_summary,
    weighted_ce, flat_history_row, predict,
)
from microcture.task1_formal.baselines import fit_baseline, predict_baseline
from microcture.task1_formal.data import apply_condition, prepare_inner, prepare_outer
from microcture.task1_formal.runner import frozen_fold, outer_neural_endpoint, guard_output as formal_guard_output

from .protocol import (
    PROTOCOL, job_directory, endpoint_policies, planned_jobs, prepare_manifest,
    json_signature, project_path, sha256,
)
from .selection import InnerCESelector, endpoint_selection_records, selection_signature


class RunStopped(RuntimeError):
    """An interrupted or I/O-failed scientific fit is retained, without retry."""


STAGING_PREFIX = "microcture-reevaluation-fit-"
STAGING_CONTEXT = "STAGING_CONTEXT.json"
STAGING_OWNER = "microcture.task1_reevaluation"


def write_staging_context(output: Path, root: Path, run_dir: Path, job: dict) -> None:
    write_json(output / STAGING_CONTEXT, {
        "owner": STAGING_OWNER, "project_root": str(root.resolve()),
        "run_dir": str(run_dir.resolve()), "job": job,
        "state": "unpublished scientific staging; never automatically retry this fit",
    })


def unpublished_tmp_staging(root: Path, run_dir: Path) -> dict[tuple, list[Path]]:
    found = defaultdict(list)
    for directory in Path("/tmp").glob(STAGING_PREFIX + "*"):
        if directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / STAGING_CONTEXT
        if marker.is_symlink() or not marker.is_file():
            continue
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or value.get("owner") != STAGING_OWNER
                    or value.get("project_root") != str(root.resolve())
                    or value.get("run_dir") != str(run_dir.resolve())):
                continue
            job = value["job"]
            job_directory(run_dir, job)
            found[job["condition"], job["model"], job["seed"], job["fold"]].append(directory)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(found)


def failed_staging_directory(run_dir: Path, job: dict) -> Path:
    return run_dir / "failed_staging" / job_directory(run_dir, job).relative_to(run_dir / "runs")


def preserve_failed_staging(output: Path, run_dir: Path, job: dict,
                            error: BaseException, root: Path) -> Path:
    destination = failed_staging_directory(run_dir, job)
    if not output.exists():
        return job_directory(run_dir, job)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Previous failed staging must be preserved: {destination}")
        shutil.move(str(output), str(destination))
        print(f"Unsealed scientific staging retained at {destination}; use a fresh run directory.",
              file=sys.stderr, flush=True)
        return destination
    except (OSError, KeyboardInterrupt) as preserve_error:
        try:
            marker = output / STAGING_CONTEXT
            context = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
            if (not isinstance(context, dict) or context.get("owner") != STAGING_OWNER
                    or context.get("project_root") != str(root.resolve())
                    or context.get("run_dir") != str(run_dir.resolve())
                    or context.get("job") != job):
                write_staging_context(output, root, run_dir, job)
        except (OSError, KeyboardInterrupt, ValueError, TypeError) as marker_error:
            print(f"Cannot persist the fallback marker ({type(marker_error).__name__}: {marker_error}); "
                  "automatic retry prevention cannot be persisted under this double storage fault.",
                  file=sys.stderr, flush=True)
        print(f"Cannot persist failed staging ({type(preserve_error).__name__}: {preserve_error}); "
              f"numeric files remain at {output}. Original error: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return output


def guard_output(root: Path, run_dir: Path, formal_dir: Path | None = None,
                 readout_dir: Path | None = None) -> Path:
    result = formal_guard_output(root, run_dir)
    protected = (formal_dir or root / "results/task1_formal_v1",
                 readout_dir or root / "results/task1_readout_v1",
                 root / "results/task1_morphology_probe_v1")
    for original in protected:
        original = original.resolve()
        if result == original or original in result.parents or result in original.parents:
            raise ValueError(f"Reevaluation output overlaps retained scientific evidence: {original}")
    if (result / "manifest.json").is_file():
        existing = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("protocol") != PROTOCOL:
            raise ValueError("Reevaluation cannot reuse another experiment directory")
    return result


def write_artifact_index(output: Path, job: dict, manifest: dict) -> None:
    """Seal all fit and endpoint files with relative recursive names."""
    paths = sorted(output.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Scientific artifacts must not contain symbolic links")
    records = [{"name": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in paths if path.is_file() and path != output / "ARTIFACTS.json"]
    write_json(output / "ARTIFACTS.json", {
        "job": job, "experiment_signature": manifest["experiment_signature"],
        "files": records, "signature": json_signature(records),
    })


def verify_artifact_index(directory: Path, job: dict, manifest: dict) -> None:
    index = json.loads((directory / "ARTIFACTS.json").read_text(encoding="utf-8"))
    if (index.get("job") != job or index.get("experiment_signature") != manifest["experiment_signature"]
            or index.get("signature") != json_signature(index.get("files"))):
        raise ValueError(f"Scientific artifact index differs from the fit/protocol: {directory}")
    names = set()
    for record in index["files"]:
        name = record["name"]
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or not relative.parts or name in names or name == "ARTIFACTS.json"):
            raise ValueError(f"Unsafe/duplicate indexed relative artifact: {name!r}")
        names.add(name)
        path = directory / relative
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"]
                or sha256(path) != record["sha256"]):
            raise ValueError(f"Scientific artifact changed or missing: {path}")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError(f"Scientific artifact tree contains a symbolic link: {directory}")
    actual = {path.relative_to(directory).as_posix() for path in paths if path.is_file()}
    if actual != names | {"ARTIFACTS.json"}:
        raise ValueError(f"Scientific recursive inventory changed: {directory}")


def publish_fold(source: Path, destination: Path) -> None:
    """Publish recursive endpoints; keep all scientific staging on copy failure."""
    if destination.exists():
        raise FileExistsError(f"Never overwrite an existing scientific fit: {destination}")
    try:
        os.replace(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        destination.mkdir()
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if path.is_symlink():
                raise ValueError("Cannot publish a symbolic-link scientific artifact")
            if path.is_dir():
                (destination / relative).mkdir(parents=True, exist_ok=True)
            elif relative.as_posix() != "metrics.json":
                shutil.copy2(path, destination / relative)
        # A partial cross-filesystem publication is never a completed fit.
        # Its source and partial destination remain on exception, without deletion.
        shutil.copy2(source / "metrics.json", destination / "metrics.json")


def _checkpoint_payload(model, job: dict, model_config: dict, epoch: int,
                        policy: str, config: dict) -> dict:
    return {"state_dict": model.state_dict(), "backbone": job["backbone"],
            "head_version": job["head"], "model_config": model_config,
            "job": job, "epoch": epoch, "policy": policy,
            "selection_signature": selection_signature(config)}


def fit_diagnostic(job: dict, train: dict, validation: dict, config: dict,
                   model_config: dict, device: torch.device, output: Path) -> dict:
    """Unchanged frozen optimization loop plus immediate best-CE weight saves."""
    settings = config["training"]
    seed_everything(job["seed"], settings["deterministic_algorithms"])
    model = build_controlled_model(job["backbone"], model_config, job["head"]).to(device)
    initial_state = state_fingerprint(model)
    common_state = state_fingerprint(model, common_only=True)
    write_json(output / "initialization.json", {
        "initial_state_sha256": initial_state, "common_initial_state_sha256": common_state,
        "seed": job["seed"], "job": job,
    })
    x, length, y = augmented_arrays(train, settings["augmentation"])
    counts = np.bincount(train["y"], minlength=3)
    if np.any(counts == 0):
        raise ValueError("Every frozen training fold needs the three classes")
    weights = len(train["y"]) / (3 * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(job["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(length), torch.from_numpy(y),
                      torch.arange(len(y))),
        batch_size=settings["batch_size"], shuffle=True, generator=generator,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    order_digest = hashlib.sha256()
    history = []
    epsilon = config["acceptance"]["activation_epsilon"]
    selector = InnerCESelector(settings["epochs"])
    best_path = output / "endpoints/inner_ce/weights.pt"
    final_path = output / "endpoints/final40/weights.pt"
    best_path.parent.mkdir(parents=True)
    final_path.parent.mkdir(parents=True)
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
            # Save the actual current weights immediately. Saving does not
            # sample RNG; this operation never sees an outer score or CHID F1.
            if selector.consider(epoch, row["weighted_validation_ce"]):
                torch.save(_checkpoint_payload(model, job, model_config, epoch, "inner_ce", config), best_path)
            write_json(output / "training_progress.json", {
                "job": job, "last_completed_epoch": epoch,
                "minibatch_order_sha256": order_digest.hexdigest(),
                "class_weights": weights.tolist(), "selected_epoch_so_far": selector.selected_epoch,
                "minimum_weighted_inner_validation_ce_so_far": selector.minimum_ce,
                "outer_images_used": False,
            })
            print(f"  {job['name']} epoch={epoch:02d} fixed_train_CE={row['weighted_train_ce']:.4f} "
                  f"val_CHID_F1={validation_diag['metrics']['per_class']['CHID']['f1']:.4f} "
                  f"head_active={train_diag['head']['effective_fraction']:.3f} "
                  f"val_logit_var={validation_diag['centered_logit_between_structure_var']:.3g}", flush=True)
    acceptance = learning_acceptance(history, config)
    selections = endpoint_selection_records(history, config)
    if selector.selected_epoch != selections["inner_ce"]["selected_epoch"]:
        raise ValueError("Saved best-CE weight epoch differs from the complete trajectory")
    torch.save(_checkpoint_payload(model, job, model_config, settings["epochs"], "final40", config), final_path)
    result = {
        "status": "completed", "job": job, "acceptance": acceptance,
        "initial_state_sha256": initial_state, "common_initial_state_sha256": common_state,
        "minibatch_order_sha256": order_digest.hexdigest(), "class_weights": weights.tolist(),
        "parameter_spec": model_spec(job["backbone"], model_config, job["head"]),
        "initial_train_ce": history[0]["weighted_train_ce"], "final_train_ce": history[-1]["weighted_train_ce"],
        "train_diagnostics": history[-1]["train_diagnostics"],
        "validation_diagnostics": history[-1]["validation_diagnostics"],
        "epochs": settings["epochs"], "outer_images_used": False,
        "checkpoint_selection": "paired final40 and weighted inner-validation CE; forty epochs completed",
        "selection_signature": selection_signature(config), "endpoint_selections": selections,
        "fit_units": 1, "endpoint_count": 2,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def restore_endpoint(job: dict, config: dict, checkpoint: Path, device: torch.device,
                     policy: str, selected_epoch: int):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if (saved.get("job") != job or saved.get("epoch") != selected_epoch
            or saved.get("model_config") != config["models"] or saved.get("head_version") != "H3"
            or saved.get("backbone") != job["backbone"] or saved.get("policy") != policy
            or saved.get("selection_signature") != selection_signature(config)
            or policy not in ("final40", "inner_ce") or not 1 <= selected_epoch <= 40):
        raise ValueError("Checkpoint is not this preregistered fixed endpoint")
    if policy == "final40" and selected_epoch != config["training"]["epochs"]:
        raise ValueError("The final40 policy must use the last registered epoch")
    model = build_controlled_model(job["backbone"], config["models"], "H3").to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def _identity_fields(job: dict, preprocessing: dict, fold: dict, manifest: dict) -> dict:
    return {"job": job, "experiment_signature": manifest["experiment_signature"],
            "source_signature": manifest["source_signature"], "dataset_signature": manifest["dataset_signature"],
            "preprocessing_signature": preprocessing["signature"],
            "train_structures": len(fold["train_ids"]), "validation_structures": len(fold["validation_ids"]),
            "test_structures": len(fold["test_ids"])}


def fit_one(job: dict, train: dict, validation: dict, preprocessing: dict, dataset: dict,
            fold: dict, config: dict, manifest: dict, output: Path, device: torch.device) -> dict:
    """Fix every inner endpoint before constructing one shared outer tensor."""
    write_json(output / "protocol.json", {
        "job": job, "training": config["training"], "baseline": config["baseline"],
        "acceptance": config["acceptance"], "preprocessing": preprocessing,
        "selection": config["selection"], "selection_signature": selection_signature(config),
        "evaluation_scope": manifest["evaluation_scope"],
    })
    started, result, model = time.perf_counter(), None, None
    stage, outer_constructed = "inner_fit", False
    identity = _identity_fields(job, preprocessing, fold, manifest)
    endpoints = {}
    try:
        if job["kind"] == "neural":
            selected_train = apply_condition(train, job["condition"])
            selected_validation = apply_condition(validation, job["condition"])
            result = fit_diagnostic(job, selected_train, selected_validation, config, config["models"], device, output)
            history = json.loads((output / "history.json").read_text(encoding="utf-8"))["epochs"]
            for policy in endpoint_policies(job):
                chosen = result["endpoint_selections"][policy]
                row = history[chosen["selected_epoch"]]
                endpoints[policy] = {**chosen, "status": "completed",
                                     "train_diagnostics": row["train_diagnostics"],
                                     "validation_diagnostics": row["validation_diagnostics"]}
        else:
            directory = output / "endpoints/baseline"
            model, baseline = fit_baseline(job["model"], train, validation, config, directory)
            result = {**baseline, "job": job, "fit_units": 1, "endpoint_count": 1,
                      "selection_signature": selection_signature(config)}
            endpoints["baseline"] = {**baseline, "policy": "baseline", "selected_epoch": None,
                                      "selection_signature": selection_signature(config)}
        if result["status"] != "completed":
            raise ValueError("Inner fit did not produce all registered endpoints")
        # No prepare_outer call may occur until every policy's weight and
        # inner-only selection marker have been frozen and hashed.
        checkpoint_hashes = {}
        for policy, value in endpoints.items():
            directory = output / "endpoints" / policy
            checkpoint = directory / ("weights.pt" if job["kind"] == "neural" else "model.joblib")
            if not checkpoint.is_file():
                raise ValueError(f"Missing registered {policy} endpoint weight")
            checkpoint_hash = sha256(checkpoint)
            value["checkpoint_sha256"] = checkpoint_hash
            checkpoint_hashes[policy] = checkpoint_hash
        for policy, value in endpoints.items():
            directory = output / "endpoints" / policy
            checkpoint = directory / ("weights.pt" if job["kind"] == "neural" else "model.joblib")
            write_json(directory / "endpoint.json", {
                **identity, "policy": policy, "selected_epoch": value["selected_epoch"],
                "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_hashes[policy],
                "selection_signature": selection_signature(config),
                "outer_images_constructed": False, "training_and_selection_complete": True,
                "both_neural_endpoints_fixed_before_outer": job["kind"] == "neural",
                "fixed_endpoint_hashes": checkpoint_hashes,
            })
        write_json(output / "endpoints_fixed.json", {
            **identity, "selection_signature": selection_signature(config),
            "checkpoint_sha256_by_policy": checkpoint_hashes,
            "selected_epoch_by_policy": {policy: value["selected_epoch"] for policy, value in endpoints.items()},
            "outer_images_constructed": False, "training_and_selection_complete": True,
        })
        stage = "inner_endpoint_validation"
        if job["kind"] == "neural":
            for policy, value in endpoints.items():
                directory = output / "endpoints" / policy
                model = restore_endpoint(job, config, directory / "weights.pt", device, policy, value["selected_epoch"])
                save_prediction_csv(directory / "validation_predictions.csv", selected_validation,
                                    predict(model, selected_validation, device, config["training"]["batch_size"]))
                del model
                model = None
        stage = "outer_evaluation"
        outer = prepare_outer(dataset, fold, preprocessing)
        outer_constructed = True
        if job["kind"] == "neural":
            outer = apply_condition(outer, job["condition"])
        for policy, value in endpoints.items():
            endpoint_started = time.perf_counter()
            directory = output / "endpoints" / policy
            if job["kind"] == "neural":
                model = restore_endpoint(job, config, directory / "weights.pt", device, policy, value["selected_epoch"])
                probabilities, diagnostic = outer_neural_endpoint(model, outer, config, device)
                diagnostic["evaluation_scope"] = f"outer prediction at fixed {policy}; no fitting or selection"
                value["outer_diagnostics"] = diagnostic
                del model
                model = None
            else:
                probabilities = predict_baseline(model, job["model"], outer)
            checkpoint = directory / ("weights.pt" if job["kind"] == "neural" else "model.joblib")
            if sha256(checkpoint) != value["checkpoint_sha256"]:
                raise ValueError("Fixed endpoint weight changed during outer evaluation")
            save_prediction_csv(directory / "outer_predictions.csv", outer, probabilities)
            value.update(identity, outer_metrics=score_samples(outer, probabilities),
                         outer_images_used=True, outer_fit_or_selection=False,
                         outer_constructed_after_fixed_endpoint=True,
                         endpoint_inference_seconds=time.perf_counter() - endpoint_started,
                         fit_units=1, endpoint_count=len(endpoints), shared_fit_unit=True)
            write_json(directory / "metrics.json", value)
        result.update(outer_images_used=True, outer_fit_or_selection=False,
                      outer_constructed_after_fixed_endpoints=True,
                      endpoints_fixed_before_outer=True,
                      completed_endpoint_policies=list(endpoints))
    except (Exception, KeyboardInterrupt) as error:
        result = {**(result or {}),
                  "status": "numerical_failure" if isinstance(error, RuntimeError) else "execution_failure",
                  "job": job, "error": str(error) or type(error).__name__, "error_type": type(error).__name__,
                  "run_stop_requested": isinstance(error, (OSError, KeyboardInterrupt)),
                  "failure_stage": stage, "outer_images_used": outer_constructed,
                  "outer_fit_or_selection": False,
                  "completed_endpoint_policies": [p for p in endpoints if (output / "endpoints" / p / "metrics.json").is_file()]}
        initialization = output / "initialization.json"
        if initialization.is_file():
            init = json.loads(initialization.read_text(encoding="utf-8"))
            result.update({key: init[key] for key in ("initial_state_sha256", "common_initial_state_sha256")})
        write_json(output / "failure.json", result)
        print(f"Failure retained: {job['model']}/{job['condition']} fold{job['fold']} seed{job['seed']}: {error}", flush=True)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    runtime = time.perf_counter() - started
    result.update(identity, runtime_seconds=runtime)
    if result["status"] == "completed":
        # Runtime describes the one shared fit. It must not be added twice.
        for policy in endpoints:
            path = output / "endpoints" / policy / "metrics.json"
            endpoint_result = json.loads(path.read_text(encoding="utf-8"))
            endpoint_result.update(runtime_seconds=runtime, runtime_scope="shared fit total; do not sum endpoint copies")
            write_json(path, endpoint_result)
    write_json(output / "metrics.json", result)
    write_artifact_index(output, job, manifest)
    return result


def assert_resume(directory: Path, job: dict, manifest: dict) -> dict | None:
    if not directory.exists():
        return None
    if not (directory / "metrics.json").is_file():
        raise ValueError(f"Unfinished scientific fit must be preserved without retry: {directory}")
    result = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    for key in ("experiment_signature", "source_signature", "dataset_signature"):
        if result.get(key) != manifest[key]:
            raise ValueError(f"Saved fit has changed source/data/config: {directory}")
    if result.get("job") != job or result.get("status") not in ("completed", "numerical_failure", "execution_failure"):
        raise ValueError(f"Saved fit metadata is inconsistent: {directory}")
    verify_artifact_index(directory, job, manifest)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if (protocol.get("job") != job
            or protocol.get("preprocessing", {}).get("signature") != result.get("preprocessing_signature")
            or protocol.get("selection") != manifest["config"]["selection"]):
        raise ValueError(f"Saved preprocessing/selection protocol is inconsistent: {directory}")
    if result["status"] == "completed":
        policies = endpoint_policies(job)
        if result.get("completed_endpoint_policies") != list(policies):
            raise ValueError(f"Completed fit lacks all registered endpoints: {directory}")
        fixed = json.loads((directory / "endpoints_fixed.json").read_text(encoding="utf-8"))
        checkpoint_hashes = {
            policy: sha256(directory / "endpoints" / policy /
                           ("weights.pt" if job["kind"] == "neural" else "model.joblib"))
            for policy in policies
        }
        if (fixed.get("job") != job or fixed.get("checkpoint_sha256_by_policy") != checkpoint_hashes
                or fixed.get("selection_signature") != selection_signature(manifest["config"])
                or fixed.get("outer_images_constructed") is not False
                or fixed.get("training_and_selection_complete") is not True
                or set(fixed.get("selected_epoch_by_policy", {})) != set(policies)):
            raise ValueError(f"Shared pre-outer endpoint fixation evidence differs: {directory}")
        if job["kind"] == "neural":
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))["epochs"]
            selections = endpoint_selection_records(history, manifest["config"])
            if result.get("endpoint_selections") != selections:
                raise ValueError("Saved endpoint selections differ from the complete inner trajectory")
            if history[-1]["weighted_train_ce"] != result.get("final_train_ce"):
                raise ValueError("Saved final training CE differs from its trajectory")
        for policy in policies:
            endpoint = directory / "endpoints" / policy
            for name in ("endpoint.json", "metrics.json", "outer_predictions.csv", "validation_predictions.csv"):
                if not (endpoint / name).is_file():
                    raise ValueError(f"Missing completed scientific endpoint evidence: {endpoint / name}")
            value = json.loads((endpoint / "metrics.json").read_text(encoding="utf-8"))
            marker = json.loads((endpoint / "endpoint.json").read_text(encoding="utf-8"))
            checkpoint = endpoint / ("weights.pt" if job["kind"] == "neural" else "model.joblib")
            if (not checkpoint.is_file() or value.get("status") != "completed" or value.get("job") != job
                    or value.get("policy") != policy or marker.get("policy") != policy
                    or value.get("checkpoint_sha256") != sha256(checkpoint)
                    or marker.get("checkpoint_sha256") != value["checkpoint_sha256"]
                    or value.get("selected_epoch") != marker.get("selected_epoch")
                    or fixed["selected_epoch_by_policy"].get(policy) != value.get("selected_epoch")
                    or marker.get("fixed_endpoint_hashes") != checkpoint_hashes
                    or value.get("selection_signature") != selection_signature(manifest["config"])
                    or marker.get("outer_images_constructed") is not False
                    or marker.get("training_and_selection_complete") is not True):
                raise ValueError(f"Saved endpoint metadata/weight differs from its fixed selection: {endpoint}")
            if job["kind"] == "neural" and marker.get("both_neural_endpoints_fixed_before_outer") is not True:
                raise ValueError("Neural endpoint lacks the shared pre-outer fixation marker")
            if job["kind"] == "neural" and value.get("selected_epoch") != selections[policy]["selected_epoch"]:
                raise ValueError("Saved endpoint epoch differs from inner-only selection")
    return result


def run_jobs(config: dict, root: Path, run_dir: Path, requested_device: str = "auto", *,
             stage: str = "all", model: str = "all", condition: str | None = None,
             fold: int | None = None, seed: int | None = None) -> dict:
    """Real training; invoke only through the user-run local CLI."""
    run_dir = guard_output(root, run_dir, project_path(root, config["formal_run_dir"]),
                           project_path(root, config["readout_run_dir"]))
    config = {**config, "run_dir": str(run_dir.relative_to(root.resolve()))}
    jobs = planned_jobs(config, stage, model, condition, fold, seed)
    if not jobs:
        raise ValueError("No registered fits match the stage/model/condition/fold/seed")
    manifest, metadata = prepare_manifest(root, config, requested_device)
    device = resolve_device(requested_device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["training"]["cpu_threads"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir), threadpool_limits(limits=config["training"]["cpu_threads"]):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Source/config/environment changed; preserve this run and use a fresh --run-dir")
        else:
            write_json(manifest_path, manifest)
        pending = []
        tmp_staging = unpublished_tmp_staging(root, run_dir)
        for job in jobs:
            identity = (job["condition"], job["model"], job["seed"], job["fold"])
            if identity in tmp_staging:
                raise ValueError(f"Unpublished scientific /tmp staging prevents automatic retry: {tmp_staging[identity]}")
            if failed_staging_directory(run_dir, job).exists():
                raise ValueError(f"Preserved unsealed staging prevents automatic retry: {failed_staging_directory(run_dir, job)}")
            existing = assert_resume(job_directory(run_dir, job), job, manifest)
            if existing is None:
                pending.append(job)
            else:
                print(f"Skip saved {existing['status']}: {job['condition']}/{job['model']}/fold{job['fold']}/seed{job['seed']}", flush=True)
        if pending:
            dataset = load_dataset(project_path(root, config["cache_dir"]))
            if dataset["signature"] != manifest["dataset_signature"]:
                raise ValueError("Loaded count cache differs from the frozen metadata")
            grouped = defaultdict(list)
            for job in pending:
                grouped[job["fold"]].append(job)
            for number, fold_jobs in sorted(grouped.items()):
                frozen = frozen_fold(root, config, manifest, number)
                train, validation, preprocessing = prepare_inner(dataset, frozen)
                saved_fold = {"frozen_fold": frozen, "preprocessing": preprocessing}
                saved_fold["signature"] = json_signature(saved_fold)
                fold_path = run_dir / "folds" / f"fold_{number}.json"
                if fold_path.is_file():
                    if json.loads(fold_path.read_text(encoding="utf-8")) != saved_fold:
                        raise ValueError("Saved shared preprocessing changed")
                else:
                    write_json(fold_path, saved_fold)
                for job in fold_jobs:
                    destination = job_directory(run_dir, job)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Start {job['condition']}/{job['model']} fold={job['fold']} seed={job['seed']}", flush=True)
                    output = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir="/tmp"))
                    published = False
                    try:
                        write_staging_context(output, root, run_dir, job)
                        result = fit_one(job, train, validation, preprocessing, dataset, frozen,
                                         config, manifest, output, device)
                        publish_fold(output, destination)
                        verify_artifact_index(destination, job, manifest)
                        published = True
                        if result.get("run_stop_requested"):
                            raise RunStopped(f"Stopped after {result.get('error_type')}; failed fit and trajectory retained at {destination}")
                    except RunStopped:
                        raise
                    except BaseException as error:
                        retained = preserve_failed_staging(output, run_dir, job, error, root)
                        raise RunStopped(f"Stopped after {type(error).__name__}; scientific staging retained at {retained}. "
                                         "No automatic retry; inspect evidence and use a fresh run directory.") from error
                    finally:
                        if published and output.exists():
                            shutil.rmtree(output)
                del train, validation
            del dataset
        from .evaluation import summarize_reevaluation
        return summarize_reevaluation(run_dir, metadata)


def check_protocol(root: Path, config: dict, run_dir: Path, requested_device: str = "auto") -> dict:
    run_dir = guard_output(root, run_dir, project_path(root, config["formal_run_dir"]),
                           project_path(root, config["readout_run_dir"]))
    config = {**config, "run_dir": str(run_dir.relative_to(root.resolve()))}
    manifest, metadata = prepare_manifest(root, config, requested_device)
    jobs = planned_jobs(config)
    main_jobs = planned_jobs(config, stage="main")
    return {"status": "checked", "training_launched": False, "real_images_preprocessed": False,
            "source_signature": manifest["source_signature"], "experiment_signature": manifest["experiment_signature"],
            "run_dir": str(run_dir), "n_structures": len(metadata["structures"]),
            "fit_count": len(jobs), "job_count": len(jobs),
            "endpoint_count": sum(len(endpoint_policies(job)) for job in jobs),
            "main_fit_count": len(main_jobs),
            "main_endpoint_count": sum(len(endpoint_policies(job)) for job in main_jobs),
            "models": {name: model_spec(name, config["models"], "H3") for name in config["backbones"]},
            "training": config["training"], "selection": config["selection"],
            "evaluation_scope": manifest["evaluation_scope"]}
