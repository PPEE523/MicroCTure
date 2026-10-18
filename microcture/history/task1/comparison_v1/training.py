"""User-run training; hyperparameters/checkpoints use inner validation only.

Outer-fold tensors are constructed only after a selection is frozen. We keep
the selected inner-training model (no refit using validation labels), so all
four methods receive exactly the same training examples and normalization.
Completed folds can be resumed; interrupted folds restart from scratch.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_dataset, prepare_fold, split_train_validation, transform_heldout
from .evaluation import grouped_predictions, metrics, save_prediction_csv, summarize_results
from .models import build_model, model_spec

PRIMARY_MODELS = ("pca_lr", "m2", "m2plus", "cnn_transformer")


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Replace small metadata atomically and clean temporary files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir="/tmp",
                                         prefix=".metadata_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            os.replace(temporary, path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # /tmp may be a different filesystem. This fallback is not atomic;
            # a failed metadata write is reported rather than scored as a run.
            shutil.copyfile(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("protocol", "cache_dir", "run_dir", "data", "models", "training", "baseline"):
        if key not in config:
            raise ValueError(f"Missing config field: {key}")
    training, baseline = config["training"], config["baseline"]
    for key in ("batch_size", "max_epochs", "patience", "cpu_threads"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0 or training["min_delta"] < 0:
        raise ValueError("Invalid optimizer/early-stop setting")
    seeds = training["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training.seeds must contain distinct nonnegative integers")
    if not baseline["pca_components"] or any(type(n) is not int or n < 1 for n in baseline["pca_components"]):
        raise ValueError("baseline.pca_components must be positive integers")
    if not baseline["C"] or any(c <= 0 for c in baseline["C"]):
        raise ValueError("baseline.C must be positive")
    for name in PRIMARY_MODELS[1:]:
        model_spec(name, config["models"])
    if config["data"]["output_size"] != 64 or config["data"]["window_bins"] != [64, 256]:
        raise ValueError("The four-model protocol requires 64/256-bin windows and 64x64 inputs")
    return config


def source_signature(root: Path) -> str:
    paths = sorted((root / "microcture" / "task1").glob("*.py"))
    paths.append(root / "scripts" / "run_task1_comparison.py")
    return json_hash({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths})


def environment_record() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run scripts/check_environment.py or use --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def training_arrays(data: dict, augmentation: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = data["x"], data["y"]
    if augmentation == "reverse":
        # Genome direction reversal: flip both axes, scales and masks together.
        return np.concatenate((x, x[..., ::-1, ::-1]), axis=0), np.concatenate((y, y))
    if augmentation != "none":
        raise ValueError(f"Unknown augmentation: {augmentation}")
    return np.ascontiguousarray(x), y.copy()


def score_samples(data: dict, probabilities: np.ndarray) -> dict:
    grouped = grouped_predictions(data, probabilities)
    return {**metrics(grouped["y"], grouped["probabilities"]),
            "replicate_disagreement": grouped["replicate_disagreement"]}


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # Small (32-token) attention is cheap with the math implementation;
        # avoid fused-backend backward incompatibilities in strict mode.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.mha.set_fastpath_enabled(False)


def predict_neural(model: nn.Module, samples: dict, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(samples["x"]), batch_size):
            x = torch.from_numpy(samples["x"][start:start + batch_size]).to(device)
            outputs.append(model(x).softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def fit_neural(name: str, train: dict, validation: dict, config: dict, seed: int,
               device: torch.device, output: Path) -> tuple[nn.Module, dict, dict]:
    settings = config["training"]
    seed_everything(seed, settings["deterministic_algorithms"])
    model = build_model(name, config["models"]).to(device)
    x, y = training_arrays(train, config["data"]["train_augmentation"])
    counts = np.bincount(y, minlength=3)
    if np.any(counts == 0):
        raise ValueError("Every training subset must contain all three classes")
    weights = torch.tensor(len(y) / (3 * counts), dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=settings["batch_size"], shuffle=True, generator=generator,
                        num_workers=0, pin_memory=device.type == "cuda")
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    checkpoint = output / "best.pt"
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "training_loss", "validation_macro_f1",
                                                     "validation_accuracy", "epoch_seconds", "selected"])
        writer.writeheader()
        for epoch in range(1, settings["max_epochs"] + 1):
            started = time.perf_counter()
            model.train()
            total_loss, total_weight = 0.0, 0.0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(inputs), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                batch_weight = float(weights[labels].sum())
                total_loss += float(loss.detach()) * batch_weight
                total_weight += batch_weight
            probabilities = predict_neural(model, validation, device, settings["batch_size"])
            val = score_samples(validation, probabilities)
            # Structural macro-F1 is the only selection criterion. Ties keep
            # the earliest checkpoint, rather than consulting outer labels.
            selected = val["macro_f1"] > best_score + settings["min_delta"]
            average_loss = total_loss / total_weight
            if selected:
                best_score, best_loss, best_epoch, stale = val["macro_f1"], average_loss, epoch, 0
                torch.save({"state_dict": model.state_dict(), "name": name,
                            "model_config": config["models"], "epoch": epoch, "seed": seed}, checkpoint)
            else:
                stale += 1
            writer.writerow({"epoch": epoch, "training_loss": average_loss,
                             "validation_macro_f1": val["macro_f1"], "validation_accuracy": val["accuracy"],
                             "epoch_seconds": time.perf_counter() - started, "selected": int(selected)})
            stream.flush()
            print(f"    {name} seed={seed} epoch={epoch:03d} loss={average_loss:.4f} "
                  f"inner_macro_F1={val['macro_f1']:.4f} best_epoch={best_epoch}", flush=True)
            if stale >= settings["patience"]:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    validation_score = score_samples(validation, predict_neural(model, validation, device, settings["batch_size"]))
    selection = {"criterion": "inner_validation_structure_macro_f1", "selected_epoch": best_epoch,
                 "epochs_run": epoch, "best_training_loss": best_loss,
                 "validation_macro_f1": best_score, "refit_on_outer_training": False,
                 "class_weights": weights.cpu().tolist()}
    return model, validation_score, selection


def fit_pca(train: dict, validation: dict, config: dict, output: Path) -> tuple[Pipeline, dict, dict]:
    settings = config["baseline"]
    x, y = training_arrays(train, config["data"]["train_augmentation"])
    x = x.reshape(len(x), -1)
    vx = validation["x"].reshape(len(validation["x"]), -1)
    scaler = StandardScaler().fit(x)
    scaled, validation_scaled = scaler.transform(x), scaler.transform(vx)
    best, best_score, search = None, -1.0, []
    for components in settings["pca_components"]:
        if components >= min(scaled.shape):
            raise ValueError(f"PCA components={components} too large for this training fold")
        pca = PCA(n_components=components, svd_solver="randomized", random_state=settings["seed"])
        projected = pca.fit_transform(scaled)
        validation_projected = pca.transform(validation_scaled)
        for c in settings["C"]:
            classifier = LogisticRegression(C=c, solver="lbfgs", class_weight="balanced",
                                            max_iter=settings["max_iter"], tol=settings["tol"],
                                            random_state=settings["seed"])
            # A nonconverged candidate must not silently enter the comparison.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(projected, y)
            if not np.array_equal(classifier.classes_, [0, 1, 2]):
                raise ValueError("Unexpected baseline class ordering")
            score = score_samples(validation, classifier.predict_proba(validation_projected))
            search.append({"pca_components": components, "C": c, "macro_f1": score["macro_f1"]})
            if score["macro_f1"] > best_score + 1e-12:
                best_score = score["macro_f1"]
                best = (pca, classifier, score, components, c)
            print(f"    pca_lr components={components} C={c:g} inner_macro_F1={score['macro_f1']:.4f}", flush=True)
    pca, classifier, score, components, c = best
    pipeline = Pipeline([("scaler", scaler), ("pca", pca), ("classifier", classifier)])
    joblib.dump(pipeline, output / "model.joblib")
    selection = {"criterion": "inner_validation_structure_macro_f1", "pca_components": components,
                 "C": c, "validation_macro_f1": best_score, "search": search,
                 "refit_on_outer_training": False, "class_weight": "balanced",
                 "fit_scope": "scaler, PCA and logistic regression fitted on inner training only"}
    return pipeline, score, selection


@contextmanager
def run_lock(run_dir: Path):
    path = run_dir / ".training.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process is using this run directory; run models sequentially.") from exc
        try:
            yield
        finally:
            path.unlink(missing_ok=True)


def publish_fold(source: Path, destination: Path) -> None:
    """Commit one finished fold, keeping one-off staging files under /tmp."""
    try:
        os.replace(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        destination.mkdir()
        try:
            for path in source.iterdir():
                if path.name != "metrics.json":
                    shutil.copy2(path, destination / path.name)
            # The completion marker is last when cross-filesystem moves cannot
            # be atomic. Concurrent training is prevented by the run lock.
            write_json(destination / "metrics.json", json.loads((source / "metrics.json").read_text(encoding="utf-8")))
        except BaseException:
            shutil.rmtree(destination)  # Only this invocation's unfinished fold.
            raise


def run_training(config: dict, root: Path, cache_dir: Path, run_dir: Path, model_name: str,
                 fold_numbers: list[int] | None = None, requested_device: str = "auto") -> dict:
    dataset = load_dataset(cache_dir)
    if dataset["config"] != config["data"]:
        raise ValueError("Config differs from the prepared cache protocol; use its exact data settings or a new cache")
    models = list(PRIMARY_MODELS) if model_name == "all" else [model_name]
    if any(name not in PRIMARY_MODELS for name in models):
        raise ValueError("Choose pca_lr, m2, m2plus, cnn_transformer or all")
    device = resolve_device(requested_device)
    # Required by CUDA deterministic matrix multiplication; set before model use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["training"]["cpu_threads"])
    seeds_by_model = {name: ([config["baseline"]["seed"]] if name == "pca_lr"
                            else config["training"]["seeds"]) for name in PRIMARY_MODELS}
    fold_ids = [fold["fold"] for fold in dataset["folds"]]
    selected_folds = fold_ids if fold_numbers is None else fold_numbers
    if not selected_folds or len(set(selected_folds)) != len(selected_folds) or not set(selected_folds) <= set(fold_ids):
        raise ValueError(f"--fold must contain distinct fold numbers from {fold_ids}")
    manifest = {"protocol": config["protocol"], "dataset_signature": dataset["signature"],
                "source_signature": source_signature(root), "config": config,
                "models": list(PRIMARY_MODELS), "seeds": config["training"]["seeds"],
                "model_seeds": seeds_by_model, "folds": fold_ids, "n_structures": len(dataset["y"]),
                "label_order": dataset["label_order"], "environment": environment_record(),
                "evaluation_scope": dataset["metadata"]["evaluation_scope"],
                "selection_protocol": "inner grouped holdout selects parameters/checkpoint; outer fold scored once; no refit",
                "requested_device": requested_device}
    manifest["run_signature"] = json_hash(manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir), threadpool_limits(limits=config["training"]["cpu_threads"]):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("Run config/source/environment changed. Keep this evidence and choose a new --run-dir.")
        else:
            write_json(manifest_path, manifest)
        for outer in dataset["folds"]:
            fold = outer["fold"]
            if fold not in selected_folds:
                continue
            todo = []
            for name in models:
                for seed in seeds_by_model[name]:
                    destination = run_dir / "runs" / name / f"seed_{seed}" / f"fold_{fold}"
                    if destination.exists():
                        result = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                        if result.get("run_signature") != manifest["run_signature"]:
                            raise ValueError(f"Existing fold has a different run signature: {destination}")
                        print(f"Skip completed {name} seed={seed} fold={fold}", flush=True)
                    else:
                        todo.append((name, seed, destination))
            if not todo:
                continue
            print(f"Preparing train-only transforms for outer fold {fold}", flush=True)
            fold_path = run_dir / "folds" / f"fold_{fold}.json"
            if fold_path.exists():
                stored = json.loads(fold_path.read_text(encoding="utf-8"))
                train_ids, validation_ids = stored["train_ids"], stored["validation_ids"]
                if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != set(outer["train_ids"]):
                    raise ValueError("Stored inner split does not partition this outer training fold")
            else:
                train_ids, validation_ids = split_train_validation(dataset, outer["train_ids"], config["data"],
                                                                   config["training"]["inner_split_seed"] + fold)
            train, validation, preprocessing = prepare_fold(dataset, train_ids, validation_ids, config["data"])
            fold_metadata = {"fold": fold, "train_ids": train_ids, "validation_ids": validation_ids,
                             "test_ids": outer["test_ids"], "preprocessing": preprocessing}
            if fold_path.exists() and json.loads(fold_path.read_text(encoding="utf-8")) != fold_metadata:
                raise ValueError("Stored inner split/preprocessing differs from the current fold")
            write_json(fold_path, fold_metadata)
            for name, seed, destination in todo:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Only finished fold artifacts enter the evidence directory.
                with tempfile.TemporaryDirectory(prefix=f"microcture_task1_fold_{fold}_", dir="/tmp") as temporary:
                    output = Path(temporary)
                    print(f"Start {name} seed={seed} fold={fold} device={'cpu' if name == 'pca_lr' else device}", flush=True)
                    started = time.perf_counter()
                    if name == "pca_lr":
                        model, validation_score, selection = fit_pca(train, validation, config, output)
                        spec = {"name": name, "input_shape": [2, 2, 64, 64], "parameters": None,
                                "selected_pca_components": selection["pca_components"],
                                "logistic_coefficients": int(model[-1].coef_.size + model[-1].intercept_.size)}
                    else:
                        model, validation_score, selection = fit_neural(name, train, validation, config, seed, device, output)
                        spec = model_spec(name, config["models"])
                    fit_seconds = time.perf_counter() - started
                    write_json(output / "selection.json", selection)
                    # No outer tensor/probability has been consulted above.
                    test = transform_heldout(dataset, outer["test_ids"], preprocessing)
                    if name == "pca_lr":
                        probabilities = model.predict_proba(test["x"].reshape(len(test["x"]), -1))
                    else:
                        probabilities = predict_neural(model, test, device, config["training"]["batch_size"])
                    test_score = score_samples(test, probabilities)
                    save_prediction_csv(output / "predictions.csv", test, probabilities)
                    result = {"model": name, "seed": seed, "fold": fold, "protocol": config["protocol"],
                              "dataset_signature": dataset["signature"], "run_signature": manifest["run_signature"],
                              "source_signature": manifest["source_signature"],
                              "preprocessing_signature": preprocessing["signature"], "model_spec": spec,
                              "validation": validation_score, "test": test_score, "selection": selection,
                              "fit_seconds": fit_seconds, "runtime_seconds": time.perf_counter() - started,
                              "device": "cpu" if name == "pca_lr" else str(device),
                              "parameter_count": spec["parameters"], "train_structures": len(train_ids),
                              "validation_structures": len(validation_ids), "test_structures": len(outer["test_ids"]),
                              "training_replicate_examples": len(train["y"]),
                              "training_augmentation": config["data"]["train_augmentation"]}
                    write_json(output / "metrics.json", result)
                    publish_fold(output, destination)
                    print(f"Finished {name} seed={seed} fold={fold} outer_macro_F1={test_score['macro_f1']:.4f} "
                          f"fit_seconds={fit_seconds:.1f}", flush=True)
                    del model, test
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        return summarize_results(run_dir)
