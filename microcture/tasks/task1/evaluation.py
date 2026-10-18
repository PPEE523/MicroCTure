"""Task 1 structure-level metrics, OOF verification and comparison summaries.

Replicate probabilities are averaged once per structure. Failed and constant
models remain visible; only complete registered OOF sets receive aggregate scores."""
from __future__ import annotations

from collections import defaultdict
import csv
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from microcture.layout import PROJECT_ROOT, resolve


# Common structure-level evaluation.
CLASS_NAMES = ("OPCID", "CHIN", "CHID")


_PROBABILITY_COLUMNS = tuple(f"p_{name}" for name in CLASS_NAMES)


def _probabilities(values: Any, n: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (n, len(CLASS_NAMES)):
        raise ValueError(f"Expected probabilities with shape ({n}, 3), got {probabilities.shape}.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must all be finite.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Each probability row must sum to 1.")
    return probabilities


def _integer_array(values: Any, name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (n is not None and len(array) != n):
        raise ValueError(f"{name} must be a one-dimensional array with {n} entries.")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integers.")
    return array.astype(np.int64, copy=False)


def grouped_predictions(data: dict, probabilities: Any) -> dict:
    """Average replicate probabilities once per annotated structure.

    Required data keys are ``structure_id``, ``y`` and ``group_id``. An optional
    ``replicate`` array gives names for the per-replicate records in CSV output.
    First occurrence determines structure order. Labels and leakage groups must
    agree across the replicates belonging to one structure.
    """
    structure_ids = np.asarray(data["structure_id"], dtype=str)
    if structure_ids.ndim != 1 or len(structure_ids) == 0:
        raise ValueError("structure_id must be a nonempty one-dimensional array.")
    if np.any(structure_ids == ""):
        raise ValueError("structure_id must not be empty.")
    n = len(structure_ids)
    y = _integer_array(data["y"], "y", n)
    if np.any((y < 0) | (y >= 3)):
        raise ValueError("y must use the fixed class indices 0, 1, 2.")
    group_ids = _integer_array(data["group_id"], "group_id", n)
    probabilities = _probabilities(probabilities, n)
    replicates = np.asarray(data.get("replicate", np.arange(n).astype(str)), dtype=str)
    if replicates.ndim != 1 or len(replicates) != n:
        raise ValueError("replicate must contain one entry per input row.")

    indices: dict[str, list[int]] = defaultdict(list)
    for index, structure_id in enumerate(structure_ids):
        indices[str(structure_id)].append(index)
    labels, groups, means, counts, disagree, replicate_records = [], [], [], [], [], []
    for structure_id, rows in indices.items():
        if len(np.unique(y[rows])) != 1:
            raise ValueError(f"Replicates of {structure_id!r} have inconsistent labels.")
        if len(np.unique(group_ids[rows])) != 1:
            raise ValueError(f"Replicates of {structure_id!r} have inconsistent group_id.")
        labels.append(int(y[rows[0]]))
        groups.append(int(group_ids[rows[0]]))
        means.append(probabilities[rows].mean(axis=0))
        counts.append(len(rows))
        row_predictions = probabilities[rows].argmax(axis=1)
        disagree.append(len(np.unique(row_predictions)) > 1)
        replicate_records.append([
            {"replicate": str(replicates[row]), "probabilities": probabilities[row].tolist(),
             "predicted": int(probabilities[row].argmax())}
            for row in rows
        ])
    means_array = np.asarray(means, dtype=np.float64)
    multiple = int(np.count_nonzero(np.asarray(counts) > 1))
    disagreements = int(np.count_nonzero(disagree))
    return {
        "structure_id": np.asarray(list(indices), dtype=str),
        "y": np.asarray(labels, dtype=np.int64),
        "probabilities": means_array,
        "predicted": means_array.argmax(axis=1),
        "group_id": np.asarray(groups, dtype=np.int64),
        "replicate_count": np.asarray(counts, dtype=np.int64),
        "replicate_disagree": np.asarray(disagree, dtype=bool),
        "replicate_probabilities": replicate_records,
        "replicate_disagreement": {
            "total_structures": len(indices),
            "multi_replicate_structures": multiple,
            "disagreement_count": disagreements,
            "disagreement_rate": disagreements / multiple if multiple else None,
        },
    }


def metrics(y: Any, probabilities: Any) -> dict:
    """Score fixed OPCID, CHIN, CHID indices; absent-class recall/F1 are zero."""
    labels = _integer_array(y, "y")
    if len(labels) == 0:
        raise ValueError("Cannot score an empty set of structures.")
    if np.any((labels < 0) | (labels >= 3)):
        raise ValueError("y must use the fixed class indices 0, 1, 2.")
    probabilities = _probabilities(probabilities, len(labels))
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positives = np.diag(confusion)
    precision = np.divide(true_positives, predicted_count, out=np.zeros(3), where=predicted_count > 0)
    recall = np.divide(true_positives, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {
        "n_structures": len(labels),
        "class_names": list(CLASS_NAMES),
        "accuracy": float(np.mean(labels == predicted)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "balanced_accuracy_definition": "mean recall over fixed three classes; absent-class recall is zero",
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist(),
    }


def save_prediction_csv(path: Path | str, data: dict, probabilities: Any) -> dict:
    """Write each structure once and retain its replicate probabilities."""
    result = grouped_predictions(data, probabilities)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["structure_id", "y_true", "y_pred", "group_id", *_PROBABILITY_COLUMNS,
               "replicate_count", "replicate_disagree", "replicate_probabilities_json"]
    # An existing fold is evidence; callers must choose a new run/fold directory.
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, structure_id in enumerate(result["structure_id"]):
            row = {"structure_id": str(structure_id), "y_true": int(result["y"][i]),
                   "y_pred": int(result["predicted"][i]), "group_id": int(result["group_id"][i]),
                   "replicate_count": int(result["replicate_count"][i]),
                   "replicate_disagree": int(result["replicate_disagree"][i]),
                   "replicate_probabilities_json": json.dumps(result["replicate_probabilities"][i], ensure_ascii=False)}
            row.update({column: float(result["probabilities"][i, j])
                        for j, column in enumerate(_PROBABILITY_COLUMNS)})
            writer.writerow(row)
    return result


def _read_prediction_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"structure_id", "y_true", "y_pred", "group_id", *_PROBABILITY_COLUMNS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prediction file {path} lacks {sorted(required - set(reader.fieldnames or []))}.")
        rows = list(reader)
    ids = np.asarray([row["structure_id"] for row in rows], dtype=str)
    if len(ids) != len(np.unique(ids)):
        raise ValueError(f"Prediction file {path} repeats a structure.")
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray([[float(row[column]) for column in _PROBABILITY_COLUMNS] for row in rows])
    score = metrics(labels, probabilities)
    predictions = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise ValueError(f"Prediction file {path} has inconsistent y_pred.")
    return {"structure_id": ids, "y": labels, "probabilities": probabilities,
            "group_id": np.asarray([int(row["group_id"]) for row in rows], dtype=np.int64),
            "metrics": score, "rows": rows}


def _write_json(path: Path, value: dict) -> None:
    # Only derived report files are replaced; no run evidence or source data are removed.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir="/tmp",
                                         prefix="microcture-summary-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        try:
            temporary.replace(path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # Cross-filesystem fallback is not atomic; this summary is rebuildable.
            shutil.copyfile(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# Full registered comparison and ablation evidence verification.
_EVENTS = ("constant_probability", "constant_centered_logits", "low_centered_logit_variance",
           "single_class_argmax", "majority_concentration")
_FAILED = {"numerical_failure", "execution_failure", "failed", "failure"}
_COLLISION_IDS = {"CHID_18", "CHIN_172", "CHID_25", "CHIN_243"}
_PROJECT_ROOT = PROJECT_ROOT


def _project_path(value: str | Path) -> Path:
    return resolve(_PROJECT_ROOT, value)


def _guard_run(run_dir: Path) -> None:
    """Reject links before reading evidence or replacing any derived output."""
    if any(path.is_symlink() for path in (run_dir, *run_dir.parents)):
        raise ValueError("Formal summaries cannot read or write through a symbolic-link run path")
    for directory, children, files in os.walk(run_dir, followlinks=False):
        for name in (*children, *files):
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"Symbolic links are forbidden inside a formal run: {Path(directory) / name}")
    if (run_dir / "ARTIFACTS.json").exists():
        raise ValueError("A sealed job directory cannot be used as the formal summary root")
    # The runs/ tree legitimately contains seals. Derived oof/ is never a fit
    # destination; refuse accidental placement of a sealed job anywhere in it.
    if (run_dir / "oof").is_dir() and any((run_dir / "oof").rglob("ARTIFACTS.json")):
        raise ValueError("Derived OOF output would overlap an ARTIFACTS-sealed job")


def _guard_derived(path: Path) -> None:
    if any((ancestor / "ARTIFACTS.json").exists() for ancestor in path.parents):
        raise ValueError(f"Derived output cannot replace files inside an ARTIFACTS-sealed job: {path}")


def _write_derived_json(path: Path, value: dict) -> None:
    _guard_derived(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    _guard_derived(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def _describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "population_std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Aggregate values must be finite")
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "population_std": float(array.std(ddof=0)), "min": float(array.min()), "max": float(array.max())}


def _identity(metadata: dict) -> dict[str, dict]:
    structures = metadata.get("structures")
    groups = metadata.get("group_id")
    if not isinstance(structures, list) or not structures or not isinstance(groups, list) or len(groups) != len(structures):
        raise ValueError("Metadata requires nonempty structures and one group_id per structure")
    result = {}
    for row, group in zip(structures, groups):
        sid = str(row["structure_id"])
        if sid in result:
            raise ValueError(f"Metadata repeats structure {sid}")
        name = str(row["type"])
        if name not in CLASS_NAMES:
            raise ValueError(f"Unknown class {name}")
        start, end = int(row["start"]), int(row["end"])
        if end <= start:
            raise ValueError(f"Invalid interval for {sid}")
        result[sid] = {**row, "structure_id": sid, "y": CLASS_NAMES.index(name), "group_id": int(group),
                       "start": start, "end": end, "length_bp": end - start}
    return result


def _fold_ids(manifest: dict, identity: dict[str, dict], run_dir: Path) -> dict[int, dict]:
    stated = manifest.get("fold_ids")
    if not isinstance(stated, dict):
        # Compatibility with independently prepared manifests; matrices are never read.
        frozen = _project_path(manifest["config"]["frozen_run_dir"])
        stated = {str(fold): _read_json(frozen / "folds" / f"fold_{fold}.json")
                  for fold in manifest["config"]["folds"]}
    folds = {}
    owners: dict[str, int] = {}
    group_owners: dict[int, int] = {}
    for key, value in stated.items():
        fold = int(key)
        sections = {}
        for name in ("train_ids", "validation_ids", "test_ids"):
            ids = [str(sid) for sid in value[name]]
            if len(ids) != len(set(ids)) or not set(ids).issubset(identity):
                raise ValueError(f"Fold {fold} has duplicate/unknown {name}")
            sections[name] = ids
        sets = [set(sections[name]) for name in ("train_ids", "validation_ids", "test_ids")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)) or set.union(*sets) != set(identity):
            raise ValueError(f"Fold {fold} does not partition all annotated structures")
        group_sets = [{identity[sid]["group_id"] for sid in subset} for subset in sets]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Fold {fold} splits a spatial group across train/validation/test")
        for sid in sections["test_ids"]:
            group = identity[sid]["group_id"]
            if sid in owners or group in group_owners and group_owners[group] != fold:
                raise ValueError("Frozen outer folds repeat a structure or split a spatial group")
            owners[sid] = fold
            group_owners[group] = fold
        folds[fold] = sections
    if set(owners) != set(identity):
        raise ValueError("Frozen outer folds must cover every annotated structure exactly once")
    return folds


def _same_score(stored: dict, calculated: dict, name: str) -> None:
    for key in ("n_structures", "accuracy", "macro_f1", "balanced_accuracy"):
        if key not in stored or not np.isclose(stored[key], calculated[key], rtol=0, atol=1e-7):
            raise ValueError(f"{name}.{key} differs from structure predictions")
    if stored.get("confusion_matrix") != calculated["confusion_matrix"]:
        raise ValueError(f"{name}.confusion_matrix differs from structure predictions")
    for cls in CLASS_NAMES:
        for key in ("precision", "recall", "f1", "support"):
            if not np.isclose(stored["per_class"][cls][key], calculated["per_class"][cls][key], rtol=0, atol=1e-7):
                raise ValueError(f"{name}.{cls}.{key} differs from structure predictions")


def _history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = _read_json(path).get("epochs")
    if not isinstance(result, list):
        raise ValueError("history.json requires epochs list")
    epochs = [row["epoch"] for row in result]
    if epochs != list(range(len(result))):
        raise ValueError("history epochs must be contiguous starting from zero")
    return result


def _load_job(run_dir: Path, job: dict, manifest: dict, identity: dict, folds: dict) -> dict:
    from .ablation import learning_acceptance
    from .protocol import job_directory

    directory = job_directory(run_dir, job)
    record = {"job": job, "status": "pending", "directory": str(directory.relative_to(run_dir))}
    path = directory / "metrics.json"
    if not path.is_file():
        return record
    value = _read_json(path)
    if value.get("job") != job:
        raise ValueError("Stored job differs from the planned job")
    for name in ("experiment_signature", "source_signature", "dataset_signature"):
        if name not in manifest or value.get(name) != manifest[name]:
            raise ValueError(f"{name} differs from manifest")
    if not value.get("preprocessing_signature"):
        raise ValueError("A preprocessing signature is required")
    status = value.get("status")
    if status not in {"completed", *_FAILED}:
        raise ValueError(f"Unknown stored status {status!r}")
    record.update(status="execution_failure" if status == "execution_failure" else "numerical_failure" if status in _FAILED else "completed", metrics=value)
    if job["kind"] == "neural":
        history = _history(directory / "history.json")
        record["history"] = history
        if status == "completed":
            expected_epochs = int(manifest["config"].get("training", {}).get("epochs", 40))
            if len(history) != expected_epochs + 1 or value.get("epochs") != expected_epochs:
                raise ValueError("Completed neural fit lacks the preregistered full epoch trajectory")
            if not np.isclose(value["final_train_ce"], history[-1]["weighted_train_ce"], atol=1e-10, rtol=0):
                raise ValueError("Final train CE differs from trajectory")
            if not np.isclose(value["initial_train_ce"], history[0]["weighted_train_ce"], atol=1e-10, rtol=0):
                raise ValueError("Initial train CE differs from trajectory")
            for name in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256"):
                if not value.get(name):
                    raise ValueError(f"Completed neural fit lacks {name}")
            for view in ("train", "validation"):
                if value.get(f"{view}_diagnostics") != history[-1].get(f"{view}_diagnostics"):
                    raise ValueError(f"Final {view} diagnostics differ from the full trajectory")
            recalculated = learning_acceptance(history, manifest["config"])
            if value.get("acceptance") != recalculated:
                raise ValueError("Stored learning acceptance differs from recomputation on the full trajectory")
    if status != "completed":
        return record
    predictions = _read_prediction_csv(directory / "outer_predictions.csv")
    expected_ids = set(folds[job["fold"]]["test_ids"])
    if set(predictions["structure_id"]) != expected_ids:
        raise ValueError("Outer predictions must match exactly the frozen test structure IDs")
    for sid, label, group in zip(predictions["structure_id"], predictions["y"], predictions["group_id"]):
        if (int(label), int(group)) != (identity[str(sid)]["y"], identity[str(sid)]["group_id"]):
            raise ValueError(f"Outer identity/label/group differs from frozen metadata for {sid}")
    for field, name in (("train_structures", "train_ids"), ("validation_structures", "validation_ids"),
                        ("test_structures", "test_ids")):
        if value.get(field) != len(folds[job["fold"]][name]):
            raise ValueError(f"{field} differs from frozen fold")
    _same_score(value["outer_metrics"], predictions["metrics"], "outer_metrics")
    if job["kind"] == "neural":
        diagnostics = value["outer_diagnostics"]
        _same_score(diagnostics["metrics"], predictions["metrics"], "outer_diagnostics.metrics")
        expected_counts = dict(zip(CLASS_NAMES, np.bincount(predictions["probabilities"].argmax(axis=1), minlength=3).tolist()))
        if diagnostics.get("prediction_counts") != expected_counts:
            raise ValueError("outer_diagnostics prediction counts differ from CSV")
        if diagnostics.get("num_predicted_classes") != sum(count > 0 for count in expected_counts.values()):
            raise ValueError("outer_diagnostics predicted-class count differs from CSV")
        majority = max(expected_counts.values()) / len(predictions["y"])
        if not np.isclose(diagnostics.get("majority_prediction_fraction", -1), majority, atol=1e-10, rtol=0):
            raise ValueError("outer_diagnostics majority fraction differs from CSV")
        distribution = diagnostics.get("probabilities", {})
        expected_range = float(np.ptp(predictions["probabilities"], axis=0).max())
        if "max_coordinate_range" not in distribution or not np.isclose(distribution["max_coordinate_range"], expected_range, atol=1e-10, rtol=0):
            raise ValueError("outer_diagnostics probability range differs from CSV")
        expected_variance = float(np.var(predictions["probabilities"], axis=0, ddof=0).mean())
        if "probability_between_structure_var" in diagnostics and not np.isclose(diagnostics["probability_between_structure_var"], expected_variance, atol=1e-10, rtol=0):
            raise ValueError("outer_diagnostics probability variance differs from CSV")
    runtime = value.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or not np.isfinite(runtime) or runtime < 0:
        raise ValueError("runtime_seconds must be finite and nonnegative")
    record["predictions"] = predictions
    return record


def _events(diagnostic: dict, settings: dict) -> dict[str, bool]:
    epsilon = settings.get("constant_epsilon", 1e-6)
    def constant(name: str, fallback: str) -> bool:
        distribution = diagnostic.get(name, {})
        if "max_coordinate_range" in distribution:
            return bool(float(distribution["max_coordinate_range"]) <= epsilon)
        return bool(diagnostic.get(fallback, False))
    variance = diagnostic.get("centered_logit_between_structure_var")
    return {
        "constant_probability": constant("probabilities", "probabilities_constant"),
        "constant_centered_logits": constant("centered_logits", "centered_logits_constant"),
        "low_centered_logit_variance": variance is not None and float(variance) <= settings.get("logit_variance_threshold", 1e-6),
        "single_class_argmax": diagnostic.get("num_predicted_classes") == 1,
        "majority_concentration": diagnostic.get("majority_prediction_fraction", 0) >= settings.get("majority_warning_fraction", .95),
    }


def _event_summary(records: list[dict], view: str, settings: dict) -> dict:
    initial, epoch_events, terminal, ever = [], [], [], []
    observed_cases = 0
    for record in records:
        value = record.get("metrics", {})
        if view == "outer":
            if record["status"] != "completed" or "outer_diagnostics" not in value:
                continue
            event = _events(value["outer_diagnostics"], settings)
            epoch_events.append(event)
            terminal.append(event)
            ever.append(event)
            observed_cases += 1
            continue
        history = record.get("history", [])
        present = [row for row in history if isinstance(row.get(f"{view}_diagnostics"), dict)]
        if not present:
            continue
        observed_cases += 1
        initial.extend(_events(row[f"{view}_diagnostics"], settings) for row in present if row["epoch"] == 0)
        trained = [_events(row[f"{view}_diagnostics"], settings) for row in present if row["epoch"] > 0]
        epoch_events.extend(trained)
        ever.append({key: any(event[key] for event in trained) for key in _EVENTS})
        if record["status"] == "completed":
            terminal.append(_events(value[f"{view}_diagnostics"], settings))
    result = {"observed_cases": observed_cases, "observed_events": len(epoch_events), "final_observed_cases": len(terminal),
              "measurement": "one endpoint per completed job" if view == "outer" else "epochs > 0; epoch zero is separate; failed traces remain included",
              "initial_observed_cases": len(initial)}
    for key in _EVENTS:
        count = sum(event[key] for event in epoch_events)
        result[key] = {"events": count, "event_rate": count / len(epoch_events) if epoch_events else None,
                       "ever_cases": sum(event[key] for event in ever), "ever_rate": sum(event[key] for event in ever) / len(ever) if ever else None,
                       "final_cases": sum(event[key] for event in terminal), "final_rate": sum(event[key] for event in terminal) / len(terminal) if terminal else None,
                       "initial_cases": sum(event[key] for event in initial)}
    return result


def _neural_summary(records: list[dict], config: dict) -> dict:
    completed = [record for record in records if record["status"] == "completed"]
    settings = config.get("acceptance", {})
    window = settings.get("window", 10)
    epsilon = settings.get("gradient_norm_epsilon", 1e-10)
    losses, cvs, regressions = [], [], []
    gradients = {name: {"observed_cases": 0, "effective_cases": 0, "tail_mean_norm_min": []}
                 for name in ("total", "encoder", "head")}
    for record in completed:
        value = record["metrics"]
        losses.append(float(value["final_train_ce"]))
        history = record["history"]
        trained = [row for row in history if row["epoch"] > 0]
        tail = trained[-window:]
        tail_losses = np.asarray([row["weighted_train_ce"] for row in tail], dtype=float)
        if len(tail_losses) == window and np.isfinite(tail_losses).all():
            cvs.append(float(tail_losses.std(ddof=0) / max(abs(tail_losses.mean()), 1e-12)))
            all_losses = np.asarray([row["weighted_train_ce"] for row in trained], dtype=float)
            best = min(float(all_losses[i:i + window].mean()) for i in range(len(all_losses) - window + 1))
            regressions.append(float((tail_losses.mean() - best) / max(best, 1e-12)))
        for name in gradients:
            entries = [row.get("gradient_norms", {}).get(name, {}) for row in tail]
            present = len(entries) == window and all("mean" in entry for entry in entries)
            if not present:
                continue
            target = gradients[name]
            target["observed_cases"] += 1
            effective = all(entry.get("finite") is True and np.isfinite(entry["mean"]) and entry["mean"] > epsilon
                            and entry.get("nonzero_fraction", 0) > 0 for entry in entries)
            target["effective_cases"] += int(effective)
            if all(np.isfinite(entry["mean"]) for entry in entries):
                target["tail_mean_norm_min"].append(float(min(entry["mean"] for entry in entries)))
    for value in gradients.values():
        value["tail_mean_norm_min"] = _describe(value["tail_mean_norm_min"])
    return {
        "expected_cases": len(records), "completed_cases": len(completed),
        "numerical_failure_cases": sum(record["status"] == "numerical_failure" for record in records),
        "execution_failure_cases": sum(record["status"] == "execution_failure" for record in records),
        "invalid_cases": sum(record["status"] == "invalid" for record in records),
        "pending_cases": sum(record["status"] == "pending" for record in records),
        "strict_learning_passes": sum((record["metrics"].get("acceptance") or {}).get("accepted") is True for record in completed),
        "strict_learning_pass_rate_completed": sum((record["metrics"].get("acceptance") or {}).get("accepted") is True for record in completed) / len(completed) if completed else None,
        "last_train_ce": _describe(losses), "tail_ce_coefficient_of_variation": _describe(cvs),
        "terminal_relative_loss_regression": _describe(regressions), "effective_gradients": gradients,
        "prediction_events": {view: _event_summary(records, view, settings) for view in ("train", "validation", "outer")},
        "denominator_note": "Each completed fold/seed is one case. Missing and numerical failures are explicit; they are not scored as zero or removed from expected coverage.",
    }


def _baseline_outer_summary(records: list[dict], config: dict) -> dict:
    """Probability-only endpoint events; traditional baselines have no logits probes."""
    settings = config.get("acceptance", {})
    completed = [record for record in records if record["status"] == "completed"]
    counts = {"constant_probability": 0, "single_class_argmax": 0, "majority_concentration": 0}
    for record in completed:
        probabilities = record["predictions"]["probabilities"]
        predicted_counts = np.bincount(probabilities.argmax(axis=1), minlength=3)
        counts["constant_probability"] += int(float(np.ptp(probabilities, axis=0).max()) <= settings.get("constant_epsilon", 1e-6))
        counts["single_class_argmax"] += int(np.count_nonzero(predicted_counts) == 1)
        counts["majority_concentration"] += int(predicted_counts.max() / len(probabilities) >= settings.get("majority_warning_fraction", .95))
    return {"expected_cases": len(records), "observed_cases": len(completed),
            "measurement": "one saved outer endpoint per completed fold; probability-only checks from CSV",
            "denominator_note": "One deterministic seed. Baseline logits, head activity and gradients are not measured.",
            **{name: {"cases": count, "rate": count / len(completed) if completed else None,
                      "denominator": len(completed)} for name, count in counts.items()}}


def _seed_agreement(seeds: list[int], oof: dict[int, dict], identity: dict) -> dict | None:
    if len(seeds) < 2 or any(seed not in oof for seed in seeds):
        return None
    ids = sorted(identity)
    labels = np.asarray([[int(oof[seed]["by_id"][sid]["y_pred"]) for sid in ids] for seed in seeds])
    pairwise = [float(np.mean(labels[i] == labels[j])) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    class_agreement = {}
    for cls in CLASS_NAMES:
        mask = np.asarray([identity[sid]["type"] == cls for sid in ids])
        class_agreement[cls] = {"support": int(mask.sum()), "unanimous_rate": float(np.all(labels[:, mask] == labels[0, mask], axis=0).mean()) if mask.any() else None}
    return {"n_structures": len(ids), "n_seeds": len(seeds),
            "unanimous_rate": float(np.all(labels == labels[0], axis=0).mean()),
            "pairwise_agreement_mean": float(np.mean(pairwise)), "pairwise_agreement": pairwise,
            "per_class": class_agreement}


def _error_rows(rows: list[dict], identity: dict, model: str, condition: str, seed: int) -> list[dict]:
    output = []
    for row in rows:
        sid = row["structure_id"]
        true, predicted = int(row["y_true"]), int(row["y_pred"])
        if {true, predicted} != {1, 2} and sid not in _COLLISION_IDS:
            continue
        source = identity[sid]
        output.append({"model": model, "condition": condition, "seed": seed, "fold": row["fold"],
                       "structure_id": sid, "type": source["type"], "predicted_type": CLASS_NAMES[predicted],
                       "start": source["start"], "end": source["end"], "length_bp": source["length_bp"],
                       "group_id": source["group_id"], "CHIN_CHID_confusion": {true, predicted} == {1, 2},
                       "known_same_center_pair": sid in _COLLISION_IDS,
                       **{f"p_{cls}": float(row[f"p_{cls}"]) for cls in CLASS_NAMES}})
    return output


def _pairwise(records: dict[tuple, dict], jobs: list[dict], complete_oof: dict[tuple, dict]) -> dict:
    requests = []
    neural = [job for job in jobs if job["kind"] == "neural"]
    for model in dict.fromkeys(job["model"] for job in neural):
        for alternative in ("I0", "G0"):
            pairs = [(job, {**job, "condition": alternative, "input_variant": alternative})
                     for job in neural if job["model"] == model and job["condition"] == "I3"]
            requests.append((f"{model}:I3-{alternative}", pairs, model, "I3", model, alternative))
    for condition in dict.fromkeys(job["condition"] for job in neural):
        pairs = [(job, {**job, "model": "m2plus", "backbone": "m2plus"})
                 for job in neural if job["model"] == "cnn_transformer" and job["condition"] == condition]
        requests.append((f"{condition}:cnn_transformer-m2plus", pairs, "cnn_transformer", condition, "m2plus", condition))
    summaries = {}
    all_rows = []
    for name, pairs, model_a, condition_a, model_b, condition_b in requests:
        if not pairs:
            continue
        rows = []
        for first, second in pairs:
            a = records.get((first["condition"], first["model"], first["seed"], first["fold"]))
            b = records.get((second["condition"], second["model"], second["seed"], second["fold"]))
            if a is None or b is None:
                continue
            if a["status"] != "completed" or b["status"] != "completed":
                continue
            score_a, score_b = a["predictions"]["metrics"], b["predictions"]["metrics"]
            rows.append({"comparison": name, "seed": first["seed"], "fold": first["fold"],
                         "macro_f1_delta": score_a["macro_f1"] - score_b["macro_f1"],
                         "CHID_f1_delta": score_a["per_class"]["CHID"]["f1"] - score_b["per_class"]["CHID"]["f1"],
                         "balanced_accuracy_delta": score_a["balanced_accuracy"] - score_b["balanced_accuracy"]})
        complete = len(rows) == len(pairs)
        oof_rows = []
        seeds = sorted({pair[0]["seed"] for pair in pairs})
        for seed in seeds:
            a = complete_oof.get((condition_a, model_a, seed))
            b = complete_oof.get((condition_b, model_b, seed))
            if a is None or b is None:
                continue
            oof_rows.append({"seed": seed, "macro_f1_delta": a["metrics"]["macro_f1"] - b["metrics"]["macro_f1"],
                             "CHID_f1_delta": a["metrics"]["per_class"]["CHID"]["f1"] - b["metrics"]["per_class"]["CHID"]["f1"]})
        summaries[name] = {"status": "complete" if complete else "pending", "expected_pairs": len(pairs),
                           "completed_pairs": len(rows), "fold_seed_deltas": rows,
                           "macro_f1_delta": _describe([row["macro_f1_delta"] for row in rows]) if complete else None,
                           "CHID_f1_delta": _describe([row["CHID_f1_delta"] for row in rows]) if complete else None,
                           "complete_seed_oof_deltas": oof_rows,
                           "interpretation": "Paired development-set contrasts, not independent datasets or blind-test confidence intervals."}
        all_rows.extend(rows)
    return {"comparisons": summaries, "rows": all_rows}


def summarize_formal(run_dir: Path | str, metadata: dict | None = None) -> dict:
    """Verify stored predictions and summarize only complete preregistered OOF runs.

    This function reads small metadata/trajectories and saved CSVs. It does not
    load matrices/checkpoints, call models or fit any transformations.
    """
    from .protocol import json_signature, planned_jobs

    run_dir = resolve(_PROJECT_ROOT, run_dir)
    _guard_run(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    canonical_signature = json_signature({key: value for key, value in manifest.items() if key != "experiment_signature"})
    if manifest.get("experiment_signature") != canonical_signature:
        raise ValueError("Formal manifest experiment_signature differs from its canonical contents")
    config = manifest["config"]
    if metadata is None:
        metadata = _read_json(_project_path(config["cache_dir"]) / "metadata.json")
    if metadata.get("signature") is not None and metadata["signature"] != manifest.get("dataset_signature"):
        raise ValueError("Frozen metadata signature differs from formal manifest")
    identity = _identity(metadata)
    folds = _fold_ids(manifest, identity, run_dir)
    jobs = planned_jobs(config)
    if not jobs:
        raise ValueError("Formal config must declare jobs")
    errors, records, job_rows = [], {}, []
    for job in jobs:
        key = (job["condition"], job["model"], job["seed"], job["fold"])
        if key in records:
            raise ValueError("Protocol repeats a planned job")
        try:
            if job["fold"] not in folds:
                raise ValueError("Planned job uses an unfrozen fold")
            record = _load_job(run_dir, job, manifest, identity, folds)
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            record = {"job": job, "status": "invalid", "reason": str(error)}
            errors.append({"job": job, "reason": str(error)})
        records[key] = record
    # Input ablations share exact initial state and minibatch order. All jobs
    # within one fold share the fitted transform and class weights.
    consistency = []
    by_fold: dict[int, list[dict]] = defaultdict(list)
    by_initialization: dict[tuple, list[dict]] = defaultdict(list)
    for record in records.values():
        if record["status"] != "completed":
            continue
        job = record["job"]
        by_fold[job["fold"]].append(record)
        if job["kind"] == "neural":
            by_initialization[(job["model"], job["seed"], job["fold"])].append(record)
    for fold, items in by_fold.items():
        for field in ("preprocessing_signature", "class_weights"):
            values = {json.dumps(item["metrics"].get(field), sort_keys=True) for item in items if field in item["metrics"]}
            if len(values) > 1:
                reason = f"Different {field} among fits of one fold"
                errors.append({"fold": fold, "reason": reason})
                for item in items:
                    item.update(status="invalid", reason=reason)
        consistency.append({"fold": fold, "completed_jobs": len(items)})
    for key, items in by_initialization.items():
        for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256"):
            if len({item["metrics"][field] for item in items}) > 1:
                reason = f"Input ablations have different {field}"
                errors.append({"model_seed_fold": list(key), "reason": reason})
                for item in items:
                    item.update(status="invalid", reason=reason)
    comparisons, details, complete_oof, confusion_rows, error_seed_rows = [], {}, {}, [], []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records.values():
        job = record["job"]
        grouped[(job["condition"], job["model"])].append(record)
        value = record.get("metrics", {})
        row = {"condition": job["condition"], "model": job["model"], "seed": job["seed"], "fold": job["fold"],
               "kind": job["kind"], "status": record["status"], "reason": record.get("reason"),
               "runtime_seconds": value.get("runtime_seconds"), "last_train_ce": value.get("final_train_ce"),
               "parameter_count": value.get("parameter_count", value.get("parameter_spec", {}).get("parameters")),
               "learning_pass": (value.get("acceptance") or {}).get("accepted")}
        if record["status"] == "completed":
            score = record["predictions"]["metrics"]
            predicted_counts = np.bincount(record["predictions"]["probabilities"].argmax(axis=1), minlength=3)
            row.update(macro_f1=score["macro_f1"], CHID_f1=score["per_class"]["CHID"]["f1"],
                       outer_prediction_counts=dict(zip(CLASS_NAMES, predicted_counts.tolist())))
        job_rows.append(row)
    for (condition, model), items in grouped.items():
        seeds = sorted({record["job"]["seed"] for record in items})
        seed_detail, seed_oof = {}, {}
        for seed in seeds:
            seed_records = [record for record in items if record["job"]["seed"] == seed]
            missing = [record["job"]["fold"] for record in seed_records if record["status"] == "pending"]
            failed = [record["job"]["fold"] for record in seed_records if record["status"] == "numerical_failure"]
            execution_failed = [record["job"]["fold"] for record in seed_records if record["status"] == "execution_failure"]
            invalid = [record["job"]["fold"] for record in seed_records if record["status"] == "invalid"]
            destination = run_dir / "oof" / condition / model / f"seed_{seed}"
            provenance = {"condition": condition, "model": model, "seed": seed,
                          "experiment_signature": manifest["experiment_signature"],
                          "source_signature": manifest["source_signature"], "dataset_signature": manifest["dataset_signature"],
                          "expected_structures": len(identity), "expected_folds": sorted(record["job"]["fold"] for record in seed_records)}
            if missing or failed or execution_failed or invalid:
                seed_detail[str(seed)] = {"status": "invalid" if invalid else "failed" if failed or execution_failed else "pending",
                                          "pending_folds": missing, "numerical_failure_folds": failed,
                                          "execution_failure_folds": execution_failed, "invalid_folds": invalid}
                _write_derived_json(destination / "status.json", {**provenance, **seed_detail[str(seed)],
                                                                  "oof_current": False,
                                                                  "previous_numeric_artifacts_retained": (destination / "metrics.json").exists() or (destination / "predictions.csv").exists(),
                                                                  "interpretation": "Any previous numeric OOF files are retained historical evidence and are not a valid current OOF result."})
                continue
            ids = [str(sid) for record in seed_records for sid in record["predictions"]["structure_id"]]
            if len(ids) != len(set(ids)) or set(ids) != set(identity):
                errors.append({"condition": condition, "model": model, "seed": seed, "reason": "OOF must cover each annotated structure exactly once"})
                seed_detail[str(seed)] = {"status": "invalid"}
                _write_derived_json(destination / "status.json", {**provenance, "status": "invalid", "oof_current": False,
                                                                  "reason": "OOF must cover each annotated structure exactly once",
                                                                  "previous_numeric_artifacts_retained": (destination / "metrics.json").exists() or (destination / "predictions.csv").exists()})
                continue
            rows = [{**row, "fold": record["job"]["fold"]} for record in seed_records for row in record["predictions"]["rows"]]
            rows.sort(key=lambda row: row["structure_id"])
            labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
            probabilities = np.asarray([[float(row[f"p_{name}"]) for name in CLASS_NAMES] for row in rows])
            score = metrics(labels, probabilities)
            chid = score["confusion_matrix"]
            score["CHID_counts"] = {"TP": chid[2][2], "FP": sum(chid[i][2] for i in (0, 1)),
                                    "FN": sum(chid[2][i] for i in (0, 1)), "support": sum(chid[2])}
            seed_detail[str(seed)] = {"status": "complete", "oof": score}
            oof = {"metrics": score, "rows": rows, "by_id": {row["structure_id"]: row for row in rows}}
            complete_oof[(condition, model, seed)] = oof
            seed_oof[seed] = oof
            _write_derived_json(destination / "metrics.json", {**score, **provenance,
                                                               "protocol": manifest.get("protocol"), "status": "complete", "oof_current": True,
                                                               "score_unit": "unique structures after biological-replicate probability averaging"})
            _csv(destination / "predictions.csv", rows)
            _write_derived_json(destination / "status.json", {**provenance, "status": "complete", "oof_current": True,
                                                              "n_structures": len(rows), "numeric_artifacts": ["metrics.json", "predictions.csv"]})
            confusion_rows.extend(_error_rows(rows, identity, model, condition, seed))
        complete = len(seed_oof) == len(seeds)
        scores = [seed_oof[seed]["metrics"] for seed in seeds] if complete else []
        agreement = _seed_agreement(seeds, seed_oof, identity)
        row = {"condition": condition, "model": model, "status": "complete" if complete else "pending",
               "expected_jobs": len(items), "completed_jobs": sum(record["status"] == "completed" for record in items),
               "numerical_failure_jobs": sum(record["status"] == "numerical_failure" for record in items),
               "execution_failure_jobs": sum(record["status"] == "execution_failure" for record in items),
               "invalid_jobs": sum(record["status"] == "invalid" for record in items),
               "pending_jobs": sum(record["status"] == "pending" for record in items),
               "complete_seeds": len(seed_oof), "expected_seeds": len(seeds),
               "macro_f1_mean": float(np.mean([score["macro_f1"] for score in scores])) if scores else None,
               "macro_f1_seed_sample_std": float(np.std([score["macro_f1"] for score in scores], ddof=1)) if len(scores) > 1 else None,
               "accuracy_mean": float(np.mean([score["accuracy"] for score in scores])) if scores else None,
               "accuracy_seed_sample_std": float(np.std([score["accuracy"] for score in scores], ddof=1)) if len(scores) > 1 else None,
               "balanced_accuracy_mean": float(np.mean([score["balanced_accuracy"] for score in scores])) if scores else None,
               "balanced_accuracy_seed_sample_std": float(np.std([score["balanced_accuracy"] for score in scores], ddof=1)) if len(scores) > 1 else None,
               "completed_runtime_seconds": sum(float(record["metrics"]["runtime_seconds"]) for record in items if record["status"] == "completed"),
               "seed_unanimous_rate": agreement["unanimous_rate"] if agreement else None}
        for cls in CLASS_NAMES:
            for measure in ("precision", "recall", "f1"):
                row[f"{cls}_{measure}_mean"] = float(np.mean([score["per_class"][cls][measure] for score in scores])) if scores else None
                row[f"{cls}_{measure}_seed_sample_std"] = float(np.std([score["per_class"][cls][measure] for score in scores], ddof=1)) if len(scores) > 1 else None
        detail = {"seeds": seed_detail, "seed_prediction_agreement": agreement}
        confused_ids = {prediction["structure_id"] for seed_result in seed_oof.values() for prediction in seed_result["rows"]
                        if {int(prediction["y_true"]), int(prediction["y_pred"])} == {1, 2}}
        for seed, seed_result in seed_oof.items():
            for prediction in seed_result["rows"]:
                sid = prediction["structure_id"]
                if sid not in confused_ids:
                    continue
                source = identity[sid]
                error_seed_rows.append({"condition": condition, "model": model, "seed": seed, "fold": prediction["fold"],
                                        "structure_id": sid, "type": source["type"], "predicted_type": CLASS_NAMES[int(prediction["y_pred"])],
                                        "start": source["start"], "end": source["end"], "length_bp": source["length_bp"], "group_id": source["group_id"],
                                        "CHIN_CHID_confusion": {int(prediction["y_true"]), int(prediction["y_pred"])} == {1, 2},
                                        **{f"p_{cls}": float(prediction[f"p_{cls}"]) for cls in CLASS_NAMES}})
        if items[0]["job"]["kind"] == "neural":
            detail["learning_diagnostics"] = _neural_summary(items, config)
            row["strict_learning_passes"] = detail["learning_diagnostics"]["strict_learning_passes"]
            row["strict_learning_expected_cases"] = len(items)
        else:
            detail["seed_interpretation"] = "Deterministic traditional baseline, one seed; not three independent stochastic runs."
            detail["outer_prediction_events"] = _baseline_outer_summary(items, config)
        comparisons.append(row)
        details[f"{condition}/{model}"] = detail
    paired = _pairwise(records, jobs, complete_oof)
    summary = {
        "status": "invalid" if errors else "complete" if all(row["status"] == "complete" for row in comparisons) else "pending",
        "protocol": manifest.get("protocol"), "experiment_signature": manifest["experiment_signature"],
        "dataset_signature": manifest["dataset_signature"], "source_signature": manifest["source_signature"],
        "score_unit": "one unique annotated structure, biological-replicate probabilities averaged before scoring",
        "n_structures": len(identity), "class_names": list(CLASS_NAMES),
        "class_support": {name: sum(row["type"] == name for row in identity.values()) for name in CLASS_NAMES},
        "main_score": "Full frozen-fold OOF per seed, then mean and sample SD only once every expected seed is complete.",
        "interpretation": "Developmental re-evaluation on previously inspected spatial folds; seed SD is optimization variability, not a population confidence interval.",
        "expected_jobs": len(jobs), "completed_jobs": sum(record["status"] == "completed" for record in records.values()),
        "numerical_failure_jobs": sum(record["status"] == "numerical_failure" for record in records.values()),
        "execution_failure_jobs": sum(record["status"] == "execution_failure" for record in records.values()),
        "pending_jobs": sum(record["status"] == "pending" for record in records.values()),
        "invalid_jobs": sum(record["status"] == "invalid" for record in records.values()),
        "comparison": comparisons, "models": details, "jobs": job_rows, "errors": errors,
        "fold_consistency": consistency, "paired_comparisons": paired["comparisons"],
        "constant_classifier_policy": "Completed constant-output jobs remain in every performance score. Numerical and execution failures remain separate, preserve trajectories and block complete OOF.",
    }
    _write_derived_json(run_dir / "summary.json", summary)
    _csv(run_dir / "comparison.csv", comparisons)
    _csv(run_dir / "job_review.csv", job_rows)
    _csv(run_dir / "paired_deltas.csv", paired["rows"], ["comparison", "seed", "fold", "macro_f1_delta", "CHID_f1_delta", "balanced_accuracy_delta"])
    _csv(run_dir / "CHIN_CHID_errors.csv", [row for row in confusion_rows if row["CHIN_CHID_confusion"]],
         ["model", "condition", "seed", "fold", "structure_id", "type", "predicted_type", "start", "end", "length_bp", "group_id", "CHIN_CHID_confusion", "known_same_center_pair", "p_OPCID", "p_CHIN", "p_CHID"])
    _csv(run_dir / "same_center_cases.csv", [row for row in confusion_rows if row["known_same_center_pair"]],
         ["model", "condition", "seed", "fold", "structure_id", "type", "predicted_type", "start", "end", "length_bp", "group_id", "CHIN_CHID_confusion", "known_same_center_pair", "p_OPCID", "p_CHIN", "p_CHID"])
    _csv(run_dir / "error_seed_review.csv", error_seed_rows,
         ["model", "condition", "seed", "fold", "structure_id", "type", "predicted_type", "start", "end", "length_bp", "group_id", "CHIN_CHID_confusion", "p_OPCID", "p_CHIN", "p_CHID"])
    return summary
