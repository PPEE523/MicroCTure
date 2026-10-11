"""Structure-level evaluation and report artifacts for Task 1 comparisons.

Biological replicates are averaged before scoring. All macro scores always use
the same three classes, including a class absent from one validation fold.
"""

from __future__ import annotations

import csv
import errno
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def _plot_artifacts(run_dir: Path, comparisons: list[dict], oof: dict) -> None:
    """Use noninteractive figures and discard the one-use font cache."""
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="microcture-mpl-", dir="/tmp") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            figure = Figure(figsize=(8, 4.8), layout="constrained")
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            x = np.arange(len(comparisons))
            complete = [item["status"] == "complete" for item in comparisons]
            heights = [item["macro_f1_mean"] if done else 0 for item, done in zip(comparisons, complete)]
            errors = [item["macro_f1_seed_std"] or 0 if done else 0
                      for item, done in zip(comparisons, complete)]
            axis.bar(x, heights, yerr=errors, color=["#3569a8" if done else "#d7dce3" for done in complete], capsize=5)
            for index, done in enumerate(complete):
                if not done:
                    axis.text(index, 0.05, "PENDING", ha="center", fontsize=9)
            axis.set_xticks(x, [item["model"] for item in comparisons], rotation=15)
            axis.set_ylim(0, 1)
            axis.set_ylabel("Structure-level OOF macro-F1 (fixed 3 classes)")
            axis.set_title("Mean across complete seeds; error bars: sample seed SD")
            figure.savefig(run_dir / "comparison.png", dpi=170)
            figure.clear()

            for (model, seed), value in oof.items():
                figure = Figure(figsize=(5.5, 4.8), layout="constrained")
                FigureCanvasAgg(figure)
                axis = figure.subplots()
                confusion = np.asarray(value["metrics"]["confusion_matrix"])
                heatmap = axis.imshow(confusion, cmap="Blues", vmin=0)
                for i in range(3):
                    for j in range(3):
                        axis.text(j, i, str(confusion[i, j]), ha="center", va="center",
                                  color="white" if confusion[i, j] > confusion.max() * 0.55 else "black")
                axis.set_xticks(range(3), CLASS_NAMES)
                axis.set_yticks(range(3), CLASS_NAMES)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("True")
                score = value["metrics"]["macro_f1"]
                axis.set_title(f"{model}, seed {seed}: OOF macro-F1 = {score:.3f}")
                figure.colorbar(heatmap, ax=axis, label="Annotated structures")
                destination = run_dir / "oof" / model / f"seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                figure.savefig(destination / "confusion_matrix.png", dpi=170)
                figure.clear()
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def summarize_results(run_dir: Path | str) -> dict:
    """Aggregate outer test folds as one OOF prediction per structure.

    ``manifest.json`` must state ``models``, ``model_seeds`` (or shared ``seeds``),
    and ``folds`` (count or list of indices) before a result can be complete. Fold evidence lives
    in ``runs/{model}/seed_{seed}/fold_{fold}``; the ``runs`` component is optional.
    Incomplete seeds do not contribute to the main comparison statistics.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    evidence_root = run_dir / "runs" if (run_dir / "runs").is_dir() else run_dir
    records: dict[tuple[str, int, int], dict] = {}
    errors: list[str] = []
    for path in sorted(evidence_root.glob("*/seed_*/fold_*/metrics.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            model = path.parent.parent.parent.name
            seed = int(path.parent.parent.name.removeprefix("seed_"))
            fold = int(path.parent.name.removeprefix("fold_"))
            if (str(value.get("model")), int(value.get("seed", -1)), int(value.get("fold", -1))) != (model, seed, fold):
                raise ValueError("metrics model/seed/fold disagree with their directory")
            for field in ("protocol", "dataset_signature", "run_signature", "source_signature"):
                if field in manifest and value.get(field) != manifest[field]:
                    raise ValueError(f"{field} differs from the run manifest")
            prediction = _read_prediction_csv(path.with_name("predictions.csv"))
            test = value.get("test")
            if not isinstance(test, dict):
                raise ValueError("test metrics are required for each outer fold; validation is the inner set")
            for field in ("macro_f1", "accuracy", "balanced_accuracy"):
                if field not in test or not np.isclose(float(test[field]), prediction["metrics"][field], atol=1e-7):
                    raise ValueError(f"test {field} differs from the stored structure predictions")
            if "parameter_count" not in value or "runtime_seconds" not in value:
                raise ValueError("parameter_count and runtime_seconds fields are required")
            parameter_count = value["parameter_count"]
            if parameter_count is not None and (type(parameter_count) is not int or parameter_count < 0):
                raise ValueError("parameter_count must be a nonnegative integer or null")
            runtime = value["runtime_seconds"]
            if runtime is None or not np.isfinite(float(runtime)) or float(runtime) < 0:
                raise ValueError("runtime_seconds must be finite and nonnegative")
            records[(model, seed, fold)] = {"metadata": value, "predictions": prediction}
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append(f"{path.relative_to(run_dir)}: {error}")

    models = list(manifest.get("models", sorted({key[0] for key in records})))
    seeds = [int(seed) for seed in manifest.get("seeds", sorted({key[1] for key in records}))]
    model_seeds = {model: [int(seed) for seed in manifest.get("model_seeds", {}).get(model, seeds)]
                   for model in models}
    fold_setting = manifest.get("folds", sorted({key[2] for key in records}))
    folds = list(range(fold_setting)) if isinstance(fold_setting, int) else [int(fold) for fold in fold_setting]
    seed_specified = "seeds" in manifest or ("model_seeds" in manifest and
                                           all(model in manifest["model_seeds"] for model in models))
    verified_manifest = bool(manifest_path.is_file() and models and folds and
                             all(model_seeds[model] for model in models) and seed_specified
                             and all(field in manifest for field in ("models", "folds")))
    expected_count = manifest.get("n_structures", manifest.get("structure_count"))
    comparisons, model_details, oof, pending = [], {}, {}, []
    expected_set: dict[str, tuple[int, int]] | None = None
    for model in models:
        seeds_for_model = model_seeds[model]
        details = {"seeds": {}, "pending_folds": [], "seed_variability_interpretation":
                   "training seed variability on the same evaluation protocol"}
        if model in {"pca_logreg", "pca_lr", "pca-logreg", "pca", "PCA + Logistic Regression"}:
            details["seed_variability_interpretation"] = (
                "deterministic PCA/Logistic Regression baseline; repeated identical predictions "
                "are not independent stochastic training runs")
        seed_scores, fold_scores, parameters, runtimes = [], [], [], []
        scores_by_fold: dict[int, list[float]] = defaultdict(list)
        completed_folds = 0
        for seed in seeds_for_model:
            missing = [fold for fold in folds if (model, seed, fold) not in records]
            completed_folds += len(folds) - len(missing)
            if missing:
                details["pending_folds"].append({"seed": seed, "folds": missing})
                pending.extend({"model": model, "seed": seed, "fold": fold} for fold in missing)
                details["seeds"][str(seed)] = {"status": "pending", "missing_folds": missing}
                continue
            if not verified_manifest:
                details["seeds"][str(seed)] = {"status": "unverified_manifest"}
                continue
            entries = [records[(model, seed, fold)] for fold in folds]
            predictions = [entry["predictions"] for entry in entries]
            ids = np.concatenate([item["structure_id"] for item in predictions])
            labels = np.concatenate([item["y"] for item in predictions])
            groups = np.concatenate([item["group_id"] for item in predictions])
            probs = np.concatenate([item["probabilities"] for item in predictions])
            message = None
            if len(ids) != len(np.unique(ids)):
                message = "structure occurs in multiple outer test folds"
            elif expected_count is not None and len(ids) != int(expected_count):
                message = f"OOF has {len(ids)} structures; manifest expects {expected_count}"
            fold_groups: dict[int, int] = {}
            for fold, prediction in zip(folds, predictions):
                for group in np.unique(prediction["group_id"]):
                    if int(group) in fold_groups and fold_groups[int(group)] != fold:
                        message = "leakage group occurs in multiple outer test folds"
                    fold_groups[int(group)] = fold
            identity = {str(sid): (int(y), int(group)) for sid, y, group in zip(ids, labels, groups)}
            if expected_set is not None and identity != expected_set:
                message = "OOF structures, labels or group IDs differ across compared runs"
            if message:
                errors.append(f"{model}/seed_{seed}: {message}")
                details["seeds"][str(seed)] = {"status": "invalid", "reason": message}
                continue
            if expected_set is None:
                expected_set = identity
            score = metrics(labels, probs)
            seed_scores.append(score)
            fold_scores.extend(item["metrics"]["macro_f1"] for item in predictions)
            for fold, prediction in zip(folds, predictions):
                scores_by_fold[fold].append(prediction["metrics"]["macro_f1"])
            parameters.extend(int(entry["metadata"]["parameter_count"]) for entry in entries
                              if entry["metadata"].get("parameter_count") is not None)
            runtimes.extend(float(entry["metadata"]["runtime_seconds"]) for entry in entries
                            if entry["metadata"].get("runtime_seconds") is not None)
            seed_fold_scores = [item["metrics"]["macro_f1"] for item in predictions]
            details["seeds"][str(seed)] = {"status": "complete", "oof": score,
                                            "fold_macro_f1": seed_fold_scores,
                                            "fold_macro_f1_std": float(np.std(seed_fold_scores, ddof=1)) if len(seed_fold_scores) > 1 else None}
            rows = [row for item in predictions for row in item["rows"]]
            oof[(model, seed)] = {"metrics": score, "rows": rows}
            destination = run_dir / "oof" / model / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            _write_json(destination / "metrics.json", score)
            with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        status = "complete" if verified_manifest and len(seed_scores) == len(seeds_for_model) else "pending"
        if not verified_manifest:
            status = "unverified_manifest"
        fold_means = [float(np.mean(scores_by_fold[fold])) for fold in folds if scores_by_fold[fold]]
        row = {
            "model": model, "status": status, "complete_seeds": len(seed_scores),
            "expected_seeds": len(seeds_for_model), "completed_folds": completed_folds,
            "expected_folds": len(seeds_for_model) * len(folds),
            "macro_f1_mean": float(np.mean([score["macro_f1"] for score in seed_scores])) if seed_scores else None,
            "macro_f1_seed_std": float(np.std([score["macro_f1"] for score in seed_scores], ddof=1)) if len(seed_scores) > 1 else None,
            "accuracy_mean": float(np.mean([score["accuracy"] for score in seed_scores])) if seed_scores else None,
            "balanced_accuracy_mean": float(np.mean([score["balanced_accuracy"] for score in seed_scores])) if seed_scores else None,
            "fold_macro_f1_mean": float(np.mean(fold_scores)) if fold_scores else None,
            "fold_macro_f1_std": float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else None,
            "parameter_count": float(np.mean(parameters)) if parameters else None,
            "runtime_seconds": float(np.sum(runtimes)) if runtimes else None,
        }
        for class_name in CLASS_NAMES:
            for field in ("precision", "recall", "f1"):
                row[f"{class_name}_{field}_mean"] = float(np.mean([
                    score["per_class"][class_name][field] for score in seed_scores
                ])) if seed_scores else None
        agreement = None
        if status == "complete" and len(seeds_for_model) > 1:
            ordered_ids = sorted(expected_set or {})
            aligned_predictions = []
            for seed in seeds_for_model:
                prediction_by_id = {row["structure_id"]: int(row["y_pred"])
                                    for row in oof[(model, seed)]["rows"]}
                aligned_predictions.append([prediction_by_id[sid] for sid in ordered_ids])
            aligned_predictions = np.asarray(aligned_predictions)
            unanimous_count = int(np.all(aligned_predictions == aligned_predictions[0], axis=0).sum())
            pairwise_rates = [float(np.mean(aligned_predictions[i] == aligned_predictions[j]))
                              for i in range(len(seeds_for_model)) for j in range(i + 1, len(seeds_for_model))]
            agreement = {"n_structures": len(ordered_ids), "n_seeds": len(seeds_for_model),
                         "unanimous_count": unanimous_count,
                         "unanimous_rate": unanimous_count / len(ordered_ids),
                         "pairwise_agreement_mean": float(np.mean(pairwise_rates))}
        details["seed_prediction_agreement"] = agreement
        row["unanimous_rate"] = agreement["unanimous_rate"] if agreement else None
        comparisons.append(row)
        model_details[model] = details
    summary = {
        "status": "complete" if comparisons and all(row["status"] == "complete" for row in comparisons) and not errors else "pending",
        "score_unit": "one annotated structure, averaging biological replicate probabilities",
        "main_score": "macro-F1 of concatenated outer-fold OOF predictions, then mean across complete seeds",
        "standard_deviation": {"seed": "sample SD across complete seed OOF scores; undefined for one seed",
                               "fold": "sample SD across outer-fold macro-F1 after averaging seeds within each fold; per-seed fold SD also retained"},
        "class_names": list(CLASS_NAMES), "protocol": manifest.get("protocol"),
        "dataset_signature": manifest.get("dataset_signature"),
        "run_signature": manifest.get("run_signature"), "source_signature": manifest.get("source_signature"),
        "manifest_verified": verified_manifest, "model_seeds": model_seeds, "comparison": comparisons,
        "models": model_details, "pending": pending, "errors": errors,
    }
    _write_json(run_dir / "summary.json", summary)
    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(comparisons[0]) if comparisons else ["model", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    _plot_artifacts(run_dir, comparisons, oof)
    return summary
