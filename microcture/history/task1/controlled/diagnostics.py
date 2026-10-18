"""Inner-only, structure-level diagnostics for controlled learning experiments.

Acceptance tests numerical learning and prediction collapse, not biological
recognition quality. All measurements use frozen full images in evaluation
mode; callers must supply inner training/validation samples, never outer test.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from microcture.task1.evaluation import CLASS_NAMES, grouped_predictions, metrics


DEFAULT_ACCEPTANCE = {
    "window": 10,
    "min_relative_loss_drop": 0.01,
    "logit_variance_threshold": 1e-6,
    "gradient_norm_epsilon": 1e-10,
    "activation_epsilon": 1e-6,
    "constant_epsilon": 1e-6,
    "majority_warning_fraction": 0.95,
    "max_terminal_loss_regression": 0.05,
}


def _distribution(values: np.ndarray, epsilon: float) -> dict:
    """Population variance across structures, separately for each coordinate."""
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    ranges = np.ptp(values, axis=0)
    variance = np.var(values, axis=0, ddof=0)
    return {
        "between_structure_var": float(variance.mean()),
        "per_coordinate_var": variance.tolist(),
        "per_coordinate_mean": values.mean(axis=0).tolist(),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std(ddof=0)),
        "max_coordinate_range": float(ranges.max()),
        "constant": bool(np.all(ranges <= epsilon)),
        "constant_epsilon": epsilon,
        "variance_definition": "population variance across replicate-averaged structures; mean over vector coordinates",
    }


def _activation_distribution(values: np.ndarray, epsilon: float) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    effective = np.abs(values) > epsilon
    active_units = np.any(effective, axis=0)
    norms = np.linalg.norm(values, axis=1)
    return {
        "n_replicate_observations": len(values),
        "total_units": values.shape[1],
        "positive_fraction": float(np.mean(values > 0)),
        "effective_fraction": float(effective.mean()),
        "inactive_units": int(np.count_nonzero(~active_units)),
        "active_units": int(np.count_nonzero(active_units)),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std(ddof=0)),
        "epsilon": epsilon,
        "inactive_definition": "abs(value) <= epsilon on every probed replicate; negative GELU values can be active",
    }


def _head_modules(model: nn.Module) -> tuple[Any, Any]:
    candidates = [getattr(model, "head", None), model]
    for head in candidates:
        if head is not None and isinstance(getattr(head, "activation", None), nn.Module):
            return getattr(head, "input_linear", None), head.activation
    return None, None


def diagnose_model(model: nn.Module, samples: dict, device: torch.device | str,
                   batch_size: int, *, epsilon: float = 1e-6) -> dict:
    """Measure a frozen inner sample view without augmentation or gradients.

    ``samples`` uses ``x, y, structure_id, group_id, replicate`` and an optional
    ``length`` side channel of shape [N, 1]. Probability vectors are averaged after softmax over each replicate;
    logit vectors are averaged separately. Head unit activity is observed on
    every replicate, so opposing values cannot cancel a genuinely active unit.
    The caller's per-module train/eval flags are restored even on failure.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    x = np.asarray(samples["x"])
    if x.ndim < 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("x must contain nonempty finite sample tensors")
    length = None
    if "length" in samples:
        length = np.asarray(samples["length"])
        if length.shape != (len(x), 1) or not np.isfinite(length).all():
            raise ValueError("length side channel must be finite with shape [N, 1]")
    # Validate metadata before running a model, and establish stable structure order.
    template = grouped_predictions(samples, np.full((len(x), 3), 1 / 3))
    if len(samples["y"]) != len(x):
        raise ValueError("x and metadata contain different sample counts")
    input_linear, activation = _head_modules(model)
    head_activations, head_preactivations, outputs, probability_outputs, handles = [], [], [], [], []

    def capture(destination):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor) or output.shape[0] != inputs[0].shape[0]:
                raise ValueError("Head diagnostics require a tensor with one row per example")
            destination.append(output.detach().cpu().numpy().copy())
        return hook

    if activation is not None:
        handles.append(activation.register_forward_hook(capture(head_activations)))
    if isinstance(input_linear, nn.Module):
        handles.append(input_linear.register_forward_hook(capture(head_preactivations)))
    original_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                inputs = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size], dtype=np.float32)).to(device)
                if length is None:
                    logits = model(inputs)
                else:
                    length_inputs = torch.from_numpy(np.ascontiguousarray(length[start:start + batch_size], dtype=np.float32)).to(device)
                    logits = model(inputs, length_inputs)
                if logits.ndim != 2 or logits.shape != (len(inputs), 3) or not torch.isfinite(logits).all():
                    raise ValueError("Model must return finite logits with shape (batch, 3)")
                outputs.append(logits.detach().cpu().numpy().copy())
                # Match the endpoint predictor's dtype/softmax, including near ties.
                probability_outputs.append(logits.softmax(dim=1).detach().cpu().numpy().copy())
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in original_modes:
            module.training = was_training
    logits = np.concatenate(outputs).astype(np.float64)
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    normalizers = exponentials.sum(axis=1, keepdims=True)
    probabilities = np.concatenate(probability_outputs).astype(np.float64)
    log_probabilities = shifted_logits - np.log(normalizers)
    labels = np.asarray(samples["y"], dtype=np.int64)
    nll = -log_probabilities[np.arange(len(logits)), labels]
    nll_sums = np.bincount(labels, weights=nll, minlength=3)
    label_counts = np.bincount(labels, minlength=3)
    grouped = grouped_predictions(samples, probabilities)
    ids = np.asarray(samples["structure_id"], dtype=str)
    structure_logits = np.stack([logits[ids == sid].mean(axis=0) for sid in template["structure_id"]])
    centered_logits = structure_logits - structure_logits.mean(axis=1, keepdims=True)
    logit_stats = _distribution(structure_logits, epsilon)
    centered_stats = _distribution(centered_logits, epsilon)
    probability_stats = _distribution(grouped["probabilities"], epsilon)
    predicted_counts = np.bincount(grouped["predicted"], minlength=3)
    head = {"available": bool(head_activations),
            "activation_name": type(activation).__name__ if activation is not None else None,
            "activity_basis": "all fixed replicate observations; abs activation > epsilon",
            "positive_fraction_is_dead_unit_test": isinstance(activation, nn.ReLU)}
    if head_activations:
        values = np.concatenate(head_activations)
        if len(values) != len(x) or not np.isfinite(values).all():
            raise ValueError("Head activation hook must run exactly once per example and return finite values")
        head.update(_activation_distribution(values, epsilon))
    if head_preactivations:
        values = np.concatenate(head_preactivations)
        if len(values) != len(x) or not np.isfinite(values).all():
            raise ValueError("Head preactivation hook must run exactly once per example and return finite values")
        head["preactivation"] = _activation_distribution(values, epsilon)
    return {
        "n_structures": len(grouped["y"]), "n_replicate_examples": len(x),
        "replicate_count_min": int(grouped["replicate_count"].min()),
        "replicate_count_max": int(grouped["replicate_count"].max()),
        "evaluation_view": "fixed supplied full images, eval/inference_mode, no augmentation",
        "metrics": {**metrics(grouped["y"], grouped["probabilities"]),
                    "replicate_disagreement": grouped["replicate_disagreement"]},
        "loss_support": {"unweighted_nll_sum_by_class": nll_sums.tolist(),
                         "examples_count_by_class": label_counts.tolist(),
                         "aggregation_unit": "one fixed replicate image; sum(class_weight * NLL) / sum(class_weight * example_count)"},
        "logits": logit_stats, "centered_logits": centered_stats,
        "probabilities": probability_stats, "head": head,
        "logit_between_structure_var": logit_stats["between_structure_var"],
        "centered_logit_between_structure_var": centered_stats["between_structure_var"],
        "probability_between_structure_var": probability_stats["between_structure_var"],
        "logits_constant": logit_stats["constant"],
        "centered_logits_constant": centered_stats["constant"],
        "probabilities_constant": probability_stats["constant"],
        "majority_prediction_fraction": float(predicted_counts.max() / len(grouped["y"])),
        "num_predicted_classes": int(np.count_nonzero(predicted_counts)),
        "prediction_counts": dict(zip(CLASS_NAMES, predicted_counts.tolist())),
    }


def _settings(config: dict) -> dict:
    settings = {**DEFAULT_ACCEPTANCE, **config.get("acceptance", {})}
    if type(settings["window"]) is not int or settings["window"] < 2:
        raise ValueError("acceptance.window must be an integer >= 2")
    for key in DEFAULT_ACCEPTANCE:
        if key != "window" and (not np.isfinite(settings[key]) or settings[key] < 0):
            raise ValueError(f"acceptance.{key} must be finite and nonnegative")
    if not 0 < settings["majority_warning_fraction"] <= 1:
        raise ValueError("majority_warning_fraction must lie in (0, 1]")
    if not 0 < settings["min_relative_loss_drop"] < 1:
        raise ValueError("min_relative_loss_drop must lie in (0, 1)")
    return settings


def _window_report(rows: list[dict], initial_loss: float, settings: dict, window: int) -> dict:
    tail = rows[-window:]
    losses = np.asarray([float(row["weighted_train_ce"]) for row in tail])
    window_complete = len(tail) == window
    finite_loss = bool(np.isfinite(losses).all() and np.all(losses >= 0))
    mean_loss = float(losses.mean()) if finite_loss and len(losses) else None
    relative_drop = (initial_loss - mean_loss) / initial_loss if mean_loss is not None else None
    terminal_drop = float((initial_loss - losses[-1]) / initial_loss) if finite_loss and len(losses) else None
    all_losses = np.asarray([float(row["weighted_train_ce"]) for row in rows])
    rolling_means = [float(all_losses[i:i + window].mean())
                     for i in range(len(rows) - window + 1)
                     if np.isfinite(all_losses[i:i + window]).all()]
    best_window = min(rolling_means) if rolling_means else None
    regression = ((mean_loss - best_window) / max(best_window, 1e-12)) if mean_loss is not None and best_window is not None else None
    checks = {"enough_observations": window_complete, "finite_nonnegative_loss": finite_loss,
              "relative_loss_drop": relative_drop is not None and relative_drop >= settings["min_relative_loss_drop"],
              "terminal_loss_drop": terminal_drop is not None and terminal_drop >= settings["min_relative_loss_drop"],
              "terminal_not_degraded": regression is not None and regression <= settings["max_terminal_loss_regression"]}
    views = {}
    for name in ("train", "validation"):
        diagnostics = [row[f"{name}_diagnostics"] for row in tail]
        raw_variances = [float(d["logit_between_structure_var"]) for d in diagnostics]
        centered_variances = [float(d["centered_logit_between_structure_var"]) for d in diagnostics]
        # Re-evaluate the range using preregistered epsilon, not a stale boolean.
        constant = [float(d["logits"]["max_coordinate_range"]) <= settings["constant_epsilon"] for d in diagnostics]
        centered_constant = [float(d["centered_logits"]["max_coordinate_range"]) <= settings["constant_epsilon"] for d in diagnostics]
        probability_constant = [float(d["probabilities"]["max_coordinate_range"]) <= settings["constant_epsilon"] for d in diagnostics]
        majority = [float(d["majority_prediction_fraction"]) for d in diagnostics]
        active = [bool(d["head"].get("available") and d["head"].get("active_units", 0) > 0
                       and d["head"].get("effective_fraction", 0) > 0
                       and np.isclose(d["head"].get("epsilon", float("nan")), settings["activation_epsilon"],
                                      rtol=0, atol=1e-15)) for d in diagnostics]
        variance_ok = bool(raw_variances and np.isfinite(raw_variances).all() and
                           all(v > settings["logit_variance_threshold"] for v in raw_variances))
        discriminative_ok = bool(centered_variances and np.isfinite(centered_variances).all() and
                                 all(v > settings["logit_variance_threshold"] for v in centered_variances))
        views[name] = {"logit_variance_min": min(raw_variances) if raw_variances else None,
                       "centered_logit_variance_min": min(centered_variances) if centered_variances else None,
                       "logits_constant_epochs": sum(constant),
                       "centered_logits_constant_epochs": sum(centered_constant),
                       "probabilities_constant_epochs": sum(probability_constant),
                       "majority_prediction_fraction_max": max(majority) if majority else None,
                       "majority_concentration_epochs": sum(v >= settings["majority_warning_fraction"] for v in majority),
                       "head_inactive_epochs": len(active) - sum(active),
                       "logit_variation": variance_ok and not any(constant),
                       "discriminative_variation": discriminative_ok and not any(centered_constant) and not any(probability_constant),
                       "head_activity": bool(active and all(active)),
                       "prediction_noncollapse": bool(majority and np.isfinite(majority).all() and
                                                      all(v < settings["majority_warning_fraction"] for v in majority))}
        checks[f"{name}_logit_variation"] = views[name]["logit_variation"]
        checks[f"{name}_discriminative_variation"] = views[name]["discriminative_variation"]
        checks[f"{name}_head_activity"] = views[name]["head_activity"]
    gradients = {}
    for part in ("total", "encoder", "head"):
        records = [(row.get("gradient_norms") or {}).get(part) for row in tail]
        present = all(isinstance(record, dict) for record in records)
        finite = bool(present and all(record.get("finite") and
                                     all(np.isfinite(float(record[key])) for key in ("mean", "min", "max", "nonzero_fraction"))
                                     for record in records))
        effective = bool(finite and all(float(record["mean"]) > settings["gradient_norm_epsilon"]
                                       and float(record["nonzero_fraction"]) > 0 for record in records))
        gradients[part] = {"present": present, "finite": finite, "effective_each_epoch": effective,
                           "mean_norm_min": min(float(record["mean"]) for record in records) if finite else None,
                           "nonzero_fraction_min": min(float(record["nonzero_fraction"]) for record in records) if finite else None}
        checks[f"{part}_gradient_finite_effective"] = effective
    learning = all(checks.values())
    noncollapse = all(v["prediction_noncollapse"] for v in views.values())
    return {"window": window, "epochs": [row["epoch"] for row in tail],
            "initial_weighted_train_ce": initial_loss, "mean_tail_weighted_train_ce": mean_loss,
            "relative_loss_drop": relative_drop, "terminal_relative_loss_drop": terminal_drop,
            "best_window_weighted_train_ce": best_window, "terminal_relative_loss_regression": regression,
            "checks": checks, "views": views, "gradients": gradients,
            "learning_detected": learning, "prediction_noncollapse": noncollapse,
            "accepted": learning and noncollapse,
            "failed_checks": [key for key, passed in checks.items() if not passed] +
                             [f"{name}_prediction_concentrated" for name, value in views.items() if not value["prediction_noncollapse"]]}


def learning_acceptance(history: list[dict], config: dict) -> dict:
    """Evaluate preregistered sustained-learning criteria without F1 thresholds.

    Rows must contain ``epoch, weighted_train_ce, train_diagnostics,
    validation_diagnostics, gradient_norms``. The epoch-zero fixed-view loss is
    the preferred reference; a first-training-epoch fallback is explicitly
    identified. Only epochs with ``epoch > 0`` enter observation windows.
    """
    settings = _settings(config)
    if not history:
        return {"status": "insufficient_evidence", "accepted": False,
                "learning_detected": False, "thresholds": settings, "reason": "empty history"}
    epochs = [row["epoch"] for row in history]
    if any(type(epoch) is not int or epoch < 0 for epoch in epochs) or epochs != sorted(set(epochs)):
        raise ValueError("history epochs must be distinct ascending nonnegative integers")
    initial = history[0]
    initial_loss = float(initial["weighted_train_ce"])
    if not np.isfinite(initial_loss) or initial_loss <= 0:
        raise ValueError("Initial weighted_train_ce must be finite and positive")
    rows = [row for row in history if row["epoch"] > 0]
    if not rows:
        return {"status": "insufficient_evidence", "accepted": False,
                "learning_detected": False, "thresholds": settings, "reason": "no post-initial observations"}
    windows = sorted({5, 10, settings["window"]})
    reports = {str(window): _window_report(rows, initial_loss, settings, window) for window in windows}
    primary = reports[str(settings["window"])]
    warnings = []
    for name, view in primary["views"].items():
        if view["majority_concentration_epochs"]:
            warnings.append(f"{name} predictions concentrate in one class; varying logits do not establish multi-class discrimination")
        if view["centered_logits_constant_epochs"] and not view["logits_constant_epochs"]:
            warnings.append(f"{name} raw logits vary only through a class-common offset; softmax discrimination is constant")
    status = "accepted" if primary["accepted"] else "failed"
    if len(rows) < settings["window"]:
        status = "insufficient_evidence"
    return {"status": status, "accepted": primary["accepted"],
            "learning_detected": primary["learning_detected"],
            "prediction_noncollapse": primary["prediction_noncollapse"],
            "scope": "inner fixed train/validation views only; numerical learning and noncollapse, not three-class recognition quality",
            "initial_reference_epoch": initial["epoch"],
            "initial_reference": "epoch_zero_fixed_view" if initial["epoch"] == 0 else "first_training_epoch_fallback",
            "thresholds": settings, "primary_window": settings["window"],
            "primary": primary, "windows": reports, "warnings": warnings,
            "requires_manual_trace_review": True,
            "CHID_F1_used_as_acceptance_threshold": False}
