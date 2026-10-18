"""Pure, read-only aggregation of the controlled head experiments.

This module imports no training or inference code and writes no artifacts.
Each input record is ``{case, head, status, metrics, history}``; ``metrics``
and ``history`` are the already saved dictionaries. A missing job is omitted,
not fabricated as a failed run. Population spread and event rates describe
the available fixed cases; they do not imply independent biological samples.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import math
import statistics
from typing import Any


HEADS = ("H0", "H1", "H2", "H3")
GRADIENT_GROUPS = ("total", "encoder", "head")
EVENTS = (
    "raw_logits_constant", "centered_logits_constant", "probabilities_constant",
    "single_class_predictions", "majority_concentrated", "all_head_units_inactive",
)
REQUIRED_CASE_COUNT = 6


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stats(values: list[float | None]) -> dict:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return dict(n_cases=0, mean=None, median=None, std_population=None, min=None, max=None)
    return dict(n_cases=len(finite), mean=statistics.mean(finite), median=statistics.median(finite),
                std_population=statistics.pstdev(finite), min=min(finite), max=max(finite))


def _rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _settings(config: dict) -> dict:
    training, acceptance = config.get("training", {}), config.get("acceptance", {})
    settings = {
        "epochs": training.get("epochs", 40), "window": acceptance.get("window", 10),
        "constant_epsilon": acceptance.get("constant_epsilon", 1e-6),
        "gradient_norm_epsilon": acceptance.get("gradient_norm_epsilon", 1e-10),
        "logit_variance_threshold": acceptance.get("logit_variance_threshold", 1e-6),
        "activation_epsilon": acceptance.get("activation_epsilon", 1e-6),
        "majority_warning_fraction": acceptance.get("majority_warning_fraction", 0.95),
        "min_relative_loss_drop": acceptance.get("min_relative_loss_drop", 0.01),
        "max_terminal_loss_regression": acceptance.get("max_terminal_loss_regression", 0.05),
    }
    if type(settings["epochs"]) is not int or type(settings["window"]) is not int:
        raise ValueError("epochs and window must be integers")
    if settings["window"] < 2 or settings["epochs"] < 2 * settings["window"]:
        raise ValueError("epochs must cover at least two windows of length >= 2")
    for key, value in settings.items():
        if key not in ("epochs", "window") and (not _finite(value) or value < 0):
            raise ValueError(f"Invalid finite nonnegative threshold {key}")
    if not 0 < settings["majority_warning_fraction"] <= 1 or not 0 < settings["min_relative_loss_drop"] < 1:
        raise ValueError("Invalid majority/loss-drop fraction")
    if acceptance.get("CHID_F1_threshold") is not None:
        raise ValueError("Head learning controls do not select by CHID F1")
    return settings


def _event(diag: dict, name: str, settings: dict) -> bool | None:
    epsilon = settings["constant_epsilon"]
    distribution = {"raw_logits_constant": "logits", "centered_logits_constant": "centered_logits",
                    "probabilities_constant": "probabilities"}.get(name)
    if distribution:
        value = diag.get(distribution, {}).get("max_coordinate_range")
        return value <= epsilon if _finite(value) else None
    if name == "single_class_predictions":
        counts = diag.get("prediction_counts")
        return sum(value > 0 for value in counts.values()) == 1 if counts else None
    if name == "majority_concentrated":
        value = diag.get("majority_prediction_fraction")
        return value >= settings["majority_warning_fraction"] if _finite(value) else None
    head = diag.get("head", {})
    total, inactive = head.get("total_units"), head.get("inactive_units")
    return inactive == total if total is not None and total > 0 and inactive is not None else None


def _gradient_window(rows: list[dict], group: str, settings: dict) -> dict:
    observations = []
    for row in rows:
        value = (row.get("gradient_norms") or {}).get(group, {})
        finite = bool(value.get("finite")) and all(
            _finite(value.get(key)) for key in ("mean", "min", "max", "nonzero_fraction")
        )
        effective = finite and value["mean"] > settings["gradient_norm_epsilon"] and value["nonzero_fraction"] > 0
        observations.append({"epoch": row["epoch"], "finite": finite, "effective": effective,
                             "mean_norm": value.get("mean"), "nonzero_fraction": value.get("nonzero_fraction")})
    means = [observation["mean_norm"] for observation in observations if _finite(observation["mean_norm"])]
    fractions = [observation["nonzero_fraction"] for observation in observations if _finite(observation["nonzero_fraction"])]
    return {"every_epoch_finite_and_effective": len(rows) == settings["window"] and all(value["effective"] for value in observations),
            "finite_epochs": sum(value["finite"] for value in observations),
            "effective_epochs": sum(value["effective"] for value in observations),
            "observed_epochs": len(rows), "mean_epoch_mean_norm": statistics.mean(means) if means else None,
            "minimum_epoch_mean_norm": min(means) if means else None,
            "minimum_nonzero_fraction": min(fractions) if fractions else None, "epochs": observations}


def _review(record: dict, settings: dict) -> dict:
    history = record.get("history", [])
    epochs = [row.get("epoch") for row in history]
    if any(type(epoch) is not int or epoch < 0 or epoch > settings["epochs"] for epoch in epochs) or epochs != sorted(set(epochs)):
        raise ValueError(f"Invalid epoch sequence for {record['case']}/{record['head']}")
    post = [row for row in history if row["epoch"] > 0]
    tail = post[-settings["window"]:]
    initial = next((row for row in history if row["epoch"] == 0), None)
    endpoint = next((row for row in history if row["epoch"] == settings["epochs"]), None)
    complete_history = epochs == list(range(settings["epochs"] + 1))
    losses = [row.get("weighted_train_ce") for row in post]
    valid_losses = bool(losses) and all(_finite(loss) and loss >= 0 for loss in losses)
    best, tail_mean, cv, regression = None, None, None, None
    if valid_losses and len(post) >= settings["window"]:
        width = settings["window"]
        means = [statistics.mean(losses[i:i + width]) for i in range(len(losses) - width + 1)]
        best, tail_mean = min(means), means[-1]
        cv = statistics.pstdev(losses[-width:]) / tail_mean if tail_mean > 0 else None
        regression = (tail_mean - best) / max(best, 1e-12)
    gradients = {group: _gradient_window(tail, group, settings) for group in GRADIENT_GROUPS}
    checks = {"complete_history": complete_history, "finite_nonnegative_train_loss": valid_losses,
              "terminal_not_degraded": regression is not None and regression <= settings["max_terminal_loss_regression"]}
    initial_loss = initial.get("weighted_train_ce") if initial else None
    initial_valid = _finite(initial_loss) and initial_loss > 0
    checks["relative_loss_drop"] = bool(initial_valid and tail_mean is not None and
                                         (initial_loss - tail_mean) / initial_loss >= settings["min_relative_loss_drop"])
    final_loss = endpoint.get("weighted_train_ce") if endpoint else None
    checks["terminal_loss_drop"] = bool(initial_valid and _finite(final_loss) and
                                         (initial_loss - final_loss) / initial_loss >= settings["min_relative_loss_drop"])
    for group in GRADIENT_GROUPS:
        checks[group + "_gradient_finite_effective"] = gradients[group]["every_epoch_finite_and_effective"]
    for split in ("train", "validation"):
        ds = [row.get(split + "_diagnostics", {}) for row in tail]
        complete_window = len(ds) == settings["window"]
        checks[split + "_raw_logit_variation"] = complete_window and all(
            _finite(diag.get("logit_between_structure_var"))
            and diag["logit_between_structure_var"] > settings["logit_variance_threshold"]
            and _event(diag, "raw_logits_constant", settings) is False for diag in ds
        )
        checks[split + "_discriminative_variation"] = complete_window and all(
            _finite(diag.get("centered_logit_between_structure_var"))
            and diag["centered_logit_between_structure_var"] > settings["logit_variance_threshold"]
            and _event(diag, "centered_logits_constant", settings) is False
            and _event(diag, "probabilities_constant", settings) is False for diag in ds
        )
        checks[split + "_head_activity"] = complete_window and all(
            diag.get("head", {}).get("available")
            and diag["head"].get("active_units", 0) > 0
            and diag["head"].get("effective_fraction", 0) > 0
            and _finite(diag["head"].get("epsilon"))
            and math.isclose(diag["head"]["epsilon"], settings["activation_epsilon"], rel_tol=0, abs_tol=1e-15)
            for diag in ds
        )
        checks[split + "_prediction_noncollapse"] = complete_window and all(
            _event(diag, "majority_concentrated", settings) is False for diag in ds
        )
    saved_accepted = bool(record.get("metrics", {}).get("acceptance", {}).get("accepted"))
    verified_accepted = record["status"] == "completed" and all(checks.values())
    strict_accepted = saved_accepted and verified_accepted
    terminal = record["status"] in ("numerical_failure", "failed", "execution_failure") or (
        record["status"] == "completed" and complete_history
    )
    dead = any(any(_event(row.get(split + "_diagnostics", {}), "all_head_units_inactive", settings) is True
                   for row in tail) for split in ("train", "validation"))
    constant = any(any(_event(row.get(split + "_diagnostics", {}), name, settings) is True for row in tail)
                   for split in ("train", "validation") for name in ("centered_logits_constant", "probabilities_constant"))
    concentrated = any(any(_event(row.get(split + "_diagnostics", {}), "majority_concentrated", settings) is True
                           for row in tail) for split in ("train", "validation"))
    categories = []
    if record["status"] == "numerical_failure": categories.append("numerical_failure")
    if dead: categories.append("dead_head")
    if constant: categories.append("constant_classifier")
    if concentrated: categories.append("class_concentration")
    encoder_evidence = gradients["encoder"]
    if (encoder_evidence["observed_epochs"] == settings["window"]
            and encoder_evidence["finite_epochs"] == settings["window"]
            and not encoder_evidence["every_epoch_finite_and_effective"]):
        categories.append("ineffective_encoder_gradient")
    if regression is not None and regression > settings["max_terminal_loss_regression"]: categories.append("loss_regression")
    if terminal and not strict_accepted and not categories: categories.append("other_recorded_acceptance_failure")
    return {"case": record["case"], "head": record["head"], "status": record["status"],
            "terminal": terminal, "complete_history": complete_history, "saved_accepted": saved_accepted,
            "history_acceptance_checks_pass": verified_accepted, "strict_accepted": strict_accepted,
            "saved_acceptance_agrees_with_history": saved_accepted == verified_accepted,
            "failed_checks": [name for name, value in checks.items() if not value], "failure_categories": categories,
            "final_train_ce": final_loss, "last_window_mean_train_ce": tail_mean,
            "last_window_loss_cv": cv, "last_window_loss_regression": regression,
            "best_window_train_ce": best, "gradient_window": gradients,
            "terminal_views": {
                split: {"mean_centered_logit_variance": _stats([
                    row.get(split + "_diagnostics", {}).get("centered_logit_between_structure_var") for row in tail
                ])["mean"], "mean_probability_variance": _stats([
                    row.get(split + "_diagnostics", {}).get("probability_between_structure_var") for row in tail
                ])["mean"], "mean_head_effective_fraction": _stats([
                    row.get(split + "_diagnostics", {}).get("head", {}).get("effective_fraction") for row in tail
                ])["mean"]} for split in ("train", "validation")
            },
            "final_constants": {split: {name: _event(endpoint.get(split + "_diagnostics", {}), name, settings)
                                          if endpoint else None for name in EVENTS}
                                for split in ("train", "validation")},
            "epoch_zero_events": {split: {name: _event(initial.get(split + "_diagnostics", {}), name, settings)
                                           if initial else None for name in EVENTS}
                                  for split in ("train", "validation")},
            "post_epoch_events": {split: {name: [_event(row.get(split + "_diagnostics", {}), name, settings)
                                                 for row in post] for name in EVENTS}
                                   for split in ("train", "validation")},
            "observed_post_epochs": [row["epoch"] for row in post]}


def _aggregate_events(records: list[dict], split: str, name: str) -> dict:
    streams = [record["post_epoch_events"][split][name] for record in records]
    observed = [value for stream in streams for value in stream if value is not None]
    ever = [any(value is True for value in stream) for stream in streams if any(value is not None for value in stream)]
    final = [record["final_constants"][split][name] for record in records if record["final_constants"][split][name] is not None]
    zero = [record["epoch_zero_events"][split][name] for record in records if record["epoch_zero_events"][split][name] is not None]
    return {"post_epoch_incidence": _rate(sum(value is True for value in observed), len(observed)),
            "ever_case_incidence": _rate(sum(ever), len(ever)),
            "final_epoch_case_incidence": _rate(sum(value is True for value in final), len(final)),
            "epoch_zero_case_incidence": _rate(sum(value is True for value in zero), len(zero)),
            "unknown_post_epoch_measurements": sum(value is None for stream in streams for value in stream)}


def _recipe(summary: dict, ready: bool, config: dict) -> dict:
    if not ready:
        return {"status": "defer", "recipe": None, "reason": "Await terminal results for all four heads in all six preregistered cases."}
    passed = {head: summary[head]["strict_accepted"]["numerator"] == REQUIRED_CASE_COUNT for head in HEADS}
    base_lr = config.get("training", {}).get("learning_rate", 0.001)
    if passed["H1"]:
        return {"status": "candidate", "recipe": {"head_architecture": "H1", "learning_rate": base_lr},
                "reason": "H1 passes all six cases; prefer the simpler GELU head when both H1 and H2 pass, subject to manual stability review.",
                "manual_trace_review_required": True}
    if passed["H2"]:
        return {"status": "candidate", "recipe": {"head_architecture": "H2", "learning_rate": base_lr},
                "reason": "H2 passes all six cases while H1 does not.", "manual_trace_review_required": True}
    if passed["H3"]:
        return {"status": "candidate", "recipe": {"head_architecture": "H2", "head_version": "H3",
                                                   "learning_rate": config.get("head_stage", {}).get("H3_learning_rate", 0.0001)},
                "reason": "H3 passes all six cases; H1 and H2 at the original learning rate do not.",
                "untested_comparison": "H1 plus the lower learning rate was not tested; this result does not establish that it would be worse.",
                "manual_trace_review_required": True}
    return {"status": "defer", "recipe": None, "reason": "No tested H1/H2/H3 recipe passes the sustained-learning criteria in all six cases."}


def aggregate_heads(case_records: list[dict], expected_cases: list[str], config: dict) -> dict:
    """Aggregate case endpoints, collapse incidence and paired head controls.

    Strict acceptance requires a saved acceptance and independent agreement of
    the complete fixed-duration history with the preregistered checks. Counts
    use observed terminal cases; expected cases and missing jobs are explicit.
    Recommendation is a candidate for manual review, never a highest-F1 rule.
    """
    if not expected_cases or len(expected_cases) != len(set(expected_cases)):
        raise ValueError("expected_cases must be nonempty and distinct")
    settings = _settings(config)
    seen = set()
    reviewed = []
    for record in case_records:
        key = (record.get("case"), record.get("head"))
        if key[0] not in expected_cases or key[1] not in HEADS or key in seen:
            raise ValueError(f"Unknown or duplicate head record: {key}")
        seen.add(key)
        reviewed.append(_review(record, settings))
    summaries = {}
    for head in HEADS:
        records = [record for record in reviewed if record["head"] == head]
        terminal = [record for record in records if record["terminal"]]
        completed = [record for record in records if record["status"] == "completed" and record["complete_history"]]
        failures = [record for record in terminal if not record["strict_accepted"]]
        summary = {"expected_cases": len(expected_cases), "observed_cases": len(records),
                   "terminal_observed_cases": len(terminal), "completed_cases": len(completed),
                   "missing_cases": len(expected_cases) - len(records),
                   "missing_case_names": [case for case in expected_cases if (case, head) not in seen],
                   "unfinished_observed_cases": sum(not record["terminal"] for record in records),
                   "numerical_failure_cases": sum(record["status"] == "numerical_failure" for record in records),
                   "strict_accepted": _rate(sum(record["strict_accepted"] for record in terminal), len(terminal)),
                   "observed_not_accepted_cases": len(failures),
                   "final_train_ce_by_case": _stats([record["final_train_ce"] for record in completed]),
                   "last_window_mean_train_ce_by_case": _stats([record["last_window_mean_train_ce"] for record in completed]),
                   "last_window_loss_cv_by_case": _stats([record["last_window_loss_cv"] for record in completed]),
                   "last_window_loss_regression_by_case": _stats([record["last_window_loss_regression"] for record in completed]),
                   "effective_gradient_cases": {}, "event_incidence": {},
                   "failure_category_counts": {name: sum(name in record["failure_categories"] for record in terminal)
                                               for name in ("numerical_failure", "dead_head", "constant_classifier", "class_concentration",
                                                            "ineffective_encoder_gradient", "loss_regression", "other_recorded_acceptance_failure")},
                   "case_acceptance": {record["case"]: record["strict_accepted"] for record in terminal},
                   "all_observed_terminal_cases_pass": bool(terminal) and all(record["strict_accepted"] for record in terminal)}
        for group in GRADIENT_GROUPS:
            summary["effective_gradient_cases"][group] = {
                **_rate(sum(record["gradient_window"][group]["every_epoch_finite_and_effective"] for record in completed), len(completed)),
                "case_mean_epoch_gradient_norm": _stats([record["gradient_window"][group]["mean_epoch_mean_norm"] for record in completed]),
                "case_minimum_epoch_gradient_norm": _stats([record["gradient_window"][group]["minimum_epoch_mean_norm"] for record in completed]),
            }
        for split in ("train", "validation"):
            summary["event_incidence"][split] = {name: _aggregate_events(records, split, name) for name in EVENTS}
        summary["terminal_view_case_stats"] = {
            split: {key: _stats([record["terminal_views"][split][key] for record in completed])
                    for key in ("mean_centered_logit_variance", "mean_probability_variance", "mean_head_effective_fraction")}
            for split in ("train", "validation")
        }
        summaries[head] = summary
    terminal_keys = {(record["case"], record["head"]) for record in reviewed if record["terminal"]}
    coverage = all((case, head) in terminal_keys for case in expected_cases for head in HEADS)
    ready = coverage and len(expected_cases) == REQUIRED_CASE_COUNT
    indexed = {(record["case"], record["head"]): record for record in reviewed}
    paired = []
    for case in expected_cases:
        row = {"case": case, "same_learning_rate": config.get("training", {}).get("learning_rate", 0.001),
               "single_changed_factor": "readout-plus-length LayerNorm", "pair_complete": False}
        for head in ("H1", "H2"):
            record = indexed.get((case, head))
            row[head] = None if record is None else {key: record[key] for key in (
                "status", "terminal", "strict_accepted", "final_train_ce", "last_window_mean_train_ce",
                "last_window_loss_regression", "last_window_loss_cv", "final_constants", "gradient_window", "failure_categories")}
        row["pair_complete"] = all(row[head] is not None and row[head]["terminal"] for head in ("H1", "H2"))
        paired.append(row)
    return {"comparison_status": "fullcoverage" if ready else "pending", "decision_ready": ready,
            "coverage_complete_for_supplied_expected_cases": coverage, "required_case_count": REQUIRED_CASE_COUNT,
            "expected_case_count": len(expected_cases), "expected_cases": list(expected_cases),
            "terminal_complete_cases": [case for case in expected_cases if all((case, head) in terminal_keys for head in HEADS)],
            "head_summaries": summaries, "H1_H2_same_lr_pairs": paired,
            "candidate_recommendation": _recipe(summaries, ready, config), "case_reviews": reviewed,
            "definitions": {
                "endpoint_unit": "One fixed case; equal case weight, epoch-final fixed training CE, population std ddof=0.",
                "acceptance_denominator": "Observed terminal cases only; explicit numerical failures count, pending/missing cases do not.",
                "event_denominator": "Actual known post-epoch measurements within epochs 1..configured_epochs; epoch zero reported separately.",
                "ever_case_incidence": "Cases with >=1 event / cases with at least one known post-epoch measurement.",
                "final_case_incidence": "Event at the preregistered final epoch / cases with that endpoint measurement.",
                "constant": "Maximum per-coordinate range across replicate-averaged structures <= constant_epsilon; raw/centered/probability vectors are distinct.",
                "single_class": "Exactly one predicted class; varying probabilities with a fixed argmax are not a constant classifier.",
                "gradient_effective": "Every terminal-window epoch finite=true and finite mean/min/max/fraction, mean>epsilon and nonzero batch fraction>0, for each group separately.",
                "gradient_norm_stats": "Within-case mean/min of epoch means in the terminal window, then population statistics across cases.",
                "regression": "(last-window mean - best contiguous-window mean after epoch zero) / max(best mean,1e-12).",
                "CV": "Population std / mean inside the terminal loss window; descriptive, without an added acceptance threshold.",
                "failure_categories": "Nonexclusive diagnostic categories; dead activation, constant outputs, class concentration and loss regression are distinct.",
                "selection": "All six fixed cases must finish; no best-F1 selection, no F1 acceptance threshold, manual traces still required.",
            }}
