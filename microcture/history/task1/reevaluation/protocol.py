"""Frozen paired checkpoint policies, original spatial data and model families."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from microcture.task1_formal.protocol import (ACCEPTANCE, BACKBONES, BASELINES,
    CONDITIONS, FOLDS, SEEDS, _package_signature, json_signature, project_path,
    sha256, planned_jobs as formal_jobs, prepare_manifest as prepare_formal_manifest)

PROTOCOL = "task1_paired_inner_ce_checkpoint_spatial_cv_v1"
FORMAL_SOURCE = "2de4e177d4557773bf61183c090a1a67db002325cb999a7d24d3cfa410013a8c"
FORMAL_EXPERIMENT = "8467176519680652c219a2fd993847048dddaa0344f27c8afb3ada2d0c1c5135"
READOUT_SOURCE = "414d1c39d477bf4f120ad66ddc66e44ed68a1f00c8d81b9c3d495aa5a9cf6a0a"
READOUT_EXPERIMENT = "3d9e949334434e824f1c47f431e1672fa7d7c1e9b2228210d70d877d646b9fae"
SELECTION = {"training_epochs": 40, "policies": ["final40", "inner_ce"],
             "inner_ce_epochs": [1, 40], "inner_ce_metric": "weighted_validation_ce",
             "tie_break": "earliest", "comparison_rule": "strict_less",
             "early_stopping": False, "refit": False}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _formal_config(config: dict) -> dict:
    result = {key: copy.deepcopy(value) for key, value in config.items()
              if key not in {"formal_run_dir", "readout_run_dir", "readout_audit_path", "selection"}}
    result["protocol"] = "task1_target_aware_spatial_cv_v1"
    result["training"]["checkpoint_selection"] = "final_epoch"
    return result


def validate_protocol(config: dict) -> dict:
    from microcture.task1_formal.protocol import validate_protocol as validate_formal
    _require(config.get("protocol") == PROTOCOL, "Use the frozen paired reevaluation protocol")
    _require(config.get("selection") == SELECTION, "Checkpoint policies are preregistered and cannot be changed")
    _require(config.get("training", {}).get("checkpoint_selection") == "paired_final40_inner_ce",
             "Freeze paired_final40_inner_ce checkpoint selection")
    for key in ("formal_run_dir", "readout_run_dir", "readout_audit_path"):
        _require(isinstance(config.get(key), str) and bool(config[key].strip()), f"Invalid reference path: {key}")
    validate_formal(_formal_config(config))
    expected = {"protocol", "cache_dir", "frozen_run_dir", "failed_evidence_dir", "controlled_evidence_dir",
                "run_dir", "folds", "seeds", "conditions", "backbones", "models", "training", "baseline",
                "acceptance", "formal_run_dir", "readout_run_dir", "readout_audit_path", "selection"}
    _require(set(config) == expected, "Unregistered protocol configuration fields")
    return config


def load_protocol(path: Path) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def planned_jobs(config: dict, stage: str = "all", model: str = "all", condition: str | None = None,
                 fold: int | None = None, seed: int | None = None) -> list[dict]:
    validate_protocol(config)
    _require(stage in ("all", "main", "ablations", "baselines"), "Choose main/ablations/baselines/all")
    # Reuse immutable job order and filters while adding PCA to the new main.
    tasks = formal_jobs(_formal_config(config), "all", model, condition, fold, seed)
    if stage == "main":
        tasks = [job for job in tasks if job["condition"] == "I3"
                 and (job["kind"] == "neural" or job["model"] == "pca_lr")]
    elif stage == "ablations":
        tasks = [job for job in tasks if job["kind"] == "neural" and job["condition"] in ("I0", "G0")]
    elif stage == "baselines":
        tasks = [job for job in tasks if job["kind"] == "baseline"]
    return [{**job, "stage": "reevaluation"} for job in tasks]


def endpoint_policies(job: dict) -> tuple[str, ...]:
    _require(job.get("kind") in ("neural", "baseline"), "Unknown fit kind")
    return ("final40", "inner_ce") if job["kind"] == "neural" else ("baseline",)


def job_directory(run_dir: Path, job: dict) -> Path:
    from microcture.task1_formal.protocol import job_directory as original_directory
    return original_directory(run_dir, job)


def endpoint_directory(run_dir: Path, job: dict, policy: str) -> Path:
    _require(policy in endpoint_policies(job), "Unregistered endpoint policy for this fit")
    return job_directory(run_dir, job) / "endpoints" / policy


def source_signature(root: Path) -> str:
    return _package_signature(root.resolve(), "task1_reevaluation", "run_task1_reevaluation.py")


def prepare_manifest(root: Path, config: dict, requested_device: str = "auto") -> tuple[dict, dict]:
    root = root.resolve()
    validate_protocol(config)
    formal_dir = project_path(root, config["formal_run_dir"])
    readout_dir = project_path(root, config["readout_run_dir"])
    for supplied in (config["formal_run_dir"], config["readout_run_dir"], config["readout_audit_path"]):
        path = root / supplied if not Path(supplied).is_absolute() else Path(supplied)
        _require(not any(p.is_symlink() for p in (path, *path.parents)), "Reference symlinks are forbidden")
    reference = json.loads((formal_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(reference["experiment_signature"] == FORMAL_EXPERIMENT
             and json_signature({k: v for k, v in reference.items() if k != "experiment_signature"}) == FORMAL_EXPERIMENT
             and reference["source_signature"] == FORMAL_SOURCE, "Exact formal reference is required")
    _require(formal_dir == project_path(root, reference["config"]["run_dir"]), "Formal reference moved")
    dependency, metadata = prepare_formal_manifest(root, reference["config"], requested_device)
    _require(dependency["source_signature"] == FORMAL_SOURCE, "Frozen formal code changed")
    reconstructed = _formal_config(config)
    reconstructed["run_dir"] = reference["config"]["run_dir"]
    _require(reconstructed == reference["config"], "Only checkpoint policy and new protocol/output/reference fields may differ")
    for key in ("dataset_signature", "fold_ids", "fold_signatures", "dependency_source_signature"):
        _require(dependency[key] == reference[key], f"Frozen formal {key} changed")
    readout = json.loads((readout_dir / "manifest.json").read_text(encoding="utf-8"))
    audit_path = project_path(root, config["readout_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    from microcture.task1_readout.protocol import source_signature as readout_source
    _require(readout["experiment_signature"] == READOUT_EXPERIMENT
             and json_signature({k: v for k, v in readout.items() if k != "experiment_signature"}) == READOUT_EXPERIMENT
             and readout["source_signature"] == readout_source(root) == READOUT_SOURCE,
             "Readout decision reference/source changed")
    _require(readout_dir == project_path(root, readout["config"]["run_dir"]), "Readout reference moved")
    _require(audit["status"] == "passed" and audit["experiment_signature"] == READOUT_EXPERIMENT
             and audit["source_signature"] == READOUT_SOURCE
             and audit["manifest_sha256"] == sha256(readout_dir / "manifest.json")
             and audit["checks"]["jobs"] == audit["checks"]["artifact_indices"] == 60,
             "Completed readout integrity audit is required as decision reference")
    _require(audit["dataset_signature"] == dependency["dataset_signature"], "Readout reference dataset differs")
    from .runner import guard_output
    guard_output(root, project_path(root, config["run_dir"]))
    manifest = {"protocol": PROTOCOL, "config": copy.deepcopy(config), "config_signature": json_signature(config),
        "source_signature": source_signature(root),
        "dependency_source_signature": {**dependency["dependency_source_signature"], "task1_formal": FORMAL_SOURCE},
        "formal_reference": {"run_dir": config["formal_run_dir"], "manifest_sha256": sha256(formal_dir / "manifest.json"),
                             "experiment_signature": FORMAL_EXPERIMENT, "reference_only": True, "all_models_retrained": True},
        "readout_decision_reference": {"run_dir": config["readout_run_dir"], "experiment_signature": READOUT_EXPERIMENT,
            "source_signature": READOUT_SOURCE, "audit_path": config["readout_audit_path"], "audit_sha256": sha256(audit_path),
            "O1_rejected": True, "scores_not_used_as_new_model_results": True},
        "dataset_signature": dependency["dataset_signature"], "fold_signatures": dependency["fold_signatures"],
        "fold_ids": dependency["fold_ids"], "label_order": dependency["label_order"], "n_structures": len(metadata["structures"]),
        "failed_evidence_sha256": dependency["failed_evidence_sha256"], "controlled_evidence_sha256": dependency["controlled_evidence_sha256"],
        "environment_record": dependency["environment_record"], "requested_device": requested_device,
        "selection_signature": json_signature(SELECTION), "neural_endpoint_policies": ["final40", "inner_ce"],
        "baseline_policy": "baseline", "job_count": 155, "endpoint_count": 290,
        "job_counts": {stage: len(planned_jobs(config, stage)) for stage in ("main", "ablations", "baselines", "all")},
        "endpoint_counts": {stage: sum(len(endpoint_policies(job)) for job in planned_jobs(config, stage))
                            for stage in ("main", "ablations", "baselines", "all")},
        "evaluation_scope": "Developmental paired checkpoint-policy spatial CV; previously inspected data, no independent morphology truth",
        "selection_protocol": "Same full40 fit: final40 and minimum weighted inner validation CE epochs1..40; earliest exact tie; fixed before outer; no refit"}
    manifest["experiment_signature"] = json_signature(manifest)
    return manifest, metadata
