"""Frozen 60-job O0/O1 protocol and metadata-only preflight."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from microcture.task1_formal.protocol import (ACCEPTANCE, FOLDS, SEEDS, _package_signature,
    json_signature, project_path, sha256, prepare_manifest as prepare_formal_manifest)
from .models import BETA, VARIANTS

PROTOCOL = "task1_m2plus_target_spatial_readout_v1"
CONDITIONS = ("I3", "G0")
FORMAL_SOURCE = "2de4e177d4557773bf61183c090a1a67db002325cb999a7d24d3cfa410013a8c"
FORMAL_EXPERIMENT = "8467176519680652c219a2fd993847048dddaa0344f27c8afb3ada2d0c1c5135"


def _require(value: bool, message: str):
    if not value:
        raise ValueError(message)


def validate_protocol(config: dict) -> dict:
    for key, expected in {"protocol": PROTOCOL, "variants": list(VARIANTS),
                          "conditions": list(CONDITIONS), "folds": list(FOLDS),
                          "seeds": list(SEEDS), "gate_beta": BETA,
                          "acceptance": ACCEPTANCE}.items():
        _require(config.get(key) == expected, f"Freeze {key}={expected!r}")
    for key in ("folds", "seeds"):
        _require(all(type(value) is int for value in config[key]), f"{key} must contain integers")
    for key in ("run_dir", "formal_run_dir", "cache_dir", "frozen_run_dir"):
        _require(isinstance(config.get(key), str) and bool(config[key].strip()), f"Invalid {key}")
    _require(isinstance(config.get("training"), dict) and isinstance(config.get("models"), dict),
             "Training and model settings must be explicit")
    fixed = {"epochs": 40, "learning_rate": 1e-4, "weight_decay": 1e-4,
             "batch_size": 32, "cpu_threads": 4, "optimizer": "AdamW",
             "loss": "class_weighted_cross_entropy", "augmentation": "reverse",
             "dropout": 0.2, "early_stopping": False, "scheduler": False,
             "gradient_clipping": False, "head": "H3",
             "checkpoint_selection": "final_epoch", "refit": False,
             "deterministic_algorithms": True}
    _require(config["training"] == fixed, "Readout control must preserve all formal training settings")
    return config


def load_protocol(path: Path) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def planned_jobs(config: dict, variant: str = "all", condition: str | None = None,
                 fold: int | None = None, seed: int | None = None) -> list[dict]:
    validate_protocol(config)
    _require(variant == "all" or variant in VARIANTS, "Choose all/O0/O1")
    _require(condition is None or condition in CONDITIONS, "Choose I3/G0")
    _require(fold is None or type(fold) is int and fold in FOLDS, "Choose fold 0-4")
    _require(seed is None or type(seed) is int and seed in SEEDS, "Choose seed 42/43/44")
    return [{"stage": "readout", "kind": "neural", "model": name, "variant": name,
             "backbone": "m2plus", "condition": input_name, "input_variant": input_name,
             "fold": number, "seed": paired_seed, "head": "H3", "name": name,
             "learning_rate": config["training"]["learning_rate"]}
            for input_name in CONDITIONS for name in VARIANTS for paired_seed in SEEDS for number in FOLDS
            if (variant == "all" or name == variant) and (condition is None or input_name == condition)
            and (fold is None or number == fold) and (seed is None or paired_seed == seed)]


def job_directory(run_dir: Path, job: dict) -> Path:
    _require(job.get("model") in VARIANTS and job.get("condition") in CONDITIONS, "Unsafe variant/condition")
    _require(type(job.get("fold")) is int and job["fold"] in FOLDS
             and type(job.get("seed")) is int and job["seed"] in SEEDS, "Unsafe fold/seed")
    return Path(run_dir) / "runs" / job["condition"] / job["model"] / f"seed_{job['seed']}" / f"fold_{job['fold']}"


def source_signature(root: Path) -> str:
    return _package_signature(root.resolve(), "task1_readout", "run_task1_readout.py")


def prepare_manifest(root: Path, config: dict, requested_device: str = "auto") -> tuple[dict, dict]:
    """Verify frozen dependencies and IDs without count/image preprocessing."""
    root = root.resolve()
    validate_protocol(config)
    supplied_formal = Path(config["formal_run_dir"])
    supplied_formal = supplied_formal if supplied_formal.is_absolute() else root / supplied_formal
    _require(not any(path.is_symlink() for path in (supplied_formal, *supplied_formal.parents)),
             "Formal reference paths must not use symbolic links")
    formal_dir = project_path(root, config["formal_run_dir"])
    reference = json.loads((formal_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(formal_dir == project_path(root, reference["config"]["run_dir"]),
             "Formal reference must be read at its original recorded run path")
    _require(reference["experiment_signature"] == FORMAL_EXPERIMENT
             and json_signature({k: v for k, v in reference.items() if k != "experiment_signature"}) == FORMAL_EXPERIMENT,
             "The exact formal reference manifest is required")
    _require(reference["source_signature"] == FORMAL_SOURCE, "Formal reference source differs")
    dependency, metadata = prepare_formal_manifest(root, reference["config"], requested_device)
    _require(dependency["source_signature"] == FORMAL_SOURCE, "Frozen formal source changed")
    for key in ("training", "models", "acceptance", "folds", "seeds", "cache_dir", "frozen_run_dir"):
        _require(config[key] == reference["config"][key], f"Readout control must preserve formal {key}")
    for key in ("dataset_signature", "fold_signatures", "fold_ids", "dependency_source_signature"):
        _require(dependency[key] == reference[key], f"Frozen formal {key} changed")
    from .runner import guard_output
    guard_output(root, project_path(root, config["run_dir"]), formal_dir)
    manifest = {"protocol": PROTOCOL, "config": copy.deepcopy(config),
                "source_signature": source_signature(root),
                "dependency_source_signature": {**dependency["dependency_source_signature"], "task1_formal": FORMAL_SOURCE},
                "formal_reference": {"run_dir": str(formal_dir.relative_to(root)),
                                     "manifest_sha256": sha256(formal_dir / "manifest.json"),
                                     "experiment_signature": FORMAL_EXPERIMENT,
                                     "reference_only": True, "O0_retraining_required": True},
                "dataset_signature": dependency["dataset_signature"],
                "fold_signatures": dependency["fold_signatures"], "fold_ids": dependency["fold_ids"],
                "n_structures": len(metadata["structures"]), "label_order": metadata["label_order"],
                "failed_evidence_sha256": dependency["failed_evidence_sha256"],
                "controlled_evidence_sha256": dependency["controlled_evidence_sha256"],
                "environment_record": dependency["environment_record"], "requested_device": requested_device,
                "job_count": 60, "parameter_count_per_variant": 110437,
                "single_changed_factor": "fixed beta=0.5 target gate after 4x4 feature pooling",
                "selection_protocol": "fixed epoch40 endpoint; no refit/early stopping/outer tuning",
                "evaluation_scope": "developmental paired spatial CV on previously inspected data; no biological morphology proof"}
    manifest["experiment_signature"] = json_signature(manifest)
    return manifest, metadata
