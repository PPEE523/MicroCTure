"""Signed fixed Task 1 comparison and ablation rules.

New runs sign the current merged implementation. Historical runs retain their
original signatures and are never relabeled as runs of this source version."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from microcture.layout import resolve

from .data import _json_hash
from .models import DEFAULT_CONFIG
from .training import environment_record
from .ablation import neural_jobs


PROTOCOL = "task1_submission_target_aware_v1"
BACKBONES = ("m2", "m2plus", "cnn_transformer")
CONDITIONS = ("I3", "I0", "G0")
BASELINES = ("pca_lr", "length_only", "mask_only", "mask_length")
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
ACCEPTANCE = {"window": 10, "min_relative_loss_drop": 0.01, "logit_variance_threshold": 1e-6,
              "constant_epsilon": 1e-6, "gradient_norm_epsilon": 1e-10,
              "activation_epsilon": 1e-6, "majority_warning_fraction": 0.95,
              "max_terminal_loss_regression": 0.05, "CHID_F1_threshold": None}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def json_signature(value) -> str:
    return _json_hash(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root: Path, value: str | Path) -> Path:
    return resolve(root, value)


def validate_protocol(config: dict) -> dict:
    _require(config.get("protocol") == PROTOCOL, "Use the final submission protocol")
    for key, expected in (("backbones", list(BACKBONES)), ("conditions", list(CONDITIONS)),
                          ("folds", list(FOLDS)), ("seeds", list(SEEDS))):
        _require(config.get(key) == expected, f"Keep registered {key}={expected}")
    fixed = {"head": "H3", "epochs": 40, "learning_rate": 1e-4, "weight_decay": 1e-4,
             "batch_size": 32, "cpu_threads": 4, "optimizer": "AdamW",
             "loss": "class_weighted_cross_entropy", "augmentation": "reverse", "dropout": 0.2,
             "early_stopping": False, "scheduler": False, "gradient_clipping": False,
             "checkpoint_selection": "final_epoch", "refit": False, "deterministic_algorithms": True}
    for key, expected in fixed.items():
        value = config.get("training", {}).get(key)
        _require(value == expected and (not isinstance(expected, (int, bool)) or type(value) is type(expected)),
                 f"Keep registered training.{key}={expected!r}")
    for key, expected in {"names": list(BASELINES), "seed": 42, "pca_components": 16,
                          "C": 1.0, "max_iter": 3000, "tol": 1e-5}.items():
        _require(config.get("baseline", {}).get(key) == expected, f"Keep registered baseline.{key}={expected}")
    _require(config.get("models") == DEFAULT_CONFIG, "Keep all three registered backbone configurations")
    _require(config.get("acceptance") == ACCEPTANCE, "Keep the original sustained-learning thresholds")
    for key in ("cache_dir", "frozen_run_dir", "run_dir"):
        _require(isinstance(config.get(key), str) and bool(config[key]), f"Missing {key}")
    return config


def load_protocol(path: Path) -> dict:
    return validate_protocol(json.loads(resolve(path).read_text(encoding="utf-8")))


def planned_jobs(config: dict, stage="all", model="all", condition=None, fold=None, seed=None) -> list[dict]:
    validate_protocol(config)
    _require(stage in ("main", "ablations", "baselines", "all"), "Choose main/ablations/baselines/all")
    _require(model == "all" or model in BACKBONES + BASELINES, "Unknown model")
    _require(condition is None or condition in CONDITIONS, "Unknown input condition")
    _require(fold is None or (type(fold) is int and fold in FOLDS), "Use frozen fold 0-4")
    _require(seed is None or (type(seed) is int and seed in SEEDS), "Use seed 42/43/44")
    jobs = neural_jobs(stage, BACKBONES, SEEDS, FOLDS)
    names = ["pca_lr"] if stage == "main" else list(BASELINES[1:]) if stage == "baselines" else list(BASELINES) if stage == "all" else []
    jobs += [{"stage": "formal", "kind": "baseline", "model": name, "condition": "I3", "fold": number,
              "seed": 42, "input_variant": "I3", "name": name} for name in names for number in FOLDS]
    return [job for job in jobs if (model == "all" or job["model"] == model)
            and (condition is None or job["condition"] == condition) and (fold is None or job["fold"] == fold)
            and (seed is None or job["seed"] == seed)]


def job_directory(run_dir: Path, job: dict) -> Path:
    _require(job.get("model") in BACKBONES + BASELINES and job.get("condition") in CONDITIONS,
             "Unsafe job model or condition")
    _require(type(job.get("fold")) is int and job["fold"] in FOLDS
             and type(job.get("seed")) is int and job["seed"] in SEEDS, "Unsafe job fold or seed")
    return Path(run_dir) / "runs" / job["condition"] / job["model"] / f"seed_{job['seed']}" / f"fold_{job['fold']}"


def source_signature(root: Path) -> str:
    paths = sorted((Path(root) / "microcture" / "tasks" / "task1").rglob("*.py"))
    paths += [Path(root) / "microcture" / "layout.py"]
    if (Path(root) / "configs/layout.json").is_file():
        paths.append(Path(root) / "configs/layout.json")
    paths += [resolve(root, f"scripts/{name}") for name in ("run_task1_local.py", "prepare_task1.py", "predict_task1_final.py", "check_task1_final.py")]
    return json_signature({str(path.relative_to(root)): sha256(path) for path in paths})


def fold_record(directory: Path, number: int) -> tuple[dict, dict | None]:
    record = json.loads((resolve(directory) / "folds" / f"fold_{number}.json").read_text(encoding="utf-8"))
    if "frozen_fold" in record:
        return record["frozen_fold"], record.get("preprocessing")
    return record, None


def metadata_for_check(root: Path, config: dict) -> tuple[dict, list[dict], str]:
    cache = project_path(root, config["cache_dir"])
    if (cache / "metadata.json").is_file():
        metadata_path, folds_path, source = cache / "metadata.json", cache / "folds.json", "configured count cache metadata"
    else:
        reference = project_path(root, config["frozen_run_dir"])
        metadata_path, folds_path, source = reference / "metadata.json", reference / "cache_folds.json", "packaged reference metadata; no raw/cache required"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cached = json.loads(folds_path.read_text(encoding="utf-8"))
    _require(metadata["signature"] == json_signature({"metadata": {k: v for k, v in metadata.items() if k != "signature"}, "folds": cached}),
             "Dataset metadata/fold signature mismatch")
    return metadata, cached, source


def prepare_manifest(root: Path, config: dict, requested_device="auto") -> tuple[dict, dict]:
    validate_protocol(config)
    _require(requested_device in ("auto", "cpu", "cuda"), "Unknown device")
    metadata, cached, provenance = metadata_for_check(root, config)
    ids = [row["structure_id"] for row in metadata["structures"]]
    _require(len(ids) == len(set(ids)), "Duplicate annotation IDs")
    groups = dict(zip(ids, metadata["group_id"]))
    signatures, partitions, heldout = {}, {}, []
    directory = project_path(root, config["frozen_run_dir"])
    for number in FOLDS:
        record, target_fit = fold_record(directory, number)
        _require(record["fold"] == number, "Frozen fold index differs")
        sections = {key: record[key] for key in ("train_ids", "validation_ids", "test_ids")}
        combined = [sid for values in sections.values() for sid in values]
        _require(len(combined) == len(set(combined)) and set(combined) == set(ids), "Frozen fold must partition all structures")
        group_sets = [{groups[sid] for sid in values} for values in sections.values()]
        _require(all(not group_sets[i] & group_sets[j] for i in range(3) for j in range(i)), "Spatial group leakage")
        fit = record["preprocessing"]
        _require(fit["signature"] == json_signature({k: v for k, v in fit.items() if k != "signature"}), "Signal fit signature changed")
        _require(fit["dataset_signature"] == metadata["signature"] and set(fit["train_ids"]) == set(record["train_ids"]),
                 "Signal fit does not belong to these inner-training IDs/dataset")
        if target_fit is not None:
            _require(target_fit["signature"] == json_signature({k: v for k, v in target_fit.items() if k != "signature"})
                     and target_fit["frozen_fold_signature"] == json_signature(record), "Target preprocessing signature changed")
        signatures[str(number)] = json_signature(record)
        partitions[str(number)] = copy.deepcopy(sections)
        heldout.extend(sections["test_ids"])
    _require(len(heldout) == len(ids) and set(heldout) == set(ids), "Outer folds must cover each structure once")
    manifest = {"protocol": PROTOCOL, "config": copy.deepcopy(config), "source_signature": source_signature(root),
                "dataset_signature": metadata["signature"], "fold_signatures": signatures, "fold_ids": partitions,
                "n_structures": len(ids), "label_order": metadata["label_order"], "environment": environment_record(),
                "requested_device": requested_device, "metadata_source": provenance,
                "historical_reference_directory": config["frozen_run_dir"],
                "evaluation_scope": "developmental spatial CV on previously inspected folds; classification given candidate intervals, not automatic discovery",
                "selection_protocol": "fixed final epoch40, no outer tuning, no train+validation refit",
                "job_counts": {stage: len(planned_jobs(config, stage)) for stage in ("main", "ablations", "baselines", "all")}}
    manifest["experiment_signature"] = json_signature(manifest)
    return manifest, metadata
