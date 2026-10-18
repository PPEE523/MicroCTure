"""Read-only protocol checks and deterministic job planning for formal task one.

Preflight reads JSON, source files and file metadata. It never opens a count
band, builds an image, fits a transform, or creates a result directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from microcture.task1.training import environment_record


PROTOCOL = "task1_target_aware_spatial_cv_v1"
BACKBONES = ("m2", "m2plus", "cnn_transformer")
CONDITIONS = ("I3", "I0", "G0")
BASELINES = ("pca_lr", "length_only", "mask_only", "mask_length")
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
SMALL_FILE_LIMIT = 2 * 1024 * 1024
ACCEPTANCE = {
    "window": 10, "min_relative_loss_drop": 0.01,
    "logit_variance_threshold": 1e-6, "constant_epsilon": 1e-6,
    "gradient_norm_epsilon": 1e-10, "activation_epsilon": 1e-6,
    "majority_warning_fraction": 0.95, "max_terminal_loss_regression": 0.05,
    "CHID_F1_threshold": None,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _exact(actual, expected, description: str) -> None:
    # bool must not silently pass an integer setting (True == 1).
    if isinstance(expected, (bool, int)):
        _require(type(actual) is type(expected) and actual == expected,
                 f"Freeze {description}={expected!r}")
    else:
        _require(actual == expected, f"Freeze {description}={expected!r}")


def validate_protocol(config: dict) -> dict:
    """Reject silent changes to this registered comparison; return no mutation."""
    _exact(config.get("protocol"), PROTOCOL, "protocol")
    for key, expected in (("backbones", list(BACKBONES)), ("conditions", list(CONDITIONS)),
                          ("folds", list(FOLDS)), ("seeds", list(SEEDS))):
        _exact(config.get(key), expected, key)
        if key in ("folds", "seeds"):
            _require(all(type(value) is int for value in config[key]), f"{key} must contain integers")
    for key in ("cache_dir", "frozen_run_dir", "controlled_evidence_dir", "failed_evidence_dir", "run_dir"):
        _require(isinstance(config.get(key), str) and bool(config[key].strip()), f"{key} must be a path string")
    settings = config.get("training", {})
    fixed = {"epochs": 40, "learning_rate": 1e-4, "weight_decay": 1e-4,
             "batch_size": 32, "cpu_threads": 4, "optimizer": "AdamW",
             "loss": "class_weighted_cross_entropy", "augmentation": "reverse",
             "dropout": 0.2, "early_stopping": False, "scheduler": False,
             "gradient_clipping": False, "head": "H3",
             "checkpoint_selection": "final_epoch", "refit": False,
             "deterministic_algorithms": True}
    for key, value in fixed.items():
        _exact(settings.get(key), value, f"training.{key}")
    baseline = config.get("baseline", {})
    for key, value in {"names": list(BASELINES), "seed": 42, "pca_components": 16,
                       "C": 1.0, "max_iter": 3000, "tol": 1e-5}.items():
        _exact(baseline.get(key), value, f"baseline.{key}")
    _exact(config.get("acceptance"), ACCEPTANCE, "acceptance")
    _require(isinstance(config.get("models"), dict), "models must preserve the frozen backbone configuration")
    # Full resolved-path, ancestor and symlink checks are performed with the
    # explicit project root in prepare_manifest, rather than guessing a root
    # from the config filename.
    return config


def load_protocol(path: Path) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def planned_jobs(config: dict, stage: str = "all", model: str = "all", condition: str | None = None,
                 fold: int | None = None, seed: int | None = None) -> list[dict]:
    """155 registered fits; filters select subsets without changing the manifest."""
    validate_protocol(config)
    _require(stage in ("all", "main", "ablations", "baselines"), "Choose all/main/ablations/baselines")
    _require(model == "all" or model in BACKBONES + BASELINES, "Unknown registered model")
    _require(condition is None or condition in CONDITIONS, "Choose a registered I3/I0/G0 input condition")
    _require(fold is None or (type(fold) is int and fold in FOLDS), "Choose a frozen fold 0-4")
    _require(seed is None or (type(seed) is int and seed in SEEDS), "Choose a registered seed 42/43/44")
    jobs = []
    neural_conditions = (["I3"] if stage == "main" else ["I0", "G0"] if stage == "ablations"
                         else list(CONDITIONS) if stage == "all" else [])
    for input_condition in neural_conditions:
        for backbone in config["backbones"]:
            for paired_seed in config["seeds"]:
                for frozen_fold in config["folds"]:
                    jobs.append({"stage": "formal", "kind": "neural", "model": backbone,
                                 "backbone": backbone, "condition": input_condition,
                                 "input_variant": input_condition, "fold": frozen_fold,
                                 "seed": paired_seed, "head": "H3", "name": "H3",
                                 "learning_rate": config["training"]["learning_rate"]})
    if stage in ("all", "baselines"):
        for name in config["baseline"]["names"]:
            for frozen_fold in config["folds"]:
                jobs.append({"stage": "formal", "kind": "baseline", "model": name,
                             "condition": "I3", "fold": frozen_fold,
                             "seed": config["baseline"]["seed"], "input_variant": "I3", "name": name})
    return [job for job in jobs if (model == "all" or job["model"] == model)
            and (condition is None or job["condition"] == condition)
            and (fold is None or job["fold"] == fold) and (seed is None or job["seed"] == seed)]


def job_directory(run_dir: Path, job: dict) -> Path:
    _require(job.get("model") in BACKBONES + BASELINES and job.get("condition") in CONDITIONS,
             "Unsafe or unknown job model/input")
    _require(type(job.get("fold")) is int and job["fold"] in FOLDS
             and type(job.get("seed")) is int and job["seed"] in SEEDS, "Unsafe fold/seed")
    return Path(run_dir) / "runs" / job["condition"] / job["model"] / f"seed_{job['seed']}" / f"fold_{job['fold']}"


def _package_signature(root: Path, package: str, entrypoint: str) -> str:
    paths = sorted((root / "microcture" / package).glob("*.py"))
    _require(bool(paths), f"No source found for {package}")
    paths.append(root / "scripts" / entrypoint)
    return json_signature({str(path.relative_to(root)): sha256(path) for path in paths})


def source_signature(root: Path) -> str:
    return _package_signature(Path(root).resolve(), "task1_formal", "run_task1_formal.py")


def validate_run_directory(root: Path, config: dict) -> Path:
    """Reject protected descendants, protected ancestors and symlink aliases."""
    root = Path(root).resolve()
    supplied = Path(config["run_dir"])
    unresolved = supplied if supplied.is_absolute() else root / supplied
    resolved = unresolved.resolve()
    _require(root / "results" in resolved.parents, "Formal run_dir must be inside project results/")
    # Inspect the supplied components before resolve hides a symlink alias.
    current = unresolved
    for parent in (current, *current.parents):
        if parent == root:
            break
        _require(not parent.is_symlink(), "Formal run_dir and its parents must not be symlinks")
    protected = [root / "data", root / "evidence", root / "micro-c数据",
                 project_path(root, config["frozen_run_dir"]),
                 project_path(root, config["controlled_evidence_dir"]),
                 project_path(root, config["failed_evidence_dir"]),
                 root / "results/task1_controlled_v1", root / "reports/task1_controlled_review_v1"]
    for tree in (root / "results", root / "reports"):
        if tree.exists():
            protected.extend(marker.parent for marker in tree.rglob("FROZEN.json"))
    for path in protected:
        path = path.resolve()
        _require(resolved != path and path not in resolved.parents and resolved not in path.parents,
                 f"Formal run_dir overlaps protected data/evidence: {path}")
    return resolved


def _verify_freeze_metadata(root: Path, evidence_dir: Path) -> tuple[dict, dict]:
    """Check immutable inventory and hash small provenance files, without bands.

    This is deliberately not advertised as a fresh full-content verification
    of checkpoints/count bands; their full SHA256 index was frozen separately.
    """
    manifest_path = evidence_dir / "freeze_manifest.json"
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads((project_path(root, freeze["run_dir"]) / "FROZEN.json").read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest_path)
    _require(marker["evidence_manifest_sha256"] == manifest_hash, "Frozen evidence manifest differs from marker")
    checked_bytes, hashed_files = 0, 0
    indexed_paths = set()
    for record in freeze["files"]:
        path = project_path(root, record["path"])
        _require(root in path.parents, "Frozen inventory points outside the project")
        _require(str(path) not in indexed_paths, "Duplicate frozen inventory path")
        indexed_paths.add(str(path))
        _require(path.is_file() and path.stat().st_size == record["size_bytes"],
                 f"Frozen file missing/size changed: {record['path']}")
        if record["size_bytes"] <= SMALL_FILE_LIMIT:
            _require(sha256(path) == record["sha256"], f"Frozen small-file hash changed: {record['path']}")
            hashed_files += 1
            checked_bytes += record["size_bytes"]
    return freeze, {"evidence_manifest_sha256": manifest_hash, "files_stat_checked": len(freeze["files"]),
                    "small_files_sha256_checked": hashed_files, "bytes_hashed": checked_bytes,
                    "full_large_file_sha256_recheck": False,
                    "scope": "Inventory sizes and small-file SHA256; count bands and large checkpoints not opened"}


def _validate_cache_metadata(root: Path, cache_dir: Path, expected_signature: str) -> tuple[dict, list[dict]]:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    cached_folds = json.loads((cache_dir / "folds.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in metadata.items() if key != "signature"}
    _require(metadata["signature"] == expected_signature
             and json_signature({"metadata": unsigned, "folds": cached_folds}) == expected_signature,
             "Frozen dataset metadata/fold signature changed")
    _require(project_path(root, metadata["project_root"]) == root, "Cache project root changed")
    for relative, expected in metadata["input_sha256"].items():
        _require(sha256(project_path(root, relative)) == expected, f"Annotation/config source changed: {relative}")
    input_config = metadata["config"]
    for key, value in {"resolution_bp": 100, "window_bins": [64, 256], "output_size": 64,
                       "guard_bins": 10, "normalization": "oe", "train_augmentation": "reverse"}.items():
        _exact(input_config[key], value, f"cache.data.{key}")
    for sample in metadata["samples"]:
        band = cache_dir / sample["band_file"]
        band_stat = band.stat()
        _require(band_stat.st_size == sample["band_size_bytes"]
                 and band_stat.st_mtime_ns == sample["band_mtime_ns"], "Frozen count band metadata changed")
        raw = Path(sample["raw_path"])
        _require(raw.is_file(), "Original raw matrix must remain available for formal comparison")
        raw_stat = raw.stat()
        _require(raw_stat.st_size == sample["raw_size_bytes"]
                 and raw_stat.st_mtime_ns == sample["raw_mtime_ns"], "Original raw matrix metadata changed")
    return metadata, cached_folds


def _validate_folds(root: Path, config: dict, metadata: dict, cached_folds: list[dict]) -> tuple[dict, dict]:
    ids = [row["structure_id"] for row in metadata["structures"]]
    _require(len(ids) == len(set(ids)), "Unique structures required")
    _require(len(ids) == len(metadata["group_id"]), "Spatial group count differs from structures")
    lookup = dict(zip(ids, metadata["group_id"]))
    labels = {row["structure_id"]: row["type"] for row in metadata["structures"]}
    _exact(metadata["label_order"], ["OPCID", "CHIN", "CHID"], "label_order")
    cached = {row["fold"]: row for row in cached_folds}
    _require(set(cached) == set(FOLDS), "All five frozen outer folds are required")
    signatures, fold_ids, heldout = {}, {}, []
    for number in FOLDS:
        path = project_path(root, config["frozen_run_dir"]) / "folds" / f"fold_{number}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        _exact(record["fold"], number, "fold number")
        partitions = {key: record[key] for key in ("train_ids", "validation_ids", "test_ids")}
        combined = [item for values in partitions.values() for item in values]
        _require(len(combined) == len(set(combined)) and set(combined) == set(ids),
                 "Frozen train/validation/test must partition all unique structures")
        group_sets = [{lookup[item] for item in values} for values in partitions.values()]
        _require(all(not group_sets[i].intersection(group_sets[j]) for i in range(3) for j in range(i)),
                 "Spatial groups cross a frozen split")
        _require(set(partitions["test_ids"]) == set(cached[number]["test_ids"])
                 and set(partitions["train_ids"] + partitions["validation_ids"]) == set(cached[number]["train_ids"]),
                 "Original inner fold differs from the frozen outer split")
        for key, values in partitions.items():
            _require(set(labels[item] for item in values) == set(metadata["label_order"]),
                     f"Frozen {key} must contain all classes")
        fit = record["preprocessing"]
        _require(fit["dataset_signature"] == metadata["signature"], "Signal fit dataset differs")
        _require(fit["signature"] == json_signature({key: value for key, value in fit.items() if key != "signature"}),
                 "Signal preprocessing signature changed")
        _require(fit["train_ids"] == partitions["train_ids"], "Signal fit is not the exact inner-training ID list")
        _require(fit["fit_scope"] == "training structure footprints and training replicate images only",
                 "Signal statistics must belong to inner training only")
        _require(len(fit["expected"]) == len(metadata["samples"]) and len(fit["scalers"]) == 2,
                 "Frozen signal fit must contain both WT replicate expectations and scale transforms")
        for replicate in fit["expected"]:
            _require(all(math.isfinite(value) and value >= 0 for value in replicate["expected"]), "Invalid frozen OE")
            _require(replicate["training_footprint_bins"] > 0, "Empty OE training footprint")
        for scaler in fit["scalers"]:
            _require(math.isfinite(scaler["mean"]) and math.isfinite(scaler["std"]) and scaler["std"] > 0
                     and scaler["fitted_valid_pooled_pixels"] > 0, "Invalid inner-training signal scaler")
        signatures[str(number)] = json_signature(record)
        fold_ids[str(number)] = copy.deepcopy(partitions)
        heldout.extend(partitions["test_ids"])
    _require(len(heldout) == len(ids) and set(heldout) == set(ids), "Outer folds do not cover each unique structure once")
    return signatures, fold_ids


def prepare_manifest(root: Path, config: dict, requested_device: str = "auto") -> tuple[dict, dict]:
    """Build immutable run provenance without writes, image extraction or fitting."""
    root = Path(root).resolve()
    validate_protocol(config)
    _require(requested_device in ("auto", "cpu", "cuda"), "Choose auto/cpu/cuda")
    validate_run_directory(root, config)
    failed, failed_check = _verify_freeze_metadata(root, project_path(root, config["failed_evidence_dir"]))
    controlled, controlled_check = _verify_freeze_metadata(root, project_path(root, config["controlled_evidence_dir"]))
    _require(project_path(root, failed["run_dir"]) == project_path(root, config["frozen_run_dir"])
             and project_path(root, failed["cache_dir"]) == project_path(root, config["cache_dir"]),
             "Use the exact frozen comparison and count-cache paths")
    old_signature = _package_signature(root, "task1", "run_task1_comparison.py")
    control_signature = _package_signature(root, "task1_controlled", "run_task1_controlled.py")
    _require(old_signature == failed["source_signature"], "Original comparison source changed")
    _require(control_signature == controlled["source_signature"], "Controlled comparison source changed")
    _require(controlled["dataset_signature"] == failed["dataset_signature"]
             and controlled["original_run_signature"] == failed["run_signature"], "Controlled evidence uses another dataset/run")
    _require(controlled["old_evidence_reference"]["sha256"] == failed_check["evidence_manifest_sha256"],
             "Controlled evidence does not reference this failed experiment")
    _require(controlled["strict_pass_counts"]["H3"] == 6
             and controlled["strict_pass_denominator_per_head"] == 6,
             "H3 default requires all six frozen sustained-learning passes")
    control_manifest = json.loads((project_path(root, controlled["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    _require(control_manifest["experiment_signature"] == controlled["experiment_signature"]
             and json_signature({k: v for k, v in control_manifest.items() if k != "experiment_signature"})
             == control_manifest["experiment_signature"], "Controlled run manifest signature changed")
    _exact(config["acceptance"], control_manifest["config"]["acceptance"], "original acceptance thresholds")
    cases = control_manifest["config"]["head_stage"]["cases"]
    _require(len(cases) == 6, "Expected the six fixed failed development cases")
    for name, case in cases.items():
        path = project_path(root, controlled["run_dir"]) / "heads" / name / "H3/metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        job = metrics["job"]
        _require(metrics["status"] == "completed" and metrics["acceptance"]["accepted"] is True
                 and metrics["experiment_signature"] == controlled["experiment_signature"]
                 and job["head"] == "H3" and job["input_variant"] == "I3"
                 and job["learning_rate"] == 1e-4
                 and all(job[key] == case[key] for key in ("backbone", "fold", "seed")),
                 f"Frozen H3 support is missing or mismatched: {name}")
    original_manifest = json.loads((project_path(root, failed["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    _require(original_manifest["run_signature"] == failed["run_signature"]
             and json_signature({k: v for k, v in original_manifest.items() if k != "run_signature"})
             == original_manifest["run_signature"], "Original run manifest signature changed")
    _exact(config["models"], original_manifest["config"]["models"], "original backbone configuration")
    metadata, cached_folds = _validate_cache_metadata(root, project_path(root, config["cache_dir"]), failed["dataset_signature"])
    fold_signatures, fold_ids = _validate_folds(root, config, metadata, cached_folds)
    environment = environment_record()
    manifest = {
        "protocol": config["protocol"], "config": copy.deepcopy(config), "source_signature": source_signature(root),
        "dependency_source_signature": {"task1": old_signature, "task1_controlled": control_signature},
        "controlled_evidence_sha256": controlled_check["evidence_manifest_sha256"],
        "failed_evidence_sha256": failed_check["evidence_manifest_sha256"],
        "dataset_signature": metadata["signature"], "fold_signatures": fold_signatures,
        "fold_ids": fold_ids, "n_structures": len(metadata["structures"]), "label_order": metadata["label_order"],
        "environment_record": environment, "environment": environment, "requested_device": requested_device,
        "evaluation_scope": "developmental spatial-group cross-validation on previously inspected outer folds; candidate-interval classification, not automatic discovery",
        "selection_protocol": "fixed epoch 40, no early stopping, no outer tuning, no inner-validation refit",
        "controlled_support": {"head": "H3", "passed_cases": 6, "expected_cases": 6,
                               "recognition_performance_claim": False},
        "evidence_preflight": {"failed": failed_check, "controlled": controlled_check},
        "job_counts": {stage: len(planned_jobs(config, stage)) for stage in ("main", "ablations", "baselines", "all")},
    }
    manifest["experiment_signature"] = json_signature(manifest)
    return manifest, metadata
