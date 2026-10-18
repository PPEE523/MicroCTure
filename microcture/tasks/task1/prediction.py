"""Given-interval Task 1 inference; no fitting or candidate discovery.

Loads registered H3 state-dictionary checkpoints directly and checks the exact
saved signal/geometry transformations before predicting supplied intervals."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from microcture.layout import PROJECT_ROOT, resolve

from . import data
from .models import build_controlled_model
from .protocol import json_signature, sha256
from .training import reject_retained_overlap


CLASS_NAMES = ("OPCID", "CHIN", "CHID")


def load_checkpoint(checkpoint: Path, device: torch.device):
    saved = torch.load(resolve(checkpoint), map_location=device, weights_only=True)
    if saved.get("head_version") != "H3" or saved.get("epoch") != 40:
        raise ValueError("Expected an H3 fixed-epoch40 checkpoint")
    if saved.get("job", {}).get("input_variant") != "I3":
        raise ValueError("Inference requires the complete I3 input checkpoint")
    backbone = saved.get("backbone", saved.get("job", {}).get("backbone"))
    model = build_controlled_model(backbone, saved["model_config"], "H3").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved


def load_input_npz(path: Path) -> dict:
    with np.load(resolve(path), allow_pickle=False) as archive:
        samples = {key: archive[key] for key in archive.files}
    n = len(samples.get("structure_id", []))
    if not n or samples.get("x", np.empty(0)).shape != (n, 2, 3, 64, 64) or samples.get("length", np.empty(0)).shape != (n, 1):
        raise ValueError("NPZ requires structure_id, x[N,2,3,64,64] and length[N,1]")
    if not np.isfinite(samples["x"]).all() or not np.isfinite(samples["length"]).all():
        raise ValueError("Nonfinite inference inputs")
    if samples.get("replicate", np.empty(0)).shape != (n,):
        raise ValueError("NPZ requires one replicate identity per observation")
    return samples


def read_intervals(path: Path, chrom_length: int) -> list[dict]:
    with resolve(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"structure_id", "start", "end"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Candidate CSV requires structure_id,start,end, with optional center/chrom; no labels required")
    result, identifiers = [], set()
    for row in rows:
        sid = row["structure_id"]
        start, end = float(row["start"]), float(row["end"])
        center = float(row.get("center") or (start + end) / 2)
        if (not sid or sid in identifiers or not np.isfinite([start, end, center]).all()
                or not 0 <= start <= center < end <= chrom_length):
            raise ValueError(f"Invalid or duplicate candidate interval: {sid}")
        identifiers.add(sid)
        result.append({**row, "structure_id": sid, "start": start, "end": end, "center": center})
    return result


def prepare_candidates(dataset: dict, rows: list[dict], preprocessing: dict) -> dict:
    """Extract provided intervals using saved signal and length fits only.

    The candidate metadata has dummy internal labels, which are neither input
    features nor written as truth. This helper never invokes a fit operation.
    """
    if preprocessing["signature"] != json_signature({key: value for key, value in preprocessing.items() if key != "signature"}):
        raise ValueError("Target preprocessing signature changed")
    if preprocessing["dataset_signature"] != dataset["signature"]:
        raise ValueError("Checkpoint transforms and count cache belong to different source data")
    known_chrom = next((str(row["chrom"]) for row in dataset["metadata"].get("structures", []) if row.get("chrom")), None)
    if known_chrom is not None and any(row.get("chrom") and str(row["chrom"]) != known_chrom for row in rows):
        raise ValueError("Candidate chromosome differs from the prepared single-chromosome data")
    n = len(rows)
    candidate = {**dataset, "structure_id": np.asarray([row["structure_id"] for row in rows]),
                 "center_bin": np.asarray([int(row["center"] // 100) for row in rows], dtype=np.int64),
                 "y": np.zeros(n, dtype=np.int64), "group_id": np.arange(n, dtype=np.int64),
                 "metadata": {**dataset["metadata"], "structures": rows}}
    width = max(dataset["config"]["window_bins"])
    candidate["window_footprints"] = ((candidate["center_bin"][:, None] + np.arange(-width // 2, width // 2))
                                      % int(dataset["n_bins"]))
    signal = preprocessing["signal_preprocessing"]
    if signal["signature"] != json_signature({key: value for key, value in signal.items() if key != "signature"}):
        raise ValueError("Saved signal preprocessing signature changed")
    raw = data._extract_inputs(candidate, np.arange(n), signal["expected"])
    scaled = data._apply_scalers(raw, signal["scalers"])
    return data._add_target_features(candidate, scaled, data._annotation_lookup(candidate), preprocessing["length_scaler"])


def predict_samples(model, samples: dict, device: torch.device, batch_size=32) -> np.ndarray:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(samples["x"]), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(samples["x"][start:start + batch_size], dtype=np.float32)).to(device)
            length = torch.from_numpy(np.ascontiguousarray(samples["length"][start:start + batch_size], dtype=np.float32)).to(device)
            outputs.append(model(x, length).softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(outputs)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite model probabilities")
    return probabilities


def prediction_rows(samples: dict, probabilities: np.ndarray) -> list[dict]:
    n = len(samples["structure_id"])
    if probabilities.shape != (n, 3) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid three-class prediction probabilities")
    groups = defaultdict(list)
    for index, sid in enumerate(samples["structure_id"]):
        groups[str(sid)].append(index)
    rows = []
    for sid, indices in groups.items():
        mean = probabilities[indices].astype(np.float64).mean(axis=0)
        row = {"structure_id": sid, "predicted": CLASS_NAMES[int(mean.argmax())],
               "replicate_count": len(indices)}
        row.update({f"p_{name}": float(value) for name, value in zip(CLASS_NAMES, mean)})
        rows.append(row)
    return rows


def guard_prediction_output(root: Path, output: Path) -> Path:
    """Validate a fresh inference CSV before loading weights and again before writing."""
    root = Path(root).resolve()
    path = resolve(root, output)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Prediction output cannot pass through symbolic links")
    path = path.resolve()
    reject_retained_overlap(root, path, kinds=("result", "report", "cache", "raw", "source",
                                              "config", "annotation", "metadata", "protocol", "history"))
    task_results = root / "results/task1"
    if (not path.is_relative_to(task_results) or len(path.relative_to(task_results).parts) < 3
            or path.suffix.lower() != ".csv"):
        raise ValueError("Prediction output must be a fresh CSV under results/task1/<experiment>/<version>/")
    if any((parent / "FROZEN.json").exists() or (parent / "ARTIFACTS.json").exists()
           for parent in (path, *path.parents)):
        raise ValueError("Prediction output cannot enter sealed scientific evidence")
    if path.exists():
        raise FileExistsError(f"Preserve existing prediction output; choose a new file: {path}")
    return path


def save_predictions(path: Path, rows: list[dict], *, root: Path = PROJECT_ROOT) -> None:
    path = guard_prediction_output(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["structure_id", "predicted", "replicate_count", "p_OPCID", "p_CHIN", "p_CHID"])
        writer.writeheader()
        writer.writerows(rows)
