"""Task 1 common numerical, device, locking and publication helpers."""
from __future__ import annotations

from contextlib import contextmanager
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

import numpy as np
import sklearn
import torch

from microcture.layout import mappings

from .evaluation import grouped_predictions, metrics


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


def reject_retained_overlap(root: Path, output: Path, *, kinds: tuple[str, ...]) -> None:
    """Keep all migrated scientific trees intact, including markerless freezes."""
    for row in mappings(root):
        if row.get("kind") not in kinds:
            continue
        retained = (root / row["new"]).resolve()
        if output == retained or retained in output.parents or output in retained.parents:
            raise ValueError(f"Output overlaps retained scientific evidence; choose a new version: {retained}")


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
