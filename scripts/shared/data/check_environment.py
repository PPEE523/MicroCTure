"""Check the Conda/GPU environment with small, read-only data exercises.

This verifies runtime readiness only; it does not train or assess Task 1.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


ROOT = _layout_root
PACKAGES = (
    "numpy", "scipy", "pandas", "h5py", "cooler", "openpyxl",
    "scikit-learn", "matplotlib", "torch",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_conda():
    prefix = Path(sys.prefix).resolve()
    require(prefix.name == "microcture", f"Expected microcture environment, got {prefix}")
    require((prefix / "conda-meta").is_dir(), f"Not a Conda environment: {prefix}")
    require(sys.version_info[:2] == (3, 12), "Expected Python 3.12")
    return {"name": prefix.name, "prefix": str(prefix), "python": platform.python_version()}


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def check_coolers(all_coolers):
    import cooler
    import numpy as np

    config = json.loads((_layout_resolve(ROOT, "configs/data.json")).read_text(encoding="utf-8"))
    samples = list(config["samples"])
    required_ids = {"WT_37C_rep1", "WT_37C_rep2"}
    require(required_ids.issubset({s["sample_id"] for s in samples}), "Missing WT replicates in data config")
    if all_coolers:
        conditions = json.loads((_layout_resolve(ROOT, "configs/conditions.json")).read_text(encoding="utf-8"))
        seen = {s["path"] for s in samples}
        for sample in conditions["samples"]:
            if sample["path"] not in seen:
                samples.append(sample)
                seen.add(sample["path"])
        require(len(samples) == 6, f"Expected six configured Cooler samples, got {len(samples)}")
    else:
        samples = [s for s in samples if s["sample_id"] in required_ids]
        require(len(samples) == 2, "Expected two distinct WT replicates")

    region = (config["chrom"], 50000, 51000)
    expected_width = (region[2] - region[1]) // config["bin_size"]
    results = []
    for sample in samples:
        path = _layout_resolve(ROOT, sample["path"])
        require(path.is_file(), f"Missing raw Cooler: {path}")
        clr = cooler.Cooler(str(path))
        require(clr.binsize == config["bin_size"], f"Unexpected resolution: {path.name}")
        require(clr.chromnames == [config["chrom"]], f"Unexpected chromosome: {path.name}")
        require(int(clr.chromsizes[config["chrom"]]) == config["chrom_length"], f"Unexpected chromosome length: {path.name}")
        matrix = clr.matrix(balance=False).fetch(region)
        require(matrix.shape == (expected_width, expected_width), f"Unexpected local shape: {path.name}")
        require(np.isfinite(matrix).all(), f"Nonfinite counts: {path.name}")
        require((matrix >= 0).all(), f"Negative counts: {path.name}")
        require(np.array_equal(matrix, matrix.T), f"Asymmetric local counts: {path.name}")
        results.append({
            "sample_id": sample["sample_id"], "path": sample["path"],
            "bin_size": int(clr.binsize), "region": list(region), "balance": False,
            "shape": list(matrix.shape), "count_sum": int(matrix.sum()),
            "nonzero_pixels": int(np.count_nonzero(matrix)),
        })
    return {"samples_checked": len(results), "samples": results, "scope": "One 1 kb window per sample; no full pixel audit"}


def check_excel():
    import openpyxl

    config = json.loads((_layout_resolve(ROOT, "configs/data.json")).read_text(encoding="utf-8"))
    spec = config["annotations"]
    expected = {"OPCID": 68, "CHIN": 250, "CHID": 26}
    require(spec["expected_counts"] == expected, "Unexpected configured label counts")
    workbook = openpyxl.load_workbook(_layout_resolve(ROOT, spec["path"]), read_only=True, data_only=True)
    counts, seen = {}, set()
    try:
        for kind, number in spec["tables"].items():
            name = f"Supplementary Table {number}"
            rows = workbook[name].iter_rows(values_only=True)
            headers = next(rows)
            columns = {str(value).strip(): i for i, value in enumerate(headers) if value is not None}
            require(kind + "_ID" in columns, f"Missing label ID column: {name}")
            count = 0
            for line, row in enumerate(rows, 2):
                sid = row[columns[kind + "_ID"]]
                if sid is None:
                    continue
                require(isinstance(sid, str) and sid.startswith(kind + "_"), f"Invalid label ID in {name}:{line}")
                require(sid not in seen, f"Duplicate label ID: {sid}")
                seen.add(sid)
                count += 1
            counts[kind] = count
    finally:
        workbook.close()
    require(counts == expected, f"Label counts do not match: {counts}")
    return {"path": spec["path"], "counts": counts, "unique_structure_ids": len(seen),
            "coordinate_status": spec["coordinate_status"], "scope": "ID/count check; coordinate convention is not verified"}


def check_gpu():
    import torch
    from torch import nn

    require(torch.cuda.is_available(), "CUDA is unavailable in PyTorch")
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    require(capability == (12, 0), f"Expected sm_120 GPU, got capability {capability}")
    require("sm_120" in arch_list, f"PyTorch wheel lacks sm_120: {arch_list}")
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3),
    ).to("cuda:0")
    inputs = torch.randn(4, 2, 64, 64, device="cuda:0", requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0], device="cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, labels)
    require(logits.shape == (4, 3) and torch.isfinite(logits).all().item(), "Invalid GPU forward output")
    require(torch.isfinite(loss).item(), "Nonfinite GPU loss")
    loss.backward()
    torch.cuda.synchronize()
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None, f"Missing gradient: {name}")
        require(torch.isfinite(parameter.grad).all().item(), f"Nonfinite gradient: {name}")
        norm = float(parameter.grad.norm().item())
        require(norm > 0, f"Zero gradient: {name}")
        gradient_norms[name] = norm
    require(inputs.grad is not None and torch.isfinite(inputs.grad).all().item(), "Invalid input gradient")
    require(inputs.grad.abs().sum().item() > 0, "Zero input gradient")
    optimizer.step()
    torch.cuda.synchronize()
    changed = [not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters())]
    require(all(changed), "Optimizer did not change all trainable parameters")
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite updated parameters")
    return {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "compute_capability": list(capability),
        "architecture": "sm_120", "wheel_architectures": arch_list,
        "loss": float(loss.item()), "gradient_norms": gradient_norms,
        "parameters_changed": sum(changed), "input_shape": list(inputs.shape),
        "forward_backward_optimizer_step": "passed",
        "scope": "One synthetic GPU step; no Task 1 training or quality evaluation",
    }


def check_sklearn():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.array([[-3, 0], [-2, 0], [0, 3], [0, 2], [3, 0], [2, 0]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=42))
    model.fit(x, y)
    probabilities = model.predict_proba(x)
    require(np.isfinite(probabilities).all(), "Nonfinite sklearn probabilities")
    require(np.allclose(probabilities.sum(axis=1), 1), "Invalid sklearn probability sums")
    require(np.array_equal(model.predict(x), y), "Sklearn synthetic three-class fit failed")
    return {"synthetic_samples": len(y), "classes": 3, "fit_predict": "passed"}


def check_matplotlib():
    import numpy as np

    # Keep Matplotlib's temporary font/cache files outside the project.
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="microcture-matplotlib-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            figure = Figure(figsize=(2, 2), dpi=80)
            canvas = FigureCanvasAgg(figure)
            figure.add_subplot().plot([0, 1, 2], [0, 1, 0])
            canvas.draw()
            pixels = np.asarray(canvas.buffer_rgba())
            require(pixels.shape == (160, 160, 4) and pixels.std() > 0, "Agg rendering failed")
            buffer = io.BytesIO()
            canvas.print_png(buffer)
            require(buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n"), "Invalid in-memory PNG")
            return {"backend": matplotlib.get_backend(), "shape": list(pixels.shape),
                    "in_memory_png_bytes": buffer.tell(), "output_images_saved": 0}
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_layout_resolve(ROOT, "reports/environment_check.json"), help="Fresh JSON report; existing evidence is never replaced")
    parser.add_argument("--all-coolers", action="store_true", help="Also check the four condition samples")
    args = parser.parse_args()
    args.output = _layout_resolve(ROOT, args.output)
    require(args.output.suffix.lower() == ".json", "Output must be a JSON file")
    from prepare_data import require_new_outputs
    require_new_outputs([args.output])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": False, "model_quality_assessed": False,
        "scope": "Conda environment, dependencies, small raw data reads and a synthetic GPU optimization step",
        "environment": {
            "python": platform.python_version(), "executable": sys.executable,
            "prefix": sys.prefix, "platform": platform.platform(),
            "kernel_release": platform.release(),
            "wsl": "microsoft" in (platform.release() + platform.version()).lower(),
        },
        "checks": {}, "failures": [],
    }
    checks = (
        ("conda", check_conda), ("packages", package_versions),
        ("coolers", lambda: check_coolers(args.all_coolers)),
        ("excel", check_excel), ("gpu", check_gpu),
        ("sklearn", check_sklearn), ("matplotlib", check_matplotlib),
    )
    for name, check in checks:
        try:
            report["checks"][name] = {"passed": True, "details": check()}
        except Exception as error:
            failure = {"check": name, "error_type": type(error).__name__, "message": str(error)}
            report["checks"][name] = {"passed": False, "error": failure}
            report["failures"].append(failure)
    report["checks_passed"] = not report["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    print(json.dumps({
        "checks_passed": report["checks_passed"],
        "checks": {name: check["passed"] for name, check in report["checks"].items()},
        "failures": report["failures"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
