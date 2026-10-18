"""Review retained task 5/6 predictions, without training or loading weights.

The input workspace is read-only.  Historical checkpoint verification remains a
historical claim; this entry point checks saved arrays and coordinates instead.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


ROOT = _layout_root
HISTORICAL_REVISION = '3bbd4309dae0f2d2e4efe55f3398b575238975ba'
SEEDS = (42, 43, 44)
TASK5_MODELS = ("bilinear", "bicubic") + tuple(
    f"{family}_{seed}" for family in ("residual_cnn", "plain_cnn") for seed in SEEDS)
TASK6_MODELS = ("mds",) + tuple(
    f"{family}_{seed}" for family in ("geometry", "graph") for seed in SEEDS)


def read_csv(path):
    with _layout_resolve(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    value = hashlib.sha256()
    with _layout_resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_once(path, text):
    """Allow identical reruns, but never replace a previous scientific file."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Existing output differs; use a new report directory: {path}")
        return
    with path.open("x") as handle:
        handle.write(text)


def write_json(path, value):
    write_once(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def output_guard(data_root, report_dir):
    from optional_common import fresh_report_path
    data_root = Path(data_root).resolve()
    if data_root != ROOT and Path(report_dir).resolve().is_relative_to(data_root):
        raise ValueError('Report directory must be outside the read-only input workspace')
    report_dir = fresh_report_path(report_dir, root=ROOT, data_root=data_root)
    report_dir.mkdir(parents=True, exist_ok=False)
    return data_root, report_dir


def result_root(data_root, task=None):
    """Resolve a complete legacy task name rather than its namespace parent."""
    from optional_common import input_path
    data_root = Path(data_root).resolve()
    if task is not None:
        try:
            return input_path(data_root, f'results/{task}')
        except FileNotFoundError:
            # Inventory must enumerate absent files instead of claiming readiness.
            return _layout_resolve(data_root / 'results' / task, root=data_root)
    legacy = data_root / 'data/legacy_experiments/results'
    return legacy if legacy.is_dir() or _layout_resolve(legacy / 'task5', root=data_root).is_dir() else data_root / 'results'


def inventory(data_root):
    base = result_root(data_root)
    required = {
        "task5": ["manifest.json", "metrics.csv", "summary.csv", "windows.csv"]
        + [f"{name}_reconstruction.npz" for name in TASK5_MODELS],
        "task6": ["manifest.json", "metrics.csv", "evaluation_inputs.npz"]
        + [f"{name}_{suffix}" for name in TASK6_MODELS
           for suffix in ("coordinates.csv", "maps.npz")],
    }
    files, historical_inputs, checkpoints = [], [], []
    for task, names in required.items():
        directory = result_root(data_root, task)
        for name in names:
            path = directory / name
            row = dict(task=task, path=str(path), exists=path.is_file())
            if row["exists"]:
                row.update(bytes=path.stat().st_size, sha256=digest(path))
            files.append(row)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            for relative, expected in manifest.get("input_sha256", {}).items():
                # Data paths stay in the input workspace. Source files removed
                # by the earlier cleanup may be checked against this clone.
                candidate = _layout_resolve(data_root / relative, root=data_root)
                if not candidate.is_file() and relative.startswith(("scripts/", "configs/")):
                    candidate = _layout_resolve(ROOT / relative)
                row = dict(task=task, historical_path=relative, checked_path=str(candidate),
                           expected_sha256=expected, exists=candidate.is_file())
                if row["exists"]:
                    row.update(actual_sha256=digest(candidate))
                    row["matches"] = row["actual_sha256"] == expected
                if relative.startswith(('scripts/', 'configs/')):
                    # Code intentionally changed in this maintenance branch.
                    # Validate the historical source in Git, while reporting
                    # current-source differences separately from data changes.
                    row['current_worktree_matches'] = row.get('matches', False)
                    snapshot = subprocess.run(['git', 'show', f'{HISTORICAL_REVISION}:{relative}'],
                                              cwd=ROOT, capture_output=True)
                    row['historical_snapshot_revision'] = HISTORICAL_REVISION
                    row['historical_snapshot_matches'] = bool(snapshot.returncode == 0 and
                        hashlib.sha256(snapshot.stdout).hexdigest() == expected)
                    row['matches'] = row['historical_snapshot_matches']
                    row['validation_basis'] = 'historical Git source, not current modified code'
                else:
                    row['validation_basis'] = 'current read-only saved data bytes'
                historical_inputs.append(row)
        models = TASK5_MODELS[2:] if task == "task5" else TASK6_MODELS[1:]
        checkpoints.extend(dict(task=task, path=str(directory / f"{name}.pt"),
                                exists=(directory / f"{name}.pt").is_file()) for name in models)
    missing = [row["path"] for row in files if not row["exists"]]
    return dict(schema_version=1, data_root=str(data_root), result_root=str(base),
                result_directories={task: str(result_root(data_root, task)) for task in required},
                status="saved_inputs_available" if not missing else "missing_saved_inputs",
                required_saved_files=files, missing_saved_inputs=missing,
                historical_input_signatures=historical_inputs, checkpoints=checkpoints,
                missing_checkpoints=[row["path"] for row in checkpoints if not row["exists"]],
                checkpoint_restoration_performed=False, training_performed=False,
                raw_preprocessing_performed=False,
                interpretation="Inventory only; availability does not certify biological validity.")


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def image_metrics(prediction, target, mask, window=7):
    valid = mask & np.triu(np.ones_like(mask, dtype=bool), 2)
    if not valid.any():
        raise ValueError("No valid upper-triangle evaluation pixels")
    mse = float(np.mean((prediction[valid] - target[valid]) ** 2))
    ux, uy = uniform_filter(prediction, window), uniform_filter(target, window)
    vx = np.maximum(uniform_filter(prediction ** 2, window) - ux ** 2, 0)
    vy = np.maximum(uniform_filter(target ** 2, window) - uy ** 2, 0)
    covariance = uniform_filter(prediction * target, window) - ux * uy
    ssim = ((2 * ux * uy + .01 ** 2) * (2 * covariance + .03 ** 2)
            / ((ux ** 2 + uy ** 2 + .01 ** 2) * (vx + vy + .03 ** 2)))
    interior = valid & (uniform_filter(mask.astype(float), window, mode="constant") > 1 - 1e-8)
    return dict(mse=mse, psnr=float(-10 * np.log10(max(mse, 1e-12))),
                ssim=float(ssim[interior].mean()) if interior.any() else None,
                shape_correlation=correlation(prediction[valid], target[valid]),
                evaluated_pixels=int(valid.sum()), ssim_pixels=int(interior.sum()))


def profile(matrix, mask, flank):
    values = np.full(len(matrix), np.nan)
    for cut in range(flank, len(matrix) - flank):
        block = matrix[cut - flank:cut, cut:cut + flank]
        keep = mask[cut - flank:cut, cut:cut + flank]
        if keep.sum() >= flank * flank // 2:
            values[cut] = block[keep].mean()
    return values


def structure_metrics(prediction, target, mask, row, cfg):
    predicted = profile(prediction, mask, cfg["profile_flank_bins"])
    actual = profile(target, mask, cfg["profile_flank_bins"])
    shifts = []
    for boundary in ("start", "end"):
        center = round((int(row[boundary]) - int(row["start_bin"]) * 100)
                       / cfg["target_resolution_bp"])
        ids = np.arange(max(0, center - cfg["boundary_radius_bins"]),
                        min(len(predicted), center + cfg["boundary_radius_bins"] + 1))
        ids = ids[np.isfinite(predicted[ids]) & np.isfinite(actual[ids])]
        if len(ids):
            shifts.append(abs(int(ids[np.argmin(predicted[ids])]) - int(ids[np.argmin(actual[ids])]))
                          * cfg["target_resolution_bp"])
    return dict(profile_correlation=correlation(predicted, actual),
                boundary_shift_bp=float(np.mean(shifts)) if shifts else None,
                evaluated_boundaries=len(shifts))


def compare_metrics(record, calculated, errors):
    for key, value in calculated.items():
        saved = record[key]
        if value is None:
            if saved not in (None, ""):
                raise ValueError(f"Expected empty {key}, got {saved}")
            continue
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite recomputed {key}")
        expected = float(saved)
        if not np.isclose(expected, value, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Saved {key}={expected} differs from recomputed {value}")
        errors[key] = max(errors.get(key, 0.), abs(expected - value))


def review_task5(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    cfg = manifest["config"]
    windows, saved = read_csv(directory / "windows.csv"), read_csv(directory / "metrics.csv")
    if {row["model"] for row in saved} != set(TASK5_MODELS):
        raise ValueError("Unexpected task5 model roster")
    test_indices = [i for i, row in enumerate(windows) if row["split"] == "test"]
    errors, recomputed, identities = {}, [], []
    for name in TASK5_MODELS:
        selected = [row for row in saved if row["model"] == name]
        with np.load(directory / f"{name}_reconstruction.npz", allow_pickle=False) as z:
            prediction, target, mask, indices = z["prediction"], z["target"], z["mask"], z["window_indices"]
            if prediction.shape != target.shape or prediction.shape != mask.shape:
                raise ValueError("Task5 saved prediction/target/mask shapes differ")
            if mask.dtype != np.bool_ or prediction.ndim != 3:
                raise ValueError("Task5 requires boolean masks and image stacks")
            if len(selected) != len(indices) or indices.tolist() != test_indices:
                raise ValueError("Task5 saved prediction indices differ from test metadata")
            if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
                raise ValueError("Task5 prediction outside finite [0,1]")
            np.testing.assert_allclose(prediction, prediction.transpose(0, 2, 1), atol=1e-6)
            np.testing.assert_allclose(z["contact_per_million"],
                                       np.expm1(prediction * manifest["normalization"]["log_scale"]))
            for record, index, pred, actual, valid in zip(selected, indices, prediction, target, mask):
                row = windows[int(index)]
                for key in ("structure_id", "type", "split", "replicate", "group_id"):
                    if record[key] != row[key]:
                        raise ValueError(f"Task5 identity mismatch: {key}")
                calculated = image_metrics(pred, actual, valid, cfg["ssim_window"])
                calculated.update(structure_metrics(pred, actual, valid, row, cfg))
                compare_metrics(record, calculated, errors)
                recomputed.append(dict(model=name, structure_id=row["structure_id"],
                                       replicate=row["replicate"], **calculated))
        identities.append(dict(model=name, observations=len(selected)))
    for summary in read_csv(directory / "summary.csv"):
        selected = [row for row in saved if row["model"] == summary["model"]
                    and row["replicate"] == summary["replicate"]
                    and (summary["type"] == "all" or row["type"] == summary["type"])]
        if len(selected) != int(summary["n"]):
            raise ValueError("Task5 summary sample count mismatch")
        values = {key: float(np.mean([float(row[key]) for row in selected if row[key] != ""]))
                  for key in ("psnr", "ssim", "shape_correlation", "profile_correlation", "boundary_shift_bp")}
        compare_metrics(summary, values, errors)
    unique = [row for row in windows if row["split"] == "test" and row["replicate"] == "1"]
    return dict(status="saved_predictions_recomputed", metric_rows=len(recomputed),
                models=identities, unique_test_positions=len(unique),
                test_class_counts={kind: sum(row["type"] == kind for row in unique)
                                   for kind in ("OPCID", "CHIN", "CHID")},
                maximum_absolute_errors=errors, checkpoint_restoration_performed=False), recomputed


def within_distance_correlation(predicted, actual, separation):
    """Exactly the old weighted mean of correlations, with grouped moments."""
    sizes = np.bincount(separation)
    sx = np.bincount(separation, weights=predicted, minlength=len(sizes))
    sy = np.bincount(separation, weights=actual, minlength=len(sizes))
    meanx = np.divide(sx, sizes, out=np.zeros_like(sx), where=sizes > 0)
    meany = np.divide(sy, sizes, out=np.zeros_like(sy), where=sizes > 0)
    dx, dy = predicted - meanx[separation], actual - meany[separation]
    covariance = np.bincount(separation, weights=dx * dy, minlength=len(sizes))
    vx = np.bincount(separation, weights=dx * dx, minlength=len(sizes))
    vy = np.bincount(separation, weights=dy * dy, minlength=len(sizes))
    keep = (sizes >= 3) & (vx > 0) & (vy > 0)
    if not keep.any():
        return None
    return float(np.average(covariance[keep] / np.sqrt(vx[keep] * vy[keep]), weights=sizes[keep]))


def contact_metrics(xyz, a, b, use, counts, cfg, fitted):
    distances = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    predicted = np.maximum((np.maximum(distances, 1e-6) * fitted["distance_scale"])
                           ** (-1 / cfg["contact_exponent"]) - cfg["pseudocount"], 0)
    actual = counts[a, b]
    target = (actual + cfg["pseudocount"]) ** (-cfg["contact_exponent"]) / fitted["distance_scale"]
    separation = np.minimum(b - a, len(xyz) - (b - a))
    background = np.asarray(fitted["log_contact_background"])[separation]
    px, py = np.log1p(predicted), np.log1p(actual)
    return dict(pairs=int(use.sum()), log_distance_rmse=float(np.sqrt(np.mean(
        (np.log(np.maximum(distances[use], 1e-6)) - np.log(target[use])) ** 2))),
        contact_log_pearson=correlation(px[use], py[use]),
        contact_spearman=float(spearmanr(predicted[use], actual[use]).statistic),
        distance_adjusted_contact_pearson=correlation((px - background)[use], (py - background)[use]),
        within_distance_pearson=within_distance_correlation(px[use], py[use], separation[use]))


def review_task6(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    cfg, fitted = manifest["config"], manifest["fitted"]
    records = read_csv(directory / "metrics.csv")
    if {row["model"] for row in records} != set(TASK6_MODELS):
        raise ValueError("Unexpected task6 model roster")
    with np.load(directory / "evaluation_inputs.npz", allow_pickle=False) as z:
        a, b, roles = z["a"], z["b"], z["roles"]
        counts = [z["counts_rep1"], z["counts_rep2"]]
    if len(set(zip(a.tolist(), b.tolist()))) != len(a) or not (a < b).all():
        raise ValueError("Task6 duplicated or non-upper-triangle evaluation pairs")
    if set(roles.tolist()) != {0, 1, 2}:
        raise ValueError("Unexpected task6 partitions")
    errors, recomputed = {}, []
    for name in TASK6_MODELS:
        coordinates = read_csv(directory / f"{name}_coordinates.csv")
        if len(coordinates) != manifest["bins"]:
            raise ValueError("Task6 node count differs from manifest")
        if int(coordinates[0]["start"]) != 0 or int(coordinates[-1]["end"]) != manifest["chromosome_length"]:
            raise ValueError("Task6 coordinates do not cover the chromosome")
        if any(left["end"] != right["start"] for left, right in zip(coordinates, coordinates[1:])):
            raise ValueError("Task6 coordinate bins are not contiguous")
        xyz = np.array([[float(row[key]) for key in ("x", "y", "z")] for row in coordinates])
        if not np.isfinite(xyz).all():
            raise ValueError("Nonfinite task6 coordinates")
        distances = squareform(pdist(xyz))
        predicted = np.maximum((np.maximum(distances, 1e-6) * fitted["distance_scale"])
                               ** (-1 / cfg["contact_exponent"]) - cfg["pseudocount"], 0)
        np.fill_diagonal(predicted, 0)
        with np.load(directory / f"{name}_maps.npz", allow_pickle=False) as z:
            np.testing.assert_allclose(distances, z["distances"], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(predicted, z["predicted_contacts"], rtol=1e-7, atol=1e-7)
        selected = [row for row in records if row["model"] == name]
        if len(selected) != 6 or len({(row["replicate"], row["partition"]) for row in selected}) != 6:
            raise ValueError("Task6 expected both replicates and all three partitions exactly once")
        for record in selected:
            use = roles == ("train", "validation", "test").index(record["partition"])
            calculated = contact_metrics(xyz, a, b, use, counts[int(record["replicate"]) - 1], cfg, fitted)
            compare_metrics(record, calculated, errors)
            recomputed.append(dict(model=name, replicate=record["replicate"],
                                   partition=record["partition"], **calculated))
    return dict(status="saved_coordinates_recomputed", models=len(TASK6_MODELS),
                bins=manifest["bins"], metric_rows=len(recomputed),
                pair_counts=manifest["pair_counts"], maximum_absolute_errors=errors,
                checkpoint_restoration_performed=False), recomputed


def review(data_root, report_dir):
    inputs = inventory(data_root)
    write_json(report_dir / "inputs.json", inputs)
    if inputs["missing_saved_inputs"]:
        raise ValueError("Required saved inputs are missing; see inputs.json")
    mismatches = [row for row in inputs["historical_input_signatures"] if row.get("matches") is False]
    if mismatches:
        raise ValueError("Historical inputs changed; see inputs.json and use a separately reviewed version")
    base = Path(inputs["result_root"])
    task5, rows5 = review_task5(Path(inputs["result_directories"]["task5"]))
    task6, rows6 = review_task6(Path(inputs["result_directories"]["task6"]))
    result = dict(schema_version=1, status="retained_numerical_results_recomputed", task5=task5, task6=task6,
                  provenance=dict(script_sha256=digest(__file__), python=platform.python_version(),
                                  numpy=np.__version__, input_inventory="inputs.json"),
                  training_performed=False, raw_preprocessing_performed=False,
                  checkpoint_restoration_performed=False,
                  missing_checkpoints=inputs["missing_checkpoints"],
                  biological_reliability_certified=False,
                  limitations=["历史结果复算，不是新训练或独立盲测。",
                               "任务五为800→200 bp合成退化；53个测试位置、其中CHID仅4个。",
                               "任务六为同染色体传导式边留出与相对坐标，不验证真实单细胞三维形态。",
                               "历史task56_verification.json的权重恢复记录不代表本轮恢复了权重。"])
    write_once(report_dir / "task5_recomputed_metrics.csv", csv_text(rows5))
    write_once(report_dir / "task6_recomputed_metrics.csv", csv_text(rows6))
    write_json(report_dir / "review.json", result)
    write_once(report_dir / "analysis.md", "# 任务五、六保留成果复算\n\n"
               "本轮只读取历史保存矩阵和坐标，没有训练、原始数据扫描或恢复checkpoint。\n\n"
               f"任务五：{task5['metric_rows']}行逐窗口指标复算通过，"
               f"{task5['unique_test_positions']}个测试位置，CHID {task5['test_class_counts']['CHID']}个。\n\n"
               f"任务六：{task6['models']}组、每组{task6['bins']}节点坐标，"
               f"{task6['metric_rows']}行距离和接触指标复算通过。\n\n"
               f"缺失checkpoint {len(inputs['missing_checkpoints'])}个；"
               "当前可以核验保存结果，不能声称已恢复原权重。文件清单和hash见inputs.json，"
               "最大复算误差见review.json，逐行结果见两个recomputed_metrics.csv。\n\n"
               "合成图像质量提高不等于边界位置或生物形态可靠恢复；"
               "三维接触嵌入不等于真实单细胞构象。旧实验和本轮复算均属于开发性证据。\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "review"))
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Read-only old workspace containing legacy results")
    parser.add_argument("--report-dir", type=Path, default=_layout_resolve(ROOT / "reports/task56_retained_review_v1"))
    args = parser.parse_args(argv)
    try:
        data_root, report_dir = output_guard(args.data_root, args.report_dir)
        if args.command == "check":
            result = inventory(data_root)
            write_json(report_dir / "inputs.json", result)
            code = 2 if result["missing_saved_inputs"] else 0
            print(json.dumps(dict(status=result["status"], missing_saved_inputs=len(result["missing_saved_inputs"]),
                                  missing_checkpoints=len(result["missing_checkpoints"]),
                                  report_dir=str(report_dir)), ensure_ascii=False))
            return code
        result = review(data_root, report_dir)
        print(json.dumps(dict(status=result["status"], task5_rows=result["task5"]["metric_rows"],
                              task6_rows=result["task6"]["metric_rows"], report_dir=str(report_dir)),
                         ensure_ascii=False))
        return 0
    except (ValueError, OSError, AssertionError, KeyError) as error:
        print(f"Review failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
