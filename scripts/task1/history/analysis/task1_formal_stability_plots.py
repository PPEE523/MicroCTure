"""Plot saved formal-CV diagnostics without model loading or result mutation.

Every trajectory is one spatial fold/seed. Medians are descriptive medians
over available fold/seed cases at each epoch, not pooled OOF performance.
CHID prediction cells use the saved unique-structure OOF probabilities' argmax.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ("m2", "m2plus", "cnn_transformer")
MODEL_LABELS = {"m2": "M2", "m2plus": "M2+", "cnn_transformer": "CNN-Transformer"}
CONDITIONS = ("I0", "G0", "I3")
CLASS_NAMES = ("OPCID", "CHIN", "CHID")
SEEDS = (42, 43, 44)


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _nested(row: dict, keys: tuple[str, ...]) -> float:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return math.nan
        value = value.get(key)
    return _finite(value)


def _saved_histories(run_dir: Path, model: str) -> list[dict]:
    histories = []
    for seed in SEEDS:
        for fold in range(5):
            path = run_dir / "runs" / "I3" / model / f"seed_{seed}" / f"fold_{fold}" / "history.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("epochs", []) if isinstance(payload, dict) else payload
            if rows:
                histories.append({"seed": seed, "fold": fold, "rows": rows})
    return histories


def _curve_matrix(histories: list[dict], keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Align actual epoch values; missing measurements stay NaN, never zero."""
    epochs = sorted({int(row["epoch"]) for history in histories for row in history["rows"]})
    values = np.full((len(histories), len(epochs)), np.nan)
    index = {epoch: position for position, epoch in enumerate(epochs)}
    for case, history in enumerate(histories):
        for row in history["rows"]:
            values[case, index[int(row["epoch"])]] = _nested(row, keys)
    return np.asarray(epochs), values


def _plot_curves(ax, histories: list[dict], keys: tuple[str, ...], color: str,
                 label: str, linestyle: str = "-", *, thin: bool = True) -> None:
    epochs, matrix = _curve_matrix(histories, keys)
    if not matrix.size or not np.isfinite(matrix).any():
        return
    if thin:
        for values in matrix:
            ax.plot(epochs, values, color=color, alpha=0.15, linewidth=0.65,
                    linestyle=linestyle)
            zero = np.isfinite(values) & (values == 0)
            if zero.any():
                ax.scatter(epochs[zero], values[zero], marker="x", color="#b2182b",
                           s=9, alpha=0.8, zorder=5)
    # Avoid nanmedian's all-NaN warning for epoch zero gradient measurements.
    median = np.asarray([np.median(column[np.isfinite(column)])
                         if np.isfinite(column).any() else np.nan
                         for column in matrix.T])
    ax.plot(epochs, median, color=color, linewidth=2, linestyle=linestyle, label=label)


def _save(fig, output_dir: Path, name: str, caption: str) -> dict:
    import matplotlib.pyplot as plt

    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"name": name, "path_png": str(png), "path_pdf": str(pdf), "caption": caption}


def _training_plot(run_dir: Path, output_dir: Path) -> dict | None:
    import matplotlib.pyplot as plt

    histories_by_model = {model: _saved_histories(run_dir, model) for model in MODELS}
    if not any(histories_by_model.values()):
        return None
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    titles = ("Fixed-view weighted CE", "Inner validation CHID F1",
              "Encoder gradient norm", "Centered-logit variance")
    for row_index, model in enumerate(MODELS):
        histories = histories_by_model[model]
        loss_ax, f1_ax, grad_ax, variance_ax = axes[row_index]
        _plot_curves(loss_ax, histories, ("weighted_train_ce",), "#0072B2", "Train median")
        _plot_curves(loss_ax, histories, ("weighted_validation_ce",), "#D55E00", "Inner-val median", "--")
        loss_ax.set_yscale("symlog", linthresh=0.01)
        _plot_curves(f1_ax, histories, ("validation_diagnostics", "metrics", "per_class", "CHID", "f1"),
                     "#009E73", "Inner-val median")
        f1_ax.set_ylim(-0.035, 1.035)
        _plot_curves(grad_ax, histories, ("gradient_norms", "encoder", "mean"),
                     "#CC79A7", "Batch-mean median")
        grad_ax.set_yscale("symlog", linthresh=1e-10)
        grad_ax.axhline(1e-10, color="0.45", linewidth=0.7, linestyle=":")
        _plot_curves(variance_ax, histories, ("train_diagnostics", "centered_logit_between_structure_var"),
                     "#0072B2", "Train median")
        _plot_curves(variance_ax, histories, ("validation_diagnostics", "centered_logit_between_structure_var"),
                     "#D55E00", "Inner-val median", "--")
        variance_ax.set_yscale("symlog", linthresh=1e-6)
        variance_ax.axhline(1e-6, color="0.45", linewidth=0.7, linestyle=":")
        for column_index, ax in enumerate(axes[row_index]):
            if row_index == 0:
                ax.set_title(titles[column_index], fontsize=11)
            if row_index == 2:
                ax.set_xlabel("Epoch (0 = initialization)")
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7, loc="best")
        loss_ax.set_ylabel(f"{MODEL_LABELS[model]}\n{len(histories)} fold/seed cases")
    fig.suptitle("I3 / H3: saved training trajectories, fixed 40-epoch endpoint", fontsize=15)
    fig.text(0.5, 0.015,
             "Faint lines = individual cases; thick lines = epoch-wise medians. "
             "F1 is inner validation, not outer OOF. Red crosses retain exact zeros. "
             "CE, gradient and variance use symlog axes.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    return _save(fig, output_dir, "training_diagnostics",
                 "I3/H3 三骨干的15条固定 fold/seed 轨迹及逐轮中位数；F1仅为内层验证，"
                 "梯度为训练批次均值，中心化logits方差按唯一结构计算。symlog与红色零值标记保留失活信号。")


def _diagnostic_matrix(summary: dict, kind: str) -> tuple[np.ndarray, list[list[str]]]:
    """Return case-level counts / explicit observed or expected denominators."""
    values = np.full((len(MODELS), len(CONDITIONS)), np.nan)
    labels = [["pending" for _ in CONDITIONS] for _ in MODELS]
    for row, model in enumerate(MODELS):
        for column, condition in enumerate(CONDITIONS):
            diagnostics = summary.get("models", {}).get(f"{condition}/{model}", {}).get("learning_diagnostics", {})
            if not diagnostics:
                continue
            if kind == "strict":
                numerator = diagnostics.get("strict_learning_passes")
                denominator = diagnostics.get("expected_cases", 15)
            else:
                view = diagnostics.get("prediction_events", {}).get("validation", {})
                denominator = view.get("final_observed_cases", 0)
                numerator = view.get(kind, {}).get("final_cases")
            if numerator is None or not denominator:
                continue
            values[row, column] = numerator / denominator
            labels[row][column] = f"{numerator}/{denominator}"
    return values, labels


def _stability_plot(summary: dict, output_dir: Path) -> dict | None:
    import matplotlib.pyplot as plt

    if not any(summary.get("models", {}).get(f"{condition}/{model}", {}).get("learning_diagnostics")
               for model in MODELS for condition in CONDITIONS):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    specs = (("strict", "Strict learning passes", "YlGn"),
             ("constant_probability", "Final inner-val constant probabilities", "OrRd"),
             ("single_class_argmax", "Final inner-val single-class predictions", "OrRd"))
    for ax, (kind, title, cmap) in zip(axes, specs):
        values, labels = _diagnostic_matrix(summary, kind)
        color_map = plt.get_cmap(cmap).copy()
        color_map.set_bad("#e4e4e4")
        ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=color_map, aspect="auto")
        ax.set_xticks(range(3), CONDITIONS)
        ax.set_yticks(range(3), [MODEL_LABELS[model] for model in MODELS])
        ax.set_title(title, fontsize=10)
        for row in range(3):
            for column in range(3):
                ax.text(column, row, labels[row][column], ha="center", va="center", fontsize=13,
                        color="white" if np.isfinite(values[row, column]) and values[row, column] > 0.7 else "black")
        ax.tick_params(length=0)
    fig.suptitle("Learning acceptance and endpoint prediction collapse", fontsize=14)
    footnotes = []
    for model in MODELS:
        observations = []
        for condition in CONDITIONS:
            view = summary.get("models", {}).get(f"{condition}/{model}", {}).get("learning_diagnostics", {}).get("prediction_events", {}).get("validation", {})
            event = view.get("single_class_argmax", {})
            if view:
                observations.append(f"{condition}: {event.get('events', 0)}/{view.get('observed_events', 0)} epochs, "
                                    f"{event.get('ever_cases', 0)}/{view.get('observed_cases', 0)} cases")
        footnotes.append(f"{MODEL_LABELS[model]} inner-val single-class events: " + "; ".join(observations))
    fig.text(0.5, 0.012,
             "Strict: expected-case denominator; endpoint: observed completed-case denominator. "
             "Incomplete cases are never zero-filled.\n" + "\n".join(footnotes) +
             "\nPost-initialization events only. Noncollapse alone does not establish recognition quality or generalization.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.23, 1, 0.93))
    return _save(fig, output_dir, "stability_overview",
                 "三输入条件×三骨干持续学习通过数与末轮内层验证常数概率、单类预测案例数分开统计；"
                 "图下注记保留第1–40轮曾发生的单类事件，不把末轮恢复等同于全程稳定。")


def _load_oof(run_dir: Path, model: str, seed: int) -> dict[str, dict]:
    path = run_dir / "oof" / "I3" / model / f"seed_{seed}" / "predictions.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        result = {}
        for row in csv.DictReader(handle):
            identifier = row["structure_id"]
            if identifier in result:
                raise ValueError(f"Duplicate OOF structure {identifier}: {path}")
            result[identifier] = {**row, "y_true": int(row["y_true"]), "y_pred": int(row["y_pred"])}
        return result


def _natural_id(identifier: str) -> tuple[str, int]:
    prefix, _, suffix = identifier.rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else 0


def _chid_grid(oof: dict[tuple[str, int], dict[str, dict]]) -> tuple[list[str], np.ndarray, list[str]]:
    """Sort across all models/seeds by correct-count; no best-model selection."""
    columns = [(model, seed) for model in MODELS for seed in SEEDS]
    metadata: dict[str, dict] = {}
    for predictions in oof.values():
        for identifier, row in predictions.items():
            if row["y_true"] == 2:
                previous = metadata.get(identifier)
                if previous and (str(previous.get("group_id")), str(previous.get("fold"))) != (str(row.get("group_id")), str(row.get("fold"))):
                    raise ValueError(f"Inconsistent OOF group/fold for {identifier}")
                metadata[identifier] = row
    identifiers = sorted(metadata, key=lambda identifier: (
        sum(oof.get(column, {}).get(identifier, {}).get("y_pred") == 2 for column in columns),
        _natural_id(identifier)))
    matrix = np.full((len(identifiers), len(columns)), np.nan)
    labels = []
    for row_index, identifier in enumerate(identifiers):
        row = metadata[identifier]
        labels.append(f"{identifier}  [g{row.get('group_id', '?')}, f{row.get('fold', '?')}]")
        for column_index, column in enumerate(columns):
            if identifier in oof.get(column, {}):
                prediction = oof[column][identifier]
                if prediction["y_true"] != 2:
                    raise ValueError(f"Inconsistent OOF true class for {identifier}")
                matrix[row_index, column_index] = prediction["y_pred"]
    return identifiers, matrix, labels


def _chid_plot(run_dir: Path, summary: dict, output_dir: Path) -> dict | None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    oof = {(model, seed): _load_oof(run_dir, model, seed) for model in MODELS for seed in SEEDS}
    identifiers, matrix, labels = _chid_grid(oof)
    if not identifiers:
        return None
    fig = plt.figure(figsize=(12.5, max(7.5, len(identifiers) * 0.25)))
    grid = fig.add_gridspec(1, 2, width_ratios=(2.7, 1.4), wspace=0.28)
    ax = fig.add_subplot(grid[0])
    note_ax = fig.add_subplot(grid[1])
    color_map = ListedColormap(("#E69F00", "#56B4E9", "#009E73"))
    color_map.set_bad("#e4e4e4")
    ax.imshow(np.ma.masked_invalid(matrix), cmap=color_map,
              norm=BoundaryNorm((-0.5, 0.5, 1.5, 2.5), 3), aspect="auto", interpolation="nearest")
    short_labels = {"m2": "M2", "m2plus": "M2+", "cnn_transformer": "CNN-T"}
    ax.set_xticks(range(9), [f"{short_labels[model]}\ns{seed}" for model in MODELS for seed in SEEDS], fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted class from unique-structure outer OOF", labelpad=8)
    ax.set_ylabel("True CHID (sorted by correct count across all 9 predictions)")
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="white", linewidth=2.5)
    for row in range(len(identifiers)):
        for column in range(9):
            if np.isfinite(matrix[row, column]):
                ax.text(column, row, "O" if matrix[row, column] == 0 else "N" if matrix[row, column] == 1 else "D",
                        ha="center", va="center", fontsize=7, color="white" if matrix[row, column] == 2 else "black")
    note_ax.axis("off")
    note_ax.text(0, 1, "Complete-OOF seed metrics", fontsize=12, weight="bold", va="top")
    y = 0.94
    for model in MODELS:
        model_summary = summary.get("models", {}).get(f"I3/{model}", {})
        note_ax.text(0, y, MODEL_LABELS[model], fontsize=11, weight="bold", va="top")
        y -= 0.055
        for seed in SEEDS:
            seed_summary = model_summary.get("seeds", {}).get(str(seed), {})
            f1 = _nested(seed_summary, ("oof", "per_class", "CHID", "f1"))
            values = oof[(model, seed)]
            chid = [row for row in values.values() if row["y_true"] == 2]
            correct = sum(row["y_pred"] == 2 for row in chid)
            f1_text = f"{f1:.3f}" if math.isfinite(f1) else "pending"
            note_ax.text(0, y, f"s{seed}: CHID F1 {f1_text}; TP {correct}/{len(chid)}", fontsize=9, va="top")
            y -= 0.042
        full_predictions = [oof[(model, seed)] for seed in SEEDS]
        complete = all(len(rows) == summary.get("n_structures", 344) for rows in full_predictions)
        same_ids = complete and len({frozenset(rows) for rows in full_predictions}) == 1
        if same_ids:
            ids = set(full_predictions[0])
            unanimous = sum(len({rows[identifier]["y_pred"] for rows in full_predictions}) == 1 for identifier in ids)
            chid_ids = [identifier for identifier in ids if full_predictions[0][identifier]["y_true"] == 2]
            unanimous_chid = sum(len({rows[identifier]["y_pred"] for rows in full_predictions}) == 1 for identifier in chid_ids)
            agreement = f"All-class agreement: {unanimous}/{len(ids)} = {unanimous/len(ids):.1%}\nCHID agreement: {unanimous_chid}/{len(chid_ids)}"
        else:
            agreement = "Agreement pending: incomplete seed OOF"
        note_ax.text(0, y, agreement, fontsize=9, va="top")
        y -= 0.095
    note_ax.text(0, max(0.015, y), f"Agreement includes consistent errors.\nCHID F1 counts FP from all {summary['n_structures']} structures.\nTags = frozen groups/folds; gray = missing.\nCNN-T = CNN-Transformer.",
                 fontsize=8, va="top", color="0.35")
    fig.legend(handles=[Patch(facecolor=color_map(index), label=f"{CLASS_NAMES[index]} ({'OND'[index]})") for index in range(3)],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.008))
    fig.suptitle(f"I3: seed stability of all {len(identifiers)} true CHID structures", fontsize=14, y=0.99)
    fig.subplots_adjust(top=0.93, bottom=0.15, left=0.2, right=0.98)
    return _save(fig, output_dir, "CHID_prediction_stability",
                 f"{len(identifiers)}个真实CHID×三骨干×三个种子的唯一结构OOF预测，按全部9次预测的正确数排序，"
                 "同时显示冻结组/折。右侧CHID F1以完整344结构计算；一致率包含一致错误，不等同正确率。")


def plot_stability(run_dir: Path, summary: dict, output_dir: Path) -> list[dict]:
    """Write reusable PNG/PDF report figures; never fit, predict, or alter results."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Stability figures must be written outside the scientific result directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for result in (_training_plot(run_dir, output_dir),
                   _stability_plot(summary, output_dir),
                   _chid_plot(run_dir, summary, output_dir)):
        if result is not None:
            figures.append(result)
    return figures
