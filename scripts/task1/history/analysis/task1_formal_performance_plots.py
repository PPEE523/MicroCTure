"""Publication-ready figures from saved, verified formal OOF metrics.

This module performs no model inference, fitting, or result-directory writes.
Each seed contributes one complete spatial-fold OOF result. Error bars are
sample standard deviations across optimization seeds, never confidence bounds.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MODELS = ("pca_lr", "m2", "m2plus", "cnn_transformer")
NEURAL_MODELS = MODELS[1:]
LABELS = {"pca_lr": "PCA + LR", "m2": "M2", "m2plus": "M2+",
          "cnn_transformer": "CNN-Transformer", "length_only": "Length LR",
          "mask_only": "Mask LR", "mask_length": "Mask + length LR"}
COLORS = {"pca_lr": "#727272", "m2": "#0072B2", "m2plus": "#009E73",
          "cnn_transformer": "#CC79A7", "I0": "#0072B2", "G0": "#E69F00",
          "I3": "#009E73"}
SEED_MARKERS = ("o", "s", "^")


def complete_seed_scores(summary: dict, condition: str, model: str) -> tuple[list[tuple[int, dict]], str | None]:
    """Return all expected complete seed OOF scores, or an explicit reason.

    Refusing partial aggregation prevents selecting a favorable subset of
    seeds or treating fold-by-seed results as independent observations.
    """
    key = f"{condition}/{model}"
    detail = summary.get("models", {}).get(key)
    if detail is None:
        return [], f"{key}: not present"
    candidates = [row for row in summary.get("comparison", [])
                  if row.get("condition") == condition and row.get("model") == model]
    if len(candidates) != 1:
        return [], f"{key}: missing or repeated comparison row"
    row = candidates[0]
    seeds = detail.get("seeds", {})
    expected = int(row.get("expected_seeds", 0))
    if row.get("status") != "complete" or expected <= 0 or len(seeds) != expected:
        return [], f"{key}: {row.get('complete_seeds', 0)}/{expected} complete seeds"
    output = []
    classes = list(summary["class_names"])
    support = summary["class_support"]
    for seed, value in sorted(seeds.items(), key=lambda item: int(item[0])):
        if value.get("status") != "complete" or not isinstance(value.get("oof"), dict):
            return [], f"{key}/seed_{seed}: incomplete OOF"
        score = value["oof"]
        matrix = np.asarray(score["confusion_matrix"], dtype=float)
        if (matrix.shape != (len(classes), len(classes)) or not np.isfinite(matrix).all()
                or (matrix < 0).any() or not np.equal(matrix, np.round(matrix)).all()
                or int(score["n_structures"]) != int(summary["n_structures"])
                or not np.array_equal(matrix.sum(axis=1), [support[name] for name in classes])):
            raise ValueError(f"{key}/seed_{seed}: OOF support or confusion counts differ from summary")
        for field in ("macro_f1", "accuracy", "balanced_accuracy"):
            numeric = float(score[field])
            if not np.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"{key}/seed_{seed}: invalid {field}")
        for name in classes:
            for field in ("precision", "recall", "f1"):
                numeric = float(score["per_class"][name][field])
                if not np.isfinite(numeric) or not 0 <= numeric <= 1:
                    raise ValueError(f"{key}/seed_{seed}: invalid {name} {field}")
        output.append((int(seed), score))
    return output, None


def _value(score: dict, metric: str) -> float:
    if metric == "macro_f1":
        return float(score[metric])
    return float(score["per_class"]["CHID"][metric])


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    data = np.asarray(values, dtype=float)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else None


def performance_payload(summary: dict) -> dict:
    """Pure numerical payload useful for tests and external table reuse."""
    output: dict[str, Any] = {"groups": {}, "paired": {}, "missing": []}
    for condition in ("I0", "G0", "I3"):
        models = (*NEURAL_MODELS, *MODELS[:1], "length_only", "mask_only", "mask_length") if condition == "I3" else NEURAL_MODELS
        for model in models:
            scores, reason = complete_seed_scores(summary, condition, model)
            if reason:
                output["missing"].append(reason)
                continue
            key = f"{condition}/{model}"
            group = {"seeds": [seed for seed, _ in scores], "scores": dict(scores), "metrics": {}}
            for metric in ("macro_f1", "precision", "recall", "f1"):
                values = [_value(score, metric) for _, score in scores]
                mean, sd = _mean_sd(values)
                group["metrics"][metric] = {"values": values, "mean": mean, "sample_sd": sd}
            matrices = np.asarray([score["confusion_matrix"] for _, score in scores], dtype=float)
            row_support = matrices.sum(axis=2, keepdims=True)
            fractions = np.divide(matrices, row_support, out=np.zeros_like(matrices), where=row_support != 0)
            group["confusion_mean_count"] = matrices.mean(axis=0)
            group["confusion_mean_row_fraction"] = fractions.mean(axis=0)
            output["groups"][key] = group
    requests = [(f"{LABELS[model]}: I3 - {alternative}", f"I3/{model}", f"{alternative}/{model}")
                for alternative in ("I0", "G0") for model in NEURAL_MODELS]
    requests.append(("I3: Transformer - M2+", "I3/cnn_transformer", "I3/m2plus"))
    for label, first, second in requests:
        a, b = output["groups"].get(first), output["groups"].get(second)
        if a is None or b is None:
            continue
        if a["seeds"] != b["seeds"]:
            raise ValueError(f"Paired comparison requires the same complete seeds: {label}")
        output["paired"][label] = {"seeds": a["seeds"], "metrics": {}}
        for metric in ("macro_f1", "f1"):
            values = [float(x - y) for x, y in zip(a["metrics"][metric]["values"], b["metrics"][metric]["values"])]
            mean, sd = _mean_sd(values)
            output["paired"][label]["metrics"][metric] = {"values": values, "mean": mean, "sample_sd": sd}
    return output


def _score_axis(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Score")
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _score_group(ax: Any, x: float, values: list[float], color: str, label: bool = False) -> None:
    mean, sd = _mean_sd(values)
    offsets = np.linspace(-.10, .10, len(values)) if len(values) > 1 else [0]
    for index, (offset, value) in enumerate(zip(offsets, values)):
        ax.scatter(x + offset, value, marker=SEED_MARKERS[index % len(SEED_MARKERS)],
                   s=43, color=color, edgecolors="white", linewidths=.5, zorder=3)
    if sd is not None:
        ax.errorbar(x, mean, yerr=sd, fmt="D", ms=5, color="#202020", capsize=5,
                    linewidth=1.3, zorder=4)
    else:
        ax.scatter(x, mean, marker="D", s=32, facecolors="none", edgecolors="#202020", zorder=4)
    if label:
        top = min(.985, mean + (sd or 0) + .035)
        ax.text(x, top, f"{mean:.3f}" + (f" ± {sd:.3f}" if sd is not None else ""),
                ha="center", va="bottom", fontsize=9)


def _missing(ax: Any, x: float) -> None:
    ax.text(x, .12, "Incomplete\nOOF", ha="center", va="bottom", fontsize=9, color="#777777")


def _footnote(summary: dict) -> str:
    support = summary["class_support"]
    return (f"Each point: one complete spatial OOF seed, {summary['n_structures']} unique structures "
            f"(CHID n={support['CHID']}). Diamonds: mean; whiskers: sample SD across seeds, not CI.\n"
            "Neural models: H3 / 1e-4; PCA/LR: one deterministic seed. Previously inspected folds: development re-evaluation.")


def _seed_legend(groups: dict) -> list[Line2D]:
    seeds = sorted({seed for group in groups.values() for seed in group["seeds"]})
    return [Line2D([], [], marker=SEED_MARKERS[index % 3], linestyle="none", color="#666666",
                   markersize=6, label=f"Seed {seed}") for index, seed in enumerate(seeds)] + [
        Line2D([], [], marker="D", linestyle="none", color="#202020", markersize=5, label="Mean")]


def _save(fig: Any, output_dir: Path, name: str, caption: str, missing: list[str]) -> dict:
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {"name": name, "path_png": str(png), "path_pdf": str(pdf), "caption": caption,
            "status": "partial" if missing else "complete", "missing": missing}


def plot_performance(run_dir: Path, summary: dict, output_dir: Path) -> list[dict]:
    """Export five figure families from complete saved seed-level OOF scores."""
    # run_dir is provenance only; writing into a scientific run is prohibited.
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Performance figures must be written outside the scientific result directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = performance_payload(summary)
    groups = payload["groups"]
    artifacts = []
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        missing = []
        for ax, metric, title in zip(axes.flat, ("macro_f1", "precision", "recall", "f1"),
                                     ("Macro F1", "CHID precision", "CHID recall", "CHID F1")):
            _score_axis(ax, title)
            for index, model in enumerate(MODELS):
                group = groups.get(f"I3/{model}")
                if group is None:
                    _missing(ax, index)
                    if f"I3/{model}" not in missing:
                        missing.append(f"I3/{model}")
                else:
                    _score_group(ax, index, group["metrics"][metric]["values"], COLORS[model], label=True)
            ax.set_xticks(range(len(MODELS)), [LABELS[model] for model in MODELS])
            ax.tick_params(axis="x", labelsize=9)
            ax.set_xlim(-.5, len(MODELS) - .5)
        fig.suptitle("Target-aware I3: four-model comparison", fontsize=15, y=.985)
        fig.text(.5, .02, _footnote(summary), ha="center", va="bottom", fontsize=9)
        fig.legend(handles=_seed_legend(groups), loc="lower center", bbox_to_anchor=(.5, .062),
                   ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .96), h_pad=2.0)
        artifacts.append(_save(fig, output_dir, "main_I3_performance",
                               "完整I3四模型比较；点是单个种子的全空间折OOF，菱形为均值，误差线为跨种子样本标准差。PCA+LR为单个确定性种子。", missing))

        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
        missing = []
        for column, model in enumerate(NEURAL_MODELS):
            for row, metric in enumerate(("macro_f1", "f1")):
                ax = axes[row, column]
                _score_axis(ax, LABELS[model] + (": Macro F1" if row == 0 else ": CHID F1"))
                for x, condition in enumerate(("I0", "G0", "I3")):
                    group = groups.get(f"{condition}/{model}")
                    if group is None:
                        _missing(ax, x)
                        if f"{condition}/{model}" not in missing:
                            missing.append(f"{condition}/{model}")
                    else:
                        _score_group(ax, x, group["metrics"][metric]["values"], COLORS[condition], label=True)
                ax.set_xticks(range(3), ["I0\nNo target geometry", "G0\nGeometry only", "I3\nSignal + geometry"])
                ax.set_xlim(-.5, 2.5)
                ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("Matched-backbone input ablations", fontsize=15, y=.985)
        fig.text(.5, .015, _footnote(summary), ha="center", va="bottom", fontsize=9)
        fig.legend(handles=_seed_legend(groups), loc="lower center", bbox_to_anchor=(.5, .055),
                   ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .115, 1, .96), h_pad=2)
        artifacts.append(_save(fig, output_dir, "ablation_performance",
                               "I0清空目标掩码和长度；G0清空signal和validity，仅保留几何信息；I3完整输入。每个骨干采用相同分类头、学习率和完整OOF评价。", missing))

        paired = payload["paired"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
        labels = list(paired)
        extent = max([abs(value) for item in paired.values() for metric in item["metrics"].values()
                      for value in [*metric["values"], metric["mean"] - (metric["sample_sd"] or 0),
                                    metric["mean"] + (metric["sample_sd"] or 0)]], default=.1)
        limit = min(1, max(.1, np.ceil((extent + .025) * 10) / 10))
        for ax, metric, title in zip(axes, ("macro_f1", "f1"), ("Paired macro F1 difference", "Paired CHID F1 difference")):
            ax.axvline(0, color="#777777", linewidth=1)
            ax.grid(axis="x", color="#DDDDDD", linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title)
            ax.set_xlim(-limit, limit)
            ax.set_xlabel("Left term - right term (score units)")
            for index, label in enumerate(labels):
                item = paired[label]["metrics"][metric]
                offsets = np.linspace(-.12, .12, len(item["values"]))
                for j, (offset, value) in enumerate(zip(offsets, item["values"])):
                    ax.scatter(value, index + offset, s=35, marker=SEED_MARKERS[j % 3],
                               color=("#0072B2", "#E69F00", "#009E73")[j % 3], zorder=3)
                ax.errorbar(item["mean"], index, xerr=item["sample_sd"], fmt="D", color="#222222",
                            ms=5, capsize=4, zorder=4)
                ax.text(limit * .98, index + .23, f"mean {item['mean']:+.3f}", ha="right", va="top", fontsize=8)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_ylim(len(labels) - .5, -.6)
        fig.suptitle("Same-seed complete OOF contrasts", fontsize=15, y=.99)
        fig.text(.5, .01, "Each point pairs the same seed's full OOF results. Whiskers: paired-difference sample SD, not CI.\n"
                 "Positive I3 - G0 indicates added signal + validity value; Transformer - M2+ compares backbones under I3.",
                 ha="center", va="bottom", fontsize=9)
        handles = _seed_legend(groups)
        for index, handle in enumerate(handles[:-1]):
            handle.set_color(("#0072B2", "#E69F00", "#009E73")[index % 3])
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065),
                   ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, .95))
        paired_missing = [label for label in ("I3: Transformer - M2+",)
                          if label not in paired]
        if len(paired) < 7:
            paired_missing.append(f"{len(paired)}/7 complete paired contrasts")
        if not labels:
            for ax in axes:
                ax.text(.5, .5, "No complete paired OOF", transform=ax.transAxes, ha="center")
        artifacts.append(_save(fig, output_dir, "paired_OOF_differences",
                               "同种子完整OOF配对差值；每种子只出现一次，不将fold×seed当成独立数据集。I3−G0检验恢复signal与validity整体的额外价值，Transformer−M2+比较I3下的骨干差异。", paired_missing))

        classes = summary["class_names"]
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
        missing = []
        images = []
        for ax, model in zip(axes.flat, MODELS):
            group = groups.get(f"I3/{model}")
            if group is None:
                ax.text(.5, .5, "Incomplete seed OOF", ha="center", transform=ax.transAxes)
                missing.append(f"I3/{model}")
                ax.set_axis_off()
                continue
            fractions = group["confusion_mean_row_fraction"]
            counts = group["confusion_mean_count"]
            images.append(ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="equal"))
            for row in range(len(classes)):
                for column in range(len(classes)):
                    ax.text(column, row, f"{100 * fractions[row, column]:.1f}%\n({counts[row, column]:.1f})",
                            ha="center", va="center", fontsize=11,
                            color="white" if fractions[row, column] > .52 else "#202020")
            ax.set_xticks(range(len(classes)), classes)
            ax.set_yticks(range(len(classes)), [f"{name}\n(n={summary['class_support'][name]})" for name in classes])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f"{LABELS[model]}: {len(group['seeds'])} complete OOF seed(s)", fontsize=11)
            ax.tick_params(axis="both", length=0)
        fig.suptitle("I3 confusion matrices: row fractions and mean counts", fontsize=15, y=.985)
        fig.text(.5, .01, "Percentages: row-normalize each seed's confusion matrix, then average across seeds.\n"
                 "Parentheses: mean count per seed; seeds repeat the same structures and are not pooled into a larger sample.",
                 ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .07, .91, .96), w_pad=3, h_pad=2)
        if images:
            color_ax = fig.add_axes((.93, .20, .018, .60))
            fig.colorbar(images[0], cax=color_ax, label="Mean row fraction")
        artifacts.append(_save(fig, output_dir, "main_confusion_matrices",
                               "四模型I3混淆矩阵。百分比先在每种子的真实类别行内归一再平均；括号为单种子的平均计数，不能把三个种子视为1032个独立样本。", missing))

        geometry_models = ("length_only", "mask_only", "mask_length")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        missing = []
        for ax, metric, title in zip(axes, ("macro_f1", "f1"), ("Geometry LR: Macro F1", "Geometry LR: CHID F1")):
            _score_axis(ax, title)
            for x, model in enumerate(geometry_models):
                group = groups.get(f"I3/{model}")
                if group is None:
                    _missing(ax, x)
                    if model not in missing:
                        missing.append(model)
                else:
                    _score_group(ax, x, group["metrics"][metric]["values"], "#727272", label=True)
            ax.set_xticks(range(3), [LABELS[model] for model in geometry_models])
            ax.set_xlim(-.5, 2.5)
            ax.tick_params(axis="x", labelsize=9)
        fig.suptitle("Signal-free traditional geometry controls", fontsize=14, y=.985)
        fig.text(.5, .01, f"Each point: one deterministic complete OOF seed, {summary['n_structures']} unique structures.\n"
                 "These linear baselines supplement G0 neural controls; weak LR does not exclude nonlinear geometry shortcuts.",
                 ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, .94))
        artifacts.append(_save(fig, output_dir, "geometry_LR_baselines",
                               "三组不含信号的几何LR对照，均为单个确定性种子的完整OOF。应联合相同神经骨干的G0解释；几何LR弱不等于排除几何捷径。", missing))
    return artifacts
