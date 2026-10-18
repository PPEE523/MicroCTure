"""Readable analysis of saved formal predictions; no fitting or inference.

The report and its small numerical companion live outside the scientific run.
Every displayed score is checked against a complete saved OOF prediction CSV.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median, stdev


CLASSES = ("OPCID", "CHIN", "CHID")
NEURAL = ("m2", "m2plus", "cnn_transformer")
MAIN = ("pca_lr", *NEURAL)
LABELS = {"pca_lr": "PCA + Logistic Regression", "m2": "M2", "m2plus": "M2+",
          "cnn_transformer": "CNN-Transformer", "length_only": "仅长度 LR",
          "mask_only": "仅掩码 LR", "mask_length": "掩码 + 长度 LR"}


def _stat(values: list[float]) -> dict:
    return {"n": len(values), "mean": mean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None,
            "values": values}


def _score_cell(value: dict) -> str:
    sd = value["sample_sd"]
    return f"{value['mean']:.3f}" + (f" ± {sd:.3f}" if sd is not None else "（单次）")


def _csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_metadata(run_dir: Path, summary: dict, predictions: dict) -> tuple[dict, dict]:
    """Use the cache bound to the formal manifest, never an unsealed error export."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_signature"] != summary["dataset_signature"]:
        raise ValueError("Formal dataset signature differs from report summary")
    cache_dir = Path(manifest["config"]["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = _layout_root / cache_dir
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["signature"] != summary["dataset_signature"]:
        raise ValueError("Cached metadata signature differs from formal dataset")
    structures, groups = metadata["structures"], metadata["group_id"]
    if len(structures) != len(groups):
        raise ValueError("Frozen metadata group count differs from structure count")
    identity = {}
    reference = predictions[("I3", "m2", 42)]
    for row, group in zip(structures, groups):
        sid = row["structure_id"]
        if sid in identity or sid not in reference:
            raise ValueError("Repeated or missing frozen metadata identity")
        saved, start, end = reference[sid], int(row["start"]), int(row["end"])
        if end <= start or int(saved["y_true"]) != CLASSES.index(row["type"]) or int(saved["group_id"]) != int(group):
            raise ValueError(f"Frozen interval/class/group differs from OOF: {sid}")
        fold = int(saved["fold"])
        if sid not in manifest["fold_ids"][str(fold)]["test_ids"]:
            raise ValueError(f"Frozen outer fold differs from OOF: {sid}")
        identity[sid] = {"structure_id": sid, "start": start, "end": end,
                         "length_bp": end - start, "group_id": int(group), "fold": fold}
    if len(identity) != len(reference):
        raise ValueError("Frozen metadata does not cover complete OOF")
    return identity, {"metadata_path": str(metadata_path), "metadata_sha256": _sha256(metadata_path),
                      "dataset_signature": metadata["signature"], "n_spatial_groups": len(set(groups)),
                      "coordinate_status": metadata.get("coordinate_status")}


def _scores_from_rows(rows: list[dict]) -> dict:
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    for row in rows:
        true, predicted = int(row["y_true"]), int(row["y_pred"])
        if not 0 <= true < 3 or not 0 <= predicted < 3:
            raise ValueError("Invalid class in saved prediction CSV")
        probabilities = [float(row[f"p_{name}"]) for name in CLASSES]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("Nonfinite or out-of-range saved probabilities")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=2e-6):
            raise ValueError("Saved probabilities do not sum to one")
        if predicted != max(range(3), key=probabilities.__getitem__):
            raise ValueError("Saved y_pred does not match argmax")
        matrix[true][predicted] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, support, predicted = matrix[i][i], sum(matrix[i]), sum(r[i] for r in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": support}
    return {"confusion_matrix": matrix, "per_class": per_class,
            "macro_f1": mean(item["f1"] for item in per_class.values()),
            "accuracy": sum(matrix[i][i] for i in range(3)) / len(rows),
            "balanced_accuracy": mean(item["recall"] for item in per_class.values())}


def _verified_predictions(run_dir: Path, summary: dict) -> tuple[dict, dict]:
    """Independently reconstruct all complete OOF scores without model access."""
    predictions, reference, files = {}, None, []
    for key, details in summary["models"].items():
        condition, model = key.split("/")
        for seed, saved in details["seeds"].items():
            if saved.get("status") != "complete":
                raise ValueError(f"Report requires complete OOF: {key}/seed_{seed}")
            path = run_dir / "oof" / condition / model / f"seed_{seed}" / "predictions.csv"
            rows = _csv_rows(path)
            identities = {r["structure_id"]: (int(r["y_true"]), int(r["group_id"]),
                                             int(r["fold"])) for r in rows}
            if len(identities) != len(rows) or len(rows) != summary["n_structures"]:
                raise ValueError(f"Repeated or missing structure in {path}")
            if reference is None:
                reference = identities
            elif identities != reference:
                raise ValueError(f"OOF label/group/fold identities differ: {path}")
            actual, expected = _scores_from_rows(rows), saved["oof"]
            if actual["confusion_matrix"] != expected["confusion_matrix"]:
                raise ValueError(f"OOF confusion matrix differs from summary: {path}")
            for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
                if not math.isclose(actual[metric], float(expected[metric]), abs_tol=1e-12):
                    raise ValueError(f"OOF {metric} differs from summary: {path}")
            for name in CLASSES:
                if actual["per_class"][name]["support"] != summary["class_support"][name]:
                    raise ValueError(f"OOF support differs from dataset: {path}")
                for metric in ("precision", "recall", "f1"):
                    if not math.isclose(actual["per_class"][name][metric],
                                        float(expected["per_class"][name][metric]), abs_tol=1e-12):
                        raise ValueError(f"OOF {name} {metric} differs from summary: {path}")
            predictions[(condition, model, int(seed))] = {r["structure_id"]: r for r in rows}
            files.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
    return predictions, {"verified_complete_oof_files": len(files),
                         "verified_oof_prediction_rows": len(files) * len(reference or {}),
                         "identical_label_group_fold_mapping": True,
                         "all_scores_match_saved_oof_predictions": True,
                         "prediction_file_hashes": files}


def _group_stats(summary: dict, condition: str, model: str) -> dict:
    detail = summary["models"][f"{condition}/{model}"]
    seeds = sorted((int(seed), value["oof"]) for seed, value in detail["seeds"].items())
    if any(value["status"] != "complete" for value in detail["seeds"].values()):
        raise ValueError(f"Cannot aggregate incomplete seeds: {condition}/{model}")
    metrics = {"macro_f1": _stat([float(oof["macro_f1"]) for _, oof in seeds])}
    for metric in ("precision", "recall", "f1"):
        metrics[f"CHID_{metric}"] = _stat([float(oof["per_class"]["CHID"][metric]) for _, oof in seeds])
    matrices = [oof["confusion_matrix"] for _, oof in seeds]
    metrics["CHIN_to_CHID_mean_count"] = mean(matrix[1][2] for matrix in matrices)
    metrics["CHID_to_CHIN_mean_count"] = mean(matrix[2][1] for matrix in matrices)
    metrics["CHID_counts_mean"] = {name: mean(oof["CHID_counts"][name] for _, oof in seeds)
                                  for name in ("TP", "FP", "FN")}
    agreement = detail.get("seed_prediction_agreement")
    metrics["all_class_unanimous_rate"] = agreement.get("unanimous_rate") if agreement else None
    metrics["CHID_unanimous_rate"] = agreement.get("per_class", {}).get("CHID", {}).get("unanimous_rate") if agreement else None
    return {"condition": condition, "model": model, "seeds": [seed for seed, _ in seeds], **metrics}


def _learning_rows(summary: dict) -> list[dict]:
    rows = []
    for condition in ("I0", "G0", "I3"):
        for model in NEURAL:
            detail = summary["models"][f"{condition}/{model}"]["learning_diagnostics"]
            row = {"condition": condition, "model": model,
                   "strict_learning_passes": detail["strict_learning_passes"],
                   "expected_cases": detail["expected_cases"],
                   "last_train_ce": detail["last_train_ce"],
                   "tail_ce_coefficient_of_variation": detail["tail_ce_coefficient_of_variation"],
                   "terminal_relative_loss_regression": detail["terminal_relative_loss_regression"],
                   "effective_gradients": detail["effective_gradients"],
                   "prediction_events": detail["prediction_events"]}
            rows.append(row)
    return rows


def _chronic_cases(metadata: dict, predictions: dict) -> tuple[list[dict], dict]:
    result, per_model = [], {}
    reference = predictions[("I3", "m2", 42)]
    for model in NEURAL:
        per_model[model] = sorted((sid for sid, row in reference.items()
                                  if int(row["y_true"]) == 2
                                  and all(int(predictions[("I3", model, seed)][sid]["y_pred"]) != 2
                                          for seed in (42, 43, 44))), key=lambda sid: int(sid.split("_")[-1]))
    common = sorted(set(per_model["m2plus"]) & set(per_model["cnn_transformer"]),
                    key=lambda sid: int(sid.split("_")[-1]))
    for sid in common:
        source = metadata[sid]
        row = {"structure_id": sid, **{name: int(source[name]) for name in
               ("start", "end", "length_bp", "group_id", "fold")}}
        for model in NEURAL:
            row[f"{model}_seeds_42_43_44"] = "/".join(CLASSES[int(predictions[("I3", model, seed)][sid]["y_pred"])]
                                                     for seed in (42, 43, 44))
        result.append(row)
    return result, per_model


def _late_loss_and_early_single(run_dir: Path) -> tuple[list[dict], list[dict]]:
    late, early = [], []
    for model in NEURAL:
        by_epoch = {20: [], 40: []}
        events = {view: [] for view in ("train", "validation")}
        for seed in (42, 43, 44):
            for fold in range(5):
                for condition in ("I3", "G0"):
                    path = run_dir / "runs" / condition / model / f"seed_{seed}" / f"fold_{fold}" / "history.csv"
                    for row in _csv_rows(path):
                        epoch = int(row["epoch"])
                        if condition == "I3" and epoch in by_epoch:
                            by_epoch[epoch].append(row)
                        if condition == "G0" and epoch > 0:
                            for view in events:
                                if float(row[f"{view}_majority_fraction"]) == 1.0:
                                    events[view].append(epoch)
        if any(len(rows) != 15 for rows in by_epoch.values()):
            raise ValueError(f"Missing complete inner loss trajectories for {model}")
        late.append({"model": model, "case_count": 15,
                     "epoch20": {name: median(float(row[name]) for row in by_epoch[20])
                                  for name in ("weighted_train_ce", "weighted_validation_ce")},
                     "epoch40": {name: median(float(row[name]) for row in by_epoch[40])
                                  for name in ("weighted_train_ce", "weighted_validation_ce")}})
        early.append({"model": model, "condition": "G0",
                      **{view: {"single_class_event_count": len(values), "epochs": sorted(set(values))}
                         for view, values in events.items()}})
    return late, early


def _same_center(predictions: dict) -> list[dict]:
    result = []
    for pair in (("CHID_18", "CHIN_172"), ("CHID_25", "CHIN_243")):
        for model in NEURAL:
            correct_seeds = []
            for seed in (42, 43, 44):
                if all(predictions[("I3", model, seed)][sid]["y_pred"] ==
                       predictions[("I3", model, seed)][sid]["y_true"] for sid in pair):
                    correct_seeds.append(seed)
            result.append({"pair": list(pair), "model": model,
                           "both_correct_seed_count": len(correct_seeds), "correct_seeds": correct_seeds})
    return result


def _paired(summary: dict, first: str, second: str) -> dict:
    a, b = summary["models"][first]["seeds"], summary["models"][second]["seeds"]
    if set(a) != set(b):
        raise ValueError(f"Unmatched complete OOF seeds: {first}, {second}")
    seeds = sorted(a, key=int)
    return {"first": first, "second": second, "seeds": [int(seed) for seed in seeds],
            "macro_f1_delta": _stat([a[seed]["oof"]["macro_f1"] - b[seed]["oof"]["macro_f1"] for seed in seeds]),
            "CHID_f1_delta": _stat([a[seed]["oof"]["per_class"]["CHID"]["f1"]
                                   - b[seed]["oof"]["per_class"]["CHID"]["f1"] for seed in seeds])}


def write_analysis_report(run_dir: Path, summary: dict, output_dir: Path,
                          figures: list[dict], verification: dict) -> dict:
    """Write a Chinese report and compact audited numerical evidence outside run_dir."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or output_dir.is_relative_to(run_dir):
        raise ValueError("Analysis output must be outside the saved scientific run")
    if summary.get("status") != "complete" or summary["completed_jobs"] != summary["expected_jobs"]:
        raise ValueError("This final analysis requires all preregistered tasks to be complete")
    predictions, oof_check = _verified_predictions(run_dir, summary)
    metadata, metadata_source = _frozen_metadata(run_dir, summary, predictions)
    main = [_group_stats(summary, "I3", model) for model in MAIN]
    ablations = [_group_stats(summary, condition, model) for model in NEURAL
                 for condition in ("I0", "G0", "I3")]
    geometry = [_group_stats(summary, "I3", model)
                for model in ("length_only", "mask_only", "mask_length")]
    learning = _learning_rows(summary)
    chronic, chronic_per_model = _chronic_cases(metadata, predictions)
    centers = _same_center(predictions)
    late_loss, early_single = _late_loss_and_early_single(run_dir)
    pairs = [_paired(summary, f"I3/{model}", f"{alternative}/{model}")
             for model in NEURAL for alternative in ("I0", "G0")]
    transformer_delta = _paired(summary, "I3/cnn_transformer", "I3/m2plus")
    main_by_model = {row["model"]: row for row in main}
    accepted = sum(row["strict_learning_passes"] for row in learning)
    expected_neural = sum(row["expected_cases"] for row in learning)
    event_counts = {}
    for view in ("train", "validation", "outer"):
        events = [row["prediction_events"][view] for row in learning]
        event_counts[view] = {"observed_events": sum(value["observed_events"] for value in events),
                              **{event: sum(value[event]["events"] for value in events)
                                 for event in ("constant_probability", "constant_centered_logits", "single_class_argmax")},
                              "final_single_class_cases": sum(value["single_class_argmax"]["final_cases"] for value in events)}
    # This prose is a conclusion about this completed protocol, not a generic
    # template that may silently repeat its claims on different results.
    if (summary["expected_jobs"] != 155 or expected_neural != 135 or accepted != expected_neural
            or summary["n_structures"] != 344 or summary["class_support"] != {"OPCID": 68, "CHIN": 250, "CHID": 26}
            or metadata_source["n_spatial_groups"] != 49
            or any(summary[name] for name in ("pending_jobs", "invalid_jobs", "numerical_failure_jobs", "execution_failure_jobs"))):
        raise ValueError("Saved coverage differs from the completed run interpreted by this report")
    if (set(chronic_per_model["m2plus"]) != {"CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19"}
            or chronic_per_model["m2plus"] != chronic_per_model["cnn_transformer"]
            or any(predictions[("I3", model, seed)][row["structure_id"]]["y_pred"] != "1"
                   for model in ("m2plus", "cnn_transformer") for seed in (42, 43, 44) for row in chronic)):
        raise ValueError("Chronic CHID cases differ; current interpretation needs review")
    if any(value["constant_probability"] or value["constant_centered_logits"] or value["final_single_class_cases"]
           for value in event_counts.values()):
        raise ValueError("Prediction collapse found; current interpretation needs review")
    for row in learning:
        for view in ("train", "validation"):
            actual = row["prediction_events"][view]
            if row["condition"] in ("I0", "I3") and actual["single_class_argmax"]["events"]:
                raise ValueError("I0/I3 single-class predictions found; current interpretation needs review")
        if row["condition"] == "G0":
            trajectory = next(item for item in early_single if item["model"] == row["model"])
            for view in ("train", "validation"):
                if trajectory[view]["single_class_event_count"] != row["prediction_events"][view]["single_class_argmax"]["events"]:
                    raise ValueError("G0 early trajectory events differ from saved summary")
                if trajectory[view]["epochs"] and max(trajectory[view]["epochs"]) >= 31:
                    raise ValueError("G0 single-class predictions persist into acceptance window")
    if any(item["both_correct_seed_count"] != (0 if item["model"] == "m2" and item["pair"][0] == "CHID_25" else 3)
           for item in centers) or any(predictions[("I3", "m2", seed)]["CHIN_243"]["y_pred"] != "2" for seed in (42, 43, 44)):
        raise ValueError("Same-center evidence differs; current interpretation needs review")
    if (main_by_model["m2plus"]["macro_f1"]["mean"] != max(row["macro_f1"]["mean"] for row in main)
            or main_by_model["m2plus"]["CHID_f1"]["mean"] != max(row["CHID_f1"]["mean"] for row in main)
            or any(value >= 0 for value in transformer_delta["CHID_f1_delta"]["values"])):
        raise ValueError("Main model rankings differ; current interpretation needs review")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "chronic_CHID_cases.csv"
    if chronic:
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(chronic[0]))
            writer.writeheader()
            writer.writerows(chronic)
    table = ["| 模型 | macro-F1 | CHID precision | CHID recall | CHID F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in main:
        table.append("| " + " | ".join([LABELS[row["model"]], *[_score_cell(row[name]) for name in
                     ("macro_f1", "CHID_precision", "CHID_recall", "CHID_f1")]]) + " |")
    m2, plus, transformer = (main_by_model[model] for model in NEURAL)
    lines = ["# 任务一正式比较：结果分析与可视化", "",
             f"本轮全部{summary['completed_jobs']}项任务已完成：{expected_neural}次神经网络训练、{summary['completed_jobs']-expected_neural}项传统基线；无待办、执行失败、数值失败或无效任务。"
             f"持续学习验收通过 {accepted}/{expected_neural}，此前分类头退化的问题在本轮末期验收中未复现。"
             "在相同 I3/H3/1e-4 协议下，M2+ 的 macro-F1 和 CHID F1 均值最高；Transformer 尚未提供额外收益。"
             "M2 的 CHID 召回更高，代价是更多误报。当前结果支持继续以 M2+ 做错误分析，不能据此宣布 CHID 稳定识别已经达标。", "",
             "**四模型主比较**", "",
             *table, "",
             f"每个神经网络数值先由同一种子的5个固定空间折组成完整 OOF，在{summary['n_structures']}个唯一标注结构（{metadata_source['n_spatial_groups']}个空间组）上评分，再对 seeds 42/43/44 求均值和样本标准差。"
             "每个结构的两个生物学重复先平均预测概率，仅计一个样本。PCA+LR只登记一个确定种子，没有跨种子标准差。"
             "误差条表示优化种子的波动，不是总体置信区间，也不把15个 fold/seed 当成15个独立测试集。", "",
             f"M2+ 相比 M2，macro-F1 提高 {(plus['macro_f1']['mean'] - m2['macro_f1']['mean'])*100:.2f} 个百分点，"
             f"CHID F1 提高 {(plus['CHID_f1']['mean'] - m2['CHID_f1']['mean'])*100:.2f} 个百分点。"
             f"每个种子的 CHIN→CHID 误报平均由 {m2['CHIN_to_CHID_mean_count']:.2f} 个降至 {plus['CHIN_to_CHID_mean_count']:.2f} 个；"
             f"CHID→CHIN 漏检平均由 {m2['CHID_to_CHIN_mean_count']:.2f} 个增至 {plus['CHID_to_CHIN_mean_count']:.2f} 个。"
             "这些是三个种子的平均计数，非额外样本或单次混淆矩阵。CHID 总误报还包括 OPCID→CHID。", "",
             f"Transformer−M2+ 的同种子完整 OOF 差值：macro-F1 均值 {transformer_delta['macro_f1_delta']['mean']*100:+.2f} 个百分点，"
             f"CHID F1 均值 {transformer_delta['CHID_f1_delta']['mean']*100:+.2f} 个百分点；三个种子的 CHID F1 差值均为负。"
             "这支持‘本实现、该协议下尚未看到 Attention 的额外价值’；两者参数量也不同，不足以推断 Transformer 一般无效，或把所有差异完全归因于 Attention。", "",
             "**输入消融与几何基线**", "",
             "| 骨干 | I0 macro-F1 / CHID F1 | G0 macro-F1 / CHID F1 | I3 macro-F1 / CHID F1 |",
             "| --- | ---: | ---: | ---: |"]
    for model in NEURAL:
        lookup = {row["condition"]: row for row in ablations if row["model"] == model}
        lines.append("| " + " | ".join([LABELS[model], *[
            f"{lookup[c]['macro_f1']['mean']:.3f} / {lookup[c]['CHID_f1']['mean']:.3f}" for c in ("I0", "G0", "I3")]]) + " |")
    lines.extend(["", "I0保留信号和有效性通道、将目标掩码与长度归零；G0保留掩码和长度、将信号和有效性通道归零；I3使用完整输入。"
                  "I3−I0同时增加掩码与长度，不能分开归因。I3−G0同时恢复接触图信号和有效性，衡量的是它们整体的增量，不是严格分离的图案效应。", "",
                  "| 几何 Logistic Regression | macro-F1 | CHID F1 |",
                  "| --- | ---: | ---: |"])
    for row in geometry:
        lines.append(f"| {LABELS[row['model']]} | {row['macro_f1']['mean']:.3f} | {row['CHID_f1']['mean']:.3f} |")
    increments = [next(row for row in pairs if row["first"] == f"I3/{model}" and row["second"] == f"G0/{model}") for model in NEURAL]
    positive_pairs = []
    for model in NEURAL:
        pair = summary["paired_comparisons"][f"{model}:I3-G0"]["fold_seed_deltas"]
        positive_pairs.append((model, sum(row["macro_f1_delta"] > 0 for row in pair), len(pair)))
    lines.extend(["", "I3相对G0的完整 OOF macro-F1 增益依次为 " + "、".join(
                  f"{LABELS[model]} {delta['macro_f1_delta']['mean']*100:.2f} 个百分点" for model, delta in zip(NEURAL, increments)) +
                  "。相同 fold/seed 配对的正增益数为 " + "、".join(f"{LABELS[model]} {count}/{total}" for model, count, total in positive_pairs) +
                  "。几何单独输入有预测能力，但不足以解释完整输入的表现；这也不说明完整模型完全没有利用长度或掩码。", "",
                  "**学习是否持续、预测是否退化**", "",
                  "| 输入 / 骨干 | 通过数 | 末轮固定训练 CE 均值（范围） | 尾部 encoder / head 有效梯度案例 |",
                  "| --- | ---: | ---: | ---: |"])
    for row in learning:
        ce, grads = row["last_train_ce"], row["effective_gradients"]
        lines.append(f"| {row['condition']} / {LABELS[row['model']]} | {row['strict_learning_passes']}/{row['expected_cases']} | "
                     f"{ce['mean']:.4f}（{ce['min']:.4f}–{ce['max']:.4f}） | "
                     f"{grads['encoder']['effective_cases']}/{grads['encoder']['observed_cases']} / "
                     f"{grads['head']['effective_cases']}/{grads['head']['observed_cases']} |")
    lines.extend(["", f"训练后40轮共 {event_counts['train']['observed_events']} 次训练视图、{event_counts['validation']['observed_events']} 次内层验证视图检查，"
                  f"恒定概率输出和恒定中心化 logits 均为0次。所有{event_counts['outer']['observed_events']}个末轮外层预测也无单类 argmax、恒定概率或恒定中心化 logits。"
                  "I3与I0在训练后每轮的训练/内层验证视图均无单类 argmax。", "",
                  "G0存在早期短暂单类 argmax：" + "；".join(
                      f"{LABELS[row['model']]}训练/验证{row['train']['single_class_event_count']}/{row['validation']['single_class_event_count']}次，"
                      + "发生轮次" + "/".join(str(epoch) for epoch in sorted(set(row['train']['epochs']) | set(row['validation']['epochs'])))
                      for row in early_single) + "（每组每个视图均600次训练后检查）。"
                  "这些早期模型的结构间概率和中心化 logits 仍有变化；单一 argmax 不等于所有样本概率完全相同。"
                  "G0末轮和最后10轮均没有该现象，135次验收均通过。epoch 0的初始化诊断另行保留，不计入训练后40轮统计。"
                  "学习通过数说明本轮模型能够学习，识别质量必须另看 OOF CHID F1、召回和错误案例。", "",
                  "曲线同时展示固定训练/内层验证损失、内层验证 CHID F1、encoder 梯度和中心化 logits 方差。"
                  "淡线为15次真实训练，粗线为中位数；内层曲线不能当成外层最终性能。更低训练 loss 也不自动意味着更好的 CHID 泛化。", "",
                  "| I3骨干 | epoch20→40 固定训练 CE 中位数 | epoch20→40 内层验证 CE 中位数 |",
                  "| --- | ---: | ---: |"])
    for row in late_loss:
        lines.append(f"| {LABELS[row['model']]} | {row['epoch20']['weighted_train_ce']:.4f} → {row['epoch40']['weighted_train_ce']:.4f} | "
                     f"{row['epoch20']['weighted_validation_ce']:.4f} → {row['epoch40']['weighted_validation_ce']:.4f} |")
    lines.extend(["", "M2+与Transformer在后20轮继续降低训练损失，但内层验证损失回升，提示后期泛化回退或过拟合迹象；这与分类头失活的表现不同。"
                  "当前协议预先固定末轮权重，不能看到曲线后改选epoch20并替换本轮外层分数。下轮如测试早停，应预登记仅内层验证的选择规则，并让三个骨干使用相同规则。", "",
                  "**跨种子稳定性与持续漏检**", "",
                  "| 骨干 | 全部结构三种子一致 | CHID 三种子一致 | 三种子均漏检的 CHID |",
                  "| --- | ---: | ---: | --- |"])
    for row in (m2, plus, transformer):
        lines.append(f"| {LABELS[row['model']]} | {row['all_class_unanimous_rate']:.2%} | {row['CHID_unanimous_rate']:.2%} | "
                     + "、".join(chronic_per_model[row["model"]]) + " |")
    lines.extend(["", "三种子一致可以包含一致的错误，并不等于识别正确。M2+与Transformer共同持续漏检的六个结构，在三个种子中均被预测为CHIN：", "",
                  "| 结构 | 长度（kb） | 空间组 / 折 | M2三个种子预测 |",
                  "| --- | ---: | --- | --- |"])
    for row in chronic:
        lines.append(f"| {row['structure_id']} | {row['length_bp']/1000:.2f} | {row['group_id']} / {row['fold']} | {row['m2_seeds_42_43_44']} |")
    lines.extend(["", "五个结构较短（3.52–5.25 kb），CHID_11长9.66 kb；现有证据不能只按长度确定失败原因。"
                  "应优先逐例核对候选边界、CHID/CHIN标注含义、目标掩码与两尺度接触图，再判断是否需要更大上下文或新的结构证据。"
                  "[保留的持续漏检数值表](chronic_CHID_cases.csv)包含区间坐标、空间组、折和三个种子预测。", "",
                  "同中心的 CHID_18/CHIN_172 与 CHID_25/CHIN_243，M2+和Transformer都在三个种子中正确区分整对结构。"
                  "M2正确区分第一对，但第二对的CHIN_243在三个种子中均误判为CHID。目标感知输入解决了旧输入的相同图像歧义，仍不保证每个骨干都能可靠利用这项信息。", "",
                  "**结论的适用范围与下一步**", "",
                  "本轮是此前已检查过的固定空间折上的开发性重评，外层数据没有参与本轮拟合或末轮选择，但并非全新的盲测。"
                  "只有26个唯一CHID，各折5–6个；三个种子共享这26个结构，不能把它们当成78个新增CHID。标注区间存在重叠与空间相关，不能假定每个结构统计独立。"
                  "缓存区间坐标仍为 provisional（待复核），逐例检查时须保留这一限制。"
                  "当前任务仍是给定候选区间的三分类，尚未验证全基因组候选发现。", "",
                  "建议将 M2+ 作为后续分析的主要骨干，保留 M2 的高召回结果作为取舍依据。先完成六个持续漏检结构及稳定误报结构的逐例审查，"
                  "需要改标注或输入时另立版本，并保持所有模型的比较协议一致。任何基于当前外层结果的架构或阈值修改，都需要在之后的独立评估中重新验证。", "",
                  "**图表**", ""])
    figure_records = []
    for figure in figures:
        png, pdf = Path(figure["path_png"]).resolve(), Path(figure["path_pdf"]).resolve()
        if not png.is_file() or not pdf.is_file():
            raise ValueError(f"Report figure has not been generated: {figure['name']}")
        png_relative, pdf_relative = os.path.relpath(png, output_dir), os.path.relpath(pdf, output_dir)
        lines.extend([f"**{figure['name']}**", "", str(figure["caption"]), "",
                      f"![{figure['name']}]({png_relative})", "", f"[矢量 PDF]({pdf_relative})", ""])
        figure_records.append({"name": figure["name"], "png": png_relative, "pdf": pdf_relative,
                               "caption": figure["caption"], "png_sha256": _sha256(png), "pdf_sha256": _sha256(pdf)})
    lines.extend(["**来源与核验**", "",
                  f"原结果目录：`{run_dir}`。报告只读取保存的预测、统计和轨迹，不重新训练、推理或变更原结果。"
                  f"独立重算 {oof_check['verified_complete_oof_files']} 份完整 OOF CSV，"
                  f"共 {oof_check['verified_oof_prediction_rows']} 行，分类指标和混淆矩阵均与保存汇总一致；标签、空间组与折映射完全一致。", "",
                  f"实验签名：`{summary['experiment_signature']}`。训练源签名：`{summary['source_signature']}`。"
                  "更细核验及精简数值见 [analysis_summary.json](analysis_summary.json)。", ""])
    analysis = {"protocol": summary["protocol"], "result_directory": str(run_dir),
                "experiment_signature": summary["experiment_signature"], "source_signature": summary["source_signature"],
                "dataset_signature": summary["dataset_signature"], "n_structures": summary["n_structures"],
                "frozen_metadata_source": metadata_source,
                "class_support": summary["class_support"], "completed_jobs": summary["completed_jobs"],
                "expected_jobs": summary["expected_jobs"], "score_unit": summary["score_unit"],
                "seed_sd_interpretation": summary["interpretation"], "main_I3": main,
                "ablations": ablations, "geometry_LR": geometry, "paired_complete_oof_deltas": pairs,
                "transformer_minus_m2plus": transformer_delta,
                "learning": [{**{key: row[key] for key in
                                  ("condition", "model", "strict_learning_passes", "expected_cases", "last_train_ce",
                                   "tail_ce_coefficient_of_variation", "terminal_relative_loss_regression", "effective_gradients")},
                              "prediction_events": {view: {
                                  "observed_events": events["observed_events"],
                                  "observed_cases": events["observed_cases"],
                                  **{name: {key: events[name][key] for key in ("events", "ever_cases", "final_cases")}
                                     for name in ("constant_probability", "constant_centered_logits", "single_class_argmax")}}
                                  for view, events in row["prediction_events"].items()}}
                             for row in learning],
                "I3_epoch20_to_40_loss_medians": late_loss, "G0_early_single_class_epochs": early_single,
                "prediction_event_totals": event_counts, "strict_learning_passes": accepted,
                "strict_learning_expected_cases": expected_neural, "chronic_CHID_by_model": chronic_per_model,
                "shared_chronic_CHID_cases": chronic, "same_center_pair_recognition": centers,
                "figures": figure_records, "verification": {"artifact_verification": verification, **oof_check}}
    report_path, summary_path = output_dir / "analysis.md", output_dir / "analysis_summary.json"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return {"path_md": str(report_path), "path_summary": str(summary_path),
            "path_chronic": str(table_path) if chronic else None, "n_figures": len(figures),
            "verified_oof_files": oof_check["verified_complete_oof_files"]}
