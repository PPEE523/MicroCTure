#!/usr/bin/env python3
"""Read-only verification and six-case reporting, without changing training code.

Writes derived reports only; no image preprocessing, inference, fitting or
modification of the resumable controlled experiment is performed.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
from scripts.task1_controlled_reporting import aggregate_heads
from microcture.task1.evaluation import _read_prediction_csv
from microcture.task1.training import source_signature
from microcture.task1_controlled.diagnostics import learning_acceptance
from microcture.task1_controlled.evidence import json_signature, sha256
from microcture.task1_controlled.runner import controlled_source_signature, job_directory, planned_jobs


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_frozen_metadata(manifest: dict) -> dict:
    """Hash only small metadata/source files, never the image/count arrays."""
    config = manifest["config"]
    freeze_path = project_path(config["evidence_dir"]) / "freeze_manifest.json"
    if sha256(freeze_path) != manifest["frozen_evidence_sha256"]:
        raise ValueError("Frozen evidence index changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["dataset_signature"] != manifest["dataset_signature"] or freeze["run_signature"] != manifest["original_run_signature"]:
        raise ValueError("Frozen dataset/run changed")
    if source_signature(ROOT) != freeze["source_signature"]:
        raise ValueError("Original preprocessing/evaluation source changed")
    cache_metadata = project_path(config["cache_dir"]) / "metadata.json"
    folds = {config["input_stage"]["fold"], *(case["fold"] for case in config["head_stage"]["cases"].values())}
    paths = [cache_metadata, *(project_path(config["frozen_run_dir"]) / "folds" / f"fold_{fold}.json" for fold in sorted(folds))]
    frozen_records = {record["path"]: record for record in freeze["files"]}
    for path in paths:
        expected = frozen_records[str(path.relative_to(ROOT))]
        if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Frozen metadata changed: {path}")
    metadata = json.loads(cache_metadata.read_text(encoding="utf-8"))
    if metadata["signature"] != manifest["dataset_signature"] or metadata["label_order"] != ["OPCID", "CHIN", "CHID"]:
        raise ValueError("Frozen cache metadata signature/class order changed")
    return metadata


def verify_protocol(directory: Path, job: dict, result: dict, manifest: dict) -> dict:
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if protocol["job"] != job or protocol["training"] != config["training"] or protocol["acceptance"] != config["acceptance"]:
        raise ValueError(f"Saved protocol differs from preregistration: {directory}")
    preprocessing = protocol["preprocessing"]
    signature = json_signature({key: value for key, value in preprocessing.items() if key != "signature"})
    if signature != preprocessing["signature"] or signature != result["preprocessing_signature"]:
        raise ValueError(f"Preprocessing signature mismatch: {directory}")
    frozen = Path(config["frozen_run_dir"])
    if not frozen.is_absolute():
        frozen = ROOT / frozen
    fold = json.loads((frozen / "folds" / f"fold_{job['fold']}.json").read_text(encoding="utf-8"))
    if preprocessing["frozen_fold_signature"] != json_signature(fold):
        raise ValueError(f"Frozen fold changed: {directory}")
    if preprocessing["fold"] != job["fold"] or preprocessing["dataset_signature"] != manifest["dataset_signature"]:
        raise ValueError(f"Fold/dataset mismatch: {directory}")
    if preprocessing["signal_preprocessing"] != fold["preprocessing"]:
        raise ValueError(f"Frozen signal transformation changed: {directory}")
    if preprocessing["frozen_signal_preprocessing_signature"] != fold["preprocessing"]["signature"]:
        raise ValueError(f"Frozen signal signature changed: {directory}")
    for split in ("train", "validation"):
        ids = preprocessing[f"{split}_ids"]
        if len(ids) != len(set(ids)) or set(ids) != set(fold[f"{split}_ids"]):
            raise ValueError(f"Frozen {split} structure IDs changed: {directory}")
        if result[f"{split}_structures"] != len(ids):
            raise ValueError(f"Recorded {split} count differs from frozen IDs: {directory}")
    if set(fold["test_ids"]) & (set(preprocessing["train_ids"]) | set(preprocessing["validation_ids"])):
        raise ValueError(f"Outer structures entered a development split: {directory}")
    if set(preprocessing["train_ids"]) & set(preprocessing["validation_ids"]):
        raise ValueError(f"Inner training and validation overlap: {directory}")
    if any(preprocessing[key] is not False for key in ("outer_images_used", "signal_statistics_refitted", "splits_refitted")) or result["outer_images_used"] is not False:
        raise ValueError(f"Outer evaluation/refitting is forbidden in this control: {directory}")
    if preprocessing["length_scaler"]["fitted_unique_structures"] != len(preprocessing["train_ids"]):
        raise ValueError(f"Length fit count differs from training scope: {directory}")
    return preprocessing


def read_verified_record(run_dir: Path, job: dict, manifest: dict, metadata: dict | None = None) -> tuple[dict, list[dict]] | None:
    directory = job_directory(run_dir, job)
    if not (directory / "metrics.json").exists():
        return None
    result = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if result["experiment_signature"] != manifest["experiment_signature"] or result["job"] != job:
        raise ValueError(f"Job/protocol mismatch: {directory}")
    preprocessing = verify_protocol(directory, job, result, manifest)
    if result["status"] != "completed":
        history = json.loads((directory / "history.json").read_text())["epochs"] if (directory / "history.json").exists() else []
        return result, history
    prediction = _read_prediction_csv(directory / "validation_predictions.csv")
    if set(prediction["structure_id"].tolist()) != set(preprocessing["validation_ids"]):
        raise ValueError(f"Prediction IDs are not the frozen inner-validation IDs: {directory}")
    if metadata is None:
        metadata = verify_frozen_metadata(manifest)
    targets = {structure["structure_id"]: (metadata["label_order"].index(structure["type"]), int(group))
               for structure, group in zip(metadata["structures"], metadata["group_id"], strict=True)}
    for row in prediction["rows"]:
        if (int(row["y_true"]), int(row["group_id"])) != targets[row["structure_id"]]:
            raise ValueError(f"Saved true label/spatial group differs from frozen metadata: {directory}")
    expected_metrics = result.get("validation_diagnostics", {}).get("metrics", result.get("validation_metrics"))
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "confusion_matrix", "per_class"):
        if prediction["metrics"][key] != expected_metrics[key]:
            raise ValueError(f"Validation CSV/metrics mismatch for {key}: {directory}")
    if job["kind"] == "geometry":
        return result, []
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))["epochs"]
    epochs = manifest["config"]["training"]["epochs"]
    if [row["epoch"] for row in history] != list(range(epochs + 1)):
        raise ValueError(f"Missing/duplicate diagnostic epochs: {directory}")
    if learning_acceptance(history, manifest["config"]) != result["acceptance"]:
        raise ValueError(f"Stored acceptance differs from unchanged preregistered rules: {directory}")
    if history[-1]["weighted_train_ce"] != result["final_train_ce"]:
        raise ValueError(f"Final CE differs from the last trace point: {directory}")
    if history[0]["weighted_train_ce"] != result["initial_train_ce"] or result["epochs"] != epochs:
        raise ValueError(f"Initial CE/epoch count mismatch: {directory}")
    if not (directory / "final.pt").is_file():
        raise ValueError(f"Missing final checkpoint: {directory}")
    if history[-1]["validation_diagnostics"] != result["validation_diagnostics"]:
        raise ValueError(f"Final validation diagnostic differs from the last trace point: {directory}")
    if history[-1]["train_diagnostics"] != result["train_diagnostics"]:
        raise ValueError(f"Final train diagnostic differs from the last trace point: {directory}")
    initialization = json.loads((directory / "initialization.json").read_text(encoding="utf-8"))
    if any(initialization[key] != result[key] for key in ("initial_state_sha256", "common_initial_state_sha256")):
        raise ValueError(f"Initialization evidence mismatch: {directory}")
    return result, history


def input_review(job: dict, record: dict) -> dict:
    metric = record.get("validation_diagnostics", {}).get("metrics", record.get("validation_metrics", {}))
    chid = metric.get("per_class", {}).get("CHID", {})
    confusion = metric.get("confusion_matrix", [[0] * 3 for _ in range(3)])
    correct = confusion[2][2]
    false_positive = sum(confusion[row][2] for row in (0, 1))
    row = {"name": job["name"], "kind": job["kind"], "status": record["status"],
           "accepted": (record.get("acceptance") or {}).get("accepted"),
           "macro_f1": metric.get("macro_f1"), "CHID_precision": chid.get("precision"),
           "CHID_recall": chid.get("recall"), "CHID_f1": chid.get("f1"),
           "CHID_support": chid.get("support"), "CHID_correct": correct,
           "CHID_false_positive": false_positive, "CHID_false_negative": (chid.get("support", 0) - correct),
           "final_train_ce": record.get("final_train_ce"),
           "validation_prediction_counts": {name: sum(confusion[r][c] for r in range(3))
                                            for c, name in enumerate(("OPCID", "CHIN", "CHID"))}}
    if record.get("acceptance"):
        primary = record["acceptance"].get("primary", {})
        row["acceptance_failed_checks"] = primary.get("failed_checks", [])
        row["tail_relative_loss_drop"] = primary.get("relative_loss_drop")
        row["terminal_relative_loss_regression"] = primary.get("terminal_relative_loss_regression")
        row["numerical_acceptance"] = primary
        for split, view in primary.get("views", {}).items():
            row[f"{split}_minimum_centered_logit_variance"] = view["centered_logit_variance_min"]
            row[f"{split}_maximum_majority_fraction"] = view["majority_prediction_fraction_max"]
        row["minimum_encoder_epoch_mean_gradient"] = primary.get("gradients", {}).get("encoder", {}).get("mean_norm_min")
    return row


def verify_pairing(records: list[dict]) -> dict:
    completed = [item for item in records if item["metrics"]["status"] == "completed"]
    groups = {}
    for item in completed:
        groups.setdefault(item["case"], []).append(item["metrics"])
    result = {}
    for case, items in groups.items():
        checks = {"common_initial_weights": len({m["common_initial_state_sha256"] for m in items}) == 1,
                  "minibatch_order": len({m["minibatch_order_sha256"] for m in items}) == 1,
                  "preprocessing": len({m["preprocessing_signature"] for m in items}) == 1,
                  "class_weights": len({tuple(m["class_weights"]) for m in items}) == 1}
        heads = {m["job"]["head"]: m for m in items}
        if case == "input_reference":
            checks["all_input_full_initial_weights"] = len({m["initial_state_sha256"] for m in items}) == 1
        for first, second in (("H0", "H1"), ("H2", "H3")):
            checks[f"{first}_{second}_full_initial_weights"] = (
                heads[first]["initial_state_sha256"] == heads[second]["initial_state_sha256"]
                if first in heads and second in heads else None)
        if any(value is False for value in checks.values()):
            raise ValueError(f"Pairing mismatch: {case}")
        result[case] = {"completed_versions": len(items), "comparison_ready": len(items) == 4, "checks": checks}
    return result


def aggregate_csv_rows(aggregate: dict) -> list[dict]:
    """Keep the most useful numbers scalar and give each rate its denominator."""
    rows = []
    for head, summary in aggregate["head_summaries"].items():
        row = {"head": head, "expected_cases": summary["expected_cases"],
               "completed_cases": summary["completed_cases"], "missing_cases": summary["missing_cases"],
               "numerical_failure_cases": summary["numerical_failure_cases"],
               "strict_pass_cases": summary["strict_accepted"]["numerator"],
               "strict_observed_terminal_cases": summary["strict_accepted"]["denominator"]}
        for metric in ("final_train_ce", "last_window_loss_cv", "last_window_loss_regression"):
            row.update({f"{metric}_{key}": value for key, value in summary[metric + "_by_case"].items()})
        for group, stats in summary["effective_gradient_cases"].items():
            for key in ("numerator", "denominator", "rate"):
                row[f"{group}_effective_gradient_{key}"] = stats[key]
            row[f"{group}_mean_gradient_norm"] = stats["case_mean_epoch_gradient_norm"]["mean"]
            row[f"{group}_minimum_epoch_mean_norm"] = stats["case_minimum_epoch_gradient_norm"]["min"]
        for split, events in summary["event_incidence"].items():
            for event in ("centered_logits_constant", "probabilities_constant", "single_class_predictions", "majority_concentrated"):
                for measure in ("post_epoch_incidence", "ever_case_incidence", "final_epoch_case_incidence"):
                    for key in ("numerator", "denominator", "rate"):
                        row[f"{split}_{event}_{measure}_{key}"] = events[event][measure][key]
        row.update({f"failure_{key}_cases": value for key, value in summary["failure_category_counts"].items()})
        rows.append(row)
    return rows


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def write_markdown(path: Path, report: dict) -> None:
    lines = ["# Target-aware 与分类头控制核对", "", "本报告仅读取已保存预测、指标和轨迹；没有训练、重预测或修改续跑源码。", "",
             "## 目标输入与几何基线", "", "| 输入 | 数值验收 | macro-F1 | CHID P/R/F1 | CHID正确/支持 | 误报 |", "| --- | --- | ---: | --- | --- | ---: |"]
    for row in report["input_review"]:
        if row["status"] != "completed":
            continue
        passed = "—" if row["accepted"] is None else ("通过" if row["accepted"] else "未通过")
        lines.append(f"| {row['name']} | {passed} | {row['macro_f1']:.3f} | {row['CHID_precision']:.3f}/{row['CHID_recall']:.3f}/{row['CHID_f1']:.3f} | {row['CHID_correct']}/{row['CHID_support']} | {row['CHID_false_positive']} |")
    scope = report.get("input_scope", {})
    if scope:
        lines += ["", f"输入控制使用固定 fold {scope['fold']} 的内层验证：{scope['validation_structures']} 个结构，{scope['CHID_structures']} 个 CHID，仅 {scope['CHID_independent_groups']} 个 CHID 空间组。"]
        if scope["all_exact_conflicting_structures_in_outer_test"]:
            lines += ["", "两组历史相同图像/不同标签的实例均在该 fold 外层测试中；当前内层指标不能证明模型已经识别对这两组对象。掩码预检只证明输入表达能够区分它们。"]
    lines += ["", "几何基线弱不能排除非线性模型利用几何捷径；I1/I3 的 F1 升高也不能直接等同于 CHID 召回改善。目标 mask 本身携带长度信息。", "",
              "### I0–I3 的数值验收", "", "以下为最后 10 轮；全部原定检查与阈值见 `summary.json`。固定训练 CE 下降至少 1%，同长度窗口损失回退不超过 5%；判别 logits 方差大于 1e-6，encoder 轮均梯度大于 1e-10，多数类别比例小于 95%。头部活动与其余梯度亦已复算。", "",
              "| 输入 | 末轮CE | 尾窗CE下降 | 尾窗损失回退 | 训练/验证判别方差最小值 | encoder最小轮均梯度 | 训练/验证多数占比最大值 |", "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for row in report["input_review"]:
        if row["kind"] != "neural" or row["status"] != "completed":
            continue
        variance = "/".join(f"{row[f'{split}_minimum_centered_logit_variance']:.3g}" for split in ("train", "validation"))
        majority = "/".join(f"{row[f'{split}_maximum_majority_fraction']:.1%}" for split in ("train", "validation"))
        lines.append(f"| {row['name']} | {row['final_train_ce']:.6f} | {row['tail_relative_loss_drop']:.1%} | {row['terminal_relative_loss_regression']:.1%} | {variance} | {row['minimum_encoder_epoch_mean_gradient']:.3g} | {majority} |")
    lines += ["",
              "## 分类头的末轮预测", "", "计数按 OPCID/CHIN/CHID 排列；训练与验证分开。", "",
              "| 案例 | 头 | 严格验收 | 训练分布 | 验证分布 | 末轮CE | CHID F1 | 未通过原因 |", "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
    for row in report["case_head_review"]:
        if row["status"] != "completed":
            continue
        counts = lambda value: "/".join(str(value[k]) for k in ("OPCID", "CHIN", "CHID"))
        lines.append(f"| {row['case']} | {row['head']} | {'通过' if row['accepted'] else '未通过'} | {counts(row['train_prediction_counts'])} | {counts(row['validation_prediction_counts'])} | {row['final_train_ce']:.6f} | {row['validation_CHID_f1']:.3f} | {', '.join(row['failed_checks']) or '—'} |")
    aggregate = report["head_aggregate"]
    lines += ["", "## 六案例汇总与选择边界", "", "缺失案例不算失败。epoch 0 单独保留；epoch 事件率合并实际观测的训练后 epoch，以观测数为分母，短失败轨迹贡献较少观测；案例发生率按可观测案例计算。末轮损失等统计以案例等权，标准差是案例间总体标准差。损失回退与恒定输出是不同失败原因；H0 的近零损失 CV 反映常数停滞，不能当作持续学习稳定。", "",
              "| 头 | 已完成/预定 | 持续学习通过/已观测 | 末轮CE均值±总体SD | 有效梯度案例 total/encoder/head | 验证恒定概率末轮案例 | 验证恒定概率epoch | 验证单类预测epoch | 末窗CE变异系数均值 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rate = lambda value: f"{value['numerator']}/{value['denominator']}"
    for head, value in aggregate["head_summaries"].items():
        loss = value["final_train_ce_by_case"]
        loss_text = "—" if loss["mean"] is None else f"{loss['mean']:.6f}±{loss['std_population']:.6f}"
        gradients = "/".join(str(value["effective_gradient_cases"][group]["numerator"]) for group in ("total", "encoder", "head"))
        events = value["event_incidence"]["validation"]
        cv = value["last_window_loss_cv_by_case"]["mean"]
        cv_text = "—" if cv is None else f"{cv:.1%}"
        lines.append(f"| {head} | {value['completed_cases']}/{value['expected_cases']} | {rate(value['strict_accepted'])} | {loss_text} | {gradients}（分母{value['completed_cases']}） | {rate(events['probabilities_constant']['final_epoch_case_incidence'])} | {rate(events['probabilities_constant']['post_epoch_incidence'])} | {rate(events['single_class_predictions']['post_epoch_incidence'])} | {cv_text} |")
    lines += ["", "训练/验证事件率、有效梯度范数、案例间离散度与 H1/H2 同学习率配对细节见 `summary.json`；可复制表格见 `head_aggregate.csv`。", "",
              f"覆盖状态：`{aggregate['comparison_status']}`；六案例最终决策就绪：`{aggregate['decision_ready']}`。", "",
              "当前分类头决策（完整覆盖后仍需人工检查原始轨迹）：", "", "```json", json.dumps(aggregate["candidate_recommendation"], ensure_ascii=False, indent=2), "```", "",
              "H3 与 H2 架构相同但学习率不同。若仅 H3 稳定，可以支持 H2 结构与低学习率的组合；不能据此断言未测过的 H1+低学习率不如 H2。", "",
              "下一轮 M2、M2+ 和 Transformer 必须共用 I3、目标 mask、训练拟合长度变换，以及预先声明的共同 head/LR、划分、损失、训练与选择规则，并全部重跑。旧 I0/H0 结果只作历史失败记录，不能加入新协议架构对比表。", "",
              "给定人工候选区间的分类仍不等于自动候选发现，已经查看过的外层结果不构成新的盲测。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(run_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if json_signature({key: value for key, value in manifest.items() if key != "experiment_signature"}) != manifest["experiment_signature"]:
        raise ValueError("Experiment manifest signature mismatch")
    config = manifest["config"]
    current = controlled_source_signature(ROOT)
    if current != manifest["source_signature"]:
        raise ValueError("Controlled source changed since training; restore the exact source before claiming unchanged acceptance")
    metadata = verify_frozen_metadata(manifest)
    inputs, input_neural_records, case_records, endpoint_rows, pending = [], [], [], [], []
    for job in planned_jobs(config, "input"):
        found = read_verified_record(run_dir, job, manifest, metadata)
        if found is None:
            pending.append({"stage": "input", "name": job["name"]})
            continue
        metrics, history = found
        inputs.append(input_review(job, metrics))
        if job["kind"] == "neural":
            input_neural_records.append({"case": "input_reference", "head": job["name"], "metrics": metrics})
    for case in config["head_stage"]["cases"]:
        for job in planned_jobs(config, "heads", case):
            found = read_verified_record(run_dir, job, manifest, metadata)
            if found is None:
                pending.append({"stage": "heads", "case": case, "name": job["name"]})
                continue
            metrics, history = found
            case_records.append({"case": case, "head": job["head"], "status": metrics["status"],
                                 "metrics": metrics, "history": history})
            row = {"case": case, "head": job["head"], "status": metrics["status"],
                   "accepted": metrics.get("acceptance", {}).get("accepted", False),
                   "final_train_ce": metrics.get("final_train_ce"),
                   "failed_checks": metrics.get("acceptance", {}).get("primary", {}).get("failed_checks", [])}
            if metrics["status"] == "completed":
                for split in ("train", "validation"):
                    diag = metrics[f"{split}_diagnostics"]
                    row[f"{split}_prediction_counts"] = diag["prediction_counts"]
                    row[f"{split}_centered_logit_variance"] = diag["centered_logit_between_structure_var"]
                    row[f"{split}_head_effective_fraction"] = diag["head"]["effective_fraction"]
                row["terminal_relative_loss_regression"] = metrics["acceptance"]["primary"]["terminal_relative_loss_regression"]
                row["validation_CHID_f1"] = metrics["validation_diagnostics"]["metrics"]["per_class"]["CHID"]["f1"]
            endpoint_rows.append(row)
    pairing = {**verify_pairing(input_neural_records), **verify_pairing(case_records)}
    aggregate = aggregate_heads(case_records, list(config["head_stage"]["cases"]), config)
    reviews = {(row["case"], row["head"]): row for row in aggregate["case_reviews"]}
    for row in endpoint_rows:
        review = reviews[(row["case"], row["head"])]
        row["history_acceptance_agrees"] = review["saved_acceptance_agrees_with_history"]
        row["failure_categories"] = review["failure_categories"]
        row["last_window_loss_cv"] = review["last_window_loss_cv"]
        for group in ("total", "encoder", "head"):
            row[f"{group}_gradient_effective"] = review["gradient_window"][group]["every_epoch_finite_and_effective"]
            row[f"{group}_minimum_epoch_mean_gradient_norm"] = review["gradient_window"][group]["minimum_epoch_mean_norm"]
        if row["status"] == "completed" and not row["history_acceptance_agrees"]:
            raise ValueError(f"Independent aggregation disagrees with stored acceptance: {row['case']}/{row['head']}")
    reference_fold = json.loads((project_path(config["frozen_run_dir"]) / "folds" / f"fold_{config['input_stage']['fold']}.json").read_text(encoding="utf-8"))
    structures = {row["structure_id"]: {**row, "group_id": group}
                  for row, group in zip(metadata["structures"], metadata["group_id"], strict=True)}
    chid_ids = [key for key in reference_fold["validation_ids"] if structures[key]["type"] == "CHID"]
    conflicting_ids = {key for pair in manifest["target_scope_preflight"]["historical_center_only_conflicts"] for key in pair["structures"]}
    input_scope = {"fold": config["input_stage"]["fold"], "validation_structures": len(reference_fold["validation_ids"]),
                   "CHID_structures": len(chid_ids), "CHID_independent_groups": len({structures[key]["group_id"] for key in chid_ids}),
                   "all_exact_conflicting_structures_in_outer_test": bool(conflicting_ids) and conflicting_ids <= set(reference_fold["test_ids"])}
    report = {"scope": "read-only verification of saved diagnostics and predictions; no training, preprocessing or inference",
              "experiment_signature": manifest["experiment_signature"], "source_signature": current,
              "source_signature_unchanged": True, "frozen_metadata_and_inner_ids_verified": True,
              "run_dir": str(run_dir), "pending": pending,
              "input_review": inputs, "input_scope": input_scope,
              "case_head_review": endpoint_rows, "head_aggregate": aggregate,
              "pairing_checks": pairing,
              "selection_scope": "numeric stability on deliberately chosen failed development cases; not a population performance estimate",
              "formal_comparison_constraint": "Same I3/new target-aware protocol and same chosen head/LR across newly rerun M2, M2+, Transformer; exclude old H0 experiment numbers"}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", report)
    write_json(output_dir / "head_decision.json", aggregate["candidate_recommendation"])
    csv_file(output_dir / "input_review.csv", inputs)
    csv_file(output_dir / "case_head_review.csv", endpoint_rows)
    csv_file(output_dir / "head_aggregate.csv", aggregate_csv_rows(aggregate))
    csv_file(output_dir / "head_pairs.csv", aggregate["H1_H2_same_lr_pairs"])
    write_markdown(output_dir / "report.md", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_controlled_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/task1_controlled_review_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "FROZEN.json").exists():
        parser.error("This report is frozen. Read the existing report, run scripts/freeze_task1_controlled_evidence.py verify, or choose a new --output-dir.")
    for protected in (ROOT / "results", ROOT / "data", ROOT / "evidence", ROOT / "micro-c数据"):
        if output == protected or protected in output.parents:
            raise ValueError("Derived reports must not overwrite experiment evidence or data")
    report = build_report(args.run_dir.resolve(), output)
    print(f"Verified input jobs: {len(report['input_review'])}; head jobs: {len(report['case_head_review'])}; pending jobs: {len(report['pending'])}")
    print(json.dumps(report["head_aggregate"]["candidate_recommendation"], ensure_ascii=False, indent=2))
    print(f"Report: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
