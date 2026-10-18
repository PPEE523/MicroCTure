#!/usr/bin/env python3
"""Join recorded anonymous visual observations to saved numerical evidence.

This is an exploratory screening report, not new biological ground truth.
It never refits a transform, runs a model, changes labels, or writes results/.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
from scripts.build_task1_morphology_evidence import verify_case


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def report(review_dir: Path, evidence_dir: Path) -> dict:
    review_dir, evidence_dir = review_dir.resolve(), evidence_dir.resolve()
    if ROOT / "reports" not in review_dir.parents:
        raise ValueError("Review reports must use a separate reports/ subdirectory")
    if any(path.is_symlink() for path in review_dir.rglob("*")):
        raise ValueError("Symbolic-link report artifacts are forbidden")
    audit = read_json(review_dir / "summary.json")
    assignment = read_json(review_dir / "visual_review_assignment.json")
    cases = {row["case_id"]: row for row in csv_rows(review_dir / "cases.csv")}
    priority = {row["case_id"] for row in csv_rows(review_dir / "priority_cases.csv")}
    observations = {}
    sources = [review_dir / "summary.json", review_dir / "cases.csv",
               review_dir / "priority_cases.csv", review_dir / "visual_review_assignment.json",
               evidence_dir / "manifest.json", evidence_dir / "status.json"]
    for reviewer, identifiers in assignment["reviewers"].items():
        path = review_dir / f"visual_review_{reviewer[-1]}.json"
        value = read_json(path)
        if value.get("status") != "complete" or set(value["reviewed_case_ids"]) != set(identifiers):
            raise ValueError(f"Incomplete recorded visual observations: {reviewer}")
        rows = value["observations"]
        if len(rows) != len(identifiers) or {row["case_id"] for row in rows} != set(identifiers):
            raise ValueError(f"Observation coverage changed: {reviewer}")
        for row in rows:
            case_id = row["case_id"]
            if case_id in observations or row.get("image_examined") is not True:
                raise ValueError(f"Duplicate or unexamined visual record: {case_id}")
            observations[case_id] = {**row, "reviewer": reviewer}
        sources.append(path)
    if set(observations) != priority or not priority.issubset(cases):
        raise ValueError("Visual records must cover the entire registered priority set")
    status = read_json(evidence_dir / "status.json")
    if not priority.issubset(status["completed_case_ids"]):
        raise ValueError("Priority evidence generation is incomplete")
    joined = []
    for case_id in sorted(priority):
        source, observation = cases[case_id], observations[case_id]
        directory = evidence_dir / "blind" / case_id
        evidence = verify_case(directory, case_id)
        if evidence["exact_model_signal_and_mask_verified"] is not True:
            raise ValueError(f"Evidence is not the actual saved input protocol: {case_id}")
        fine = [row for row in evidence["measurements"] if row["width"] == 64]
        coarse = [row for row in evidence["measurements"] if row["width"] == 256]
        if len(fine) != 2 or len(coarse) != 2:
            raise ValueError(f"Missing WT/scale views: {case_id}")
        joined.append({
            "case_id": case_id, "structure_id": source["structure_id"],
            "annotation_type": source["annotation_type"], "group_id": int(source["group_id"]),
            "fold": int(source["fold"]), "length_bp": int(source["length_bp"]),
            "m2plus_stable_wrong": bool(int(source["m2plus_stable_wrong"])),
            "m2plus_correct_seed_count": int(source["m2plus_correct_seed_count"]),
            "fine_target_coverage_fraction": min(row["interval_coverage_fraction"] for row in fine),
            "coarse_target_coverage_fraction": min(row["interval_coverage_fraction"] for row in coarse),
            "fine_target_valid_fraction_WT1_WT2": [row["target_contact_valid_fraction"] for row in fine],
            "coarse_target_valid_fraction_WT1_WT2": [row["target_contact_valid_fraction"] for row in coarse],
            "display_saturation": [{"replicate": row["replicate"], "width": row["width"],
                                    "fractions": row["display_saturation_fraction"]} for row in evidence["measurements"]],
            "image": str((directory / "contact_evidence.png").relative_to(ROOT)),
            "geometry_image": str((directory / "geometry_and_validity.png").relative_to(ROOT)),
            "visual_observation": observation,
            "biological_label_changed": False,
        })
        sources.extend((directory / "evidence.json", directory / "ARTIFACTS.json"))
    by_id = {row["structure_id"]: row for row in joined}
    critical_ids = ["CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19",
                    "OPCID_66", "CHIN_51", "CHIN_73"]
    if not set(critical_ids).issubset(by_id):
        raise ValueError("Required registered systematic-error cases were not reviewed")
    counts = {name: dict(Counter(str(row["visual_observation"].get(name, "unreported")) for row in joined))
              for name in ("replicate_agreement", "pooling_loss", "confidence", "unique_class_interpretation")}
    run_manifest_path = Path(audit["source_run"]) / "manifest.json"
    manifest = read_json(run_manifest_path)
    sources.append(run_manifest_path)
    by_structure = {row["structure_id"]: row for row in cases.values()}
    training_support = []
    for fold, split in sorted(manifest["fold_ids"].items(), key=lambda item: int(item[0])):
        chid = [by_structure[sid] for sid in split["train_ids"] if by_structure[sid]["annotation_type"] == "CHID"]
        training_support.append({"fold": int(fold), "inner_training_CHID": len(chid),
                                 "inner_training_short_CHID": sum(int(row["length_bp"]) <= 6400 for row in chid),
                                 "min_training_CHID_length_bp": min(int(row["length_bp"]) for row in chid)})
    probe_path = review_dir / "shape_probe_summary.json"
    shape_probe = read_json(probe_path) if probe_path.is_file() else None
    if shape_probe is not None:
        sources.append(probe_path)
    payload = {
        "protocol": "task1_exploratory_morphology_reevaluation_v1",
        "status": "priority_visual_screening_complete",
        "n_registered": len(cases), "n_visually_screened": len(joined),
        "n_not_visually_screened": len(cases) - len(joined),
        "scope": "Literature-constrained AI visual screening on previously inspected data; not independent expert ground truth or a new blind evaluation",
        "class_support_screened": dict(Counter(row["annotation_type"] for row in joined)),
        "visual_record_counts": counts, "numerical_audit_statistics": audit["statistics"],
        "frozen_inner_training_CHID_support": training_support,
        "systematic_error_cases": [by_id[sid] for sid in critical_ids],
        "all_screened_cases": joined,
        "shape_sensitivity_probe": shape_probe,
        "labels_and_model_results_unchanged": True, "new_training_run": False,
        "fixed_weight_sensitivity_inference_run": shape_probe is not None,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "report_builder_sha256": sha256(Path(__file__)),
    }
    lines = ["# 任务一：形态证据与系统性错误的探索性复评", "",
             f"2026-10-13，按用户要求由助手直接开展。本轮数值审计覆盖全部{len(cases)}个唯一标注结构；"
             f"首批{len(joined)}例局部证据已生成并逐图筛查，其中包括全部26个CHID、三个稳定CHID误报与匹配参照。"
             f"其余{len(cases)-len(joined)}例仍未逐图审查。原标注、权重和预测未修改，未训练新模型。", "",
             "这是一轮依据原论文的AI图像观察与数值核验。原类别与预测在匿名观察记录完成后合并。"
             "三个并行助手各审不同病例，不能把分工当作三位专家的独立一致性检验；主助手已知历史错误，整个流程也不是全新盲测。"
             "可见图形相容性不升级为新的生物学真值，无法唯一判断的病例保留不确定。", "",
             "## 已确认与仍未确认的证据", "",
             "课程标注Excel与作者官方补充表逐字节一致，标注版本来源已经核实；坐标端点约定仍为provisional。"
             "[来源核验](../task1_morphology_annotation_source.json)、[原论文定义与审查准则](../../docs/TASK1_MORPHOLOGY_CRITERIA.md)。", "",
             "两WT分开展示，保留100bp计数→100bp log1p(OE)→大尺度4×4加权压缩→实际64像素标准化输入，"
             "另展示mask/validity。色标在全部病例中固定，显示饱和比例单列；它不是模型输入截断。"
             "100bp缓存不能用来证明原10bp细节完全保真。本轮没有以Grad-CAM替代形态证据。", "",
             f"匿名记录中，两WT主组织相近为{counts['replicate_agreement'].get('present', 0)}/{len(joined)}例；"
             f"池化后可见细节平滑为{counts['pooling_loss'].get('present', 0)}/{len(joined)}例。"
             "这种平滑也出现在预测正确的参照中，不能据此直接认定它导致漏检。"
             "视觉记录没有产生可替换作者标签的唯一三分类真值。", "",
             "## 系统性错误与几何关联", "",
             "M2+三个种子都预测正确297例、都预测错误20例；稳定CHID漏检6例。稳定CHID误报为OPCID_66、CHIN_51、CHIN_73。"
             "85例有同seed跨骨干分歧，这些均按唯一结构计数。", "",
             "六个稳定CHID漏检集中于fold0/4各3例。10个≤6.4kb的CHID中5个稳定漏检，16个更长CHID中1个稳定漏检；"
             "但fold1/2也含短CHID而没有稳定漏检，不能据此只归因长度。全部CHID在25.6kb视野内目标覆盖完整；"
             "六漏检中仅CHID_11在6.4kb视野被截断（覆盖66.3%），其余五例细视野目标完整。"
             "边界覆盖完整不等于信号足够或形态可辨，目标外上下文也可能不足。", "",
             "五折inner训练均有16–17个CHID，均包含短CHID。fold0训练有7个≤6.4kb的CHID，fold4有6个，"
             "所以也不能把稳定错误直接归因于这些fold完全没见过短CHID。空间组间的形态变化、标注层级和少量独立结构仍需结合图证据判断。", "",
             "## 九个稳定错误的逐例图像记录", "",
             "下表为匿名图像观察完成后的揭盲对照；不是自动重标注。点击ID可看双WT、双尺度与预处理对照。", "",
             "| 原标注ID | 长度kb / fold | 两WT组织 | 压缩后可见细节变化 | 图像观察 |",
             "| --- | --- | --- | --- | --- |"]
    for sid in critical_ids:
        row = by_id[sid]
        obs = row["visual_observation"]
        description = str(obs.get("morphology_description", "未记录")).replace("|", "/").replace("\n", " ")
        link = "../../" + row["image"]
        lines.append(f"| [{sid}]({link}) | {row['length_bp']/1000:.2f} / {row['fold']} | "
                     f"{obs.get('replicate_agreement', 'unreported')} | {obs.get('pooling_loss', 'unreported')} | {description} |")
    lines.extend(["", "全部匿名观察、质量限制、置信度和揭盲对应保存在[visual_observations.csv](visual_observations.csv)与"
                  "[reevaluation.json](reevaluation.json)。present/absent/unclear是筛查用的定性记录，不是经专家验证的诊断阈值。"
                  "压缩必然平滑像素，只有发现可重复且与判别有关的信息损失，才能支持以增加分辨率为优化依据。", ""])
    if shape_probe is not None:
        lines.extend(["## 已保存M2+的形态敏感性复评", "",
                      "15个fold/seed端点的逐WT概率、双WT均值和预测类别均精确复现原保存结果（最大概率差0）。"
                      "随后在全部344结构上，用三个预定置换repeat比较目标区与背景区，三个endpoint seed分别保留完整OOF。"
                      "本轮固定权重推理约12秒，没有训练。", "",
                      "只重排signal，保留mask、length、validity及各区域内每个64像素网格偏移层的信号值多重集；"
                      "上下三角同步、选择像素数匹配，实际改变数可以不同。该控制并非保留原10/100bp的完整距离分布。"
                      "全部344结构在每个seed/repeat都有实际变化的ROI与背景；某个WT/尺度不可匹配时仍明确记录，不能将它单独作为零敏感性证据。", "",
                      "| 固定权重输入 | macro-F1 | CHID F1 |",
                      "| --- | ---: | ---: |"])
        for condition, label in (("baseline", "原输入"), ("roi", "目标区空间置换"), ("background", "匹配背景空间置换")):
            result = shape_probe["conditions"][condition]
            lines.append(f"| {label} | {result['macro_f1']['mean']:.3f} | {result['CHID_f1']['mean']:.3f} |")
        lines.extend(["", "先对每个endpoint seed内的三个置换repeat求均值，再汇总三个seed；不把repeat、WT或seed当成新的生物学样本。", "",
                      "![形态扰动复评](shape_probe.png)", "", "[矢量PDF](shape_probe.pdf)", "",
                      "目标置换平均使CHID F1降低约4.49个百分点，背景置换约2.86个百分点，说明空间排列对部分预测有作用。"
                      "但macro-F1上目标/背景的强弱在不同seed并不完全一致，六个持续漏检CHID在所有seed/repeat及两种扰动下仍为CHIN，"
                      "三个稳定误报也保持原误类。不能据此宣称目标形态已被可靠识别，或认为扩大分辨率一定能修复这些错误。", "",
                      "该置换保留目标轮廓、区域强度分布和部分粗尺度组织，因此低敏感性也不证明模型完全不使用形态。"
                      "它属于分布外敏感性诊断，不是生物学因果实验；完整数值与匹配支持位于"
                      "`results/task1_morphology_probe_v1/`。", ""])
    lines.extend(["## 下一步如何优化与复评", "",
                  "当前图像记录与已经完成的目标/背景匹配敏感性控制，支持把目标特征读出作为下一项待检验假说，"
                  "尚未证明系统性错误就是背景干扰，也未证明增加分辨率必然修复。"
                  "不能仅凭当前错误表现同时改模型、LR、损失和数据，也不能用助手的形态猜测替换作者标签。", "",
                  "M2+优化的O0应是同协议原版参照。优化后统一重跑PCA+LR、M2、原版M2+、优化版M2+与Transformer，"
                  "沿用相同数据版本、输入、空间分组、种子、拟合隔离和权重选择规则。"
                  "用户没有新样本，因此最终结果仍注明开发性复评；已看过的区域换折不会变成独立盲测。", "",
                  "本轮优先准备参数量一致的目标空间读出控制：O0原M2+与O1固定beta=0.5的目标/全局混合门控，"
                  "不改H3、学习率、损失或训练轮数；I3/G0同时比较以检查新增几何捷径。"
                  "该版本尚未训练或采纳为正式模型，当前运行交付见[任务一说明](../../docs/TASK1.md)。", "",
                  "完整结果与后续规则见[任务一说明](../../docs/TASK1.md)。本轮尚未证明三类生物学形态可靠识别，"
                  "也尚未实施新的模型优化训练。", ""])
    (review_dir / "reevaluation.md").write_text("\n".join(lines), encoding="utf-8")
    (review_dir / "reevaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table = [{**{key: row[key] for key in ("case_id", "structure_id", "annotation_type", "fold", "group_id", "length_bp",
                                          "m2plus_correct_seed_count", "image")}, **row["visual_observation"]} for row in joined]
    fields = list(dict.fromkeys(key for row in table for key in row))
    with (review_dir / "visual_observations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in table)
    return {"status": payload["status"], "registered": len(cases), "visually_screened": len(joined),
            "report": str(review_dir / "reevaluation.md"), "biological_labels_changed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", default="reports/task1_morphology_review_v1")
    parser.add_argument("--evidence-dir", default="reports/task1_morphology_evidence_v1")
    args = parser.parse_args()
    print(json.dumps(report(project_path(args.review_dir), project_path(args.evidence_dir)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
