#!/usr/bin/env python3
"""Assemble the independently audited O0/O1 results without training/inference."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path) -> dict:
    if ROOT / "reports" not in output.resolve().parents:
        raise ValueError("Use a separate reports/ directory")
    audit = read(output / "integrity.json")
    errors = read(output / "error_learning.json")
    figures = read(output / "figures_summary.json")
    source_run = ROOT / "results/task1_readout_v1"
    manifest = read(source_run / "manifest.json")
    if audit["status"] != "passed" or figures["status"] != "complete":
        raise ValueError("Independent audit and figures must be complete")
    if digest(source_run / "manifest.json") != audit["manifest_sha256"]:
        raise ValueError("Audited run manifest changed")
    for record in (audit, figures):
        for key in ("experiment_signature", "source_signature", "dataset_signature"):
            if record[key] != manifest[key]:
                raise ValueError(f"Different {key} in the independent result records")
    with (output / "case_changes.csv").open(encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))
    cases_i3 = {row["structure_id"]: row for row in cases if row["condition"] == "I3"}
    if len(cases_i3) != 344 or len(errors["seed_summaries"]) != 12:
        raise ValueError("The complete structure set and twelve complete OOF are required")
    comparisons = errors["performance_comparison_recomputed_from_confusion_counts"]
    if len(comparisons) != 4:
        raise ValueError("Four complete condition/variant groups are required")
    historical = errors["O0_historical_formal_replication"]
    if not all(historical[key] for key in ("all_outer_predictions_fields_exact",
                                           "all_validation_predictions_fields_exact",
                                           "all_history_numerics_exact_except_wall_time")):
        raise ValueError("The O0 reference did not reproduce the frozen arithmetic")
    stable_counts = {version: sum(row[f"{version}_stable_wrong"] == "True" for row in cases_i3.values())
                     for version in ("O0", "O1")}
    new_stable = [sid for sid, row in cases_i3.items() if row["new_stable_wrong"] == "True"]
    restored = [sid for sid, row in cases_i3.items() if row["old_stable_wrong_fixed_all_three_seeds"] == "True"]
    critical = ("CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19",
                "OPCID_66", "CHIN_51", "CHIN_73")
    i3_deltas = audit["paired_complete_oof_deltas"]["I3:O1-O0"]
    payload = {"status": "complete", "source_run": str(source_run.relative_to(ROOT)),
               "experiment_signature": manifest["experiment_signature"],
               "source_signature": manifest["source_signature"],
               "decision": {"adopt_O1": False, "retained_architecture": "original M2+ / H3",
                            "reason": "All three complete I3 OOF seeds lose macro-F1 and CHID F1; recall mean is unchanged and false positives increase.",
                            "next_changed_factor": "checkpoint selection only; full 40-epoch budget, paired final40 and inner-validation-CE endpoints",
                            "next_training_executed": False},
               "stable_wrong_counts": stable_counts, "new_stable_wrong_ids": new_stable,
               "old_stable_wrong_fixed_all_seeds": restored,
               "comparison": comparisons, "paired_complete_oof_deltas": i3_deltas,
               "critical_case_changes": [cases_i3[sid] for sid in critical],
               "limits": ["Previously inspected developmental data, not independent biological validation.",
                          "Rejects this fixed beta=.5 feature gate; does not invalidate Target-aware input or every ROI architecture.",
                          "Minimum inner-validation CE epochs in old histories are diagnostic only; only final weights were saved."],
               "sources_sha256": {str((output / name).relative_to(ROOT)): digest(output / name)
                                  for name in ("integrity.json", "error_learning.json", "figures_summary.json", "case_changes.csv")},
               "report_builder_sha256": digest(Path(__file__))}
    lines = ["# M2+目标空间读出控制：结果、错误与下一轮决策", "",
             "2026-10-13。用户完成60次训练后，只读核验全部结果并生成四组PNG/PDF。结论：**不采用本轮O1固定目标加权，保留原M2+骨干与H3**。这不是分类头再次失活：60/60训练均通过持续学习，但O1的完整I3评价在三个seed上均变差。", "",
             "## 实验完整性与比较口径", "",
             "60/60训练完成，0缺失/损坏/数值或执行失败。600个科学文件共68,000,412字节的SHA-256一致；60条epoch0–40轨迹、诊断及验收复算相符，15组fold/seed四路初始化、minibatch、预处理和类别权重完全配对。12份完整OOF均包含344个唯一结构；双WT先平均概率，seed不是新的生物学样本。O0的30个I3/G0训练精确复现旧正式结果的预测和数值轨迹（耗时除外）。", "",
             "完整性依据见[integrity.json](integrity.json)。端点先于outer的证据来自封存标记与已签名源码调用顺序，未将文件时间戳当作独立执行证明。下面均值±样本SD来自三个完整OOF，不是五折分数平均或置信区间。", "",
             "## 性能与几何控制", "",
             "| 输入/版本 | macro-F1 | CHID precision | CHID recall | CHID F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        def value(name: str) -> str:
            return f"{row[name+'_mean']:.4f}±{row[name+'_seed_sample_std']:.4f}"
        lines.append(f"| {row['condition']}/{row['variant']} | {value('macro_F1')} | {value('CHID_precision')} | {value('CHID_recall')} | {value('CHID_F1')} |")
    lines.extend(["", "![性能](performance.png)", "", "[性能PDF](performance.pdf)", "",
                  "I3 O1−O0的完整OOF macro-F1平均下降2.22个百分点，CHID F1下降5.58个百分点。三个seed两项均下降；CHID recall的均值相同，却不是每个seed的召回都相同。G0的macro-F1也下降2.22个百分点，CHID F1约下降0.18个百分点。这没有支持当前固定目标权重改善形态识别；I3−G0仍是接触signal+validity通道的整体增量，不能称生物学因果效应。", "",
                  "![同seed完整OOF配对差值](paired_effects.png)", "", "[配对差值PDF](paired_effects.pdf)", "",
                  "## 最终预测分布与系统性错误", "",
                  "| I3版本/seed | 预测OPCID / CHIN / CHID | CHID TP / FP / FN | CHIN→CHID |",
                  "| --- | --- | --- | ---: |"])
    for row in errors["seed_summaries"]:
        if row["condition"] != "I3":
            continue
        counts = row["predicted_class_counts"]
        # Records use named class counts; keep the declared class order explicit.
        distribution = " / ".join(str(counts[name]) for name in errors["class_order"])
        lines.append(f"| {row['variant']}/{row['seed']} | {distribution} | {row['CHID_TP']} / {row['CHID_FP']} / {row['CHID_FN']} | {row['CHIN_to_CHID']} |")
    lines.extend(["", "O0→O1的CHID误报数在seed42/43/44分别为13→19、16→24、9→12；其中CHIN→CHID分别10→17、11→19、5→10。CHID TP分别18→16、17→19、16→16，均值没有增益。", "",
                  f"三seed稳定全错由{stable_counts['O0']}增到{stable_counts['O1']}例：新增{len(new_stable)}例，旧稳定错误只有{len(restored)}例实现全seed修复（{', '.join(restored)}）。总体预测一致率从90.41%升至91.28%，一致错误增加，不能把更低seed分数SD或更高预测一致率解读为更可靠。", "",
                  "| 事先登记案例 | O0 seeds42/43/44 | O1 seeds42/43/44 | O1正确种子数 |",
                  "| --- | --- | --- | ---: |"])
    for sid in critical:
        row = cases_i3[sid]
        lines.append(f"| {sid} | {row['O0_predictions_seed_42_43_44']} | {row['O1_predictions_seed_42_43_44']} | {row['O1_correct_seeds']}/3 |")
    lines.extend(["", "CHID_12只在seed43修复，其他五个旧CHID漏检仍全预测CHIN。CHIN_51/73仍全误报CHID；OPCID_66在两个seed由CHID改为CHIN，依旧判错，不能算修复。CHID_2与CHID_21成为新的全seed CHIN漏检。全部344例×两输入的变动保存在[case_changes.csv](case_changes.csv)，新增稳定错误为：" + ", ".join(new_stable) + "。", "",
                  "![最终混淆、预测分布与26个CHID](predictions_and_errors.png)", "", "[预测分布PDF](predictions_and_errors.pdf)", "",
                  "## 学习与过拟合证据", "",
                  "四组严格学习通过数均15/15；所有训练轮encoder/head/total梯度均有效，末10轮及末轮无常数概率/logits、单类预测或分类头失活，outer末轮也无单类预测。G0/O0的epochs1–3仍有训练18、验证19次单类argmax事件；它们是早期预测集中，不能写成全程没有单类事件，也没有发展为末轮退化。", "",
                  "| 输入/版本 | 末轮train CE mean±fit SD | 末10轮最小head梯度 | 内层val CE上升的fit（20→40） |",
                  "| --- | ---: | ---: | ---: |"])
    for row in errors["trajectory_summaries"]:
        ce = row["train_ce_40"]
        lines.append(f"| {row['condition']}/{row['variant']} | {ce['mean']:.5f}±{ce['sample_std']:.5f} | {row['head_gradient_mean_tail_min']['min']:.4f} | {row['inner_validation_ce_increased_20_to_40_jobs']}/15 |")
    lines.extend(["", "I3/O0的固定train CE中位数20→40为0.10857→0.03094，inner-val CE为0.29713→0.39682，12/15次验证回升；O1则14/15回升。原正式Transformer也有11/15验证回升，但M2只有1/15，中位CE继续改善，不能把三骨干统称都普遍过拟合。完整数值见[error_learning.json](error_learning.json)。", "",
                  "![学习轨迹与数值稳定性](learning.png)", "", "[学习PDF](learning.pdf)", "",
                  "## 后续统一复评的唯一新因素", "",
                  "拒绝当前beta=0.5的O1门控，不在这批结果上临时搜索beta。保留I3、H3/lr1e-4、原三个骨干、标注和空间分组。下一轮仅比较权重选择：仍训练完整40轮，保留final40，同时预登记epochs1–40最低weighted inner-validation CE的inner_ce权重，严格更低才替换、同值取最早，epoch0排除。三个神经骨干使用同一选择规则；PCA/几何LR保持原固定训练配方。", "",
                  "同一个fit的两个权重在读取outer图之前都确定，末轮对照和新规则都重新评价；不early-stop、refit或用外层F1挑轮次。旧轨迹的最小CE轮只能提示下一轮方案，因为旧目录只保留末轮权重，不能事后宣称早期权重的outer分数。", "",
                  "新统一复评的协议与本地命令见[任务一说明](../../docs/TASK1.md)。该轮代码准备与检查独立进行，真实训练仍由用户本地运行；本报告没有启动训练。没有新样本和专家标签复核，所有结果仍属开发性复评，不能承诺checkpoint选择会提高F1或解决CHIN/CHID语义歧义。", ""])
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "complete", "adopt_O1": False, "report": str(output / "analysis.md")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_readout_review_v1")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
