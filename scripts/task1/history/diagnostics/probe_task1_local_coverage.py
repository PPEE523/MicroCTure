#!/usr/bin/env python3
"""Four fixed examples: frozen local coverage versus indexed raw upper tails.

Reads only bin1 owner rows of selected bad100bp cells and five fixed good
controls. It never asks Cooler to reflect a full-chromosome row, which could
silently scan a large CSR prefix. Missing lower-triangle contributions remain
unmeasured; an upper zero is never called a zero full-genome marginal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from microcture.task1.data import load_dataset
from microcture.task1_formal.protocol import json_signature, sha256
from microcture.task1_native_roi_v2.data import native_source_records

CASES = ("CHID_8", "CHID_11", "CHIN_51", "OPCID_66")


def require(value, message):
    if not value:
        raise ValueError(message)


def physical_circular_distance(row, columns, chromosome_length):
    """Distance between actual bin centers, including the terminal partial bin."""
    columns = np.asarray(columns, dtype=np.int64)
    first_start = row * 10
    center = (first_start + min(first_start + 10, chromosome_length)) / 2
    starts = columns * 10
    centers = (starts + np.minimum(starts + 10, chromosome_length)) / 2
    direct = np.abs(centers - center)
    return np.minimum(direct, chromosome_length - direct)


def frozen_coverage(dataset, index, replicate_index):
    bins = dataset["window_footprints"][index]
    n_bins = dataset["n_bins"]
    difference = (bins[None, :] - bins[:, None]) % n_bins
    distance = np.minimum(difference, n_bins - difference)
    anchor = np.where(difference <= n_bins // 2, bins[:, None], bins[None, :])
    count = np.asarray(dataset["bands"][replicate_index][anchor, distance])
    exposure = dataset["bin_occupancy"][bins] > 0
    supported = (distance >= 2) & exposure[:, None] & exposure[None, :]
    local_sum = np.where(supported, count, 0).sum(axis=0)
    return bins, local_sum, (local_sum > 0) & exposure


def continuous_runs(values):
    values = np.sort(np.unique(np.asarray(values, dtype=np.int64)))
    if not len(values): return []
    cut = np.flatnonzero(np.diff(values) != 1) + 1
    return [(int(part[0]), int(part[-1] + 1)) for part in np.split(values, cut)]


def indexed_upper_rows(handle, native_rows, chromosome_length):
    """One physical read per contiguous selected owner range; no reflection."""
    results = {}
    pixel_count = 0
    for first, last in continuous_runs(native_rows):
        offsets = handle["indexes/bin1_offset"][first:last + 1]
        low, high = int(offsets[0]), int(offsets[-1])
        columns = handle["pixels/bin2_id"][low:high].astype(np.int64)
        stored_rows = handle["pixels/bin1_id"][low:high].astype(np.int64)
        counts = handle["pixels/count"][low:high].astype(np.int64)
        expected_rows = np.repeat(np.arange(first, last), np.diff(offsets))
        require(np.array_equal(stored_rows, expected_rows) and np.all(columns >= stored_rows), "Upper CSR row index differs")
        require(np.all(counts >= 0), "Invalid raw counts")
        pixel_count += high - low
        for row in range(first, last):
            a, b = int(offsets[row - first] - low), int(offsets[row - first + 1] - low)
            js, value = columns[a:b], counts[a:b]
            require(np.all(np.diff(js) > 0), "A selected raw row contains duplicate/unsorted columns")
            distance = physical_circular_distance(row, js, chromosome_length)
            near = int(value[distance < 200].sum())
            local = int(value[(distance >= 200) & (distance <= 25600)].sum())
            far = int(value[distance > 25600].sum())
            total = int(value.sum())
            require(near + local + far == total, "Physical distance categories do not partition selected upper row")
            results[row] = {"native_row": row, "start_bp": row * 10,
                "end_bp": min((row + 1) * 10, chromosome_length), "stored_upper_nnz": len(value),
                "upper_count_total": total, "upper_count_lt_200bp": near,
                "upper_count_200_to_25600bp": local, "upper_count_gt_25600bp": far,
                "diagonal_count_once": int(value[js == row].sum()),
                "lower_contribution": "not_measured", "full_genome_row_total": None}
    require(set(results) == set(int(v) for v in native_rows), "Raw owner selection changed")
    return results, {"indexed_owner_runs": continuous_runs(native_rows), "selected_native_rows": len(results),
                     "stored_pixels_read": pixel_count, "reflected_pixels_generated": 0}


def probe(output):
    dataset = load_dataset(ROOT / "data/cache/task1_comparison_v1")
    source_records = native_source_records(dataset)
    workbook = ROOT / "micro-c数据/标注数据.xlsx"
    workbook_sha = sha256(workbook)
    require(workbook_sha == dataset["metadata"]["input_sha256"]["micro-c数据/标注数据.xlsx"], "Original annotation workbook changed")
    folders = {number: json.loads((ROOT / f"results/task1_comparison_v1/folds/fold_{number}.json").read_text()) for number in range(5)}
    lookup = {str(sid): index for index, sid in enumerate(dataset["structure_id"])}
    cases = []
    total_pixels, total_rows, total_bad_cells, positive_bad_cells = 0, 0, 0, 0
    started = time.perf_counter()
    for sid in CASES:
        index = lookup[sid]
        annotation = dataset["metadata"]["structures"][index]
        start, end = int(annotation["start"]), int(annotation["end"])
        fold = next(f for f in folders.values() if sid in f["train_ids"])
        for replicate_index, source in enumerate(source_records):
            bins, local_sum, good = frozen_coverage(dataset, index, replicate_index)
            inside = (bins * 100 < end) & ((bins + 1) * 100 > start)
            bad = np.sort(bins[inside & ~good])
            all_good = np.sort(bins[inside & good])
            chosen = all_good[np.linspace(0, len(all_good) - 1, min(5, len(all_good)), dtype=int)] if len(all_good) else np.array([], dtype=int)
            cells = np.sort(np.r_[bad, chosen])
            native_rows = np.concatenate([np.arange(cell * 10, min((cell + 1) * 10, (dataset["metadata"]["chrom_length"] + 9) // 10)) for cell in cells])
            with h5py.File(ROOT / source["path"], "r") as handle:
                rows, io = indexed_upper_rows(handle, native_rows, dataset["metadata"]["chrom_length"])
            cell_records = []
            source_sum = {int(b): float(v) for b, v in zip(bins, local_sum)}
            for cell in cells:
                selected = [rows[int(r)] for r in range(int(cell * 10), int((cell + 1) * 10)) if int(r) in rows]
                sums = {key: sum(r[key] for r in selected) for key in ("upper_count_total", "upper_count_lt_200bp", "upper_count_200_to_25600bp", "upper_count_gt_25600bp", "diagonal_count_once")}
                category = "frozen_bad" if cell in set(bad) else "fixed_good_control"
                if category == "frozen_bad":
                    total_bad_cells += 1
                    positive_bad_cells += int(sums["upper_count_total"] > 0)
                cell_records.append({"coarse_bin": int(cell), "start_bp": int(cell * 100), "end_bp": int((cell + 1) * 100),
                    "target_overlap_bp": int(max(0, min((cell + 1) * 100, end) - max(cell * 100, start))),
                    "category": category, "frozen_local_256bin_supported_count_sum": source_sum[int(cell)],
                    **sums, "selected_raw_rows_with_positive_upper": sum(r["upper_count_total"] > 0 for r in selected),
                    "raw_global_missing_excluded_by_positive_upper": sums["upper_count_total"] > 0,
                    "upper_zero_does_not_establish_full_row_zero": True, "native_rows": selected})
            total_pixels += io["stored_pixels_read"]
            total_rows += io["selected_native_rows"]
            cases.append({"structure_id": sid, "annotation_class": annotation["type"], "replicate": source["replicate"],
                "start_bp": start, "end_bp": end, "first_available_inner_training_fold": fold["fold"],
                "case_in_inner_train": True, "frozen_fold_sha256": sha256(ROOT / f"results/task1_comparison_v1/folds/fold_{fold['fold']}.json"),
                "roi_coarse_bins": int(inside.sum()), "frozen_bad_coarse_bins": len(bad),
                "good_control_selection": "up to5 evenly spaced indices among physically sorted frozen-good ROI-overlapping100bp cells",
                "cells": cell_records, "reader_io": io})
    result = {"status": "complete_upper_subset_diagnostic", "script_sha256": sha256(Path(__file__)),
        "dataset_signature": dataset["signature"], "annotation_workbook_sha256": workbook_sha, "raw_sources": source_records,
        "fixed_cases": list(CASES), "frozen_coverage_rule": "Old256x256100bp window; sum counts at circular coarse distance>=2; zero sum=>bad, also terminal-bin occupancy mask",
        "native_reader": "Only indexed bin1 upper-storage owner tails; all j>=i columns, no symmetric reflection, no global pixel scan",
        "distance_rule": "actual raw-bin midpoint; min(abs(mid_i-mid_j), chrom_length-abs(mid_i-mid_j)); terminal partial bin retains its real width",
        "chromosome_length_bp": dataset["metadata"]["chrom_length"], "categories": "d<200;200<=d<=25600;d>25600 bp",
        "denominator": "Each cell is sum of its selected bin1 row endpoint counts; diagonal once. Counts are upper-storage subsets, not full symmetric row marginals or global stored-count sums.",
        "limitations": ["Lower endpoint contributions are not measured and never filled with zeros.",
            "Positive upper counts exclude complete raw-genome absence for that row/cell; zero upper counts do not establish full-genome absence.",
            "Upper-only subset size depends on genomic position; these counts are not comparable full-row quality scores across loci.",
            "A frozen local coverage rule is not a measurement of mappability, sequencing failure or biological absence.",
            "Counts from good controls are descriptive, not an independent quality label or model evaluation."],
        "total_selected_native_rows": total_rows, "stored_pixels_read": total_pixels,
        "frozen_bad_cells_across_two_WT": total_bad_cells, "bad_cells_with_positive_upper": positive_bad_cells,
        "runtime_seconds_small_row_probe": time.perf_counter() - started, "cases": cases}
    result["signature"] = json_signature(result)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "local_coverage_diagnostic.json"
    require(not json_path.exists(), "Preserve previous numerical coverage diagnostic; choose a new output directory")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# 冻结局部coverage与原始计数：四例只读诊断", "", "本次只查固定四例候选的第一inner-train fold；读取所有frozen bad100bp cells，并为每例/WT固定抽取最多5个good100bp对照。没有模型训练、推理或标签修改。", "", "## 实际读取范围", "", f"共读取{total_rows}个原生10bp bin1 rows的upper-storage tails、{total_pixels:,}条stored pixels。只用bin1_offset定位这些行，不调用Cooler的全染色体对称反射查询，不扫描完整pixel表。", "", "只测j≥i的原始upper部分；没有测量j<i的lower贡献，因此没有任何完整全基因组row-total。正数可以排除该行完全无原始计数，upper=0不能证明完整row=0。物理圆距用真实bp中心与染色体长度4,641,652计算，末端不足10bp的bin也按真实边界处理。", "", "## 逐cell数值", "", "| 结构/WT/fold | 100bp cell | 类型 | 冻结局部支持count | upper总计 | <200bp | 200–25.6kb | >25.6kb |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        for cell in case["cells"]:
            lines.append(f"| {case['structure_id']}/WT{case['replicate']}/fold{case['first_available_inner_training_fold']} | [{cell['start_bp']},{cell['end_bp']}) | {cell['category']} | {cell['frozen_local_256bin_supported_count_sum']:.0f} | {cell['upper_count_total']} | {cell['upper_count_lt_200bp']} | {cell['upper_count_200_to_25600bp']} | {cell['upper_count_gt_25600bp']} |")
    if positive_bad_cells:
        finding = f"两WT合计{total_bad_cells}个frozen bad cell中，{positive_bad_cells}个存在正upper原始计数；仅这些正计数cell可排除原始全基因组完全无计数。其余upper=0的cell仍缺少lower测量，不能断言全局缺失。"
    else:
        finding = f"两WT合计{total_bad_cells}个frozen bad cell全部upper=0。本次没有正upper的坏cell来排除全局完全无计数，也没有测量lower来证明完整row=0；因此，全局缺失状态仍未判定。"
    good_controls = [cell for case in cases for cell in case["cells"] if cell["category"] == "fixed_good_control"]
    good_finding = f"另列的{len(good_controls)}个good对照均为局部规则保留的cell，其中{sum(cell['upper_count_total']>0 for cell in good_controls)}个存在正upper计数；这些对照不构成坏cell具有正计数的证据。"
    lines += ["", "## 能支持的结论", "", finding, "", good_finding, "", "冻结规则测量的是指定25.6kb窗口内且经过coarse距离屏蔽的支持，不等同原始全局覆盖，更不等同mappability。upper-only截面大小随基因组位置变化，不能跨位置把这些数值当作完整row质量评分。good对照不负责给坏cell重新打质量标签，原mask、标注及模型均保持。每个cell求和的是selected owner rows的upper部分；不再生成对称反射，不能把这些计数当作完整全局stored-count审计。", "", f"原工作簿SHA-256：`{workbook_sha}`。完整每10bp行的计数、输入签名和原始WT SHA见[JSON](local_coverage_diagnostic.json)。", ""]
    (output / "local_coverage_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_autonomous_v1")
    args = parser.parse_args()
    result = probe(args.output)
    print(json.dumps({k: result[k] for k in ("status", "total_selected_native_rows", "stored_pixels_read", "frozen_bad_cells_across_two_WT", "bad_cells_with_positive_upper", "runtime_seconds_small_row_probe")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
