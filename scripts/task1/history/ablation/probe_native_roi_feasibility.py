#!/usr/bin/env python3
"""Small read-only native-10bp ROI feasibility; no full fit or model training.

Only selected INNER-TRAIN patches and a few owner-row calibration tiles are
read. The normalization vector used for kernel timing is explicitly a unit
fixture, not a fitted native background and not scientific model input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from microcture.task1_roi.data import roi_area_weights
from microcture.task1_formal.protocol import sha256


def require(value, message):
    if not value:
        raise ValueError(message)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not a scalar report value: {type(value).__name__}")


def _dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def common_native_roi(counts, starts, interval, frozen_good, expected, output_size=64):
    """Matched C0/C1: identical native field, support, canvas and ROI exposure.

    C0 first takes a valid/exposure mean of native log1p(OE) within actual
    100-bp cells, then broadcasts. C1 preserves the cell-internal detail.
    No dense native N-by-N matrix is constructed. Coverage comes exclusively
    from the frozen coarse footprint, not native nonzero row counts.
    """
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    good = np.asarray(frozen_good, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    n = len(starts)
    start, end = interval
    require(counts.shape == (n, n) and good.shape == (n,) and n > 0, "Shape mismatch")
    require(np.array_equal(np.diff(starts), np.full(n-1, 10)) and np.all((good == 0) | (good == 1)), "10bp bins and frozen binary coverage required")
    require(np.isfinite(counts.data).all() and np.all(counts.data >= 0), "Invalid raw counts")
    asymmetry = counts - counts.T
    require(not asymmetry.nnz or np.max(np.abs(asymmetry.data)) < 1e-12, "Symmetric raw native field required")
    a = sparse.csr_matrix(roi_area_weights(float(start), float(end), starts, resolution_bp=10, output_size=output_size))
    axis_exposure = np.maximum(0., np.minimum(starts + 10, end) - np.maximum(starts, start)) / 10
    target = sparse.diags(axis_exposure)
    weighted_coverage = a @ good
    # Full valid pair support is implicit rank one minus a 39-diagonal band.
    offsets = list(range(-19, 20))
    diagonals = [good[max(0, -d): min(n, n-d)] * good[max(0, d): min(n, n+d)] for d in offsets]
    near_band = sparse.diags(diagonals, offsets, shape=(n, n), format="csr")
    validity = np.outer(weighted_coverage, weighted_coverage) - _dense(a @ near_band @ a.T)
    validity[np.abs(validity) < 1e-12] = 0
    require(np.all((validity >= -1e-10) & (validity <= 1+1e-10)), "Invalid common support")
    validity = np.clip(validity, 0, 1)
    coo = counts.tocoo()
    distances = np.abs(coo.row - coo.col)
    keep = (distances >= 20) & (good[coo.row] > 0) & (good[coo.col] > 0)
    require(len(expected) >= n and np.isfinite(expected[20:n]).all() and np.all(expected[20:n] > 0), "Positive training-only native expectations required")
    logged = np.log1p(coo.data[keep] / expected[distances[keep]])
    weighted_log = sparse.csr_matrix((logged, (coo.row[keep], coo.col[keep])), shape=(n, n))
    fine_total = _dense(a @ weighted_log @ a.T)
    fine = np.divide(fine_total, validity, out=np.zeros_like(fine_total), where=validity > 0)
    coarse_ids = starts // 100
    unique, coarse_index = np.unique(coarse_ids, return_inverse=True)
    b = sparse.csr_matrix((np.ones(n), (np.arange(n), coarse_index)), shape=(n, len(unique)))
    projected_numerator = _dense(b.T @ target @ weighted_log @ target @ b)
    coarse_exposure = np.asarray(b.T @ (good * axis_exposure)).ravel()
    projected_denominator = np.outer(coarse_exposure, coarse_exposure) - _dense(b.T @ target @ near_band @ target @ b)
    projected = np.divide(projected_numerator, projected_denominator, out=np.zeros_like(projected_numerator), where=projected_denominator > 1e-12)
    ag = _dense(a @ sparse.diags(good) @ b)
    exclusion = near_band.tocoo()
    correction = sparse.csr_matrix((exclusion.data * projected[coarse_index[exclusion.row], coarse_index[exclusion.col]],
                                  (exclusion.row, exclusion.col)), shape=(n, n))
    coarse_total = ag @ projected @ ag.T - _dense(a @ correction @ a.T)
    coarse_total[np.abs(coarse_total) < 1e-12] = 0
    require(np.all(coarse_total >= -1e-10), "Projection produced negative weighted log(OE)")
    coarse = np.divide(np.maximum(coarse_total, 0), validity, out=np.zeros_like(coarse_total), where=validity > 0)
    mask = np.outer(np.asarray(a.sum(axis=1)).ravel(), np.asarray(a.sum(axis=1)).ravel())
    mask[np.abs(mask-1) < 1e-10] = 1
    first_c0, first_c1 = np.stack((coarse, validity, mask)).astype(np.float32), np.stack((fine, validity, mask)).astype(np.float32)
    require(np.array_equal(first_c0[1:], first_c1[1:]), "Projection changed validity/geometry")
    return first_c0, first_c1, {"native_bins": n, "coarse_cells": len(unique), "raw_sparse_nnz": counts.nnz,
        "weighted_log_sparse_nnz": weighted_log.nnz, "near_band_nnz": near_band.nnz,
        "common_validity_mean": float(validity.mean()), "target_mask_mean": float(mask.mean()),
        "mean_absolute_fixture_signal_difference": float(np.abs(coarse-fine).mean()),
        "physical_exclusion_bp": 200, "validity_and_mask_byte_identical": True,
        "native_dense_n_by_n_allocated": False,
        "signal_formula": "valid/exposure area mean of native log1p(count/E10(distance)); C0 adds actual100bp-cell valid/exposure projection"}


def frozen_coverage(metadata, bands, index, replicate):
    n_bins = metadata["n_bins"]
    bins = (int(metadata["center_bin"][index]) + np.arange(256) - 128) % n_bins
    difference = (bins[None,:] - bins[:,None]) % n_bins
    distance = np.minimum(difference, n_bins-difference)
    anchor = np.where(difference <= n_bins//2, bins[:,None], bins[None,:])
    counts = np.asarray(bands[replicate][anchor, distance])
    occupied = (bins + 1) * 100 <= metadata["chrom_length"]
    support = (distance >= 2) & occupied[:,None] & occupied[None,:]
    good = np.where(support, counts, 0).sum(axis=0) > 0
    return dict(zip(bins.tolist(), (good & occupied).tolist()))


def train_geometry(metadata, fold):
    index = {r["structure_id"]: i for i,r in enumerate(metadata["structures"])}
    n = metadata["n_bins"]
    occupancy = np.zeros(n, dtype=bool)
    centers = np.asarray(metadata["center_bin"])
    footprints = (centers[[index[sid] for sid in fold["train_ids"]],None] + np.arange(256) - 128) % n
    occupancy[footprints.ravel()] = True
    occupancy[(np.arange(n)+1)*100 > metadata["chrom_length"]] = False
    raw = np.repeat(occupancy, 10)[:(metadata["chrom_length"]+9)//10]
    boundaries = np.flatnonzero(np.diff(np.r_[False, occupancy, False])).reshape(-1,2)
    # Frozen annotations begin >49kb; no supported training pairs wrap at any
    # distance used by these 25.6kb footprints. Make that an explicit guard.
    wrap_possible = bool(raw[:2560].any() and raw[-2560:].any())
    return raw, boundaries, {"training_structures": len(fold["train_ids"]), "unique_training_100bp_bins": int(occupancy.sum()),
        "unique_training_10bp_bins": int(raw.sum()), "disjoint_contiguous_components": len(boundaries),
        "max_component_bp": int(np.max(boundaries[:,1]-boundaries[:,0])*100),
        "owner1000_raw_bin_tiles": int(np.ceil((boundaries[:,1]-boundaries[:,0])/100).sum()),
        "wrap_pairs_may_need_special_reader": wrap_possible}


def probe(root, fold_number=1, case_count=6, tile_count=4):
    import cooler
    metadata = json_read(root / "data/cache/task1_comparison_v1/metadata.json")
    fold = json_read(root / f"results/task1_comparison_v1/folds/fold_{fold_number}.json")
    data_config = json_read(root / "configs/data.json")
    require(data_config["bin_size"] == 10 and data_config["chrom_length"] == metadata["chrom_length"], "Raw metadata changed")
    train = set(fold["train_ids"])
    structures = sorted([r for r in metadata["structures"] if r["structure_id"] in train], key=lambda r: int(r["end"])-int(r["start"]))
    chosen = [structures[int(q*(len(structures)-1))] for q in np.linspace(0,1,case_count)]
    require(len({r["structure_id"] for r in chosen}) == case_count, "Duplicate feasibility cases")
    source_index = {r["structure_id"]: i for i,r in enumerate(metadata["structures"])}
    bands = [np.load(root / "data/cache/task1_comparison_v1" / r["band_file"], mmap_mode="r") for r in metadata["samples"]]
    raw_training, boundaries, geom = train_geometry(metadata, fold)
    require(not geom["wrap_pairs_may_need_special_reader"], "Current tiny reader cannot silently omit circular expected pairs")
    patches, tiles = [], []
    raw_sources = []
    for replicate, sample in enumerate(data_config["samples"]):
        path = root / sample["path"]
        old = metadata["samples"][replicate]
        require(path.stat().st_size == old["raw_size_bytes"] and path.stat().st_mtime_ns == old["raw_mtime_ns"], "Raw file changed")
        clr = cooler.Cooler(str(path))
        require(clr.binsize == 10 and clr.info["storage-mode"] == "symmetric-upper", "10bp symmetric-upper raw source required")
        raw_sources.append({"path": sample["path"], "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                            "bin_size": clr.binsize, "full_genome_pixels_read": False})
        for record in chosen:
            start, end = int(record["start"]), int(record["end"])
            # Full coarse cells are fetched, but the C0 projection's averages
            # use only ROI exposure, so no out-of-target boundary mixing.
            first, last = (start//100)*10, ((end+99)//100)*10
            timer = time.perf_counter()
            counts = clr.matrix(balance=False, sparse=True)[first:last, first:last].tocsr()
            fetch_time = time.perf_counter()-timer
            starts = np.arange(first,last,dtype=np.int64)*10
            coverage = frozen_coverage(metadata,bands,source_index[record["structure_id"]],replicate)
            good = np.asarray([coverage[int(b//100)] for b in starts], dtype=float)
            timer = time.perf_counter()
            c0,c1,facts = common_native_roi(counts,starts,(start,end),good,np.ones(len(starts)))
            kernel_time = time.perf_counter()-timer
            patches.append({"structure_id": record["structure_id"], "replicate": sample["replicate"], "in_frozen_inner_train": True,
                "length_bp": end-start, "fetch_seconds": fetch_time, "sparse_kernel_seconds": kernel_time,
                "normalization": "unit E10 kernel fixture; no native scientific expected curve fitted",
                "new_input_tensor_persisted": False, **facts})
            del counts,c0,c1
        chosen_components = [boundaries[int(q*(len(boundaries)-1))] for q in np.linspace(0,1,tile_count)]
        for comp_start,comp_end in chosen_components:
            first = int(comp_start)*10
            last = min(first+1000,int(comp_end)*10)
            col_first, col_last = first+20, min(last+2560,len(raw_training))
            timer = time.perf_counter()
            counts = clr.matrix(balance=False,sparse=True)[first:last,col_first:col_last].tocoo()
            fetch_time = time.perf_counter()-timer
            global_row,global_col = counts.row+first,counts.col+col_first
            d = global_col-global_row
            keep=(d>=20)&(d<=2559)&raw_training[global_row]&raw_training[global_col]
            sums=np.bincount(d[keep],weights=counts.data[keep],minlength=2560)
            timer=time.perf_counter()
            owner=np.arange(first,last); opportunities=[]
            for distance in range(20,2560):
                partner=owner+distance; inside=partner<len(raw_training)
                opportunities.append(int((raw_training[owner[inside]]&raw_training[partner[inside]]).sum()))
            opportunities_time=time.perf_counter()-timer
            tiles.append({"replicate":sample["replicate"],"owner_start_bin":first,"owner_end_bin":last,
                "column_start_bin":col_first,"column_end_bin":col_last,"matrix_shape":counts.shape,
                "raw_nnz":counts.nnz,"eligible_unique_pair_nnz":int(keep.sum()),
                "eligible_counts_sum":float(sums.sum()),"zero_inclusive_pair_opportunities":sum(opportunities),
                "fetch_seconds":fetch_time,"exact_owner_exposure_seconds":opportunities_time,
                "expected_curve_fitted":False})
            del counts
    geometry={}
    for f in range(5):
        _,_,geometry[str(f)]=train_geometry(metadata,json_read(root/f"results/task1_comparison_v1/folds/fold_{f}.json"))
    roi_read=np.asarray([r["fetch_seconds"] for r in patches]);kernels=np.asarray([r["sparse_kernel_seconds"] for r in patches])
    tile_read=np.asarray([r["fetch_seconds"] for r in tiles]);total_tiles=2*sum(r["owner1000_raw_bin_tiles"] for r in geometry.values())
    n_roi=344*2*5
    return {"status":"tiny_feasibility_complete","fold":fold_number,"tiny_patch_count":len(patches),"tiny_tile_count":len(tiles),
        "full_inner_expected_fit_run":False,"full_dataset_inputs_built":False,"model_training_run":False,
        "raw_sources":raw_sources,"patches":patches,"owner_tiles":tiles,"frozen_training_geometry":geometry,
        "cost_projection": {"roi_fetches_if_each_fold_reads_every_structure_two_WT":n_roi,
            "native_expected_owner1000_tiles_two_WT_all_folds":total_tiles,
            "roi_fetch_seconds_median_tiny":float(np.median(roi_read)),
            "ROI_projection_seconds_median_tiny":float(np.median(kernels)),
            "tile_fetch_seconds_median_tiny":float(np.median(tile_read)),
            "total_ROI_seconds_using_tiny_median":float(n_roi*np.median(roi_read+kernels)),
            "total_calibration_fetch_seconds_using_tiny_median":float(total_tiles*np.median(tile_read)),
            "uncertainty":"Small geometry-selected INNER-TRAIN subset; filesystem cache, varying sparse density, input construction and signatures may change full-run time. No full-run benchmark claimed."},
        "input_output":"Only first-scale 3x64x64 tensors exist temporarily in RAM; no pixel matrix/cache persisted; big-context untouched",
        "normalization_warning":"Unit E10 is used only to time/verify sparse projection algebra, never as production normalization or evidence of classification accuracy",
        "provenance":{"script_sha256":sha256(Path(__file__)),"metadata_sha256":sha256(root/"data/cache/task1_comparison_v1/metadata.json"),
            "frozen_fold_sha256":sha256(root/f"results/task1_comparison_v1/folds/fold_{fold_number}.json")}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",type=int,choices=range(5),default=1)
    parser.add_argument("--case-count",type=int,default=6)
    parser.add_argument("--tile-count",type=int,default=4)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/task1_autonomous_v1/native_resolution_feasibility.json")
    args=parser.parse_args()
    require(1<=args.case_count<=8 and 1<=args.tile_count<=4,"Only a few patches/tiles are authorized for this feasibility entry")
    require(args.output.resolve().is_relative_to((ROOT/"reports").resolve()),"Write only derived feasibility reports")
    with threadpool_limits(limits=4): data=probe(ROOT,args.fold,args.case_count,args.tile_count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    require(not args.output.exists(),"Keep prior feasibility measurements; choose another report filename")
    args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=json_scalar)+"\n",encoding="utf-8")
    print(json.dumps({k:data[k] for k in ("status","tiny_patch_count","tiny_tile_count","cost_projection")},ensure_ascii=False,default=json_scalar))


if __name__=="__main__":
    main()
