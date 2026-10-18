"""Native 10bp field with matched coarse-cell projection / fine-detail control.

Both branches share inner-training-only E10, frozen100bp coverage, physical
200bp pixel-distance support, ROI64 geometry, original context and length.
Only small expected curves are cached persistently; no image cache is written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import cooler
import numpy as np
from scipy import sparse
from scipy.fft import rfft, irfft

from microcture.task1 import data as original
from microcture.task1_controlled import data as controlled
from microcture.task1_formal import data as formal
from microcture.task1_formal.protocol import json_signature, sha256
from microcture.task1_roi.data import roi_area_weights

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "task1_native10_matched_cell_projection_v1"
ALIGNMENTS = ("C0", "C1")
MIN_DISTANCE_BINS, MAX_DISTANCE_BINS, OWNER_ROWS = 20, 2559, 1000
FROZEN_RAW_SHA256 = {
    1: "0256e10da36abfe0238bf16952bc43bc78365f43c201e6e3ba0c49bd243549a8",
    2: "ede9b4f7c0c6b40c8f098e43cdd0abbc0a8987f05e8b37051a5e3cbf143ad106",
}
_SOURCE_CACHE, _EXPECTED_CACHE = {}, {}
_INNER_MEMO = None


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def data_source_signature():
    return sha256(Path(__file__))


def _safe_cache_root(cache_dir=None):
    path = Path(cache_dir) if cache_dir is not None else ROOT / "data/task1_native_roi_v1/expected"
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Native normalizer cache symlinks forbidden")
    require(path.resolve() != (ROOT / "data/cache/task1_comparison_v1").resolve()
            and not path.resolve().is_relative_to((ROOT / "data/cache/task1_comparison_v1").resolve()),
            "Cannot write native normalizers inside frozen input cache")
    return path


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_new_json(path, value):
    """Atomic no-overwrite publication; only this helper's temporary is removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.stat().st_dev == Path("/tmp").stat().st_dev,
            "Native cache and /tmp must share a filesystem for atomic no-overwrite publication")
    fd, temporary = tempfile.mkstemp(prefix="microcture-native-expected-", suffix=".json", dir="/tmp")
    published = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_scalar)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        # Same-directory hard link publishes fully written bytes atomically,
        # fails if another process already published this scientific record.
        os.link(temporary, path)
        published = True
    finally:
        if published and Path(temporary).exists():
            Path(temporary).unlink()
        elif Path(temporary).exists():
            print(f"Native scientific JSON publication failed; evidence retained at {temporary}",flush=True)


def native_source_records(dataset):
    """Bind cooler bytes once per process; cache invalidates on stat changes."""
    records = []
    for sample in dataset["metadata"]["samples"]:
        path = ROOT / sample["path"]
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "Raw native source missing/symlink")
        stat = path.stat()
        require(stat.st_size == sample["raw_size_bytes"] and stat.st_mtime_ns == sample["raw_mtime_ns"], "Frozen raw source stat changed")
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in _SOURCE_CACHE:
            clr = cooler.Cooler(str(path))
            require(clr.binsize == 10 and clr.info["storage-mode"] == "symmetric-upper", "Native source must be10bp symmetric-upper")
            require(clr.shape[0] == (dataset["metadata"]["chrom_length"]+9)//10,
                    "Raw source bin count differs from chromosome")
            digest = sha256(path)
            require(digest == FROZEN_RAW_SHA256[sample["replicate"]],
                    "Native source bytes differ from retained original WT cooler SHA256")
            _SOURCE_CACHE[key] = {"path": sample["path"], "sample_id": sample["sample_id"], "replicate": sample["replicate"],
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest,
                "bin_size": 10, "storage_mode": "symmetric-upper"}
        records.append(dict(_SOURCE_CACHE[key]))
    require(len(records) == 2 and {r["replicate"] for r in records} == {1,2}, "Exactly two WT sources required")
    return records


def training_native_mask(dataset, frozen_fold):
    """Lift the frozen unique100bp inner-footprint union, keeping terminal mask."""
    require(dataset["config"]["resolution_bp"] == 100 and dataset["config"]["window_bins"] == [64,256]
            and dataset["config"]["output_size"] == 64, "Native extension requires frozen100bp I3 geometry")
    train = original._indices(dataset, frozen_fold["train_ids"])
    validation = original._indices(dataset, frozen_fold["validation_ids"])
    outer = original._indices(dataset, frozen_fold["test_ids"])
    original._assert_disjoint(dataset, train, validation)
    original._assert_disjoint(dataset, train, outer)
    occupancy = original._occupancy(dataset, train) & (dataset["bin_occupancy"] > 0)
    n_raw = (dataset["metadata"]["chrom_length"]+9)//10
    lifted = np.repeat(occupancy,10)[:n_raw]
    boundaries = np.flatnonzero(np.diff(np.r_[False,occupancy,False])).reshape(-1,2)*10
    require(boundaries.size and np.all(boundaries[:,1] <= n_raw), "Invalid native owner components")
    return lifted, boundaries


def fft_pair_opportunities(mask, max_distance=MAX_DISTANCE_BINS):
    """Circular zero-inclusive unique-pair opportunities; integer round guard."""
    mask = np.asarray(mask,dtype=bool)
    require(mask.ndim == 1 and 0 < max_distance < len(mask)//2, "Native distance range must be shorter than half chromosome")
    spectrum = rfft(mask.astype(np.float64),workers=1)
    floating = irfft(spectrum*spectrum.conj(),n=len(mask),workers=1)[:max_distance+1]
    rounded = np.rint(floating)
    require(np.max(np.abs(floating-rounded)) <= 1e-5 and np.all(rounded >= 0), "FFT exposure rounding failed")
    require(int(rounded[0]) == int(mask.sum()), "FFT origin does not equal occupied bins")
    return rounded.astype(np.int64)


def circular_wrap_pair_guard(mask, max_distance=MAX_DISTANCE_BINS):
    """Current candidate footprints have no eligible short cross-origin pairs.

    Raw terminal partial bins produce a sub10bp circular phase. We explicitly
    reject new datasets needing this reader rather than silently approximate
    physical distance or omit remote upper-storage columns.
    """
    mask = np.asarray(mask,dtype=bool)
    if not mask[:max_distance].any() or not mask[-max_distance:].any():
        return False
    for d in range(MIN_DISTANCE_BINS,max_distance+1):
        if np.any(mask[:d] & mask[-d:]):
            return True
    return False


def owned_native_counts(clr, training_mask, boundaries, max_distance=MAX_DISTANCE_BINS, owner_rows=OWNER_ROWS):
    """Count each nonwrapped symmetric-upper pair once, filtering both endpoints."""
    require(not circular_wrap_pair_guard(training_mask,max_distance),
            "Eligible circular native pairs need exact physical remote-column reader; current protocol refuses approximation")
    sums = np.zeros(max_distance+1,dtype=np.float64)
    facts = {"tiles":0,"raw_fetched_nnz":0,"eligible_unique_pair_nnz":0,"eligible_counts_sum":0.,
             "owner_rows":owner_rows,"orientation":"global_row < global_column; each row owned once",
             "circular_eligible_pairs":0,"bounding_rectangles_filtered_to_both_training_endpoints":True}
    for component_start,component_end in boundaries:
        for first in range(int(component_start),int(component_end),owner_rows):
            last = min(first+owner_rows,int(component_end))
            col_first,col_last = first+MIN_DISTANCE_BINS,min(last+max_distance,len(training_mask))
            if col_first >= col_last:
                continue
            matrix = clr.matrix(balance=False,sparse=True)[first:last,col_first:col_last].tocoo()
            i,j = matrix.row+first,matrix.col+col_first
            distance = j-i
            keep=(distance>=MIN_DISTANCE_BINS)&(distance<=max_distance)&training_mask[i]&training_mask[j]
            require(np.isfinite(matrix.data).all() and np.all(matrix.data >= 0), "Invalid native count tile")
            np.add.at(sums,distance[keep],matrix.data[keep])
            facts["tiles"] += 1; facts["raw_fetched_nnz"] += int(matrix.nnz)
            facts["eligible_unique_pair_nnz"] += int(keep.sum())
            facts["eligible_counts_sum"] += float(matrix.data[keep].sum(dtype=np.float64))
    require(abs(sums.sum()-facts["eligible_counts_sum"]) < 1e-6, "Owner native count totals differ")
    return sums,facts


def _expected_binding(dataset,frozen_fold,sources):
    return {"protocol":PROTOCOL,"algorithm":"inner_native_union_FFT_owned_upper_v1", "data_source_signature":data_source_signature(),
        "dataset_signature":dataset["signature"],"frozen_fold_signature":json_signature(frozen_fold),
        "train_ids":frozen_fold["train_ids"],"validation_ids":frozen_fold["validation_ids"],"outer_ids":frozen_fold["test_ids"],
        "raw_sources":sources,"min_distance_bins":MIN_DISTANCE_BINS,"max_distance_bins":MAX_DISTANCE_BINS,
        "raw_resolution_bp":10,"coarse_terminal_bin_fully_masked":True,
        "fit_scope":"Only unique raw pairs with both endpoints in frozen inner-training footprint union; separately per WT"}


def _verify_expected(record,binding):
    require(record["status"] == "complete" and record["binding"] == binding, "Native normalizer fit/source/scope changed or failed")
    require(record["signature"] == json_signature({k:v for k,v in record.items() if k != "signature"}), "Native expected numeric signature differs")
    require(len(record["records"]) == 2, "Two shared native expected records required")
    for one in record["records"]:
        counts,opportunities,expected=(np.asarray(one[k],dtype=np.float64) for k in
            ("stored_count_sum_by_distance","zero_inclusive_pair_opportunities","expected"))
        require(counts.shape == opportunities.shape == expected.shape == (MAX_DISTANCE_BINS+1,), "Native expected distance shape differs")
        computed=np.divide(counts,opportunities,out=np.zeros_like(counts),where=opportunities>0)
        require(np.array_equal(expected,computed) and np.isfinite(expected).all()
                and np.all(expected[MIN_DISTANCE_BINS:] > 0), "Invalid/zero native expected support")


def prepare_shared_expected(dataset,frozen_fold,cache_dir=None):
    """Fit once or verify a small immutable cached normalizer; never read outer for fitting."""
    started=time.perf_counter()
    sources=native_source_records(dataset)
    binding=_expected_binding(dataset,frozen_fold,sources)
    key=json_signature(binding)
    directory=_safe_cache_root(cache_dir)/data_source_signature()[:12]
    path=directory/f"fold_{int(frozen_fold['fold'])}.json"
    # A frozen fold records its number as 'fold'; guard all artifact locations.
    require(0 <= int(frozen_fold["fold"]) <= 4, "Native cache requires original five folds")
    failure=path.with_suffix(".failure.json")
    require(not failure.exists(), "Retained native normalizer failure forbids automatic retry")
    for temporary in Path("/tmp").glob("microcture-native-expected-*.json"):
        try:
            staged=read_json(temporary)
        except (OSError,ValueError):
            continue
        require(staged.get("binding") != binding,
                f"Unpublished native scientific normalizer retained at {temporary}; no automatic retry")
    if key in _EXPECTED_CACHE:
        cached=_EXPECTED_CACHE[key]
        _verify_expected(cached,binding)
        return cached
    if path.exists():
        cached=read_json(path);_verify_expected(cached,binding)
        _EXPECTED_CACHE[key]=cached
        return cached
    record={"status":"in_progress","binding":binding,"records":[]}
    try:
        occupancy,boundaries=training_native_mask(dataset,frozen_fold)
        require(not circular_wrap_pair_guard(occupancy,MAX_DISTANCE_BINS), "Unsupported exact native circular pairs; no approximate fit")
        opportunities=fft_pair_opportunities(occupancy,MAX_DISTANCE_BINS)
        # Current guard guarantees circular FFT equals direct nonwrapped pairs.
        for d in (20,21,100,255,1000,MAX_DISTANCE_BINS):
            if d > MAX_DISTANCE_BINS:
                continue
            require(opportunities[d] == int((occupancy[:-d]&occupancy[d:]).sum()), "Circular/nonwrapped exposure mismatch")
        record["unique_training_native_bins"]=int(occupancy.sum())
        record["contiguous_components"]=boundaries.tolist()
        for source in sources:
            print(f"Fit native E10 fold={frozen_fold['fold']} WT={source['replicate']} training_union_bins={occupancy.sum()}",flush=True)
            clr=cooler.Cooler(str(ROOT/source["path"]))
            sums,facts=owned_native_counts(clr,occupancy,boundaries,max_distance=MAX_DISTANCE_BINS)
            expected=np.divide(sums,opportunities,out=np.zeros_like(sums),where=opportunities>0)
            one={"replicate":source["replicate"],"sample_id":source["sample_id"],"expected":expected.tolist(),
                 "stored_count_sum_by_distance":sums.tolist(),"zero_inclusive_pair_opportunities":opportunities.tolist(),"reader":facts}
            record["records"].append(one)
            require(np.all(expected[MIN_DISTANCE_BINS:] > 0), "Inner-training native background has zero distance support/counts; no silent epsilon or smoothing")
        record.update(status="complete",runtime_seconds=time.perf_counter()-started)
        record["signature"]=json_signature(record)
        _verify_expected(record,binding)
        _write_new_json(path,record)
        _EXPECTED_CACHE[key]=record
        return record
    except BaseException as error:
        record.update(status="failure",error_type=type(error).__name__,error=str(error),runtime_seconds=time.perf_counter()-started)
        record["signature"]=json_signature({k:v for k,v in record.items() if k != "signature"})
        if not path.exists() and not failure.exists():
            _write_new_json(failure,record)
        raise


def _dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def resample_matched_native(counts,starts,interval,frozen_good,expected,output_size=64):
    """Matched mean-logOE branches, using no dense native N-by-N arrays."""
    counts=sparse.csr_matrix(counts,dtype=np.float64)
    starts=np.asarray(starts,dtype=np.int64);good=np.asarray(frozen_good,dtype=np.float64)
    expected=np.asarray(expected,dtype=np.float64);n=len(starts);start,end=interval
    require(counts.shape==(n,n) and good.shape==(n,) and n>=21,"Native field shape differs")
    require(np.array_equal(np.diff(starts),np.full(n-1,10)) and np.all((good==0)|(good==1)),"Native bins/coverage invalid")
    require(np.isfinite(counts.data).all() and np.all(counts.data>=0),"Invalid raw counts")
    asymmetry=counts-counts.T
    require(not asymmetry.nnz or np.abs(asymmetry.data).max()<1e-12,"Native ROI matrix is not symmetric")
    require(len(expected)>=n and np.isfinite(expected[20:n]).all() and np.all(expected[20:n]>0),"Positive shared E10 is required")
    a=sparse.csr_matrix(roi_area_weights(float(start),float(end),starts,resolution_bp=10,output_size=output_size))
    e=np.maximum(0.,np.minimum(starts+10,end)-np.maximum(starts,start))/10
    t=sparse.diags(e)
    offsets=list(range(-19,20))
    diagonals=[good[max(0,-d):min(n,n-d)]*good[max(0,d):min(n,n+d)] for d in offsets]
    near=sparse.diags(diagonals,offsets,shape=(n,n),format="csr")
    q=a@good
    validity=np.outer(q,q)-_dense(a@near@a.T)
    validity[np.abs(validity)<1e-12]=0
    require(np.all((validity>=-1e-10)&(validity<=1+1e-10)),"Native pair-area support invalid")
    validity=np.clip(validity,0,1)
    coo=counts.tocoo();d=np.abs(coo.row-coo.col)
    keep=(d>=20)&(good[coo.row]>0)&(good[coo.col]>0)
    values=np.log1p(coo.data[keep]/expected[d[keep]])
    field=sparse.csr_matrix((values,(coo.row[keep],coo.col[keep])),shape=(n,n))
    fine_total=_dense(a@field@a.T)
    fine=np.divide(fine_total,validity,out=np.zeros_like(fine_total),where=validity>0)
    unique,cindex=np.unique(starts//100,return_inverse=True)
    b=sparse.csr_matrix((np.ones(n),(np.arange(n),cindex)),shape=(n,len(unique)))
    numerator=_dense(b.T@t@field@t@b)
    exposure=np.asarray(b.T@(good*e)).ravel()
    denominator=np.outer(exposure,exposure)-_dense(b.T@t@near@t@b)
    projected=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12)
    ag=_dense(a@sparse.diags(good)@b)
    off=near.tocoo()
    correction=sparse.csr_matrix((off.data*projected[cindex[off.row],cindex[off.col]],(off.row,off.col)),shape=(n,n))
    coarse_total=ag@projected@ag.T-_dense(a@correction@a.T)
    coarse_total[np.abs(coarse_total)<1e-12]=0
    require(np.all(coarse_total>=-1e-10),"Projected weighted logOE is negative")
    coarse=np.divide(np.maximum(coarse_total,0),validity,out=np.zeros_like(coarse_total),where=validity>0)
    coverage=np.asarray(a.sum(axis=1)).ravel()
    mask=np.outer(coverage,coverage);mask[np.abs(mask-1)<1e-10]=1
    result={"C0":np.stack((coarse,validity,mask)).astype(np.float32),"C1":np.stack((fine,validity,mask)).astype(np.float32)}
    require(np.array_equal(result["C0"][1:],result["C1"][1:]),"Native projection altered validity/mask")
    require(np.isfinite(result["C0"]).all() and np.isfinite(result["C1"]).all(),"Native pooled input nonfinite")
    return result,{"native_bin_count":n,"source_resolution_bp":10,"source100bp_cell_count":len(unique),
        "output_size":output_size,"output_bin_width_bp":float((end-start)/output_size),
        "target_physical_coverage_fraction":float(mask.mean()),"target_validity_fraction":float(validity.mean()),
        "native_sparse_nnz":int(counts.nnz),"physical_min_pixel_distance_bp":200,
        "projection_uses_ROI_exposure_only":True,"validity_mask_shared_byte_identical":True,
        "native_dense_matrix_allocated":False,"signal_formula":"area-weighted mean(native log1p(count/E10)); C0 applies valid ROI exposure means inside actual100bp cells first"}


def _frozen_good(dataset,index,replicate):
    bins=dataset["window_footprints"][index];n=dataset["n_bins"]
    difference=(bins[None,:]-bins[:,None])%n;distance=np.minimum(difference,n-difference)
    anchor=np.where(difference<=n//2,bins[:,None],bins[None,:])
    counts=np.asarray(dataset["bands"][replicate][anchor,distance])
    occupied=dataset["bin_occupancy"][bins]>0
    support=(distance>=2)&occupied[:,None]&occupied[None,:]
    good=(np.where(support,counts,0).sum(axis=0)>0)&occupied
    return dict(zip(bins.tolist(),good.tolist()))


def _replace_native(dataset,samples,expected_record):
    out={alignment:{**samples,"x":samples["x"].copy()} for alignment in ALIGNMENTS}
    index={str(sid):i for i,sid in enumerate(dataset["structure_id"])}
    sources=expected_record["binding"]["raw_sources"]
    raw={r["replicate"]:cooler.Cooler(str(ROOT/r["path"])) for r in sources}
    rep_index={r["replicate"]:i for i,r in enumerate(dataset["metadata"]["samples"])}
    curves={r["replicate"]:np.asarray(r["expected"]) for r in expected_record["records"]}
    facts=[]
    for row,(sid,replicate) in enumerate(zip(samples["structure_id"],samples["replicate"])):
        replicate=int(replicate);number=index[str(sid)];annotation=dataset["metadata"]["structures"][number]
        start,end=int(annotation["start"]),int(annotation["end"])
        first,last=(start//100)*10,((end+99)//100)*10
        require(21 <= last-first <= MAX_DISTANCE_BINS+1,"Native ROI exceeds registered first-scale distance background")
        counts=raw[replicate].matrix(balance=False,sparse=True)[first:last,first:last].tocsr()
        starts=np.arange(first,last,dtype=np.int64)*10
        coverage=_frozen_good(dataset,number,rep_index[replicate])
        require(all(int(s//100) in coverage for s in starts),"Native ROI source is outside frozen footprint")
        good=np.asarray([coverage[int(s//100)] for s in starts],dtype=float)
        branches,record=resample_matched_native(counts,starts,(start,end),good,curves[replicate])
        require(record["target_physical_coverage_fraction"] >= 1-1e-8 and branches["C0"][1].max()>0,"Native ROI lacks target coverage or valid pixels")
        for alignment in ALIGNMENTS:
            out[alignment]["x"][row,0]=branches[alignment]
        facts.append({"structure_id":str(sid),"replicate":replicate,**record})
    for alignment in ALIGNMENTS:
        out[alignment]["roi_resampling"]=[{**r,"alignment":alignment} for r in facts]
        require(np.array_equal(out[alignment]["x"][:,1],samples["x"][:,1]),"Native source changed frozen context")
        for field in ("length","y","structure_id","group_id","replicate","sample_id"):
            require(np.array_equal(out[alignment][field],samples[field]),"Native source changed fixed metadata/length")
    return out


def _standardize(samples,scaler):
    out={**samples,"x":samples["x"].copy()}
    values=out["x"][:,0,0];valid=out["x"][:,0,1]>0
    out["x"][:,0,0]=np.where(valid,(values-scaler["mean"])/scaler["std"],0)
    require(np.isfinite(out["x"]).all(),"Native standardized first-scale input nonfinite")
    return out


def _preprocessing(base,alignment,scaler,shared):
    value={"protocol":PROTOCOL,"alignment":alignment,"base_preprocessing":base,"base_preprocessing_signature":base["signature"],
        "native_expected":shared,"native_expected_signature":shared["signature"],"native_first_signal_scaler":scaler,
        "dataset_signature":base["dataset_signature"],"frozen_fold_signature":base["frozen_fold_signature"],
        "source_resolution_bp":10,"context_source_resolution_bp":100,"second_scale_byte_identical":True,
        "length_scaler_refitted":False,"target_coverage_and_validity_shared":True,"physical_min_pixel_distance_bp":200,
        "normalization":"mean(log1p(OE)); C0 same-field100bp-cell ROI-exposure projection; C1 native detail",
        "first_scaler_fit_scope":"frozen inner-training replicate images only","images_persisted":False}
    value["signature"]=controlled._signature(value)
    return value


def prepare_native_inner(dataset,frozen_fold,alignment,cache_dir=None):
    """Return inner train/validation plus JSON-serializable fitted preprocessing."""
    global _INNER_MEMO
    require(alignment in ALIGNMENTS,"Choose C0/C1 native detail alignment")
    key=(dataset["signature"],json_signature(frozen_fold),data_source_signature(),str(_safe_cache_root(cache_dir)))
    if _INNER_MEMO is None or _INNER_MEMO[0] != key:
        train,validation,base=formal.prepare_inner(dataset,frozen_fold)
        shared=prepare_shared_expected(dataset,frozen_fold,cache_dir)
        native_train=_replace_native(dataset,train,shared)
        native_validation=_replace_native(dataset,validation,shared)
        _INNER_MEMO=(key,native_train,native_validation,base,shared)
    _,native_train,native_validation,base,shared=_INNER_MEMO
    _verify_expected(shared,_expected_binding(dataset,frozen_fold,native_source_records(dataset)))
    train,validation=native_train[alignment],native_validation[alignment]
    valid=train["x"][:,0,1]>0;values=train["x"][:,0,0][valid].astype(np.float64)
    require(len(values)>0,"No valid native training pooled pixels")
    scaler={"mean":float(values.mean()),"std":max(float(values.std(ddof=0)),1e-6),
        "fitted_valid_pooled_pixels":len(values),"train_ids":base["train_ids"],"fit_scope":"frozen inner-training replicate images only"}
    pp=_preprocessing(base,alignment,scaler,shared)
    return _standardize(train,scaler),_standardize(validation,scaler),pp


def prepare_native_outer(dataset,frozen_fold,preprocessing,alignment):
    """Transform outer ONLY after caller fixed weights; no fitting or cache writes."""
    require(alignment in ALIGNMENTS and preprocessing["protocol"]==PROTOCOL and preprocessing["alignment"]==alignment,
            "Native outer alignment/protocol differs")
    require(preprocessing["signature"]==controlled._signature({k:v for k,v in preprocessing.items() if k!="signature"}),"Native preprocessing bytes changed")
    base=preprocessing["base_preprocessing"]
    require(base["signature"]==preprocessing["base_preprocessing_signature"],"Frozen context/length preprocessing changed")
    shared=preprocessing["native_expected"]
    require(shared["signature"]==preprocessing["native_expected_signature"],"Shared native expected changed")
    _verify_expected(shared,_expected_binding(dataset,frozen_fold,native_source_records(dataset)))
    scaler=preprocessing["native_first_signal_scaler"]
    require(scaler["train_ids"]==base["train_ids"] and scaler["fit_scope"]=="frozen inner-training replicate images only"
            and np.isfinite(scaler["mean"]) and np.isfinite(scaler["std"]) and scaler["std"]>0,"Invalid native inner-training scaler")
    samples=formal.prepare_outer(dataset,frozen_fold,base)
    native=_replace_native(dataset,samples,shared)
    return _standardize(native[alignment],scaler)
