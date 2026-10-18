#!/usr/bin/env python3
"""Fixed historical anchors: inner-training R1/C0/C1 input views, no models.

Reads no prediction, loss, checkpoint or partial score. Normalization uses the
first frozen fold whose INNER-TRAIN set contains each preregistered anchor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits

from microcture.task1.data import load_dataset
from microcture.task1_controlled.data import _signature
from microcture.task1_formal.protocol import json_signature,sha256
from microcture.task1_roi import data as roi
from microcture.task1_native_roi_v2 import data as native

CASES=("CHID_8","CHID_11","CHIN_51","OPCID_66")
FROZEN_DATA_SHA="a8401c2e32629ae28c880228e991782108660cde4bb50f5b9915eacd78d18116"
SIGNAL_LIMITS=(-3.,3.)
VALIDITY_LIMITS=(0.,1.)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scaler_from_record(value):
    return {name:value[name] for name in ("mean","std","fitted_valid_pooled_pixels","fit_scope")}


def get_native_scalers(dataset,fold,native_run,validation):
    """Only preprocessing parameters; never inspect a model result/score."""
    number=fold["fold"]
    results={}
    for alignment in ("C0","C1"):
        path=native_run/"folds"/alignment/f"fold_{number}.json"
        if path.exists():
            saved=read_json(path);pp=saved["preprocessing"]
            if _signature({k:v for k,v in pp.items() if k!="signature"}) != pp["signature"]:
                raise ValueError("Saved native preprocessing signature differs")
            if saved["frozen_fold"] != fold:
                raise ValueError("Saved first-scale scaler fold differs")
            results[alignment]=(pp["native_first_signal_scaler"],str(path.relative_to(ROOT)))
        elif validation.get("status")=="passed" and validation["real_inner_validation"]["fold"]==number:
            values=validation["real_inner_validation"]
            results[alignment]=(values[f"{alignment}_first_scaler"],
                                f"reports/task1_autonomous_v1/native_data_validation_v2.json#real_inner_validation/{alignment}_first_scaler")
    if len(results)!=2:
        # Fit exactly the preregistered training-only preprocessing, with no
        # models or outer images. Shared E10 must already exist, not fit here.
        curve_path=ROOT/"data/task1_native_roi_v2/expected"/FROZEN_DATA_SHA[:12]/f"fold_{number}.json"
        if not curve_path.exists():
            raise ValueError("Build and verify scientific inner-only E10 before plotting")
        for alignment in ("C0","C1"):
            if alignment not in results:
                train,validation_samples,pp=native.prepare_native_inner(dataset,fold,alignment)
                results[alignment]=(pp["native_first_signal_scaler"],
                                    "same frozen inner-training-only algorithm recomputed for display; no model/outer")
                del train,validation_samples
    for scaler,reference in results.values():
        if scaler["train_ids"]!=fold["train_ids"] or scaler["fit_scope"]!="frozen inner-training replicate images only":
            raise ValueError("Native scaler is not fitted exclusively on the chosen inner-training set")
    return results


def build_views(native_run):
    if native.data_source_signature()!=FROZEN_DATA_SHA:
        raise ValueError("Frozen native data implementation changed")
    dataset=load_dataset(ROOT/"data/cache/task1_comparison_v1")
    sources=native.native_source_records(dataset)
    import cooler
    raw={r["replicate"]:cooler.Cooler(str(ROOT/r["path"])) for r in sources}
    folds={number:read_json(ROOT/f"results/task1_comparison_v1/folds/fold_{number}.json") for number in range(5)}
    lookup={str(sid):i for i,sid in enumerate(dataset["structure_id"])}
    validation=read_json(ROOT/"reports/task1_autonomous_v1/native_data_validation_v2.json")
    prepared={};views=[];records=[]
    for sid in CASES:
        fold=next((fold for fold in folds.values() if sid in fold["train_ids"]),None)
        if fold is None:raise ValueError("Fixed anchor has no frozen inner-training occurrence")
        number=fold["fold"]
        if number not in prepared:
            r1path=ROOT/f"results/task1_roi_v1/folds/R1/fold_{number}.json"
            saved=read_json(r1path);r1pp=saved["preprocessing"]
            if saved["frozen_fold"]!=fold or _signature({k:v for k,v in r1pp.items() if k!="signature"})!=r1pp["signature"]:
                raise ValueError("R1 preprocessor/fold differs")
            path=ROOT/"data/task1_native_roi_v2/expected"/FROZEN_DATA_SHA[:12]/f"fold_{number}.json"
            shared=read_json(path)
            native._verify_expected(shared,native._expected_binding(dataset,fold,sources))
            prepared[number]={"R1":r1pp,"R1_ref":str(r1path.relative_to(ROOT)),"native":shared,
                              "native_ref":str(path.relative_to(ROOT)),"native_sha256":sha256(path),
                              "scalers":get_native_scalers(dataset,fold,native_run,validation)}
        fitted=prepared[number];index=lookup[sid];annotation=dataset["metadata"]["structures"][index]
        start,end=int(annotation["start"]),int(annotation["end"])
        case={"structure_id":sid,"annotation_class":annotation["type"],"start_bp":start,"end_bp":end,"length_bp":end-start,
              "first_available_inner_training_fold":number,"case_in_inner_train":True,
              "selection":"Fixed before model results; first frozen fold containing anchor in INNER-TRAIN",
              "frozen_fold_sha256":sha256(ROOT/f"results/task1_comparison_v1/folds/fold_{number}.json"),
              "training_ids_signature":json_signature(fold["train_ids"]),"native_expected_signature":fitted["native"]["signature"],
              "native_expected_file":fitted["native_ref"],"native_expected_file_sha256":fitted["native_sha256"],
              "R1_preprocessing_reference":fitted["R1_ref"],"panels":[]}
        case_views=[]
        for rep_index,source in enumerate(sources):
            replicate=source["replicate"]
            base=fitted["R1"]["base_preprocessing"]
            signal,validity,starts=roi._native_source(dataset,index,rep_index,base["signal_preprocessing"]["expected"])
            r1,facts=roi.resample_roi(signal,validity,starts,start,end,chrom_length=dataset["metadata"]["chrom_length"])
            first,last=(start//100)*10,((end+99)//100)*10
            counts=raw[replicate].matrix(balance=False,sparse=True)[first:last,first:last].tocsr()
            starts10=np.arange(first,last,dtype=np.int64)*10
            coverage=native._frozen_good(dataset,index,rep_index)
            good=np.asarray([coverage[int(s//100)] for s in starts10],dtype=float)
            expected=np.asarray(next(r for r in fitted["native"]["records"] if r["replicate"]==replicate)["expected"])
            branches,native_facts=native.resample_matched_native(counts,starts10,(start,end),good,expected)
            if not np.array_equal(branches["C0"][1:],branches["C1"][1:]):
                raise ValueError("Matched native support/geometry differs")
            for name,channels in (("R1",r1),("C0",branches["C0"]),("C1",branches["C1"])):
                if name=="R1":
                    scaler=fitted["R1"]["roi_first_signal_scaler"];reference=fitted["R1_ref"]
                else:scaler,reference=fitted["scalers"][name]
                z=channels[0].copy();valid=channels[1]>0
                z=np.where(valid,(z-scaler["mean"])/scaler["std"],0).astype(np.float32)
                case_views.append((name,replicate,z,channels[1].copy()))
                case["panels"].append({"source":name,"replicate":replicate,"first_scaler":scaler_from_record(scaler),
                    "first_scaler_reference":reference,"validity_mean":float(channels[1].mean()),
                    "validity_nonzero_fraction":float(valid.mean()),"signal_display":"model first-scale z-score; same +/-3 cap everywhere",
                    "zscore_range":[float(z.min()),float(z.max())],
                    "fraction_valid_pixels_below_shared_cap":float(np.mean(z[valid]<SIGNAL_LIMITS[0])),
                    "fraction_valid_pixels_above_shared_cap":float(np.mean(z[valid]>SIGNAL_LIMITS[1])),
                    "unstandardized_logOE_range":[float(channels[0].min()),float(channels[0].max())],
                    "source_resolution_bp":100 if name=="R1" else 10,
                    "native_cell_projection":name=="C0","geometry_mask_mean":float(channels[2].mean())})
        views.append(case_views);records.append(case)
    return views,{"cases":records,"case_order":list(CASES),"data_source_signature":FROZEN_DATA_SHA,
        "dataset_signature":dataset["signature"],"raw_sources":sources,
        "source_policy":"R1 uses frozen E100/coarse support and is reference only; C0/C1 share E10 and physical200/coarse-coverage support",
        "caps":{"signal_zscore":list(SIGNAL_LIMITS),"validity":list(VALIDITY_LIMITS),"per_case_rescaling":False},
        "selection_scope":"Predeclared historical errors, presented as training-input views; no independent expert truth or heldout accuracy claim",
        "model_loaded":False,"training_run":False,"model_predictions_or_partial_scores_read":False,
        "contact_matrices_persisted":False,"script_sha256":sha256(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run",type=Path,default=ROOT/"results/task1_native_roi_v2")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/task1_autonomous_v1")
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT/"reports").resolve()):
        raise ValueError("Write derived panels only under reports/")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    with threadpool_limits(limits=2):views,report=build_views(args.native_run)
    fig,axes=plt.subplots(8,6,figsize=(16.5,16.2))
    fig.subplots_adjust(left=.115,right=.92,bottom=.085,top=.915,wspace=.025,hspace=.15)
    for case_number,(case,panels) in enumerate(zip(report["cases"],views)):
        for col,(name,rep,z,validity) in enumerate(panels):
            signal_image=axes[2*case_number,col].imshow(z,cmap="coolwarm",vmin=-3,vmax=3,interpolation="nearest")
            validity_image=axes[2*case_number+1,col].imshow(validity,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            if case_number==0:
                description={"R1":"Original100bp ROI (reference)","C0":"Native field ->100bp cells","C1":"Native10bp detail"}[name]
                axes[0,col].set_title(f"WT {rep} / {name}\n{description}",fontsize=9)
            for row in (2*case_number,2*case_number+1):
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if case_number==3:
                axes[7,col].set_xticks([0,63],["0","1"],fontsize=7)
        label=f"{case['structure_id']}\n{case['length_bp']/1000:.2f} kb / fold {case['first_available_inner_training_fold']}\nSignal"
        axes[2*case_number,0].set_ylabel(label,rotation=0,ha="right",va="center",labelpad=8,fontsize=9)
        axes[2*case_number+1,0].set_ylabel("Validity",rotation=0,ha="right",va="center",labelpad=8,fontsize=9)
    cax=fig.add_axes([.936,.56,.012,.30]);fig.colorbar(signal_image,cax=cax,label="Train-standardized log1p(OE), shared cap")
    cax=fig.add_axes([.936,.17,.012,.24]);fig.colorbar(validity_image,cax=cax,label="Valid pair-area fraction")
    fig.suptitle("Fixed historical anchors: original ROI100 reference and matched native cell-detail control",fontsize=13,y=.969)
    fig.text(.5,.946,"First frozen INNER-TRAIN fold per case; both WT replicates; identical caps, no per-case contrast fitting",ha="center",fontsize=10)
    fig.text(.5,.054,"Horizontal/vertical axes: relative supplied target coordinate 0–1. R1 uses different E100 normalization and coarse diagonal support.",ha="center",fontsize=8)
    fig.text(.5,.039,"C0/C1 share E10, physical200 support and frozen coarse coverage; first-scale scalers follow the same inner-training-only rule.",ha="center",fontsize=8)
    fig.text(.5,.024,"Annotation IDs are historical error anchors, not independent expert truth. These training-input examples do not establish morphology reliability.",ha="center",fontsize=8)
    args.output.mkdir(parents=True,exist_ok=True)
    report["artifacts"]=[]
    for suffix in ("png","pdf"):
        path=args.output/f"native_shape_examples.{suffix}"
        fig.savefig(path,dpi=175 if suffix=="png" else None,bbox_inches="tight")
        report["artifacts"].append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path)})
    plt.close(fig)
    (args.output/"native_shape_examples.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","cases":list(CASES),"model_loaded":False,"output":str(args.output/"native_shape_examples.png")},ensure_ascii=False))


if __name__=="__main__":
    main()
