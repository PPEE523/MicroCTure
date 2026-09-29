"""Verify top-three reference selection, spatial independence, metrics and outputs."""
import json
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from discover_task2_v2 import ROOT,read_csv
from discover_task2 import load_band,extract,correlation
from discovery_v2_core import bin_roles,train_background
from review_novelty import shape_feature
from audit_morphology import remove_row_effects


def main():
    cfg=json.loads((ROOT/'configs/morphology_audit.json').read_text());out=ROOT/cfg['output_dir']
    manifest=json.loads((out/'run_manifest.json').read_text())
    for p,digest in manifest['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    evidence=read_csv(f"{cfg['output_dir']}/evidence.csv");matches=read_csv(f"{cfg['output_dir']}/top3_reference_matches.csv")
    refs=[r for r in read_csv(cfg['references_source']) if r['role']=='0'];candidates=read_csv(cfg['candidate_source'])
    data=json.loads((ROOT/'configs/data.json').read_text());source=json.loads((ROOT/'configs/discovery_v2.json').read_text())
    n=(data['chrom_length']+99)//100;roles=bin_roles(n,source);bands=[train_background(load_band(s,n,256),roles) for s in data['samples']]
    rfs=np.array([shape_feature(*extract(bands[0],int(r['start_bin']),int(r['width_bins']))[1:]) for r in refs])
    occupied_candidates=np.zeros(n,bool)
    for c in candidates:occupied_candidates[int(c['start_bp'])//100:(int(c['start_bp'])+int(c['window_length_bp']))//100]=True
    assert len(evidence)==21 and len(matches)==63
    for e in evidence:
        name=e['window_id'];start=int(e['start_bp'])//100;width=int(e['width_bp'])//100
        matrices=[extract(b,start,width) for b in bands]
        feature=shape_feature(*matrices[0][1:])
        available=[j for j,r in enumerate(refs) if int(r['width_bins'])==width]
        dist=cdist(feature[None],rfs[available],metric='cosine')[0]
        ordered=np.array(available)[np.argsort(dist,kind='stable')]
        chosen=[];occupied=np.zeros(n,bool)
        for j in ordered:
            lo=int(refs[j]['start_bin'])
            if not occupied[lo:lo+width].any():chosen.append(int(j));occupied[lo:lo+width]=True
            if len(chosen)==3:break
        saved=[r for r in matches if r['window_id']==name]
        assert [r['reference_id'] for r in saved]==[refs[j]['structure_id'] for j in chosen]
        for row,j in zip(saved,chosen):
            lo=int(refs[j]['start_bin']);assert not occupied_candidates[lo:lo+width].any()
            ds=[]
            for rep,b in enumerate(bands):
                other=extract(b,lo,width)
                ds.append(float(cdist(shape_feature(*matrices[rep][1:])[None],shape_feature(*other[1:])[None],metric='cosine')[0,0]))
            np.testing.assert_allclose(ds,[float(row['rep1_cosine_distance']),float(row['rep2_same_reference_cosine_distance'])])
        common=matrices[0][2]&matrices[1][2]
        logs=[np.log1p(m[1]) for m in matrices]
        residuals=[]
        for rep,m in enumerate(matrices):
            residual,effect=remove_row_effects(logs[rep],m[2],cfg['row_effect_iterations']);residuals.append(residual)
            np.testing.assert_allclose((residual+effect)[m[2]],logs[rep][m[2]])
        np.testing.assert_allclose(float(e['log_oe_replication']),correlation(logs[0],logs[1],common))
        np.testing.assert_allclose(float(e['residual_replication']),correlation(*residuals,common))
    images=list(out.rglob('*.png'));assert len(images)==25
    for path in images:
        with Image.open(path) as image:image.verify()
    report=dict(status='passed',candidates=21,independently_checked_reference_matches=63,figures=25,source_hashes_match=True,
                reference_contexts_disjoint_from_candidates=True,top3_reference_contexts_mutually_disjoint=True,independent_confirmation=False)
    (ROOT/'reports/morphology_audit_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
