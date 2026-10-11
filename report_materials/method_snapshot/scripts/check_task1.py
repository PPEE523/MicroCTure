"""Verify task-one outputs and independently reconstruct structure confusion counts."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_task1 import OUT,ROOT,SPECS


def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    evaluations=json.loads((OUT/'evaluation.json').read_text())
    assert len(evaluations)==21
    for r in evaluations:
        name,seed=r['model'],r['seed']
        rows=list(csv.DictReader((OUT/f'{name}_{seed}_predictions.csv').open()))
        assert len(rows)==53 and len({x['structure_id'] for x in rows})==53
        cm=np.zeros((3,3),int)
        for row in rows:
            p=np.array([float(row[k]) for k in ['OPCID','CHIN','CHID']])
            assert np.isfinite(p).all() and np.isclose(p.sum(),1,atol=1e-6)
            assert int(row['predicted'])==p.argmax()
            cm[int(row['true']),int(row['predicted'])]+=1
        np.testing.assert_equal(cm,r['test']['confusion_matrix'])
        np.testing.assert_equal(cm.sum(1),[10,39,4])
        if name in SPECS:
            history=json.loads((OUT/f'{name}_{seed}_history.json').read_text())
            assert len(history)==r['epochs_run']<=100
            assert max(x['validation_macro_f1'] for x in history)==r['validation_macro_f1']
            checkpoint=torch.load(OUT/f'{name}_{seed}.pt',weights_only=False)
            assert checkpoint['summary']['best_epoch']==r['best_epoch']
    for path in OUT.glob('*.png'):
        with Image.open(path) as image:image.verify()
    checks=dict(runs_verified=21,cnn_checkpoints=18,structures_per_test=53,source_hashes_match=True,
                confusion_counts_match=True,validation_best_epochs_match=True,pngs_verified=len(list(OUT.glob('*.png'))))
    (ROOT/'reports/task1_verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
