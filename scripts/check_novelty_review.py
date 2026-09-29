"""Verify calibration separation, nearest references and all 21 review decisions."""
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from discover_task2_v2 import ROOT, read_csv
from discover_task2 import load_band, extract
from discovery_v2_core import bin_roles, train_background
from review_novelty import shape_feature


def main():
    cfg = json.loads((ROOT/'configs/novelty_review.json').read_text())
    source = json.loads((ROOT/cfg['source_config']).read_text())
    data = json.loads((ROOT/'configs/data.json').read_text())
    out = ROOT/cfg['output_dir']
    manifest = json.loads((out/'run_manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    previous = json.loads((ROOT/'results/task2_v2/run_manifest.json').read_text())
    for sample in data['samples']:
        stat = (ROOT/sample['path']).stat()
        expected = previous['samples'][sample['sample_id']]
        assert stat.st_size == expected['size'] and stat.st_mtime_ns == expected['mtime_ns']
    candidates = read_csv(cfg['candidate_source'])
    rows = read_csv(f"{cfg['output_dir']}/candidate_review.csv")
    refs = read_csv(f"{cfg['output_dir']}/known_references.csv")
    cal = read_csv(f"{cfg['output_dir']}/known_distance_calibration.csv")
    thresholds = json.loads((out/'calibration.json').read_text())
    n = (data['chrom_length']+99)//100
    roles = bin_roles(n, source)
    bands = [train_background(load_band(s, n, 256), roles) for s in data['samples']]
    occupied = np.zeros(n, bool)
    for r in candidates:
        ids = np.arange(int(r['start_bp'])//100, (int(r['start_bp'])+int(r['window_length_bp']))//100)
        assert not occupied[ids].any()
        occupied[ids] = True
    for ref in refs:
        start, width = int(ref['start_bin']), int(ref['width_bins'])
        assert not occupied[start:start+width].any()
        assert (roles[start:start+width] == int(ref['role'])).all()
    with np.load(out/'reference_features.npz') as features:
        representations = {name: features[name] for name in ['shape', 'pca']}
    for width in source['window_bins']:
        training = [i for i, r in enumerate(refs) if int(r['width_bins']) == width and int(r['role']) == 0]
        for metric, distance in [('pca', 'euclidean'), ('shape', 'cosine')]:
            values = [r for r in cal if int(r['width_bins']) == width and r['metric'] == metric]
            used = np.zeros(n, bool)
            features = representations[metric]
            for row in values:
                i, j = int(row['calibration_index']), int(row['reference_index'])
                start = int(refs[i]['start_bin'])
                assert int(refs[i]['role']) == 1 and j in training
                assert not used[start:start+width].any()
                used[start:start+width] = True
                nearest = training[int(cdist(features[i:i+1, 0], features[training, 0], metric=distance).argmin())]
                assert j == nearest
                for rep in [0, 1]:
                    np.testing.assert_allclose(float(row[f'rep{rep+1}_distance']), cdist(features[i:i+1, rep], features[j:j+1, rep], metric=distance)[0, 0])
            if len(values) >= cfg['min_calibration_references']:
                limits = np.quantile([[float(r['rep1_distance']), float(r['rep2_distance'])] for r in values], cfg['distance_quantile'], axis=0)
                np.testing.assert_allclose(limits, thresholds[str(width)][metric])
            else:
                assert thresholds[str(width)][metric] is None
    with np.load(ROOT/'results/task2_v2/shape_representation.npz') as fit:
        mean, components = fit['pca_mean'], fit['components']
    assert len(rows) == len(candidates) == 21
    for row, candidate in zip(rows, candidates):
        assert row['window_id'] == candidate['window_id']
        start, width = int(row['start_bp'])//100, int(row['window_length_bp'])//100
        fs = np.array([shape_feature(*extract(b, start, width)[1:]) for b in bands])
        ps = (fs-mean)@components.T
        training = [i for i, r in enumerate(refs) if int(r['width_bins']) == width and int(r['role']) == 0]
        decisions = []
        for metric, distance, features in [('pca', 'euclidean', ps), ('shape', 'cosine', fs)]:
            ref = representations[metric]
            j = training[int(cdist(features[0:1], ref[training, 0], metric=distance).argmin())]
            assert refs[j]['structure_id'] == row[f'{metric}_nearest_known']
            ds = [float(cdist(features[rep:rep+1], ref[j:j+1, rep], metric=distance)[0, 0]) for rep in [0, 1]]
            np.testing.assert_allclose(ds, [float(row[f'{metric}_rep1_distance']), float(row[f'{metric}_rep2_distance'])])
            threshold = thresholds[str(width)][metric]
            decision = row['quality_pass']=='True' and threshold is not None and all(d > t for d, t in zip(ds, threshold))
            assert row[f'{metric}_outside_both'] == str(decision)
            decisions.append(decision)
        assert row['priority_distinct_shape'] == str(all(decisions))
        with Image.open(out/'comparisons'/f"{row['window_id']}.png") as image:
            image.verify()
    result = dict(status='passed', candidates=21, figures=21, references=len(refs),
                  calibration_distance_rows=len(cal), reference_contexts_exclude_all_candidates=True,
                  calibration_windows_disjoint_within_scale=True, nearest_references_and_distances_recomputed=True,
                  source_hashes_match=True, independent_biological_confirmation=False)
    (ROOT/'reports/novelty_review_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
