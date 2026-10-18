"""Portable candidate evidence must survive independent decision recomputation."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from microcture.tasks.task2.validation import read_csv,write_csv,sha,json_write
from microcture.tasks.task2.validation_delivery import verify_snapshot

ROOT=Path(__file__).resolve().parents[1]


class ValidationSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='microcture-candidate-snapshot-',dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'evidence'
        shutil.copytree(ROOT/'submission/task2/validation/frozen21_v1',self.path)

    def reseal(self):
        path=self.path/'ARTIFACTS.json';manifest=json.loads(path.read_text())
        for r in manifest['files']:
            r.update(sha256=sha(self.path/r['path']),bytes=(self.path/r['path']).stat().st_size)
        json_write(path,manifest)
        p=self.path/'COMPLETED.json';done=json.loads(p.read_text());done['artifact_manifest_sha256']=sha(path);json_write(p,done)

    def test_portable_full_grid_and_thresholds_recompute_without_feature_arrays(self):
        r=verify_snapshot(self.path)
        self.assertEqual(r['threshold_decisions_recomputed'],1134)
        self.assertEqual(r['scale_cells_recomputed'],126)
        self.assertFalse(r['retrieval_feature_arrays_replayed'])

    def test_threshold_change_cannot_hide_behind_updated_hash(self):
        path=self.path/'candidate_variants.csv';rows=read_csv(path);rows[0]['cutoff']=.123
        write_csv(path,rows);self.reseal()
        with self.assertRaisesRegex(ValueError,'threshold mismatch'):verify_snapshot(self.path)

    def test_partial_quality_grid_cannot_turn_into_complete_negative(self):
        path=self.path/'candidate_scale_evidence.csv';rows=read_csv(path)
        row=next(r for r in rows if r['full_cell_above_fraction']=='');row['full_cell_above_fraction']=0
        write_csv(path,rows);self.reseal()
        with self.assertRaisesRegex(ValueError,'scale evidence mismatch'):verify_snapshot(self.path)

    def test_absent_retrieval_class_cannot_become_zero_accuracy(self):
        path=self.path/'known_retrieval_summary.csv';rows=read_csv(path)
        row=next(r for r in rows if r['type']=='CHID');row['top1']=0
        write_csv(path,rows);self.reseal()
        with self.assertRaisesRegex(ValueError,'Missing retrieval class'):verify_snapshot(self.path)


if __name__=='__main__':unittest.main()
