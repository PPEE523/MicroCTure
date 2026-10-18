"""Output protection must fail before network, raw scans or evidence changes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from microcture.layout import bootstrap
bootstrap()
import contact_band
import download_conditions
import optional_common
import plot_examples
import prepare_conditions
import prepare_data


class SharedOutputGuardsTest(unittest.TestCase):
    def test_original_annotation_export_is_refused_before_auditing(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            config=root/'config.json'
            config.write_text(json.dumps({'annotations':{'coordinate_status':'provisional'},'samples':[]}))
            retained=root/'data/processed/structures.csv'
            retained.parent.mkdir(parents=True)
            retained.write_bytes(b'retained annotations\n')
            with mock.patch.object(prepare_data,'ROOT',root), \
                 mock.patch.object(sys,'argv',['prepare_data.py','--config',str(config)]), \
                 mock.patch.object(prepare_data,'audit_sample') as audit, \
                 mock.patch.object(prepare_data,'annotations') as annotations:
                with self.assertRaises(FileExistsError):
                    prepare_data.main()
                audit.assert_not_called()
                annotations.assert_not_called()
            self.assertEqual(retained.read_bytes(),b'retained annotations\n')

    def test_condition_audit_does_not_enter_an_existing_report(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            retained=root/'reports/shared/conditions/saved_v1'
            retained.mkdir(parents=True)
            with mock.patch.object(prepare_conditions,'ROOT',root), \
                 mock.patch.object(sys,'argv',['prepare_conditions.py','--report-dir',str(retained)]), \
                 mock.patch.object(prepare_conditions,'audit_sample') as audit, \
                 mock.patch.object(prepare_conditions.gzip,'open') as raw:
                with self.assertRaises(FileExistsError):
                    prepare_conditions.main()
                audit.assert_not_called()
                raw.assert_not_called()
            self.assertEqual(list(retained.iterdir()),[])

    def test_local_plotting_refuses_old_output_before_reading_pixels(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            (root/'configs').mkdir()
            (root/'configs/data.json').write_text('{}')
            retained=root/'results/task1/exploration/saved_v1'
            retained.mkdir(parents=True)
            cache=root/'data/shared/cache/new_v1'
            argv=['plot_examples.py','--output-dir',str(retained),'--cache-dir',str(cache)]
            with mock.patch.object(plot_examples,'ROOT',root), \
                 mock.patch.object(sys,'argv',argv), \
                 mock.patch.object(plot_examples,'expected_background') as scan, \
                 mock.patch.object(plot_examples,'aggregate_local') as pixels:
                with self.assertRaises(FileExistsError):
                    plot_examples.main()
                scan.assert_not_called()
                pixels.assert_not_called()
            self.assertFalse(cache.exists())

    def test_existing_csv_and_internal_symlink_are_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            csv=root/'annotations.csv'
            csv.write_bytes(b'original\n')
            with self.assertRaises(FileExistsError):
                prepare_data.write_csv(csv,[{'label':'changed'}])
            self.assertEqual(csv.read_bytes(),b'original\n')
            target=root/'results/task5/inference/v1'
            target.mkdir(parents=True)
            alias=target.parent/'alias'
            alias.symlink_to(target,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                optional_common.fresh_output_path(alias/'new',root=root)
            self.assertFalse((target/'new').exists())

    def test_download_help_does_not_start_downloads(self):
        with mock.patch.object(sys,'argv',['download_conditions.py','--help']), \
             mock.patch.object(download_conditions.concurrent.futures,'ThreadPoolExecutor') as workers, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                download_conditions.main()
            self.assertEqual(error.exception.code,0)
            workers.assert_not_called()


class ReadOnlyBandCacheTest(unittest.TestCase):
    def test_stale_or_missing_cache_never_scans_even_with_old_allow_build_flag(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            raw=root/'raw.cool'
            raw.write_bytes(b'raw bytes must not be decoded')
            cache=root/'cache'
            cache.mkdir()
            sample={'path':'raw.cool','sample_id':'WT'}
            saved=cache/'WT.npz'
            np.savez_compressed(saved,signature=json.dumps({'version':0}),band=np.zeros((8,4)))
            before=saved.read_bytes()
            for flag in (False,True):
                with self.subTest(allow_build=flag), mock.patch.object(contact_band,'make_band') as scan:
                    with self.assertRaisesRegex(FileNotFoundError,'no raw scan'):
                        contact_band.load_band(sample,8,4,data_root=root,cache_root=cache,allow_build=flag)
                    scan.assert_not_called()
                    self.assertEqual(saved.read_bytes(),before)
            saved.unlink()
            with mock.patch.object(contact_band,'make_band') as scan:
                with self.assertRaises(FileNotFoundError):
                    contact_band.load_band(sample,8,4,data_root=root,cache_root=cache,allow_build=True)
                scan.assert_not_called()
                self.assertEqual(list(cache.iterdir()),[])

    def test_matching_saved_cache_is_reused_without_changing_bytes(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            raw=root/'raw.cool'
            raw.write_bytes(b'not opened as HDF5')
            sample={'path':'raw.cool','sample_id':'WT'}
            expected=np.arange(32,dtype=np.float32).reshape(8,4)
            saved=root/'WT.npz'
            np.savez_compressed(saved,signature=json.dumps(contact_band.cache_signature(raw,4)),band=expected)
            before=saved.read_bytes()
            with mock.patch.object(contact_band,'make_band') as scan:
                actual=contact_band.load_band(sample,8,4,data_root=root,cache_root=root)
                scan.assert_not_called()
            np.testing.assert_array_equal(actual['band'],expected)
            self.assertEqual(saved.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
