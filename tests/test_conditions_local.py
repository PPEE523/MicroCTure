"""Fixtures for unchanged band arithmetic and the new read-only local entry."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from microcture.layout import bootstrap, resolve
bootstrap()
SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
import contact_band
import run_conditions_local


class ContactBandTests(unittest.TestCase):
    def source(self, base, length=800):
        path = base / 'fixture.cool'
        a = np.array([0, 0, 20, 30, 0, 10])
        b = np.array([20, 60, 50, 70, 0, 20])
        v = np.array([4, 3, 5, 2, 7, 1])
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('pixels/bin1_id', data=a)
            handle.create_dataset('pixels/bin2_id', data=b)
            handle.create_dataset('pixels/count', data=v)
            handle.create_dataset('chroms/length', data=[length])
            handle.attrs['sum'] = int(v.sum())
        return path, a // 10, b // 10, v

    def test_circular_counts_coverage_and_zero_opportunities(self):
        with tempfile.TemporaryDirectory(prefix='microcture-band-test-') as folder:
            path, a, b, values = self.source(Path(folder))
            result = contact_band.make_band(path, 8, 4, chunk=2)
            expected = np.zeros((8, 4), np.float32)
            expected[0, 2], expected[6, 2], expected[2, 3] = 4, 3, 5
            np.testing.assert_array_equal(result['band'], expected)
            coverage = np.zeros(8)
            for i, j, value in zip(a, b, values):
                coverage[i] += value
                coverage[j] += value
            np.testing.assert_array_equal(result['coverage'], coverage)
            self.assertEqual(result['total'], 22)
            self.assertAlmostEqual(result['threshold'], .3)
            for d in [2, 3]:
                pairs = [(i, (i + d) % 8) for i in range(8) if result['valid'][i] and result['valid'][(i + d) % 8]]
                self.assertEqual(result['opportunities'][d], len(pairs))
                self.assertAlmostEqual(result['expected'][d], sum(float(expected[i, d]) for i, _ in pairs) / len(pairs))

    def test_partial_last_bin_and_extraction_keep_legacy_mask(self):
        with tempfile.TemporaryDirectory(prefix='microcture-band-test-') as folder:
            path, *_ = self.source(Path(folder), length=752)
            result = contact_band.make_band(path, 8, 4)
            self.assertFalse(result['valid'][-1])
            raw, oe, mask = contact_band.extract(result, -2, 4)
            self.assertEqual(raw[0, 2], 3)
            self.assertEqual(raw[2, 0], 3)
            self.assertFalse(mask[1].any())
            self.assertFalse(np.diag(mask).any())
            self.assertAlmostEqual(float(oe[0, 2]), raw[0, 2] / result['expected'][2], places=6)

    def test_missing_cache_never_builds_or_creates_directory(self):
        with tempfile.TemporaryDirectory(prefix='microcture-band-test-') as folder:
            root = Path(folder)
            self.source(root)
            sample = dict(sample_id='fixture', path='fixture.cool')
            with mock.patch.object(contact_band, 'make_band', side_effect=AssertionError('raw scan')):
                self.assertEqual(contact_band.inspect_band_cache(sample, 8, 4, data_root=root)['status'], 'missing')
                with self.assertRaises(FileNotFoundError):
                    contact_band.load_band(sample, 8, 4, data_root=root, allow_build=False)
            self.assertFalse((root / 'data').exists())

    def test_existing_cache_metadata_and_read_only_reload(self):
        with tempfile.TemporaryDirectory(prefix='microcture-band-test-') as folder:
            root = Path(folder)
            path, *_ = self.source(root)
            sample = dict(sample_id='fixture', path='fixture.cool')
            cache = root / 'data/cache/discovery'
            cache.mkdir(parents=True)
            expected = contact_band.make_band(path, 8, 4)
            file = cache / 'fixture.npz'
            np.savez_compressed(file, **expected, signature=json.dumps(contact_band.cache_signature(path, 4)))
            before = file.read_bytes()
            with mock.patch.object(contact_band, 'make_band', side_effect=AssertionError('raw scan')):
                info = contact_band.inspect_band_cache(sample, 8, 4, data_root=root)
                actual = contact_band.load_band(sample, 8, 4, data_root=root, allow_build=False)
            self.assertEqual(info['status'], 'ready')
            self.assertFalse(info['band_pixels_loaded'])
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(before, file.read_bytes())

    def test_stale_cache_refuses_raw_fallback(self):
        with tempfile.TemporaryDirectory(prefix='microcture-band-test-') as folder:
            root = Path(folder)
            path, *_ = self.source(root)
            sample = dict(sample_id='fixture', path='fixture.cool')
            cache = root / 'data/cache/discovery'
            cache.mkdir(parents=True)
            signature = contact_band.cache_signature(path, 4)
            signature['mtime_ns'] -= 1
            np.savez_compressed(cache / 'fixture.npz', band=np.zeros((8, 4), np.float32), signature=json.dumps(signature))
            self.assertEqual(contact_band.inspect_band_cache(sample, 8, 4, data_root=root)['status'], 'stale_or_invalid')
            with mock.patch.object(contact_band, 'make_band', side_effect=AssertionError('raw scan')):
                with self.assertRaises(FileNotFoundError):
                    contact_band.load_band(sample, 8, 4, data_root=root, allow_build=False)


class LocalEntryTests(unittest.TestCase):
    def test_import_never_loads_discovery_or_torch(self):
        code = "from microcture.layout import bootstrap; bootstrap(); import contact_band, analyze_conditions, run_conditions_local; import sys; assert not any(x in sys.modules for x in ['torch', 'discover_task2', 'ml_baselines', 'matplotlib.pyplot'])"
        subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPTS)], check=True)

    def test_discovery_reexports_same_public_band_functions(self):
        tree = ast.parse(resolve('scripts/discover_task2.py').read_text())
        exports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'contact_band']
        self.assertEqual(len(exports), 1)
        self.assertEqual({x.name for x in exports[0].names}, {'make_band', 'load_band', 'extract'})
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name in {'make_band', 'load_band', 'extract'} for node in tree.body))

    def test_output_guard_blocks_external_legacy_and_existing_paths(self):
        with tempfile.TemporaryDirectory(prefix='microcture-entry-test-') as folder:
            root = Path(folder) / 'clone'
            root.mkdir()
            for value in [root.parent / 'external', root / 'reports/new', root / 'results/task3', root / 'results/task4/v2']:
                with self.assertRaises(ValueError):
                    run_conditions_local.new_run_path(value, root=root)
            path = root / 'results/task3/condition_tracks/new_v1'
            self.assertEqual(run_conditions_local.new_run_path(path, root=root), path)
            path.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_conditions_local.new_run_path(path, root=root)
            (root / 'configs').mkdir()
            (root / 'configs/layout.json').write_text(json.dumps({'moves': [{'old': 'results/task3_old', 'new': 'results/task3/condition_tracks/frozen_v1', 'kind': 'result'}]}))
            (root / 'results/task3/condition_tracks/frozen_v1').mkdir()
            with self.assertRaisesRegex(ValueError, 'frozen'):
                run_conditions_local.new_run_path(root / 'results/task3/condition_tracks/frozen_v1/extra', root=root)

    def test_output_guard_follows_symlinks_and_input_path_rejects_escape(self):
        with tempfile.TemporaryDirectory(prefix='microcture-entry-test-') as folder:
            root = Path(folder) / 'clone'
            outside = Path(folder) / 'original'
            root.mkdir()
            outside.mkdir()
            (root / 'results').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                run_conditions_local.new_run_path(root / 'results/task34_local_v1', root=root, data_root=outside)
            with self.assertRaises(ValueError):
                run_conditions_local.input_path(root, '../original/data.csv')


if __name__ == '__main__':
    unittest.main()
