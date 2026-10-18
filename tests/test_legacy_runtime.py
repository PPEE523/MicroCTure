"""Synthetic checks for safe shared CLI paths and extracted numeric helpers."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from optional_common import fresh_output_path, input_path
from spatial_tools import assert_disjoint, pool_masked, distance


class RuntimeGuardTests(unittest.TestCase):
    def test_fresh_output_rejects_external_legacy_existing_and_symlink(self):
        with tempfile.TemporaryDirectory(prefix='microcture-runtime-test-') as folder:
            base = Path(folder)
            root, data = base / 'clone', base / 'input'
            root.mkdir()
            data.mkdir()
            for path in [data / 'results/task5_v2', root / 'reports/task6_v2', root / 'results/task2', root / 'results/task2_v2', root / 'results/task5']:
                with self.assertRaises(ValueError):
                    fresh_output_path(path, root=root, data_root=data)
            with self.assertRaises(ValueError):
                fresh_output_path(root / 'results/task5_replay_v2', root=root, data_root=data)
            new = root / 'results/task5/super_resolution/replay_v2'
            self.assertEqual(fresh_output_path(new, root=root, data_root=data), new)
            new.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                fresh_output_path(new, root=root, data_root=data)
            (root / 'results/redirect').symlink_to(data, target_is_directory=True)
            with self.assertRaises(ValueError):
                fresh_output_path(root / 'results/redirect/task6_v2', root=root, data_root=data)

    def test_preserved_input_lookup_is_read_only_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory(prefix='microcture-runtime-test-') as folder:
            base = Path(folder)
            old = base / 'data/legacy_experiments/results/task2/scan_rep1.csv'
            old.parent.mkdir(parents=True)
            old.write_text('window_id\nfixture\n')
            self.assertEqual(input_path(base, 'results/task2/scan_rep1.csv'), old)
            self.assertFalse((base / 'results').exists())
            for name in ['../outside.csv', str(old.resolve())]:
                with self.assertRaises(ValueError):
                    input_path(base, name)
            with self.assertRaises(FileNotFoundError):
                input_path(base, 'data/cache/missing.npz')

    def test_four_cli_help_and_no_arguments_never_enter_calculation(self):
        import discover_task2
        import discover_task2_v2
        import train_task5
        import train_task6
        for module in [discover_task2, discover_task2_v2, train_task5, train_task6]:
            with self.subTest(module=module.__name__), mock.patch.object(module, 'run_experiment', side_effect=AssertionError('calculation')), mock.patch.object(module, 'temporary_plotting', side_effect=AssertionError('plotting')), contextlib.redirect_stdout(io.StringIO()):
                module.main([])
                with self.assertRaises(SystemExit) as raised:
                    module.main(['--help'])
                self.assertEqual(raised.exception.code, 0)

    def test_explicit_scan_and_band_cache_are_bound_before_computation(self):
        import discover_task2
        import discover_task2_v2
        import train_task5
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        root = scripts.parent
        class StopBeforeCalculation(Exception):
            pass
        with tempfile.TemporaryDirectory(prefix='microcture-runtime-test-') as folder:
            base = Path(folder)
            data, bands = base / 'data', base / 'fresh_bands'
            structures = data / 'data/processed/structures.csv'
            structures.parent.mkdir(parents=True)
            structures.write_text('structure_id,type,start,end\n')
            scan = base / 'fresh_v1/scan_rep1.csv'
            split = base / 'frozen_split.csv'
            for module, config, signature_name in [(discover_task2,'discovery.json','signatures'),
                                                    (discover_task2_v2,'discovery_v2.json','signatures'),
                                                    (train_task5,'optional_tasks.json','file_signatures')]:
                args = SimpleNamespace(data_root=data, output_dir=base/'never_created', config=root/'configs'/config,
                                       band_cache_dir=bands, scan_file=scan, split_file=split)
                with self.subTest(module=module.__name__), mock.patch.object(module,signature_name,side_effect=StopBeforeCalculation) as signature:
                    with self.assertRaises(StopBeforeCalculation):
                        module.run_experiment(args)
                    inputs = signature.call_args.args[1]
                    self.assertIn(bands/'WT_37C_rep1.npz',inputs)
                    self.assertIn(bands/'WT_37C_rep2.npz',inputs)
                    if module is discover_task2_v2:
                        self.assertIn(scan,inputs)
                    if module is train_task5:
                        self.assertIn(split,inputs)
                    self.assertFalse(args.output_dir.exists())


class SharedSpatialTests(unittest.TestCase):
    def test_pooling_matches_manual_valid_means_and_fraction(self):
        values = np.arange(16, dtype=float).reshape(4, 4)
        mask = np.ones((4, 4), bool)
        mask[0, 0] = False
        mask[2:, 2:] = False
        result = pool_masked(values, mask, 2)
        np.testing.assert_allclose(result[0], [[(1+4+5)/3, (2+3+6+7)/4], [(8+9+12+13)/4, 0]])
        np.testing.assert_allclose(result[1], [[.75, 1], [1, 0]])
        self.assertEqual(result.dtype, np.dtype('float32'))

    def test_circular_distance_and_partition_guard(self):
        np.testing.assert_array_equal(distance(np.array([0, 1]), np.array([9, 5]), 10), [1, 4])
        footprints = [np.array([0, 1]), np.array([5, 6]), np.array([9, 10])]
        assert_disjoint(footprints, [0, 1, 2], 20, 1)
        with self.assertRaises(ValueError):
            assert_disjoint([np.array([0]), np.array([19])], [0, 1], 20, 1)


if __name__ == '__main__':
    unittest.main()
