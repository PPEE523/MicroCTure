"""Small synthetic checks only: no real raw scan or model training."""
from pathlib import Path
import hashlib
import os
import tempfile
import unittest

import h5py
import numpy as np

from microcture.task1 import data


class TaskOneDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="microcture-data-check-", dir="/tmp")
        directory = Path(cls.temporary.name)
        cls.config = data._config({"split_search_trials": 3})
        cls.n_bins = 2400
        length = cls.n_bins * 100 - 48
        coarse_a = np.repeat(np.arange(cls.n_bins), 257)
        distance = np.tile(np.arange(257), cls.n_bins)
        coarse_b = (coarse_a + distance) % cls.n_bins
        counts = (distance + 1).astype(np.int64)
        raw_n_bins = (length + 9) // 10
        source = directory / "synthetic.cool"
        with h5py.File(source, "w") as handle:
            handle.attrs.update({"storage-mode": "symmetric-upper", "bin-size": 10,
                                 "nnz": len(counts), "sum": int(counts.sum())})
            handle["chroms/name"] = np.array([b"test"])
            handle["chroms/length"] = [length]
            starts = np.arange(raw_n_bins) * 10
            handle["bins/start"] = starts
            handle["bins/end"] = np.minimum(starts + 10, length)
            handle["bins/chrom"] = np.zeros(raw_n_bins, dtype=int)
            handle["pixels/bin1_id"] = np.minimum(coarse_a, coarse_b) * 10
            handle["pixels/bin2_id"] = np.maximum(coarse_a, coarse_b) * 10
            handle["pixels/count"] = counts
        cls.total = int(counts.sum())
        cls.audit = data._raw_band(source, directory / "band.npy",
                                   {"chrom_length": length, "bin_size": 10, "chrom": "test"}, cls.config)
        cls.band = np.load(directory / "band.npy", mmap_mode="r")
        centers = np.array([500, 950, 1750, 10])
        groups, footprints = data._group_windows(centers, cls.n_bins, 256, 10)
        cls.dataset = {"n_bins": cls.n_bins, "config": cls.config, "center_bin": centers,
                       "group_id": groups, "window_footprints": footprints,
                       "bands": [cls.band, cls.band], "bin_occupancy": np.r_[np.ones(cls.n_bins - 1), 0.],
                       "structure_id": np.array(["A", "B", "C", "D"]), "y": np.array([0, 1, 2, 2]),
                       "replicate": np.array([1, 2]), "signature": "synthetic",
                       "metadata": {"samples": [{"sample_id": "rep1"}, {"sample_id": "rep2"}]}}

    @classmethod
    def tearDownClass(cls):
        del cls.band
        cls.dataset.clear()
        cls.temporary.cleanup()

    def test_circular_band_conserves_upper_storage_sum(self):
        self.assertEqual(self.audit["full_stored_count_sum"], self.total)
        self.assertEqual(self.audit["band_stored_count_sum"], self.total)
        np.testing.assert_array_equal(self.band[:, 50], np.full(self.n_bins, 51))
        # Wrapped original upper pixel (30, 2380) is canonicalised to (2380, 50).
        self.assertEqual(self.band[2380, 50], 51)
        self.assertEqual(self.band[0, 0], 1)

    def test_origin_grouping_and_leakage_rejection(self):
        groups, _ = data._group_windows(np.array([10, 2390, 700]), self.n_bins, 256, 10)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        with self.assertRaisesRegex(ValueError, "Spatial groups"):
            data._assert_disjoint({**self.dataset, "group_id": np.array([0, 0, 1, 2])},
                                  np.array([0]), np.array([1]))

    def test_heldout_changes_do_not_affect_fit_or_training_inputs(self):
        training, heldout, preprocessing = data.prepare_fold(self.dataset, ["A", "B"], ["C", "D"], self.config)
        modified = np.array(self.band)
        heldout_bins = data._occupancy(self.dataset, np.array([2, 3]))
        for distance in range(257):
            modified[heldout_bins & np.roll(heldout_bins, -distance), distance] *= 9
        changed = {**self.dataset, "bands": [modified, modified]}
        training2, heldout2, preprocessing2 = data.prepare_fold(changed, ["B", "A"], ["C", "D"], self.config)
        self.assertEqual(preprocessing["signature"], preprocessing2["signature"])
        np.testing.assert_array_equal(training["x"], training2["x"])
        self.assertFalse(np.array_equal(heldout["x"], heldout2["x"]))
        self.assertEqual(training["x"].shape, (4, 2, 2, 64, 64))
        self.assertTrue(np.isfinite(heldout["x"]).all())
        np.testing.assert_array_equal(heldout["x"], data.transform_heldout(self.dataset, ["C", "D"], preprocessing)["x"])

    def test_partial_terminal_bin_is_masked_in_fit_and_inputs(self):
        training, _, preprocessing = data.prepare_fold(self.dataset, ["A", "D"], ["B", "C"], self.config)
        modified = np.array(self.band)
        terminal = self.n_bins - 1
        for distance in range(257):
            modified[terminal, distance] *= 10_000
            modified[(terminal - distance) % self.n_bins, distance] *= 10_000
        changed = {**self.dataset, "bands": [modified, modified]}
        training2, _, preprocessing2 = data.prepare_fold(changed, ["A", "D"], ["B", "C"], self.config)
        self.assertEqual(preprocessing["signature"], preprocessing2["signature"])
        np.testing.assert_array_equal(training["x"], training2["x"])
        train_bins = data._occupancy(self.dataset, np.array([0, 3]))
        train_bins[-1] = False
        for distance in (2, 50, 256):
            opportunities = np.count_nonzero(train_bins & np.roll(train_bins, -distance))
            self.assertEqual(preprocessing["expected"][0]["zero_inclusive_pair_opportunities"][distance], opportunities)
        # D's origin-spanning 256-bin view includes terminal bin at row 117.
        sample = np.flatnonzero(training["structure_id"] == "D")[0]
        self.assertTrue(np.all(training["x"][sample, 1, 1, 117 // 4] <= 0.75))

    def test_unused_distance_256_does_not_reject_single_window(self):
        training, heldout, preprocessing = data.prepare_fold(self.dataset, ["A"], ["B"], self.config)
        self.assertEqual(preprocessing["expected"][0]["zero_inclusive_pair_opportunities"][256], 0)
        self.assertEqual(preprocessing["expected"][0]["expected"][256], 0)
        self.assertTrue(np.isfinite(training["x"]).all())
        self.assertTrue(np.isfinite(heldout["x"]).all())

    def test_loading_rejects_changed_sources_and_band(self):
        with tempfile.TemporaryDirectory(prefix="microcture-cache-check-", dir="/tmp") as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            source_files = [root / "data.json", root / "structures.csv", root / "annotations.xlsx"]
            for source in source_files:
                source.write_bytes(b"synthetic source")
            raw_source = root / "raw.cool"
            raw_source.write_bytes(b"synthetic raw identity; no pixels scanned")
            band_path = cache / "band.npy"
            np.save(band_path, self.band)
            sample = {"sample_id": "rep1", "replicate": 1, "band_file": "band.npy",
                      "band_shape": list(self.band.shape), "band_size_bytes": band_path.stat().st_size,
                      "band_mtime_ns": band_path.stat().st_mtime_ns, "raw_path": str(raw_source),
                      "raw_size_bytes": raw_source.stat().st_size, "raw_mtime_ns": raw_source.stat().st_mtime_ns}
            metadata = {"format_version": data.FORMAT_VERSION, "project_root": str(root), "config": self.config,
                        "structures": [{"structure_id": "A", "type": "OPCID"}], "center_bin": [500], "group_id": [0],
                        "n_bins": self.n_bins, "chrom_length": self.n_bins * 100 - 48,
                        "label_order": data.LABEL_ORDER, "samples": [sample],
                        "input_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in source_files}}
            metadata["signature"] = data._json_hash({"metadata": metadata, "folds": []})
            data._write_json(cache / "metadata.json", metadata)
            data._write_json(cache / "folds.json", [])
            loaded = data.load_dataset(cache)
            self.assertEqual(loaded["signature"], metadata["signature"])
            loaded.clear()
            source_files[2].write_bytes(b"modified annotation workbook")
            with self.assertRaisesRegex(ValueError, "annotation/config source changed"):
                data.load_dataset(cache)
            source_files[2].write_bytes(b"synthetic source")
            raw_stat = raw_source.stat()
            os.utime(raw_source, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns + 1_000_000))
            with self.assertRaisesRegex(ValueError, "Raw matrix source changed"):
                data.load_dataset(cache)
            os.utime(raw_source, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns))
            band_stat = band_path.stat()
            os.utime(band_path, ns=(band_stat.st_atime_ns, band_stat.st_mtime_ns + 1_000_000))
            with self.assertRaisesRegex(ValueError, "Cached numeric band changed"):
                data.load_dataset(cache)


if __name__ == "__main__":
    unittest.main()
