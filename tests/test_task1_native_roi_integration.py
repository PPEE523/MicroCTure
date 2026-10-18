"""Independent dense/owner/partition checks; no raw scans or model training."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from microcture.task1_native_roi import data as D


def dense_area(start, end, starts, canvas):
    # Independently integrate each output/source interval, including partial
    # endpoints. Avoid the production roi_area_weights helper.
    result = np.zeros((canvas, len(starts)))
    width = (end - start) / canvas
    for output in range(canvas):
        left, right = start + output * width, start + (output + 1) * width
        for source, source_left in enumerate(starts):
            result[output, source] = max(0, min(right, source_left + 10) - max(left, source_left)) / width
    return result


def dense_reference(counts, starts, interval, good, expected, canvas):
    n = len(starts)
    i, j = np.indices((n, n))
    validity = ((abs(i - j) >= 20) & good[:, None].astype(bool) & good[None, :].astype(bool)).astype(float)
    field = np.zeros_like(counts, dtype=float)
    supported = validity > 0
    field[supported] = np.log1p(counts[supported] / expected[abs(i - j)[supported]])
    a = dense_area(*interval, starts, canvas)
    amount = a @ validity @ a.T
    fine = np.divide(a @ field @ a.T, amount, out=np.zeros_like(amount), where=amount > 0)
    cell_ids = starts // 100
    unique = sorted(set(cell_ids.tolist()))
    projection = np.zeros((n, n))
    exposures = np.array([max(0, min(st + 10, interval[1]) - max(st, interval[0])) / 10 for st in starts])
    for first in unique:
        for second in unique:
            rows = np.flatnonzero(cell_ids == first)
            columns = np.flatnonzero(cell_ids == second)
            weights = validity[np.ix_(rows, columns)] * np.outer(exposures[rows], exposures[columns])
            denominator = weights.sum()
            mean = (field[np.ix_(rows, columns)] * weights).sum() / denominator if denominator > 0 else 0
            projection[np.ix_(rows, columns)] = mean
    coarse = np.divide(a @ (projection * validity) @ a.T, amount,
                       out=np.zeros_like(amount), where=amount > 0)
    mask = np.outer(a.sum(1), a.sum(1))
    return np.stack((coarse, amount, mask)), np.stack((fine, amount, mask))


class DenseNativeChecks(unittest.TestCase):
    def fixture(self):
        n = 83
        i, j = np.indices((n, n))
        counts = ((i + j) * 3 + abs(i - j) * 5) % 13
        counts = counts.astype(float)
        counts[(i + j) % 9 == 0] = 0
        # Large near-diagonal values must never leak through physical support.
        counts[abs(i - j) < 20] = 1e7
        good = np.ones(n)
        good[25:32] = 0
        return counts, 1000 + np.arange(n) * 10, (1033, 1787), good, 1 + np.arange(n) * .02

    def test_partial_endpoints_and_nonzero_cell_origin_match_dense(self):
        counts, starts, interval, good, expected = self.fixture()
        result, facts = D.resample_matched_native(sparse.csr_matrix(counts), starts, interval, good, expected, output_size=16)
        reference_c0, reference_c1 = dense_reference(counts, starts, interval, good, expected, 16)
        np.testing.assert_allclose(result["C0"], reference_c0, atol=1e-6, rtol=0)
        np.testing.assert_allclose(result["C1"], reference_c1, atol=1e-6, rtol=0)
        self.assertEqual(result["C0"][1:].tobytes(), result["C1"][1:].tobytes())
        np.testing.assert_array_equal(result["C0"][2], 1)
        self.assertFalse(facts["native_dense_matrix_allocated"])

    def test_exact_200bp_support_boundary(self):
        counts = np.zeros((41, 41))
        counts[0, 19] = counts[19, 0] = 1e9
        counts[0, 20] = counts[20, 0] = 3
        expected = np.ones(41)
        branches, _ = D.resample_matched_native(sparse.csr_matrix(counts), np.arange(41) * 10,
                                              (0, 410), np.ones(41), expected, output_size=41)
        self.assertEqual(float(branches["C1"][0, 0, 19]), 0)
        self.assertEqual(float(branches["C1"][1, 0, 19]), 0)
        self.assertAlmostEqual(float(branches["C1"][0, 0, 20]), np.log(4), places=6)
        self.assertEqual(float(branches["C1"][1, 0, 20]), 1)

    def test_zero_counts_keep_zero_inclusive_pair_support(self):
        branches, _ = D.resample_matched_native(sparse.csr_matrix((51, 51)), np.arange(51) * 10,
                                              (3, 507), np.ones(51), np.ones(51), output_size=8)
        self.assertGreater(float(branches["C1"][1].sum()), 0)
        np.testing.assert_array_equal(branches["C0"][0], 0)
        np.testing.assert_array_equal(branches["C1"][0], 0)

    def test_both_branches_symmetric_and_diagonal_values_irrelevant(self):
        counts, starts, interval, good, expected = self.fixture()
        one, _ = D.resample_matched_native(sparse.csr_matrix(counts), starts, interval, good, expected, output_size=12)
        i, j = np.indices(counts.shape)
        counts[abs(i - j) < 20] = 0
        two, _ = D.resample_matched_native(sparse.csr_matrix(counts), starts, interval, good, expected, output_size=12)
        for alignment in ("C0", "C1"):
            np.testing.assert_allclose(one[alignment], one[alignment].transpose(0, 2, 1), atol=1e-6, rtol=0)
            self.assertEqual(one[alignment].tobytes(), two[alignment].tobytes())


class DenseCooler:
    def __init__(self, values):
        self.values = values

    def matrix(self, balance, sparse):
        assert balance is False and sparse is True
        return self

    def __getitem__(self, item):
        return sparse.csr_matrix(self.values[item])


class NativeExpectedChecks(unittest.TestCase):
    def test_fft_matches_direct_circular_including_unobserved_pairs(self):
        mask = np.zeros(120, dtype=bool)
        mask[[0, 1, 3, 4, 19, 23, 42, 64, 65, 116, 117, 119]] = True
        opportunities = D.fft_pair_opportunities(mask, max_distance=35)
        direct = [sum(bool(mask[i]) and bool(mask[(i + distance) % len(mask)]) for i in range(len(mask)))
                  for distance in range(36)]
        np.testing.assert_array_equal(opportunities, direct)
        self.assertEqual(opportunities[0], int(mask.sum()))

    def test_upper_row_owners_count_each_pair_once_and_filter_endpoints(self):
        n = 140
        mask = np.zeros(n, dtype=bool)
        mask[8:30] = True
        mask[45:98] = True
        boundaries = np.array([[8, 30], [45, 98]])
        i, j = np.indices((n, n))
        values = ((i + j + abs(i - j)) % 7).astype(float)
        values[~(mask[:, None] & mask[None, :])] = 1e8
        expected = np.zeros(36)
        nnz = 0
        for first in range(n):
            for second in range(first + 20, min(n, first + 36)):
                if mask[first] and mask[second]:
                    expected[second - first] += values[first, second]
                    nnz += int(values[first, second] != 0)
        for owner_rows in (7, 50, 1000):
            sums, facts = D.owned_native_counts(DenseCooler(values), mask, boundaries,
                                              max_distance=35, owner_rows=owner_rows)
            np.testing.assert_array_equal(sums, expected)
            self.assertEqual(facts["eligible_unique_pair_nnz"], nnz)
            self.assertEqual(facts["eligible_counts_sum"], float(expected.sum()))

    def test_cross_origin_pairs_are_explicitly_rejected(self):
        mask = np.zeros(100, dtype=bool)
        mask[4:8] = True
        mask[80:84] = True
        self.assertTrue(D.circular_wrap_pair_guard(mask, max_distance=35))
        with self.assertRaises(ValueError):
            D.owned_native_counts(DenseCooler(np.ones((100, 100))), mask, np.array([[4, 8], [80, 84]]), max_distance=35)

    def test_membership_lift_uses_inner_ids_and_terminal_coarse_mask(self):
        dataset = {"config": {"resolution_bp": 100, "window_bins": [64, 256], "output_size": 64},
                   "metadata": {"chrom_length": 352}, "structure_id": np.array(["train", "val", "outer"]),
                   "bin_occupancy": np.array([1, 1, 1, 0])}
        frozen = {"train_ids": ["train"], "validation_ids": ["val"], "test_ids": ["outer"]}

        def occupancy(_dataset, indices):
            self.assertEqual(indices.tolist(), [0])
            return np.array([1, 0, 0, 1], dtype=bool)

        with patch.object(D.original, "_assert_disjoint") as disjoint, patch.object(D.original, "_occupancy", side_effect=occupancy):
            mask, boundaries = D.training_native_mask(dataset, frozen)
        self.assertEqual(disjoint.call_count, 2)
        np.testing.assert_array_equal(mask, np.r_[np.ones(10, dtype=bool), np.zeros(26, dtype=bool)])
        np.testing.assert_array_equal(boundaries, [[0, 10]])


class NativeWrapperChecks(unittest.TestCase):
    def samples(self, sid):
        return {"x": np.zeros((2, 2, 3, 64, 64), dtype=np.float32), "y": np.array([1, 1]),
                "length": np.array([[.37], [.37]], dtype=np.float32), "structure_id": np.array([sid, sid]),
                "replicate": np.array([1, 2]), "group_id": np.array([3, 3]), "sample_id": np.array(["rep1", "rep2"])}

    def test_matched_inner_branches_use_train_scalers_without_context_or_length_change(self):
        train, validation = self.samples("T"), self.samples("V")
        train["x"][:, 1] = 13
        validation["x"][:, 1] = 17
        base = {"signature": "base", "dataset_signature": "dataset", "frozen_fold_signature": "fold", "train_ids": ["T"]}
        shared = {"signature": "native expected fixture"}
        dataset = {"signature": "dataset"}
        frozen = {"fold": 0}

        def replace(_dataset, samples, _shared):
            result = {}
            for number, alignment in enumerate(("C0", "C1")):
                x = samples["x"].copy()
                x[:, 0, 0] = (2 + number) if samples["structure_id"][0] == "T" else (50 + number)
                x[:, 0, 1:] = 1
                result[alignment] = {**samples, "x": x}
            return result

        old_memo = D._INNER_MEMO
        D._INNER_MEMO = None
        try:
            with tempfile.TemporaryDirectory(prefix="microcture-native-integration-", dir="/tmp") as cache, \
                 patch.object(D.formal, "prepare_inner", return_value=(train, validation, base)), \
                 patch.object(D, "prepare_shared_expected", return_value=shared), \
                 patch.object(D, "native_source_records", return_value=[]), \
                 patch.object(D, "_expected_binding", return_value={}), \
                 patch.object(D, "_verify_expected"), \
                 patch.object(D, "_replace_native", side_effect=replace):
                c0, v0, p0 = D.prepare_native_inner(dataset, frozen, "C0", cache_dir=cache)
                c1, v1, p1 = D.prepare_native_inner(dataset, frozen, "C1", cache_dir=cache)
        finally:
            D._INNER_MEMO = old_memo
        self.assertEqual(p0["native_first_signal_scaler"]["mean"], 2)
        self.assertEqual(p1["native_first_signal_scaler"]["mean"], 3)
        self.assertEqual(c0["x"][:, 0, 1:].tobytes(), c1["x"][:, 0, 1:].tobytes())
        for samples in (c0, c1):
            self.assertEqual(samples["x"][:, 1].tobytes(), train["x"][:, 1].tobytes())
            self.assertEqual(samples["length"].tobytes(), train["length"].tobytes())
        self.assertTrue((v0["x"][:, 0, 0] > 0).all() and (v1["x"][:, 0, 0] > 0).all())


if __name__ == "__main__":
    unittest.main()
