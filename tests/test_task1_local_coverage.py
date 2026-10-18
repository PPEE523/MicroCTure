"""Upper-owner diagnostics: exact circular bp distances and no reflection."""
import unittest
import numpy as np
from scripts.probe_task1_local_coverage import physical_circular_distance, indexed_upper_rows, continuous_runs


class LocalCoverageTests(unittest.TestCase):
    def test_terminal_two_bp_bin_uses_real_chromosome_length(self):
        # Bins [0,10),[10,20),[20,30),[30,32); last center is31.
        np.testing.assert_array_equal(physical_circular_distance(0, np.array([0, 1, 2, 3]), 32), [0, 10, 12, 6])

    def test_sparse_upper_row_diagonal_once_and_lower_never_assumed_zero(self):
        handle = {"indexes/bin1_offset": np.array([0, 2, 3, 4, 5]),
                  "pixels/bin1_id": np.array([0, 0, 1, 2, 3]),
                  "pixels/bin2_id": np.array([0, 3, 1, 3, 3]),
                  "pixels/count": np.array([5, 7, 2, 11, 13])}
        result, facts = indexed_upper_rows(handle, np.array([0, 2]), 32)
        self.assertEqual(result[0]["upper_count_total"], 12)
        self.assertEqual(result[0]["diagonal_count_once"], 5)
        self.assertEqual(result[2]["upper_count_total"], 11)
        self.assertEqual(result[2]["lower_contribution"], "not_measured")
        self.assertIsNone(result[2]["full_genome_row_total"])
        self.assertEqual(facts["stored_pixels_read"], 3)
        self.assertEqual(facts["reflected_pixels_generated"], 0)

    def test_200_and_25600_bp_included_in_middle_category(self):
        handle = {"indexes/bin1_offset": np.array([0, 4]), "pixels/bin1_id": np.array([0, 0, 0, 0]),
                  "pixels/bin2_id": np.array([0, 20, 2560, 2561]), "pixels/count": np.array([1, 2, 3, 4])}
        result, _ = indexed_upper_rows(handle, np.array([0]), 100000)
        self.assertEqual(result[0]["upper_count_lt_200bp"], 1)
        self.assertEqual(result[0]["upper_count_200_to_25600bp"], 5)
        self.assertEqual(result[0]["upper_count_gt_25600bp"], 4)

    def test_runs_read_only_requested_unique_rows(self):
        self.assertEqual(continuous_runs([6, 3, 2, 2, 8]), [(2, 4), (6, 7), (8, 9)])


if __name__ == "__main__":
    unittest.main()
