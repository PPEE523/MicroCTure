"""Metadata-only regression checks for the input conflicts missed initially."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("task1_audit", Path(__file__).resolve().parents[1] / "scripts/audit_task1_data.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def structure(sid, kind, start, end, center):
    return {"structure_id": sid, "type": kind, "chrom": "test", "start": start,
            "end": end, "center": center, "length_bp": end - start}


class InputScopeAuditTests(unittest.TestCase):
    def test_distinct_nested_intervals_can_have_identical_center_only_inputs(self):
        rows = [structure("CHIN_1", "CHIN", 5000, 6100, 5550),
                structure("CHID_1", "CHID", 1000, 10120, 5560)]
        result = audit.annotation_findings(rows)
        self.assertEqual(len(result["cross_class_input_conflicts"]), 1)
        self.assertEqual(result["overlap_summary"]["distinct_contained_chin"], 1)
        self.assertEqual(result["class_distribution"]["CHID"]["fully_visible"]["6400"], 0)
        self.assertEqual(result["class_distribution"]["CHID"]["fully_visible"]["25600"], 1)

    def test_same_center_in_different_chromosome_or_same_class_is_not_label_conflict(self):
        rows = [structure("CHIN_1", "CHIN", 5000, 6100, 5550),
                structure("CHIN_2", "CHIN", 4900, 6200, 5550),
                {**structure("CHID_1", "CHID", 1000, 10100, 5550), "chrom": "other"}]
        result = audit.annotation_findings(rows)
        self.assertEqual(result["cross_class_input_conflicts"], [])


if __name__ == "__main__":
    unittest.main()
