"""Publication guards use synthetic bytes and never access research inputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('publication_audit', Path(__file__).resolve().parents[1] / 'scripts/project/audit_public_release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class PublicationAuditTests(unittest.TestCase):
    def test_tokens_are_reported_as_counts_without_values(self):
        token = b'ghp_' + b'A' * 36
        result = release.scan(token)
        self.assertEqual(result, {'github_token': 1})
        self.assertNotIn(token.decode(), str(result))

    def test_student_number_is_detected_but_genome_coordinates_are_not(self):
        number = b'3024' + b'123456'
        self.assertEqual(release.scan(number)['student_id'], 1)
        self.assertNotIn('student_id', release.scan(b'4641652 3439610 3448700'))

    def test_quoted_password_assignment_is_flagged_without_exposing_value(self):
        payload=b'{"PASSWORD":"'+b'fixture'+b'-value-123"}'
        result=release.scan(payload)
        self.assertEqual(result,{'plain_credential_assignment':1})
        self.assertNotIn('value-123',str(result))

    def test_provenance_is_separate_from_blocking_credentials(self):
        result = release.scan(b'{"source":"/home/example/project/input.csv"}')
        self.assertEqual(result, {'local_path_provenance': 1})
        self.assertFalse(set(result) & release.PATTERNS.keys())

    def test_unsafe_paths_are_rejected(self):
        for name in ('../source', '/tmp/source', 'data\\secret', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                release.safe_relative(name)

    def test_untracking_plan_preserves_local_file(self):
        with tempfile.TemporaryDirectory(prefix='microcture-public-fixture-') as folder:
            root = Path(folder)
            private = root / 'report.docx'
            private.write_bytes(b'local report')
            with patch.object(release, 'tracked_entries', return_value={'report.docx': {'mode':'100644','blob':'unused'}}):
                rows, remove, findings = release.plan(root, candidates=[])
            self.assertEqual(rows, [])
            self.assertTrue(remove[0]['exists_locally'])
            self.assertEqual(private.read_bytes(), b'local report')

    def test_internal_symlink_is_also_rejected(self):
        with tempfile.TemporaryDirectory(prefix='microcture-public-fixture-') as folder:
            root = Path(folder)
            original = root / 'source.py'; original.write_text('pass\n')
            alias = root / 'alias.py'; alias.symlink_to(original)
            with patch.object(release, 'tracked_entries', return_value={}):
                with self.assertRaisesRegex(ValueError, 'symlinks'):
                    release.plan(root, candidates=[alias])

    def test_report_in_explicit_candidates_blocks_publication(self):
        with tempfile.TemporaryDirectory(prefix='microcture-public-fixture-') as folder:
            root=Path(folder)
            report=root/'private.docx';report.write_bytes(b'local report')
            with patch.object(release,'tracked_entries',return_value={}):
                rows,remove,findings=release.plan(root,candidates=[report])
            self.assertEqual(findings[0]['findings']['forbidden_public_file'],1)
            self.assertTrue(report.exists())

    def test_output_traversal_cannot_escape_local_reports(self):
        with tempfile.TemporaryDirectory(prefix='microcture-public-fixture-') as folder:
            root=Path(folder)
            with patch.object(release,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'fresh local reports'):
                    release.audit(root/'reports/../data/unsafe')


if __name__ == '__main__':
    unittest.main()
