"""Archive regression tests use tiny temporary files, never research inputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import build_submission as builder


class SubmissionArchiveTest(unittest.TestCase):
    def test_external_output_is_rejected_before_packaging(self):
        with tempfile.TemporaryDirectory(prefix='microcture-archive-') as folder:
            root=Path(folder)/'source';root.mkdir()
            output=Path(folder)/'outside.zip'
            with mock.patch.object(builder,'ROOT',root), mock.patch.object(builder.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError, 'inside the existing MicroCTure'):
                    builder.build(output)
                run.assert_not_called()
            self.assertFalse(output.exists())

    def test_repack_does_not_include_old_manifest_or_duplicate_member(self):
        with tempfile.TemporaryDirectory(prefix='microcture-archive-') as folder:
            root=Path(folder)/'source';root.mkdir()
            code=root/'run.py';code.write_text('print(1)\n')
            previous=root/'DELIVERY_MANIFEST.json';previous.write_text('{"old":true}')
            old_guide=root/'START_HERE.md';old_guide.write_text('old guide')
            output=root/'delivery.zip'
            with mock.patch.object(builder,'ROOT',root), mock.patch.object(builder,'public_files',return_value=[code,previous,old_guide]), \
                 mock.patch.object(builder.subprocess,'run'):
                builder.build(output)
            with zipfile.ZipFile(output) as package:
                self.assertEqual(package.namelist().count('MicroCTure/DELIVERY_MANIFEST.json'),1)
                self.assertEqual(package.namelist().count('MicroCTure/START_HERE.md'),1)
                index=json.loads(package.read('MicroCTure/DELIVERY_MANIFEST.json'))
                self.assertEqual([r['path'] for r in index['files']],['run.py','START_HERE.md'])
                self.assertEqual(package.read('MicroCTure/START_HERE.md'), builder.DELIVERY_GUIDE.encode('utf-8'))

    def test_changed_source_is_rejected_and_partial_archive_removed(self):
        with tempfile.TemporaryDirectory(prefix='microcture-archive-') as folder:
            root=Path(folder)/'source';root.mkdir()
            code=root/'run.py';code.write_text('original source\n')
            output=root/'delivery.zip'
            original_read = Path.read_bytes
            reads = 0
            def changed_read(path):
                nonlocal reads
                if path == code:
                    reads += 1
                    if reads > 1:
                        return b'changed source\n'
                return original_read(path)
            with mock.patch.object(builder,'ROOT',root), mock.patch.object(builder,'public_files',return_value=[code]), \
                 mock.patch.object(builder.subprocess,'run'), mock.patch.object(Path,'read_bytes',changed_read):
                with self.assertRaisesRegex(ValueError, 'Source changed while packaging'):
                    builder.build(output)
            self.assertFalse(output.exists())

    def test_exclusive_creation_does_not_delete_another_process_file(self):
        with tempfile.TemporaryDirectory(prefix='microcture-archive-') as folder:
            root=Path(folder)/'source';root.mkdir()
            code=root/'run.py';code.write_text('pass\n')
            output=root/'delivery.zip'
            def concurrent_creation(*args,**kwargs):
                output.write_bytes(b'another process owns this file')
            with mock.patch.object(builder,'ROOT',root), mock.patch.object(builder,'public_files',return_value=[code]), \
                 mock.patch.object(builder.subprocess,'run',side_effect=concurrent_creation):
                with self.assertRaises(FileExistsError):builder.build(output)
            self.assertEqual(output.read_bytes(),b'another process owns this file')

    def test_failed_write_removes_only_its_own_partial_archive(self):
        with tempfile.TemporaryDirectory(prefix='microcture-archive-') as folder:
            root=Path(folder)/'source';root.mkdir()
            code=root/'run.py';code.write_text('pass\n')
            output=root/'delivery.zip'
            with mock.patch.object(builder,'ROOT',root), mock.patch.object(builder,'public_files',return_value=[code]), \
                 mock.patch.object(builder.subprocess,'run'), mock.patch.object(builder.zipfile.ZipFile,'writestr',side_effect=OSError('fixture')):
                with self.assertRaises(OSError):builder.build(output)
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
