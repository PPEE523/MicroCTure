"""Preserved band caches never trigger implicit scans or stale-file replacement."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from microcture.layout import bootstrap

bootstrap()
import contact_band


class ReadOnlyContactCacheTest(unittest.TestCase):
    def test_stale_cache_is_preserved_even_with_the_legacy_build_flag(self):
        with TemporaryDirectory(prefix="microcture-stale-band-", dir="/tmp") as folder:
            root = Path(folder)
            source = root / "fixture.cool"
            source.write_bytes(b"metadata-only fixture; must never scan")
            sample = {"sample_id": "fixture", "path": source.name}
            cache = root / "data/cache/discovery/fixture.npz"
            cache.parent.mkdir(parents=True)
            stale = contact_band.cache_signature(source, 4)
            stale["mtime_ns"] -= 1
            np.savez_compressed(cache, band=np.arange(32).reshape(8, 4),
                                signature=json.dumps(stale))
            before = cache.read_bytes()
            with mock.patch.object(contact_band, "make_band") as scan:
                with self.assertRaisesRegex(FileNotFoundError, "fresh cache version"):
                    contact_band.load_band(sample, 8, 4, data_root=root, allow_build=True)
                scan.assert_not_called()
            self.assertEqual(cache.read_bytes(), before)

    def test_missing_cache_fails_without_a_scan_or_output_directory(self):
        with TemporaryDirectory(prefix="microcture-missing-band-", dir="/tmp") as folder:
            root = Path(folder)
            source = root / "fixture.cool"
            source.write_bytes(b"must never scan")
            sample = {"sample_id": "fixture", "path": source.name}
            for options in ({}, {"allow_build": True}):
                with self.subTest(options=options):
                    with mock.patch.object(contact_band, "make_band") as scan:
                        with self.assertRaisesRegex(FileNotFoundError, "no raw scan"):
                            contact_band.load_band(sample, 8, 4, data_root=root, **options)
                        scan.assert_not_called()
                    self.assertFalse((root / "data").exists())


if __name__ == "__main__":
    unittest.main()
