"""Guard against one-base shifts and spreadsheet parsing regressions."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('prepare_data', Path(__file__).resolve().parents[1] / 'scripts/prepare_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CoordinatesTest(unittest.TestCase):
    def test_closed_interval_and_center(self):
        self.assertEqual(module.convert_interval(1, 10, 5, '1-based-closed'), (0, 10, 4))

    def test_half_open_and_derived_center(self):
        self.assertEqual(module.convert_interval(0, 10, None, '0-based-half-open'), (0, 10, 5))

    def test_invalid_coordinates(self):
        for values in [(1.5, 10, None), (10, 0, None), (0, 10, 10), (-1, 10, None)]:
            with self.assertRaises(ValueError):
                module.convert_interval(*values, '0-based-half-open')


if __name__ == '__main__':
    unittest.main()
