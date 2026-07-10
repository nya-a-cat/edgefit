from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


class RuntimeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "runtime_smoke.py"
        spec = importlib.util.spec_from_file_location("runtime_smoke", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.smoke = module

    def test_concrete_shape_replaces_symbolic_dimensions(self) -> None:
        self.assertEqual(self.smoke.concrete_shape(["batch", 3, None, -1]), [1, 3, 1, 1])

    def test_shape_matches_keeps_fixed_dimensions_strict(self) -> None:
        self.assertTrue(self.smoke.shape_matches(["batch", 1, 672, 672], [1, 1, 672, 672]))
        self.assertFalse(self.smoke.shape_matches(["batch", 1, 672, 672], [1, 3, 672, 672]))

    def test_compare_outputs_accepts_symbolic_dimensions(self) -> None:
        mismatches = self.smoke.compare_outputs(
            [{"name": "out", "dtype": "float32", "shape": ["batch", 1000]}],
            [{"name": "out", "dtype": "float32", "shape": [1, 1000]}],
        )
        self.assertEqual(mismatches, [])

    def test_compare_outputs_reports_dtype_and_shape_mismatch(self) -> None:
        mismatches = self.smoke.compare_outputs(
            [{"name": "out", "dtype": "float32", "shape": [1, 10]}],
            [{"name": "out", "dtype": "int64", "shape": [1, 11]}],
        )
        self.assertEqual(len(mismatches), 2)

    def test_edgefit_dtype_maps_numpy_dtype(self) -> None:
        self.assertEqual(self.smoke.edgefit_dtype(np.dtype("uint8")), "uint8")
        self.assertEqual(self.smoke.edgefit_dtype(np.dtype("float32")), "float32")


if __name__ == "__main__":
    unittest.main()