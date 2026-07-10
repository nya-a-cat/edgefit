from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


class ProfileMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "profile_matrix.py"
        spec = importlib.util.spec_from_file_location("profile_matrix", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.profile_matrix = module

    def test_counts_diagnostic_severities(self) -> None:
        diagnostics = [
            {"id": "EF0101", "severity": "error"},
            {"id": "EF0104", "severity": "warning"},
            {"id": "EF0201", "severity": "error"},
        ]

        self.assertEqual(
            self.profile_matrix.count_severities(diagnostics),
            {"error": 2, "warning": 1},
        )

    def test_renders_markdown_matrix(self) -> None:
        summary = {
            "matrix": [
                {
                    "model_id": "sample-model",
                    "target_id": "sample-target",
                    "status": "pass",
                    "error_count": 0,
                    "warning_count": 1,
                    "diagnostic_ids": ["EF0104"],
                }
            ]
        }

        markdown = self.profile_matrix.render_markdown(summary)

        self.assertIn("# EdgeFit Profile Matrix", markdown)
        self.assertIn("`sample-model`", markdown)
        self.assertIn("`sample-target`", markdown)
        self.assertIn("`EF0104`", markdown)


if __name__ == "__main__":
    unittest.main()