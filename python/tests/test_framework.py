"""Python 框架与 Rust CLI 的 canonical 结果一致性测试。"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

import edgefit


ROOT = Path(__file__).resolve().parents[2]


class FrameworkTests(unittest.TestCase):
    def test_normalized_check_matches_rust_cli(self) -> None:
        model = ROOT / "examples/models/good_tiny.edgefit.json"
        target = ROOT / "targets/esp32s3.yaml"
        python_report = edgefit.check(model, target)

        binary = ROOT / "target/debug" / ("edgefit.exe" if os.name == "nt" else "edgefit")
        completed = subprocess.run(
            [str(binary), "check", str(model), "--target", str(target), "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(python_report, json.loads(completed.stdout))

    def test_profile_validation_and_batch_preserve_order(self) -> None:
        target = edgefit.load_profile(ROOT / "targets/esp32s3.yaml")
        model = ROOT / "examples/models/good_tiny.edgefit.json"
        reports = edgefit.batch([model, model], target)

        self.assertEqual(target.target_id, "esp32s3_custom_v1")
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0], reports[1])


if __name__ == "__main__":
    unittest.main()
