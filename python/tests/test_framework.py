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

    def test_optimization_matches_rust_cli(self) -> None:
        model = ROOT / "examples/models/virtual_npu_tiny.edgefit.json"
        target = ROOT / "targets/virtual-npu.yaml"
        python_plan = edgefit.optimize(model, target)

        binary = ROOT / "target/debug" / ("edgefit.exe" if os.name == "nt" else "edgefit")
        completed = subprocess.run(
            [str(binary), "optimize", str(model), "--target", str(target), "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(python_plan, json.loads(completed.stdout))
        self.assertEqual(python_plan["status"], "pass")
        self.assertEqual(len(python_plan["segments"]), 1)

    def test_failed_optimization_plan_matches_rust_cli(self) -> None:
        model = ROOT / "examples/models/virtual_npu_spill.edgefit.json"
        target = ROOT / "targets/virtual-npu-no-spill.yaml"
        python_plan = edgefit.optimize(model, target)

        binary = ROOT / "target/debug" / ("edgefit.exe" if os.name == "nt" else "edgefit")
        completed = subprocess.run(
            [str(binary), "optimize", str(model), "--target", str(target), "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(python_plan, json.loads(completed.stdout))
        self.assertEqual(python_plan["schema"], "edgefit.optimization_plan.v1")
        self.assertEqual(python_plan["status"], "fail")

    def test_optimization_rejects_invalid_format_and_target(self) -> None:
        model = ROOT / "examples/models/virtual_npu_tiny.edgefit.json"
        target = ROOT / "targets/virtual-npu.yaml"
        with self.assertRaisesRegex(edgefit.EdgeFitError, "json or markdown"):
            edgefit.render_optimization(model, target, format="text")

        valid_target = edgefit.load_profile(target)
        invalid_target = type(valid_target)(
            path=Path("invalid-target.yaml"),
            text=valid_target.text.replace("id: generic-npu-v1", "id:"),
            target_id=valid_target.target_id,
        )
        with self.assertRaisesRegex(edgefit.EdgeFitError, "accelerator.id"):
            edgefit.optimize(model, invalid_target)


if __name__ == "__main__":
    unittest.main()
