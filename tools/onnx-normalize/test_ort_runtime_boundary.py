from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class OrtRuntimeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "ort_runtime_boundary.py"
        spec = importlib.util.spec_from_file_location("ort_runtime_boundary", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.boundary = module

    def test_parses_required_ops_and_type_constraints(self) -> None:
        text = """
# comment
ai.onnx;13;Add,Conv{"inputs": {"0": ["float"]}},Relu
!globally_allowed_types;float
com.microsoft;1;QLinearAdd
""".strip()

        parsed = self.boundary.parse_reduced_config_text(text)

        self.assertEqual(parsed[("ai.onnx", 13)], {"Add", "Conv", "Relu"})
        self.assertEqual(parsed[("com.microsoft", 1)], {"QLinearAdd"})

    def test_builds_boundary_for_toy_profile_and_model(self) -> None:
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            self.skipTest("onnx package is required")

        suffix = f"{os.getpid()}_ort_boundary"
        profile = WORKSPACE_TMP / f"{suffix}.yaml"
        corpus = WORKSPACE_TMP / f"{suffix}.json"
        model_path = WORKSPACE_TMP / f"{suffix}.onnx"
        ort_source = WORKSPACE_TMP / f"{suffix}_ort"
        fixture = ort_source / "onnxruntime" / "test" / "testdata" / "required_ops.config"
        try:
            profile.write_text(
                """
profile_version: edgefit.target.v1
metadata:
  source: sample evidence
  confidence: seed
  last_verified: 2026-07-09
target:
  id: sample_target
ops:
  allow:
    ai.onnx:
      Add:
        dtypes: [float32]
""".strip()
                + "\n",
                encoding="utf-8",
            )
            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
            y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
            z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
            node = helper.make_node("Add", ["x", "y"], ["z"])
            graph = helper.make_graph([node], "toy", [x, y], [z])
            model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
            onnx.save(model, model_path)
            corpus.write_text(
                json.dumps(
                    {
                        "schema": "edgefit.real_world_corpus.result.v1",
                        "results": [
                            {
                                "id": "toy-add",
                                "status": "pass",
                                "model_path": str(model_path),
                                "domain_ops": ["ai.onnx::Add"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text("ai.onnx;13;Add\n", encoding="utf-8")

            summary, config_text = self.boundary.build_summary(profile, corpus, ort_source)
        finally:
            profile.unlink(missing_ok=True)
            corpus.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            fixture.unlink(missing_ok=True)
            for folder in [fixture.parent, fixture.parent.parent, fixture.parent.parent.parent, ort_source]:
                try:
                    folder.rmdir()
                except OSError:
                    pass

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["profile_coverage_status"], "pass")
        self.assertEqual(summary["generated_config_roundtrip_status"], "pass")
        self.assertIn("ai.onnx;13;Add", config_text)


if __name__ == "__main__":
    unittest.main()