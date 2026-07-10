"""竞品基准编排器的纯逻辑回归测试。

本文件只覆盖工具选择、重复计时聚合所依赖的比较结构与数值边界，不启动外部工具。
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_benchmark_module():
    module_path = Path(__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location("edgefit_competitive_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class BenchmarkComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_benchmark_module()

    def test_tool_selection_uses_canonical_order(self) -> None:
        selected = self.benchmark.parse_tool_selection("onnx-tool,edgefit")

        self.assertEqual(selected, ("edgefit", "onnx-tool"))

    def test_tool_selection_rejects_unknown_and_duplicate_values(self) -> None:
        with self.assertRaises(self.benchmark.InputError):
            self.benchmark.parse_tool_selection("edgefit,edgefit")
        with self.assertRaises(self.benchmark.InputError):
            self.benchmark.parse_tool_selection("edgefit,unknown")

    def test_case_selection_preserves_manifest_order(self) -> None:
        cases = [{"id": "first"}, {"id": "second"}, {"id": "third"}]

        selected = self.benchmark.select_case_specs(cases, ["third", "first"])

        self.assertEqual([item["id"] for item in selected], ["first", "third"])
        with self.assertRaises(self.benchmark.InputError):
            self.benchmark.select_case_specs(cases, ["missing"])

    def test_generated_linear_chain_is_deterministic_and_self_describing(self) -> None:
        spec = {
            "kind": "linear_relu_chain",
            "node_count": 3,
            "tensor_elements": 16,
            "dtype": "float32",
        }
        with tempfile.TemporaryDirectory() as directory:
            first, first_path = self.benchmark.prepare_generated_model(
                "scale",
                spec,
                Path(directory) / "first",
            )
            second, second_path = self.benchmark.prepare_generated_model(
                "scale",
                spec,
                Path(directory) / "second",
            )
            data = self.benchmark.read_json(first_path)

        self.assertEqual(first["model_sha256"], second["model_sha256"])
        self.assertEqual(first["model_bytes"], second["model_bytes"])
        self.assertEqual(data["model"]["file_bytes"], first["model_bytes"])
        self.assertRegex(data["model"]["sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(len(data["graph"]["nodes"]), 3)
        self.assertEqual(len(data["graph"]["values"]), 2)

    def test_performance_expectations_fail_closed_on_missing_rss(self) -> None:
        case = {
            "id": "scale",
            "expectations": {
                "max_edgefit_duration_ms": 100,
                "max_edgefit_peak_rss_bytes": 1024,
                "expected_edgefit_node_count": 3,
                "require_deterministic_artifact": True,
            },
        }
        tools = {
            "edgefit": {
                "duration_ms": 10,
                "peak_rss_bytes": None,
                "artifact_deterministic": True,
                "observations": {"node_count": 3},
            }
        }

        result = self.benchmark.evaluate_case_expectations(case, tools)

        self.assertEqual(result["status"], "fail")
        failed = [item["name"] for item in result["checks"] if not item["passed"]]
        self.assertEqual(failed, ["max_edgefit_peak_rss_bytes"])

    def test_comparison_reports_reduction_and_peak_transition(self) -> None:
        cases = [
            self.case("before", 1000, 800, "Conv", 20),
            self.case("after", 250, 600, "QLinearConv", 10),
        ]

        comparisons = self.benchmark.build_comparisons(
            [
                {
                    "id": "quantized",
                    "baseline_case": "before",
                    "candidate_case": "after",
                    "hypothesis": "smaller model",
                }
            ],
            cases,
        )

        comparison = comparisons[0]
        self.assertEqual(comparison["status"], "complete")
        self.assertEqual(
            comparison["metrics"]["model_file_bytes"]["reduction_percent"],
            75.0,
        )
        self.assertEqual(comparison["metrics"]["planned_activation_arena_bytes"]["delta"], -200)
        self.assertEqual(comparison["candidate"]["peak_op_type"], "QLinearConv")

    @staticmethod
    def case(case_id: str, file_bytes: int, arena_bytes: int, op: str, duration: int):
        observations = {
            "verdict": "pass",
            "model_file_bytes": file_bytes,
            "initializer_bytes": file_bytes,
            "estimated_peak_activation_bytes": arena_bytes,
            "planned_activation_arena_bytes": arena_bytes,
            "quantization_representation": "qoperator" if op.startswith("Q") else "none",
            "peak_activation_node_name": "peak",
            "peak_activation_node_index": 1,
            "peak_activation_op_type": op,
        }
        return {
            "id": case_id,
            "tools": {
                "edgefit": {
                    "status": "completed",
                    "duration_ms": duration,
                    "observations": observations,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
