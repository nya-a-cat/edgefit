"""竞品基准编排器的纯逻辑回归测试。

本文件只覆盖工具选择、重复计时聚合所依赖的比较结构与数值边界，不启动外部工具。
"""

from __future__ import annotations

import importlib.util
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
