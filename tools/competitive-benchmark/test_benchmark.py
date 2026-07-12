"""竞品基准编排器的纯逻辑回归测试。

本文件只覆盖工具选择、重复计时聚合所依赖的比较结构与数值边界，不启动外部工具。
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def test_generated_int8_optimizer_chain_is_deterministic(self) -> None:
        spec = {
            "kind": "linear_op_chain",
            "node_count": 3,
            "tensor_elements": 16,
            "dtype": "int8",
            "op_type": "HardSwish",
        }
        with tempfile.TemporaryDirectory() as directory:
            first, first_path = self.benchmark.prepare_generated_model(
                "optimizer-scale",
                spec,
                Path(directory) / "first",
            )
            second, second_path = self.benchmark.prepare_generated_model(
                "optimizer-scale",
                spec,
                Path(directory) / "second",
            )
            data = self.benchmark.read_json(first_path)

        self.assertEqual(first["model_sha256"], second["model_sha256"])
        self.assertEqual(first["model_bytes"], second["model_bytes"])
        self.assertEqual(data["graph"]["inputs"][0]["dtype"], "int8")
        self.assertTrue(
            all(node["op_type"] == "HardSwish" for node in data["graph"]["nodes"])
        )

    def test_optimizer_plan_parser_and_expectations(self) -> None:
        plan = self.valid_plan()
        case = {
            "id": "optimizer",
            "expectations": {
                "expected_plan_status": "pass",
                "expected_plan_assignment_count": 1,
                "expected_plan_assignment_counts": {"npu": 1},
                "min_plan_assignment_counts": {"npu": 1},
                "expected_plan_segment_count": 1,
                "expected_plan_event_kind_counts": {
                    "load": 1,
                    "reload": 1,
                    "spill": 1,
                    "store": 1,
                },
                "required_plan_event_kinds": ["load", "reload", "spill", "store"],
                "min_plan_event_kind_counts": {"load": 1, "spill": 1},
                "min_plan_spill_bytes": 1,
                "min_plan_transfer_bytes": 1,
                "max_plan_peak_scratchpad_bytes": 128,
                "expected_plan_blockers": [],
                "expected_plan_recipe_ids": ["recipe.hardswish.v1"],
                "require_plan_latency_improvement": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            observations = self.benchmark.parse_edgefit_plan("", path)

        result = self.benchmark.evaluate_case_expectations(
            case,
            {"edgefit": {"observations": observations}},
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(observations["assignment_device_counts"], {"npu": 1})
        self.assertEqual(observations["event_kind_counts"]["spill"], 1)
        self.assertTrue(observations["latency_improved"])
        self.assertEqual(observations["target_id"], "virtual-npu")

    def test_optimizer_plan_parser_rejects_bool_as_int(self) -> None:
        plan = self.valid_plan()
        plan["events"][0]["bytes"] = True

        self.assert_plan_rejected(plan)

    def test_optimizer_plan_parser_rejects_bad_assignment(self) -> None:
        plan = self.valid_plan()
        plan["assignments"][0]["node_index"] = 1

        self.assert_plan_rejected(plan)

    def test_optimizer_plan_parser_rejects_bad_segment(self) -> None:
        plan = self.valid_plan()
        plan["segments"][0]["last_node"] = 1

        self.assert_plan_rejected(plan)

    def test_optimizer_plan_parser_rejects_bad_event(self) -> None:
        plan = self.valid_plan()
        plan["events"][0]["kind"] = "copy"

        self.assert_plan_rejected(plan)

    def test_optimizer_plan_parser_rejects_metric_mismatch(self) -> None:
        for field in ("transfer_bytes", "transfer_ns", "spill_bytes", "latency_ns"):
            with self.subTest(field=field):
                plan = self.valid_plan()
                plan["proposed"][field] += 1
                self.assert_plan_rejected(plan)

    def test_optimizer_plan_parser_rejects_pass_with_blockers(self) -> None:
        plan = self.valid_plan()
        plan["blockers"] = ["node:0 unsupported"]
        plan["proposed"]["blockers"] = 1

        self.assert_plan_rejected(plan)

    def test_optimizer_plan_parser_rejects_malformed_hash(self) -> None:
        plan = self.valid_plan()
        plan["plan_hash"] = "fnv1a64:not-hex"

        self.assert_plan_rejected(plan)

    def test_run_tool_rejects_inconsistent_repetition_exit_codes(self) -> None:
        processes = [self.finished_process(0), self.finished_process(1)]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.benchmark, "run_process", side_effect=processes):
                result = self.benchmark.run_tool(
                    "edgefit",
                    ["edgefit", "optimize"],
                    Path(directory),
                    1,
                    {0, 1},
                    lambda _output, _artifact: {"parsed": True},
                    repetitions=2,
                )

        self.assertEqual(result["status"], "runner_error")
        self.assertIn("inconsistent exit codes", result["detail"])
        self.assertEqual(result["observations"], {})

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

    def assert_plan_rejected(self, plan: dict) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(self.benchmark.InputError):
                self.benchmark.parse_edgefit_plan("", path)

    @staticmethod
    def valid_plan() -> dict:
        events = [
            {"kind": "load", "tensor": "x", "bytes": 32, "at_node": 0, "latency_ns": 20},
            {"kind": "spill", "tensor": "x", "bytes": 64, "at_node": 0, "latency_ns": 30},
            {"kind": "reload", "tensor": "x", "bytes": 64, "at_node": 0, "latency_ns": 40},
            {"kind": "store", "tensor": "y", "bytes": 32, "at_node": 0, "latency_ns": 110},
        ]
        return {
            "schema": "edgefit.optimization_plan.v1",
            "status": "pass",
            "model_sha256": "sha256:model",
            "target_id": "virtual-npu",
            "target_fingerprint": "fnv1a64:target",
            "accelerator_id": "virtual-npu",
            "confidence": "seed-simulated",
            "baseline": {"blockers": 0, "latency_ns": 1000},
            "proposed": {
                "blockers": 0,
                "latency_ns": 500,
                "launch_ns": 100,
                "compute_ns": 200,
                "transfer_ns": sum(item["latency_ns"] for item in events),
                "transfer_bytes": sum(item["bytes"] for item in events),
                "spill_bytes": sum(
                    item["bytes"] for item in events if item["kind"] == "spill"
                ),
                "peak_scratchpad_bytes": 96,
            },
            "assignments": [
                {
                    "node_index": 0,
                    "op_type": "HardSwish",
                    "device": "npu",
                    "kernel_id": "npu.hardswish.v1",
                    "recipe_id": "recipe.hardswish.v1",
                    "launch_ns": 100,
                    "compute_ns": 200,
                }
            ],
            "segments": [{"id": 0, "first_node": 0, "last_node": 0}],
            "events": events,
            "blockers": [],
            "plan_hash": "fnv1a64:0123456789abcdef",
            "future_extension": {"accepted": True},
        }

    @staticmethod
    def finished_process(exit_code: int) -> dict:
        return {
            "state": "finished",
            "exit_code": exit_code,
            "duration_ms": 1,
            "stdout": "",
            "stderr": "",
            "peak_rss_bytes": None,
        }

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
