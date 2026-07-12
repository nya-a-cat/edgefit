from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class ProfileConfidenceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "profile_confidence_gate.py"
        spec = importlib.util.spec_from_file_location("profile_confidence_gate", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.gate = module

    def test_default_machine_policy_contract_passes(self) -> None:
        self.assertTrue(self.gate.diagnostic_policy_passes(self.gate.DEFAULT_DIAGNOSTIC_POLICY))
        detail = self.gate.diagnostic_policy_detail(self.gate.DEFAULT_DIAGNOSTIC_POLICY)
        self.assertIn("schema=edgefit.diagnostic_policy.v1", detail)
        self.assertIn("ef0104=yes", detail)
        self.assertIn("severity_rules=yes", detail)
        self.assertIn("reporting=yes", detail)

    def test_seed_profile_holds_until_runtime_smoke_exists(self) -> None:
        paths = self.write_inputs(confidence="seed", include_smoke=False)
        try:
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "hold_seed")
        self.assertFalse(summary["confidence_uplift_ready"])
        self.assertIn("add runtime smoke result for the target profile", summary["next_actions"])
        self.assertEqual(self.gate.exit_code(summary), 0)

    def test_uplifted_profile_fails_without_runtime_smoke(self) -> None:
        paths = self.write_inputs(confidence="calibrated", include_smoke=False)
        try:
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "fail")
        self.assertEqual(self.gate.exit_code(summary), 1)

    def test_seed_profile_holds_until_warning_policy_exists(self) -> None:
        paths = self.write_inputs(confidence="seed", include_smoke=True, include_policy=False)
        try:
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "hold_seed")
        self.assertFalse(summary["confidence_uplift_ready"])
        self.assertIn("document warning-only diagnostic policy before confidence review", summary["next_actions"])

    def test_seed_profile_holds_when_warning_policy_is_malformed(self) -> None:
        paths = self.write_inputs(confidence="seed", include_smoke=True)
        try:
            paths[8].write_text("{not-json", encoding="utf-8")
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "hold_seed")
        self.assertFalse(summary["confidence_uplift_ready"])
        self.assertIn("document warning-only diagnostic policy before confidence review", summary["next_actions"])

    def test_seed_profile_holds_when_warning_policy_has_wrong_types(self) -> None:
        paths = self.write_inputs(confidence="seed", include_smoke=True)
        try:
            paths[8].write_text(
                json.dumps(
                    {
                        "schema": "edgefit.diagnostic_policy.v1",
                        "warning_only_diagnostics": "EF0104",
                        "gate_status": [],
                        "reporting": True,
                    }
                ),
                encoding="utf-8",
            )
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "hold_seed")
        self.assertFalse(summary["confidence_uplift_ready"])
        self.assertIn("document warning-only diagnostic policy before confidence review", summary["next_actions"])

    def test_seed_profile_holds_until_runtime_boundary_exists(self) -> None:
        paths = self.write_inputs(confidence="seed", include_smoke=True, include_boundary=False)
        try:
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "hold_seed")
        self.assertFalse(summary["confidence_uplift_ready"])
        self.assertIn("add ORT reduced-operator boundary evidence for the target profile", summary["next_actions"])

    def test_seed_profile_holds_until_public_pr_trials_exist(self) -> None:
        paths = self.write_inputs(confidence="seed", include_smoke=True, include_public_pr_trials=False)
        try:
            summary = self.gate.build_summary(*paths)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "hold_seed")
        self.assertFalse(summary["confidence_uplift_ready"])
        self.assertIn("run 3 public repository PR trials through the GitHub Action path", summary["next_actions"])

    def test_runtime_smoke_audits_policy_boundary_and_public_pr_trials_allow_uplift_review(self) -> None:
        paths = self.write_inputs(confidence="calibrated", include_smoke=True)
        try:
            summary = self.gate.build_summary(*paths)
            markdown = self.gate.render_markdown(summary)
        finally:
            self.cleanup(paths)

        self.assertEqual(summary["decision"], "uplift_ready")
        self.assertTrue(summary["confidence_uplift_ready"])
        self.assertEqual(summary["next_actions"], [])
        self.assertIn("| `runtime_smoke_verified` | `pass` |", markdown)
        self.assertIn("| `runtime_boundary_verified` | `pass` |", markdown)
        self.assertIn("| `public_pr_trials_verified` | `pass` |", markdown)
        self.assertIn("| `warning_diagnostic_policy_documented` | `pass` |", markdown)
        policy_check = next(item for item in summary["checks"] if item["id"] == "warning_diagnostic_policy_documented")
        self.assertIn("schema=edgefit.diagnostic_policy.v1", policy_check["detail"])
        self.assertIn("| `corpus_expansion_gate_verified` | `pass` |", markdown)
        self.assertIn("| `operator_support_audit_verified` | `pass` |", markdown)

    def write_inputs(
        self,
        confidence: str,
        include_smoke: bool,
        include_policy: bool = True,
        include_boundary: bool = True,
        include_public_pr_trials: bool = True,
    ) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
        suffix = f"{os.getpid()}_{confidence}_{include_smoke}_{include_policy}_{include_boundary}_{include_public_pr_trials}"
        profile = WORKSPACE_TMP / f"confidence_gate_profile_{suffix}.yaml"
        reference = WORKSPACE_TMP / f"confidence_gate_reference_{suffix}.json"
        matrix = WORKSPACE_TMP / f"confidence_gate_matrix_{suffix}.json"
        corpus_gate = WORKSPACE_TMP / f"confidence_gate_corpus_gate_{suffix}.json"
        operator_audit = WORKSPACE_TMP / f"confidence_gate_operator_audit_{suffix}.json"
        runtime = WORKSPACE_TMP / f"confidence_gate_runtime_{suffix}.json"
        smoke = WORKSPACE_TMP / f"confidence_gate_smoke_{suffix}.json"
        boundary = WORKSPACE_TMP / f"confidence_gate_boundary_{suffix}.json"
        policy = WORKSPACE_TMP / f"confidence_gate_policy_{suffix}.json"
        public_pr_trials = WORKSPACE_TMP / f"confidence_gate_public_pr_trials_{suffix}.json"

        profile.write_text(
            f"""
profile_version: edgefit.target.v1
metadata:
  source: sample evidence
  confidence: {confidence}
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
        reference.write_text(
            json.dumps(
                {
                    "target_id": "sample_target",
                    "profile_metadata": {"confidence": confidence},
                    "reference_versions": {"onnx_python_package": {"status": "match"}},
                    "missing_reference_count": 0,
                    "official_only_count": 0,
                    "runtime_only_count": 0,
                    "corpus_only_count": 0,
                    "runtime_and_corpus_count": 1,
                }
            ),
            encoding="utf-8",
        )
        matrix.write_text(
            json.dumps(
                {
                    "models": [{"id": "m1"}],
                    "matrix": [
                        {
                            "model_id": "m1",
                            "target_id": "sample_target",
                            "status": "pass",
                            "error_count": 0,
                            "warning_count": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        corpus_gate.write_text(
            json.dumps(
                {
                    "schema": "edgefit.corpus_expansion_gate.v1",
                    "status": "ready_for_profile_matrix",
                    "model_count": 1,
                    "sample_goal": 1,
                    "models_needed": 0,
                    "label_status": "complete",
                    "label_cell_count": 1,
                    "expected_label_cell_count": 1,
                    "missing_label_cell_count": 0,
                    "duplicate_label_cell_count": 0,
                    "unknown_label_cell_count": 0,
                }
            ),
            encoding="utf-8",
        )
        operator_audit.write_text(
            json.dumps(
                {
                    "schema": "edgefit.operator_support_audit.v1",
                    "status": "pass",
                    "sample_model_count": 1,
                    "sample_goal": 1,
                    "precision_recall_review": {
                        "status": "pass",
                        "labeled_cell_count": 1,
                        "mismatch_count": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        runtime.write_text(
            json.dumps(
                {
                    "schema": "edgefit.ort_runtime_evidence.result.v1",
                    "operators": [
                        {
                            "op_key": "com.microsoft::QLinearAdd",
                            "status": "pass",
                            "providers": ["CPUExecutionProvider"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        if include_smoke:
            smoke.write_text(
                json.dumps(
                    {
                        "schema": "edgefit.runtime_smoke.v1",
                        "target_id": "sample_target",
                        "status": "pass",
                        "model_count": 1,
                    }
                ),
                encoding="utf-8",
            )
        if include_boundary:
            boundary.write_text(
                json.dumps(
                    {
                        "schema": "edgefit.ort_runtime_boundary.v1",
                        "target_id": "sample_target",
                        "status": "pass",
                        "profile_operator_count": 1,
                        "required_operator_count": 1,
                        "profile_coverage_status": "pass",
                        "generated_config_roundtrip_status": "pass",
                        "missing_from_profile": [],
                        "profile_ops_not_required": [],
                    }
                ),
                encoding="utf-8",
            )
        if include_policy:
            policy.write_text(
                json.dumps(
                    {
                        "schema": "edgefit.diagnostic_policy.v1",
                        "warning_only_diagnostics": ["EF0104"],
                        "gate_status": {"warning": "pass", "error": "fail"},
                        "reporting": {
                            "sarif_includes_warnings": True,
                            "json_includes_suppressed_diagnostics": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
        if include_public_pr_trials:
            public_pr_trials.write_text(
                json.dumps(
                    {
                        "schema": "edgefit.public_pr_trial_gate.v1",
                        "status": "ready_for_confidence_review",
                        "sample_goal": 3,
                        "verified_trial_count": 3,
                        "invalid_trial_count": 0,
                        "distinct_repository_count": 3,
                        "trials_needed": 0,
                    }
                ),
                encoding="utf-8",
            )
        return profile, reference, matrix, corpus_gate, operator_audit, runtime, smoke, boundary, policy, public_pr_trials

    def cleanup(self, paths: tuple[Path, ...]) -> None:
        for path in paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()