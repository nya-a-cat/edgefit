from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class OperatorSupportAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "operator_support_audit.py"
        spec = importlib.util.spec_from_file_location("operator_support_audit", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.audit = module

    def test_builds_profile_audit_summary_with_manual_labels(self) -> None:
        suffix = str(os.getpid())
        manifest = WORKSPACE_TMP / f"support_audit_manifest_{suffix}.json"
        fixtures = WORKSPACE_TMP / f"support_audit_fixtures_{suffix}.json"
        runtime = WORKSPACE_TMP / f"support_audit_runtime_{suffix}.json"
        matrix = WORKSPACE_TMP / f"support_audit_matrix_{suffix}.json"
        labels = WORKSPACE_TMP / f"support_audit_labels_{suffix}.json"
        profile = WORKSPACE_TMP / f"support_audit_profile_{suffix}.yaml"
        try:
            manifest.write_text(json.dumps({"models": [{"id": "m1", "expected_ops": ["Add"]}]}), encoding="utf-8")
            fixtures.write_text(json.dumps({"models": [{"id": "f1", "expected_ops": ["Softmax"]}]}), encoding="utf-8")
            runtime.write_text(json.dumps({"operators": []}), encoding="utf-8")
            matrix.write_text(
                json.dumps(
                    {
                        "models": [{"id": "m1"}],
                        "matrix": [
                            {
                                "model_id": "m1",
                                "target_id": "sample_target",
                                "status": "fail",
                                "warning_count": 1,
                                "error_count": 2,
                                "unsupported_ops": ["Resize"],
                                "unsupported_dtypes": ["float32"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "model_id": "m1",
                                "target_id": "sample_target",
                                "expected_status": "fail",
                                "unsupported_ops": ["Resize"],
                                "unsupported_dtypes": ["float32"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profile.write_text(
                """
profile_version: edgefit.target.v1
metadata:
  source: sample
  confidence: seed
  last_verified: 2026-07-09
target:
  id: sample_target
ops:
  allow:
    ai.onnx:
      Add:
        dtypes: [float32]
      MissingOp:
        dtypes: [float32]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = self.audit.build_summary(
                manifest, fixtures, runtime, matrix, [profile], sample_goal=20, labels_path=labels
            )
        finally:
            for path in (manifest, fixtures, runtime, matrix, labels, profile):
                path.unlink(missing_ok=True)

        self.assertEqual(summary["status"], "needs_more_models")
        self.assertEqual(summary["sample_model_count"], 1)
        profile_summary = summary["profiles"][0]
        self.assertEqual(profile_summary["target_id"], "sample_target")
        self.assertEqual(profile_summary["allowed_without_evidence_count"], 1)
        self.assertEqual(profile_summary["matrix_fail_count"], 1)
        self.assertEqual(profile_summary["unsupported_ops"], ["Resize"])
        review = summary["precision_recall_review"]
        self.assertEqual(review["status"], "below_sample_goal")
        self.assertEqual(review["labeled_cell_count"], 1)
        self.assertEqual(review["status_match_count"], 1)
        self.assertEqual(review["unsupported_op_precision"], 1.0)
        self.assertEqual(review["unsupported_op_recall"], 1.0)
        self.assertEqual(review["unsupported_dtype_precision"], 1.0)
        self.assertEqual(review["unsupported_dtype_recall"], 1.0)

    def test_reports_missing_manual_labels(self) -> None:
        review = self.audit.build_precision_recall_review([], [], sample_goal=20)

        self.assertEqual(review["status"], "requires_manual_labels")
        self.assertEqual(review["labeled_cell_count"], 0)

    def test_renders_markdown(self) -> None:
        summary = {
            "status": "needs_more_models",
            "sample_model_count": 1,
            "sample_goal": 20,
            "observed_operator_count": 2,
            "profile_count": 1,
            "precision_recall_review": {
                "status": "below_sample_goal",
                "reason": "needs more labels",
                "labeled_cell_count": 1,
                "labeled_model_count": 1,
                "status_match_count": 1,
                "status_check_count": 1,
                "unsupported_op_precision": 1.0,
                "unsupported_op_recall": 1.0,
                "unsupported_dtype_precision": 1.0,
                "unsupported_dtype_recall": 1.0,
            },
            "profiles": [
                {
                    "target_id": "sample_target",
                    "confidence": "seed",
                    "allowed_operator_count": 2,
                    "allowed_without_evidence_count": 1,
                    "matrix_pass_count": 0,
                    "matrix_fail_count": 1,
                    "unsupported_ops": ["Resize"],
                }
            ],
        }

        markdown = self.audit.render_markdown(summary)

        self.assertIn("# EdgeFit Operator Support Audit", markdown)
        self.assertIn("`sample_target`", markdown)
        self.assertIn("`Resize`", markdown)
        self.assertIn("Unsupported-op precision/recall", markdown)
        self.assertIn("`1.0` / `1.0`", markdown)


if __name__ == "__main__":
    unittest.main()