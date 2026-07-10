from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class CorpusExpansionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "corpus_expansion_gate.py"
        spec = importlib.util.spec_from_file_location("corpus_expansion_gate", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.gate = module

    def test_reports_complete_labels_below_sample_goal(self) -> None:
        suffix = str(os.getpid())
        manifest = WORKSPACE_TMP / f"corpus_gate_manifest_{suffix}.json"
        labels = WORKSPACE_TMP / f"corpus_gate_labels_{suffix}.json"
        profile = WORKSPACE_TMP / f"corpus_gate_profile_{suffix}.yaml"
        try:
            manifest.write_text(json.dumps({"models": [{"id": "m1"}, {"id": "m2"}]}), encoding="utf-8")
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            {"model_id": "m1", "target_id": "sample_target"},
                            {"model_id": "m2", "target_id": "sample_target"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profile.write_text(profile_text("sample_target"), encoding="utf-8")

            summary = self.gate.build_summary(manifest, labels, [profile], sample_goal=20)
        finally:
            for path in (manifest, labels, profile):
                path.unlink(missing_ok=True)

        self.assertEqual(summary["status"], "needs_more_models")
        self.assertEqual(summary["label_status"], "complete")
        self.assertEqual(summary["model_count"], 2)
        self.assertEqual(summary["models_needed"], 18)
        self.assertEqual(summary["expected_label_cell_count"], 2)
        self.assertEqual(summary["missing_label_cell_count"], 0)

    def test_reports_missing_labels(self) -> None:
        suffix = str(os.getpid())
        manifest = WORKSPACE_TMP / f"corpus_gate_manifest_missing_{suffix}.json"
        labels = WORKSPACE_TMP / f"corpus_gate_labels_missing_{suffix}.json"
        profile = WORKSPACE_TMP / f"corpus_gate_profile_missing_{suffix}.yaml"
        try:
            manifest.write_text(json.dumps({"models": [{"id": "m1"}]}), encoding="utf-8")
            labels.write_text(json.dumps({"labels": []}), encoding="utf-8")
            profile.write_text(profile_text("sample_target"), encoding="utf-8")

            summary = self.gate.build_summary(manifest, labels, [profile], sample_goal=1)
        finally:
            for path in (manifest, labels, profile):
                path.unlink(missing_ok=True)

        self.assertEqual(summary["status"], "label_coverage_incomplete")
        self.assertEqual(summary["label_status"], "incomplete")
        self.assertEqual(summary["missing_label_cell_count"], 1)
        self.assertEqual(summary["missing_label_cells"], [{"model_id": "m1", "target_id": "sample_target"}])

    def test_renders_markdown(self) -> None:
        markdown = self.gate.render_markdown(
            {
                "status": "needs_more_models",
                "model_count": 2,
                "sample_goal": 20,
                "models_needed": 18,
                "target_count": 1,
                "label_cell_count": 2,
                "expected_label_cell_count": 2,
                "label_status": "complete",
                "missing_label_cell_count": 0,
                "duplicate_label_cell_count": 0,
                "unknown_label_cell_count": 0,
            }
        )

        self.assertIn("# EdgeFit Corpus Expansion Gate", markdown)
        self.assertIn("`needs_more_models`", markdown)
        self.assertIn("Missing labels", markdown)


def profile_text(target_id: str) -> str:
    return f"""
profile_version: edgefit.target.v1
metadata:
  source: sample
  confidence: seed
  last_verified: 2026-07-09
target:
  id: {target_id}
ops:
  allow:
    ai.onnx:
      Add:
        dtypes: [float32]
""".strip() + "\n"


if __name__ == "__main__":
    unittest.main()