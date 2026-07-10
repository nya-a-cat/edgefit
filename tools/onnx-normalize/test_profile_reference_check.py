from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class ProfileReferenceCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "profile_reference_check.py"
        spec = importlib.util.spec_from_file_location("profile_reference_check", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.reference = module

    def test_parses_profile_ops_and_metadata(self) -> None:
        profile = WORKSPACE_TMP / f"profile_reference_target_{os.getpid()}.yaml"
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
    com.microsoft:
      QLinearAdd:
        dtypes: [uint8]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            parsed = self.reference.parse_profile(profile)
        finally:
            profile.unlink(missing_ok=True)

        self.assertEqual(parsed["target_id"], "sample_target")
        self.assertEqual(parsed["metadata"]["confidence"], "seed")
        self.assertEqual(parsed["metadata"]["last_verified"], "2026-07-09")
        self.assertEqual(parsed["ops"], ["ai.onnx::Add", "com.microsoft::QLinearAdd"])

    def test_loads_domain_aware_corpus_op_mapping(self) -> None:
        manifest = WORKSPACE_TMP / f"profile_reference_corpus_{os.getpid()}.json"
        try:
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "m1",
                                "expected_ops": ["Add", "QLinearAdd"],
                                "expected_operator_domains": {
                                    "QLinearAdd": ["com.microsoft"]
                                },
                            },
                            {"id": "m2", "expected_ops": ["Add"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            mapping = self.reference.load_corpus_ops(manifest)
        finally:
            manifest.unlink(missing_ok=True)

        self.assertEqual(mapping["ai.onnx::Add"], ["m1", "m2"])
        self.assertEqual(mapping["com.microsoft::QLinearAdd"], ["m1"])

    def test_loads_fixture_op_mapping_and_merges_evidence(self) -> None:
        manifest = WORKSPACE_TMP / f"profile_reference_fixtures_{os.getpid()}.json"
        try:
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {"id": "f1", "expected_ops": ["Gemm"]},
                            {"id": "f2", "expected_ops": ["Gemm", "Softmax"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            fixture_mapping = self.reference.load_fixture_ops(manifest)
            merged = self.reference.merge_op_mappings(
                {"ai.onnx::Gemm": ["real"]}, fixture_mapping
            )
        finally:
            manifest.unlink(missing_ok=True)

        self.assertEqual(fixture_mapping["ai.onnx::Gemm"], ["fixture:f1", "fixture:f2"])
        self.assertEqual(merged["ai.onnx::Gemm"], ["fixture:f1", "fixture:f2", "real"])
        self.assertEqual(merged["ai.onnx::Softmax"], ["fixture:f2"])

    def test_loads_runtime_evidence_mapping(self) -> None:
        manifest = WORKSPACE_TMP / f"profile_reference_runtime_{os.getpid()}.json"
        try:
            manifest.write_text(
                json.dumps(
                    {
                        "operators": [
                            {
                                "op_key": "com.microsoft::QLinearAdd",
                                "runtime_evidence": [
                                    {"id": "cpu"},
                                    {"id": "dml"},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            mapping = self.reference.load_runtime_ops(manifest)
        finally:
            manifest.unlink(missing_ok=True)

        self.assertEqual(mapping["com.microsoft::QLinearAdd"], ["cpu", "dml"])

    def test_loads_pinned_package_version(self) -> None:
        requirements = WORKSPACE_TMP / f"profile_reference_requirements_{os.getpid()}.txt"
        try:
            requirements.write_text("\ufeffonnx==1.22.0\n", encoding="utf-8")

            version = self.reference.load_pinned_package_version(requirements, "onnx")
        finally:
            requirements.unlink(missing_ok=True)

        self.assertEqual(version, "1.22.0")
        self.assertEqual(self.reference.version_pin_status("1.22.0", version), "match")
        self.assertEqual(self.reference.version_pin_status("1.23.0", version), "mismatch")
        self.assertEqual(self.reference.version_pin_status("1.22.0", None), "unpinned")

    def test_renders_reference_markdown(self) -> None:
        summary = {
            "target_id": "sample_target",
            "profile_metadata": {
                "source": "sample evidence",
                "confidence": "seed",
                "last_verified": "2026-07-09",
            },
            "operator_count": 1,
            "missing_reference_count": 0,
            "reference_versions": {
                "onnx_python_package": {
                    "installed": "1.22.0",
                    "pinned": "1.22.0",
                    "status": "match",
                    "official_operator_count": 192,
                }
            },
            "sources": [{"id": "source", "url": "https://example.test"}],
            "operators": [
                {
                    "domain": "ai.onnx",
                    "op": "Add",
                    "op_key": "ai.onnx::Add",
                    "status": "runtime_and_corpus",
                    "official_runtime_evidence": True,
                    "runtime_evidence_ids": ["runtime"],
                    "evidence_models": ["m1"],
                }
            ],
        }

        markdown = self.reference.render_markdown(summary)

        self.assertIn("# EdgeFit Profile Reference", markdown)
        self.assertIn("`sample_target`", markdown)
        self.assertIn("**Profile confidence:** `seed`", markdown)
        self.assertIn("| `onnx` Python package | `1.22.0` | `1.22.0` | `match` |", markdown)
        self.assertIn("| Operator | Status | Evidence Models |", markdown)
        self.assertIn("`ai.onnx::Add`", markdown)
        self.assertIn("https://example.test", markdown)


if __name__ == "__main__":
    unittest.main()