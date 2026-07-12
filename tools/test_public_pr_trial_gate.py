from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class PublicPrTrialGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_path = Path("tools") / "public_pr_trial_gate.py"
        spec = importlib.util.spec_from_file_location("public_pr_trial_gate", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.gate = module

    def test_default_manifest_is_local_tmp_path(self) -> None:
        self.assertEqual(
            self.gate.DEFAULT_MANIFEST,
            Path.cwd() / "tmp" / "public_pr_trials" / "public-pr-trials.json",
        )

    def test_empty_manifest_needs_trials(self) -> None:
        manifest = self.write_manifest({"schema": "edgefit.public_pr_trials.v1", "sample_goal": 3, "trials": []})
        try:
            summary = self.gate.build_summary(manifest)
        finally:
            manifest.unlink(missing_ok=True)

        self.assertEqual(summary["status"], "needs_public_trials")
        self.assertEqual(summary["verified_trial_count"], 0)
        self.assertEqual(summary["trials_needed"], 3)
        self.assertEqual(self.gate.exit_code(summary), 0)

    def test_three_distinct_public_repositories_pass(self) -> None:
        manifest = self.write_manifest(
            {
                "schema": "edgefit.public_pr_trials.v1",
                "sample_goal": 3,
                "trials": [
                    self.trial("edgefit-demo/a", "101", "pass"),
                    self.trial("edgefit-demo/b", "102", "fail"),
                    self.trial("edgefit-demo/c", "103", "pass"),
                ],
            }
        )
        try:
            summary = self.gate.build_summary(manifest)
            markdown = self.gate.render_markdown(summary)
        finally:
            manifest.unlink(missing_ok=True)

        self.assertEqual(summary["status"], "ready_for_confidence_review")
        self.assertEqual(summary["verified_trial_count"], 3)
        self.assertEqual(summary["distinct_repository_count"], 3)
        self.assertIn("**Status:** `ready_for_confidence_review`", markdown)

    def test_invalid_trial_fails_gate(self) -> None:
        bad_trial = self.trial("edgefit-demo/a", "101", "pass")
        bad_trial["pull_request_url"] = "https://example.com/pr/101"
        bad_trial["public_repository"] = False
        manifest = self.write_manifest(
            {"schema": "edgefit.public_pr_trials.v1", "sample_goal": 3, "trials": [bad_trial]}
        )
        try:
            summary = self.gate.build_summary(manifest)
        finally:
            manifest.unlink(missing_ok=True)

        self.assertEqual(summary["status"], "invalid_evidence")
        self.assertEqual(summary["invalid_trial_count"], 1)
        self.assertEqual(self.gate.exit_code(summary), 1)
        self.assertIn("public_repository must be true", summary["trials"][0]["errors"])

    def write_manifest(self, content: dict) -> Path:
        path = WORKSPACE_TMP / f"public_pr_trials_{os.getpid()}_{len(content.get('trials', []))}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def trial(self, repository: str, number: str, result: str) -> dict:
        return {
            "id": f"{repository.replace('/', '-')}-{number}",
            "repository": repository,
            "public_repository": True,
            "pull_request_url": f"https://github.com/{repository}/pull/{number}",
            "workflow_run_url": f"https://github.com/{repository}/actions/runs/{number}",
            "commit_sha": "abcdef1",
            "target_profile": "targets/esp32s3.yaml",
            "model_path": "models/model.onnx",
            "expected_result": result,
            "actual_result": result,
            "sarif_uploaded": True,
            "job_summary_present": True,
            "outcome_clear": True,
        }


if __name__ == "__main__":
    unittest.main()
