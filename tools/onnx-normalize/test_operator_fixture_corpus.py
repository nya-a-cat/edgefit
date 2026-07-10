from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WORKSPACE_TMP = Path.cwd() / "tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)


class OperatorFixtureCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "operator_fixture_corpus.py"
        spec = importlib.util.spec_from_file_location("operator_fixture_corpus", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.fixture_corpus = module

    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_verifies_reference_operator_fixtures(self) -> None:
        manifest = Path("tools/onnx-normalize/operator_fixtures.json")
        cache = WORKSPACE_TMP / "operator_fixture_test_cache"

        summary = self.fixture_corpus.verify_manifest(manifest, cache)

        self.assertEqual(summary["schema"], "edgefit.operator_fixture.result.v1")
        self.assertEqual(
            {result["id"] for result in summary["results"]},
            {"toy-gemm-reference", "toy-resize-reference", "toy-softmax-reference"},
        )
        self.assertEqual(
            {op for result in summary["results"] for op in result["ops"]},
            {"Gemm", "Resize", "Softmax"},
        )
        self.assertEqual(
            {op for result in summary["results"] for op in result["domain_ops"]},
            {"ai.onnx::Gemm", "ai.onnx::Resize", "ai.onnx::Softmax"},
        )


if __name__ == "__main__":
    unittest.main()