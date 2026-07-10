from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


class RealWorldCorpusHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_dir = Path(__file__).parent
        sys.path.insert(0, str(module_dir))
        module_path = module_dir / "real_world_corpus.py"
        spec = importlib.util.spec_from_file_location("real_world_corpus", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.corpus = module

    def test_expected_domain_ops_defaults_to_ai_onnx_and_accepts_overrides(self) -> None:
        item = {
            "expected_ops": ["Add", "QLinearAdd"],
            "expected_operator_domains": {"QLinearAdd": ["com.microsoft"]},
        }

        self.assertEqual(
            self.corpus.expected_domain_ops(item),
            ["ai.onnx::Add", "com.microsoft::QLinearAdd"],
        )

    def test_observed_domain_ops_normalizes_empty_domain(self) -> None:
        data = {
            "graph": {
                "nodes": [
                    {"domain": "ai.onnx", "op_type": "Add"},
                    {"domain": "", "op_type": "Relu"},
                    {"domain": "com.microsoft", "op_type": "QLinearAdd"},
                ]
            }
        }

        self.assertEqual(
            self.corpus.observed_domain_ops(data),
            ["ai.onnx::Add", "ai.onnx::Relu", "com.microsoft::QLinearAdd"],
        )

    def test_prepare_direct_model_accepts_existing_model_file(self) -> None:
        cache = Path.cwd() / "tmp" / "real_world_corpus_test"
        cache.mkdir(parents=True, exist_ok=True)
        model = cache / "sample.onnx"
        model.write_bytes(b"onnx")
        try:
            path = self.corpus.prepare_direct_model(
                {"id": "sample", "model_url": "https://example.test/sample.onnx", "model_name": "sample.onnx"},
                cache,
                download=False,
            )
        finally:
            model.unlink(missing_ok=True)

        self.assertEqual(path, model)

    def test_prepare_direct_model_reports_missing_file_without_download(self) -> None:
        cache = Path.cwd() / "tmp" / "real_world_corpus_test"
        cache.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(SystemExit) as context:
            self.corpus.prepare_direct_model(
                {"id": "sample", "model_url": "https://example.test/sample.onnx", "model_name": "missing.onnx"},
                cache,
                download=False,
            )

        self.assertIn("missing model", str(context.exception))

    def test_select_models_preserves_manifest_order(self) -> None:
        models = [{"id": "first"}, {"id": "second"}, {"id": "third"}]

        selected = self.corpus.select_models(models, ["third", "first"])

        self.assertEqual([item["id"] for item in selected], ["first", "third"])

    def test_select_models_rejects_unknown_and_duplicate_ids(self) -> None:
        models = [{"id": "known"}]

        with self.assertRaises(SystemExit):
            self.corpus.select_models(models, ["missing"])
        with self.assertRaises(SystemExit):
            self.corpus.select_models(models, ["known", "known"])


if __name__ == "__main__":
    unittest.main()
