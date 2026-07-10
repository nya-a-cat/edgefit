from __future__ import annotations

import importlib.util
import sys
import tempfile
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

    def test_file_integrity_verification_does_not_normalize_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            model = cache / "boundary.onnx"
            model.write_bytes(b"not-a-normalizable-onnx")
            item = {
                "id": "boundary",
                "model_url": "https://example.test/boundary.onnx",
                "model_name": model.name,
                "model_bytes": model.stat().st_size,
                "model_sha256": self.corpus.sha256(model),
            }

            result = self.corpus.verify_model_file_integrity(item, cache, download=False)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["model_bytes"], len(b"not-a-normalizable-onnx"))
        self.assertEqual(result["model_sha256"], item["model_sha256"])

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

    def test_qlinear_global_average_pool_shape_supports_both_layouts(self) -> None:
        self.assertEqual(
            self.corpus.qlinear_global_average_pool_shape([1, 1000, 13, 13], 0),
            [1, 1000, 1, 1],
        )
        self.assertEqual(
            self.corpus.qlinear_global_average_pool_shape([1, 13, 13, 1000], 1),
            [1, 1, 1, 1000],
        )

    def test_repair_adds_schema_derived_value_info_without_overwriting_source(self) -> None:
        import onnx
        from onnx import TensorProto

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            repaired = Path(directory) / "repaired.onnx"
            onnx.save_model(self.make_pool_model(TensorProto.UINT8), source)

            evidence = self.corpus.repair_qlinear_global_average_pool_value_info(
                source,
                repaired,
                "Y_quantized",
            )

            result = onnx.load(repaired)
            value = next(item for item in result.graph.value_info if item.name == "Y_quantized")
            dtype, shape = self.corpus.value_info_metadata(value)
            self.assertEqual(dtype, TensorProto.UINT8)
            self.assertEqual(shape, [1, 1000, 1, 1])
            self.assertEqual(evidence["dtype"], "uint8")
            self.assertNotEqual(evidence["source_sha256"], evidence["repaired_sha256"])
            self.assertNotIn("Y_quantized", [item.name for item in onnx.load(source).graph.value_info])

    def test_repair_rejects_output_zero_point_type_mismatch(self) -> None:
        import onnx
        from onnx import TensorProto

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            repaired = Path(directory) / "repaired.onnx"
            onnx.save_model(self.make_pool_model(TensorProto.INT8), source)

            with self.assertRaises(SystemExit) as context:
                self.corpus.repair_qlinear_global_average_pool_value_info(
                    source,
                    repaired,
                    "Y_quantized",
                )

            self.assertIn("zero point dtype", str(context.exception))
            self.assertFalse(repaired.exists())

    @staticmethod
    def make_pool_model(output_zero_point_dtype: int):
        from onnx import TensorProto, helper

        initializers = [
            helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.1]),
            helper.make_tensor("x_zero_point", TensorProto.UINT8, [], [0]),
            helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.2]),
            helper.make_tensor("y_zero_point", output_zero_point_dtype, [], [0]),
        ]
        pool = helper.make_node(
            "QLinearGlobalAveragePool",
            ["X", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
            ["Y_quantized"],
            name="pool",
            domain="com.microsoft",
        )
        dequantize = helper.make_node(
            "DequantizeLinear",
            ["Y_quantized", "y_scale", "y_zero_point"],
            ["Y"],
            name="dequantize",
        )
        graph = helper.make_graph(
            [pool, dequantize],
            "repair-test",
            [helper.make_tensor_value_info("X", TensorProto.UINT8, [1, 1000, 13, 13])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1000, 1, 1])],
            initializer=initializers,
        )
        return helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 12), helper.make_opsetid("com.microsoft", 1)],
        )


if __name__ == "__main__":
    unittest.main()
