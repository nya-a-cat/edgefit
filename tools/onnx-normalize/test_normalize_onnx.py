from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path


class NormalizeOnnxTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_normalizes_toy_add_model(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        module_path = Path(__file__).with_name("normalize_onnx.py")
        spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Add", ["x", "y"], ["z"], name="add_0")
        graph = helper.make_graph([node], "toy_add", [x, y], [z])
        model = helper.make_model(graph)

        workspace_tmp = Path.cwd() / "tmp"
        workspace_tmp.mkdir(exist_ok=True)
        model_path = workspace_tmp / f"toy_add_adapter_test_{os.getpid()}.onnx"
        try:
            onnx.save(model, str(model_path))
            data = module.normalize(model_path)
        finally:
            model_path.unlink(missing_ok=True)

        self.assertEqual(data["schema"], "edgefit.normalized_model.v1")
        self.assertEqual(data["graph"]["nodes"][0]["op_type"], "Add")
        self.assertEqual(data["graph"]["nodes"][0]["attributes"], {})
        json.dumps(data)

    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_normalizes_stable_attribute_subset_and_preserves_unknown_evidence(self) -> None:
        import onnx
        from onnx import helper

        module_path = Path(__file__).with_name("normalize_onnx.py")
        spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        node = helper.make_node(
            "Example",
            [],
            [],
            axis=1,
            alpha=0.5,
            label="edge",
            axes=[1, 2],
            scales=[0.5, 1.0],
            labels=["a", "b"],
        )
        # TENSOR 尚未进入兼容性语义，必须作为 unknown 证据输出而不是丢弃。
        node.attribute.extend(
            [helper.make_attribute("weights", helper.make_tensor("w", 1, [1], [1.0]))]
        )

        attributes = module.node_info(node, onnx.AttributeProto)["attributes"]

        self.assertEqual(attributes["axis"], {"kind": "int", "value": "1"})
        self.assertEqual(attributes["alpha"], {"kind": "float", "value": 0.5})
        self.assertEqual(attributes["label"], {"kind": "string", "value": "edge"})
        self.assertEqual(attributes["axes"], {"kind": "ints", "value": ["1", "2"]})
        self.assertEqual(attributes["scales"], {"kind": "floats", "value": [0.5, 1.0]})
        self.assertEqual(attributes["labels"], {"kind": "strings", "value": ["a", "b"]})
        self.assertEqual(attributes["weights"]["kind"], "unknown")
        self.assertEqual(attributes["weights"]["onnx_type"], onnx.AttributeProto.TENSOR)

    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_undefined_dtype_maps_to_none(self) -> None:
        from onnx import TensorProto

        module_path = Path(__file__).with_name("normalize_onnx.py")
        spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        self.assertIsNone(module.dtype_name(TensorProto.UNDEFINED, TensorProto))

    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_rejects_loop_with_nested_body_graph(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        module_path = Path(__file__).with_name("normalize_onnx.py")
        spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
        condition_in = helper.make_tensor_value_info("condition_in", TensorProto.BOOL, [])
        condition_out = helper.make_tensor_value_info("condition_out", TensorProto.BOOL, [])
        body = helper.make_graph(
            [helper.make_node("Identity", ["condition_in"], ["condition_out"])],
            "loop_body",
            [iteration, condition_in],
            [condition_out],
        )
        trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
        condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
        loop = helper.make_node("Loop", ["trip_count", "condition"], [], body=body)
        graph = helper.make_graph([loop], "nested_loop", [trip_count, condition], [])
        model = helper.make_model(graph)

        with self.assertRaisesRegex(ValueError, "nested ONNX subgraphs are not supported"):
            module.ensure_supported_graph(model, onnx.AttributeProto)

if __name__ == "__main__":
    unittest.main()
