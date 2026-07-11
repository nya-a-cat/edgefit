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
        json.dumps(data)

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
