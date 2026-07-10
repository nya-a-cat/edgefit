from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path


class QuantizedOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_path = Path(__file__).with_name("normalize_onnx.py")
        spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.normalize = staticmethod(module.normalize)

    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_normalizes_qlinear_matmul_fixture(self) -> None:
        import onnx

        model = qlinear_matmul_model()
        workspace_tmp = Path.cwd() / "tmp"
        workspace_tmp.mkdir(exist_ok=True)
        model_path = workspace_tmp / f"toy_qlinear_matmul_{os.getpid()}.onnx"
        try:
            onnx.save(model, str(model_path))
            data = self.normalize(model_path)
        finally:
            model_path.unlink(missing_ok=True)

        self.assertEqual(data["schema"], "edgefit.normalized_model.v1")
        self.assertEqual(data["graph"]["nodes"][0]["op_type"], "QLinearMatMul")
        self.assertEqual(data["graph"]["outputs"][0]["dtype"], "uint8")
        self.assertEqual(data["graph"]["outputs"][0]["shape"], [1, 2])

        initializers = {item["name"]: item for item in data["graph"]["initializers"]}
        self.assertEqual(initializers["b"]["dtype"], "int8")
        self.assertGreater(initializers["b"].get("bytes", 0), 0)
        json.dumps(data)


def qlinear_matmul_model():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    a = helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 2])
    initializers = [
        numpy_helper.from_array(np.ones((3, 2), dtype=np.int8), name="b"),
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), name="a_scale"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), name="a_zero"),
        numpy_helper.from_array(np.array(0.03, dtype=np.float32), name="b_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="b_zero"),
        numpy_helper.from_array(np.array(0.07, dtype=np.float32), name="y_scale"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), name="y_zero"),
    ]
    node = helper.make_node(
        "QLinearMatMul",
        ["a", "a_scale", "a_zero", "b", "b_scale", "b_zero", "y_scale", "y_zero"],
        ["y"],
    )
    graph = helper.make_graph([node], "toy_qlinear_matmul", [a], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, onnx.IR_VERSION)
    return model


if __name__ == "__main__":
    unittest.main()