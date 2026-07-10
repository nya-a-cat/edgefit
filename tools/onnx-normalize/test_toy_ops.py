from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path


class ToyOnnxOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_path = Path(__file__).with_name("normalize_onnx.py")
        spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.normalize = staticmethod(module.normalize)

    @unittest.skipIf(importlib.util.find_spec("onnx") is None, "onnx package is not installed")
    def test_normalizes_report_recommended_toy_ops(self) -> None:
        cases = toy_models()
        workspace_tmp = Path.cwd() / "tmp"
        workspace_tmp.mkdir(exist_ok=True)

        for name, model, expected_shape in cases:
            with self.subTest(op=name):
                import onnx

                model_path = workspace_tmp / f"toy_{name.lower()}_{os.getpid()}.onnx"
                try:
                    onnx.save(model, str(model_path))
                    data = self.normalize(model_path)
                finally:
                    model_path.unlink(missing_ok=True)

                self.assertEqual(data["schema"], "edgefit.normalized_model.v1")
                self.assertEqual(data["graph"]["nodes"][0]["op_type"], name)
                self.assertEqual(data["graph"]["outputs"][0]["shape"], expected_shape)
                json.dumps(data)


def toy_models():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    opset = [helper.make_opsetid("", 13)]
    cases = []

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Add", ["x", "y"], ["z"])], "toy_add", [x, y], [z])
    cases.append(("Add", helper.make_model(graph, opset_imports=opset), [1, 4]))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), name="w")
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["z"])], "toy_matmul", [x], [z], [w])
    cases.append(("MatMul", helper.make_model(graph, opset_imports=opset), [1, 2]))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="w")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="b")
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node("Conv", ["x", "w", "b"], ["z"])
    graph = helper.make_graph([node], "toy_conv", [x], [z], [w, b])
    cases.append(("Conv", helper.make_model(graph, opset_imports=opset), [1, 1, 2, 2]))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    shape = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="shape")
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph([helper.make_node("Reshape", ["x", "shape"], ["z"])], "toy_reshape", [x], [z], [shape])
    cases.append(("Reshape", helper.make_model(graph, opset_imports=opset), [2, 2]))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3, 2])
    node = helper.make_node("Transpose", ["x"], ["z"], perm=[0, 2, 1])
    graph = helper.make_graph([node], "toy_transpose", [x], [z])
    cases.append(("Transpose", helper.make_model(graph, opset_imports=opset), [1, 3, 2]))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Softmax", ["x"], ["z"], axis=1)
    graph = helper.make_graph([node], "toy_softmax", [x], [z])
    cases.append(("Softmax", helper.make_model(graph, opset_imports=opset), [1, 4]))

    for _, model, _ in cases:
        model.ir_version = min(model.ir_version, onnx.IR_VERSION)
    return cases


if __name__ == "__main__":
    unittest.main()