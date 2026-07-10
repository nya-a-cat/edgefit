from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("operator_fixtures.json")
DEFAULT_CACHE = ROOT / "tmp" / "operator_fixtures"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify generated ONNX operator fixtures.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", help="Write JSON summary to this path.")
    args = parser.parse_args()

    summary = verify_manifest(Path(args.manifest), Path(args.cache))
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def verify_manifest(manifest_path: Path, cache: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "edgefit.operator_fixtures.v1":
        raise SystemExit("expected schema edgefit.operator_fixtures.v1")

    normalize = load_normalize()
    cache.mkdir(parents=True, exist_ok=True)
    results = [verify_fixture(item, cache, normalize) for item in manifest["models"]]
    return {
        "schema": "edgefit.operator_fixture.result.v1",
        "manifest": str(manifest_path),
        "results": results,
    }


def load_normalize() -> Callable[[Path], dict[str, Any]]:
    module_path = Path(__file__).with_name("normalize_onnx.py")
    spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.normalize


def verify_fixture(item: dict[str, Any], cache: Path, normalize: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    import onnx

    builder_name = item["builder"]
    try:
        builder = BUILDERS[builder_name]
    except KeyError as exc:
        raise SystemExit(f"unknown fixture builder {builder_name}") from exc

    model = builder()
    model_path = cache / f"{item['id']}.onnx"
    onnx.save(model, str(model_path))
    data = normalize(model_path)

    ops = sorted({node["op_type"] for node in data["graph"]["nodes"]})
    expected_ops = sorted(item["expected_ops"])
    if ops != expected_ops:
        raise SystemExit(f"operator mismatch for {item['id']}: {ops}")

    domain_ops = observed_domain_ops(data)
    expected_domains = expected_domain_ops(item)
    if domain_ops != expected_domains:
        raise SystemExit(f"operator domain mismatch for {item['id']}: {domain_ops}")

    outputs = data["graph"]["outputs"]
    if outputs != item["expected_outputs"]:
        raise SystemExit(f"output mismatch for {item['id']}: {outputs}")

    return {
        "id": item["id"],
        "status": "pass",
        "model_path": str(model_path),
        "ops": ops,
        "domain_ops": domain_ops,
        "outputs": outputs,
    }


def expected_domain_ops(item: dict[str, Any]) -> list[str]:
    domain_overrides = item.get("expected_operator_domains", {})
    keys = []
    for op in item["expected_ops"]:
        for domain in domain_overrides.get(op, ["ai.onnx"]):
            keys.append(op_key(domain, op))
    return sorted(keys)


def observed_domain_ops(data: dict[str, Any]) -> list[str]:
    return sorted({op_key(node.get("domain"), node["op_type"]) for node in data["graph"]["nodes"]})


def op_key(domain: str | None, op: str) -> str:
    return f"{domain or 'ai.onnx'}::{op}"


def make_gemm_model():
    import onnx
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    opset = [helper.make_opsetid("", 13)]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), name="w")
    b = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="b")
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("Gemm", ["x", "w", "b"], ["z"])
    graph = helper.make_graph([node], "toy_gemm_reference", [x], [z], [w, b])
    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = min(model.ir_version, onnx.IR_VERSION)
    return model


def make_resize_model():
    import onnx
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    opset = [helper.make_opsetid("", 13)]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    scales = numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), name="scales")
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 4, 4])
    node = helper.make_node("Resize", ["x", "", "scales"], ["z"], mode="nearest")
    graph = helper.make_graph([node], "toy_resize_reference", [x], [z], [scales])
    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = min(model.ir_version, onnx.IR_VERSION)
    return model


def make_softmax_model():
    import onnx
    from onnx import TensorProto, helper

    opset = [helper.make_opsetid("", 13)]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Softmax", ["x"], ["z"], axis=1)
    graph = helper.make_graph([node], "toy_softmax_reference", [x], [z])
    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = min(model.ir_version, onnx.IR_VERSION)
    return model


BUILDERS = {
    "Gemm": make_gemm_model,
    "Resize": make_resize_model,
    "Softmax": make_softmax_model,
}


if __name__ == "__main__":
    raise SystemExit(main())