#!/usr/bin/env python3
"""将 ONNX 模型规范化为 EdgeFit Rust 分析器使用的稳定 JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ONNX into edgefit.normalized_model.v1 JSON")
    parser.add_argument("model")
    parser.add_argument("--out", help="Output JSON path. Defaults to stdout.")
    args = parser.parse_args()

    try:
        data = normalize(Path(args.model))
    except ImportError as exc:
        raise SystemExit("onnx is required: python -m pip install onnx") from exc
    except Exception as exc:
        raise SystemExit(f"failed to normalize ONNX model: {exc}") from exc

    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def normalize(path: Path) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto

    metadata_model = onnx.load(path, load_external_data=False)
    file_bytes, model_hash, external_data_file_count = model_package_metadata(
        path, metadata_model, TensorProto
    )
    model = onnx.load(path)
    onnx.checker.check_model(model)
    ensure_supported_graph(model, onnx.AttributeProto)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        shape_inference = {"status": "pass", "error": None}
    except Exception as exc:  # ONNX 对自定义算子或不完整 schema 可能无法完成形状推断。
        inferred = model
        shape_inference = {"status": "failed", "error": str(exc)}
    graph = inferred.graph
    initializer_names = {item.name for item in graph.initializer}

    return {
        "schema": "edgefit.normalized_model.v1",
        "model": {
            "path": str(path),
            "file_bytes": file_bytes,
            "sha256": model_hash,
            "external_data_file_count": external_data_file_count,
            "opset_imports": [
                {
                    "domain": item.domain or "ai.onnx",
                    "version": int(item.version),
                }
                for item in model.opset_import
            ],
        },
        "normalization": {"shape_inference": shape_inference},
        "graph": {
            "inputs": [
                value_info(item, TensorProto)
                for item in graph.input
                if item.name not in initializer_names
            ],
            "values": [value_info(item, TensorProto) for item in graph.value_info],
            "outputs": [value_info(item, TensorProto) for item in graph.output],
            "initializers": [initializer_info(item, TensorProto) for item in graph.initializer],
            "nodes": [node_info(item, onnx.AttributeProto) for item in graph.node],
        },
    }


def model_package_metadata(path: Path, model: Any, tensor_proto: Any) -> tuple[int, str, int]:
    """统计并哈希主模型与去重后的 external-data 文件，避免预算和快照漏项。"""
    external_files: set[Path] = set()
    for initializer in model.graph.initializer:
        if initializer.data_location != tensor_proto.EXTERNAL:
            continue
        entries = {item.key: item.value for item in initializer.external_data}
        location = entries.get("location")
        if location:
            external_files.add((path.parent / location).resolve())
    package_files = [path.resolve(), *sorted(external_files, key=str)]
    digest = hashlib.sha256()
    file_bytes = 0
    for item in package_files:
        content = item.read_bytes()
        file_bytes += len(content)
        digest.update(content)
    return file_bytes, "sha256:" + digest.hexdigest(), len(external_files)


def ensure_supported_graph(model: Any, attribute_proto: Any) -> None:
    """对尚未建模的子图和稀疏 initializer 直接失败，禁止给出不完整的通过结论。"""
    if model.functions:
        raise ValueError("ONNX local functions are not supported by EdgeFit normalization")
    if model.graph.sparse_initializer:
        raise ValueError("ONNX sparse initializers are not supported by EdgeFit normalization")
    for node in model.graph.node:
        if any(
            attribute.type in (attribute_proto.GRAPH, attribute_proto.GRAPHS)
            for attribute in node.attribute
        ):
            raise ValueError(
                f"nested ONNX subgraphs are not supported: {node.domain or 'ai.onnx'}::{node.op_type}"
            )


def value_info(value: Any, tensor_proto: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    # 空 shape 代表已声明的标量；未声明 shape 则必须保留为 null，不能伪装成标量。
    shape = (
        [dim_value(dim) for dim in tensor.shape.dim]
        if tensor.HasField("shape")
        else None
    )
    return {
        "name": value.name,
        "dtype": dtype_name(tensor.elem_type, tensor_proto),
        "shape": shape,
    }


def initializer_info(value: Any, tensor_proto: Any) -> dict[str, Any]:
    item = {
        "name": value.name,
        "dtype": dtype_name(value.data_type, tensor_proto),
        "shape": list(value.dims),
    }
    if value.raw_data:
        item["bytes"] = len(value.raw_data)
    return item


def node_info(node: Any, attribute_proto: Any) -> dict[str, Any]:
    return {
        "name": node.name or None,
        "domain": node.domain or "ai.onnx",
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": {
            attribute.name: attribute_info(attribute, attribute_proto)
            for attribute in node.attribute
        },
    }


def attribute_info(attribute: Any, attribute_proto: Any) -> dict[str, Any]:
    """将可稳定比较的 ONNX 属性编码为带类型值，未知类型必须保留证据。"""
    attribute_type = attribute.type
    if attribute_type == attribute_proto.FLOAT:
        value = float(attribute.f)
        if not math.isfinite(value):
            return unknown_attribute(attribute_type, "non_finite_float")
        return {"kind": "float", "value": value}
    if attribute_type == attribute_proto.INT:
        # 十进制字符串避免 JSON number 经 f64 解析时丢失完整 int64 精度。
        return {"kind": "int", "value": str(int(attribute.i))}
    if attribute_type == attribute_proto.STRING:
        return string_attribute(attribute.s, attribute_type)
    if attribute_type == attribute_proto.FLOATS:
        values = [float(value) for value in attribute.floats]
        if not all(math.isfinite(value) for value in values):
            return unknown_attribute(attribute_type, "non_finite_float")
        return {"kind": "floats", "value": values}
    if attribute_type == attribute_proto.INTS:
        return {"kind": "ints", "value": [str(int(value)) for value in attribute.ints]}
    if attribute_type == attribute_proto.STRINGS:
        try:
            values = [value.decode("utf-8") for value in attribute.strings]
        except UnicodeDecodeError:
            return unknown_attribute(attribute_type, "non_utf8_string")
        return {"kind": "strings", "value": values}
    return unknown_attribute(attribute_type, "unmodeled_attribute_type")


def string_attribute(value: bytes, attribute_type: int) -> dict[str, Any]:
    """仅将有效 UTF-8 暴露为字符串；二进制内容不可伪装成兼容属性。"""
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError:
        return unknown_attribute(attribute_type, "non_utf8_string")
    return {"kind": "string", "value": decoded}


def unknown_attribute(attribute_type: int, reason: str) -> dict[str, Any]:
    return {"kind": "unknown", "onnx_type": int(attribute_type), "reason": reason}


def dim_value(dim: Any) -> int | str | None:
    if dim.dim_value:
        return int(dim.dim_value)
    if dim.dim_param:
        return dim.dim_param
    return None


def dtype_name(value: int, tensor_proto: Any) -> str | None:
    if value == tensor_proto.UNDEFINED:
        return None
    mapping = {
        tensor_proto.FLOAT: "float32",
        tensor_proto.FLOAT16: "float16",
        tensor_proto.BFLOAT16: "bfloat16",
        tensor_proto.DOUBLE: "float64",
        tensor_proto.INT8: "int8",
        tensor_proto.UINT8: "uint8",
        tensor_proto.INT16: "int16",
        tensor_proto.UINT16: "uint16",
        tensor_proto.INT32: "int32",
        tensor_proto.UINT32: "uint32",
        tensor_proto.INT64: "int64",
        tensor_proto.UINT64: "uint64",
        tensor_proto.BOOL: "bool",
    }
    return mapping.get(value, f"onnx_dtype_{value}")


if __name__ == "__main__":
    raise SystemExit(main())
