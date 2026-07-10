from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("real_world_corpus.json")
DEFAULT_CACHE = ROOT / "tmp" / "real_world_corpus"
QLINEAR_GLOBAL_AVERAGE_POOL_SCHEMA_SOURCE = (
    "https://github.com/microsoft/onnxruntime/blob/v1.22.0/"
    "onnxruntime/core/graph/contrib_ops/nhwc_schema_defs.cc"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EdgeFit real-world ONNX corpus entries.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--download", action="store_true", help="Download missing archives or model files from manifest URLs.")
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Only verify this model ID; repeat the option to select multiple models.",
    )
    parser.add_argument(
        "--repair-qlinear-global-average-pool",
        metavar="TENSOR",
        help="Add missing value_info for one QLinearGlobalAveragePool output.",
    )
    parser.add_argument(
        "--repair-out",
        help="Write the repaired ONNX model to this new path.",
    )
    parser.add_argument("--out", help="Write JSON summary to this path.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    selected_models = select_models(manifest["models"], args.model_id)
    validate_repair_arguments(args, selected_models)
    normalize = load_normalize()
    results = []
    for item in selected_models:
        results.append(verify_model(item, cache, args.download, normalize))

    summary = {"schema": "edgefit.real_world_corpus.result.v1", "results": results}
    if args.repair_qlinear_global_average_pool:
        source_path = prepare_model_file(selected_models[0], cache, False)
        repaired_path = Path(args.repair_out)
        repair = repair_qlinear_global_average_pool_value_info(
            source_path,
            repaired_path,
            args.repair_qlinear_global_average_pool,
        )
        normalized = normalize(repaired_path)
        repaired_value = next(
            (
                item
                for item in normalized["graph"]["values"]
                if item["name"] == args.repair_qlinear_global_average_pool
            ),
            None,
        )
        if repaired_value is None:
            raise SystemExit("repaired value_info is missing after normalization")
        if repaired_value["dtype"] != repair["dtype"] or repaired_value["shape"] != repair["shape"]:
            raise SystemExit(f"repaired value_info changed during normalization: {repaired_value}")
        repair["normalized_value_info"] = repaired_value
        summary["repair"] = repair
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def select_models(models: list[dict[str, Any]], requested_ids: list[str]) -> list[dict[str, Any]]:
    """按清单顺序筛选模型，并拒绝未知或重复 ID。"""
    if not requested_ids:
        return models
    if len(requested_ids) != len(set(requested_ids)):
        raise SystemExit("duplicate --model-id values are not allowed")
    requested = set(requested_ids)
    available = {str(item.get("id")) for item in models}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit(f"unknown model IDs: {', '.join(unknown)}")
    return [item for item in models if item.get("id") in requested]


def validate_repair_arguments(args: argparse.Namespace, models: list[dict[str, Any]]) -> None:
    """修复必须显式指定单个模型和新输出路径，避免覆盖语料事实源。"""
    requested = bool(args.repair_qlinear_global_average_pool)
    if requested != bool(args.repair_out):
        raise SystemExit("repair requires both --repair-qlinear-global-average-pool and --repair-out")
    if requested and len(models) != 1:
        raise SystemExit("repair requires exactly one selected --model-id")


def repair_qlinear_global_average_pool_value_info(
    source_path: Path,
    output_path: Path,
    tensor_name: str,
) -> dict[str, Any]:
    """依据 ORT v1.22.0 schema 和模型内证据补一个缺失的中间张量声明。"""
    import onnx
    from onnx import TensorProto, helper

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise SystemExit("repair output must not overwrite the source model")
    model = onnx.load(source_path)
    producers = [node for node in model.graph.node if tensor_name in node.output]
    if len(producers) != 1:
        raise SystemExit(f"expected one producer for {tensor_name}, found {len(producers)}")
    producer = producers[0]
    if producer.domain != "com.microsoft" or producer.op_type != "QLinearGlobalAveragePool":
        raise SystemExit(
            f"{tensor_name} producer must be com.microsoft::QLinearGlobalAveragePool"
        )
    if len(producer.input) < 5:
        raise SystemExit("QLinearGlobalAveragePool must expose input and output quantization facts")

    inferred = onnx.shape_inference.infer_shapes(model)
    input_dtype, input_shape = concrete_tensor_metadata(inferred, producer.input[0])
    if input_dtype not in (TensorProto.INT8, TensorProto.UINT8):
        raise SystemExit("QLinearGlobalAveragePool input must be int8 or uint8")
    output_zero_point_dtype, _ = concrete_tensor_metadata(inferred, producer.input[4])
    if output_zero_point_dtype != input_dtype:
        raise SystemExit("output zero point dtype does not match the pooling input dtype")

    channels_last = next(
        (int(attribute.i) for attribute in producer.attribute if attribute.name == "channels_last"),
        0,
    )
    if channels_last not in (0, 1):
        raise SystemExit("channels_last must be 0 or 1")
    output_shape = qlinear_global_average_pool_shape(input_shape, channels_last)
    consumers = [
        f"{node.domain or 'ai.onnx'}::{node.op_type}"
        for node in model.graph.node
        if tensor_name in node.input
    ]
    if not consumers:
        raise SystemExit(f"{tensor_name} has no consumer to justify an intermediate value_info")

    existing = [item for item in model.graph.value_info if item.name == tensor_name]
    if existing:
        existing_dtype, existing_shape = value_info_metadata(existing[0])
        if existing_dtype == input_dtype and existing_shape == output_shape:
            raise SystemExit(f"{tensor_name} already has the required value_info")
        retained = [item for item in model.graph.value_info if item.name != tensor_name]
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained)
    model.graph.value_info.append(
        helper.make_tensor_value_info(tensor_name, input_dtype, output_shape)
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output_path)
    repaired = onnx.load(output_path)
    onnx.checker.check_model(repaired)
    dtype_name = TensorProto.DataType.Name(input_dtype).lower()
    return {
        "schema": "edgefit.onnx_value_info_repair.v1",
        "source_model": source_path.name,
        "source_sha256": sha256(source_path),
        "repaired_model": output_path.name,
        "repaired_sha256": sha256(output_path),
        "tensor": tensor_name,
        "producer": "com.microsoft::QLinearGlobalAveragePool",
        "producer_name": producer.name or None,
        "input_tensor": producer.input[0],
        "output_zero_point": producer.input[4],
        "consumers": consumers,
        "dtype": dtype_name,
        "input_shape": input_shape,
        "shape": output_shape,
        "channels_last": bool(channels_last),
        "schema_source": QLINEAR_GLOBAL_AVERAGE_POOL_SCHEMA_SOURCE,
    }


def concrete_tensor_metadata(model: Any, tensor_name: str) -> tuple[int, list[int]]:
    """从 value_info 或 initializer 读取完整元数据；动态或缺失维度直接拒绝。"""
    for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if item.name == tensor_name:
            dtype, shape = value_info_metadata(item)
            if dtype and shape is not None and all(value > 0 for value in shape):
                return dtype, shape
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return int(initializer.data_type), [int(value) for value in initializer.dims]
    raise SystemExit(f"tensor metadata is not concrete: {tensor_name}")


def value_info_metadata(value: Any) -> tuple[int, list[int] | None]:
    tensor = value.type.tensor_type
    shape = None
    if tensor.HasField("shape"):
        shape = []
        for dimension in tensor.shape.dim:
            if not dimension.HasField("dim_value"):
                return int(tensor.elem_type), None
            shape.append(int(dimension.dim_value))
    return int(tensor.elem_type), shape


def qlinear_global_average_pool_shape(input_shape: list[int], channels_last: int) -> list[int]:
    """复现 ORT schema：保留 N/C，将全部空间维压为 1。"""
    if len(input_shape) < 2:
        raise SystemExit("QLinearGlobalAveragePool input rank must be at least 2")
    output_shape = list(input_shape)
    spatial_start = 1 if channels_last else 2
    spatial_end = len(input_shape) - 1 if channels_last else len(input_shape)
    for index in range(spatial_start, spatial_end):
        output_shape[index] = 1
    return output_shape


def load_normalize():
    module_path = Path(__file__).with_name("normalize_onnx.py")
    spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.normalize


def verify_model(item: dict[str, Any], cache: Path, download: bool, normalize) -> dict[str, Any]:
    model_path = prepare_model_file(item, cache, download)
    model_bytes = model_path.stat().st_size
    model_sha = sha256(model_path)
    if model_bytes != item["model_bytes"]:
        raise SystemExit(f"model byte mismatch for {item['id']}: {model_bytes}")
    if model_sha != item["model_sha256"]:
        raise SystemExit(f"model sha256 mismatch for {item['id']}: {model_sha}")

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
        "model_bytes": model_bytes,
        "model_sha256": model_sha,
        "node_count": len(data["graph"]["nodes"]),
        "ops": ops,
        "domain_ops": domain_ops,
        "outputs": outputs,
    }


def prepare_model_file(item: dict[str, Any], cache: Path, download: bool) -> Path:
    if "archive_name" in item:
        return prepare_archive_model(item, cache, download)
    if "model_url" in item:
        return prepare_direct_model(item, cache, download)
    raise SystemExit(f"{item.get('id', '<unknown>')} must define archive_name or model_url")


def prepare_archive_model(item: dict[str, Any], cache: Path, download: bool) -> Path:
    archive = cache / item["archive_name"]
    if not archive.exists():
        if not download:
            raise SystemExit(f"missing archive {archive}; rerun with --download")
        download_file(item["archive_url"], archive)

    archive_bytes = archive.stat().st_size
    archive_sha = sha256(archive)
    if archive_bytes != item["archive_bytes"]:
        raise SystemExit(f"archive byte mismatch for {item['id']}: {archive_bytes}")
    if archive_sha != item["archive_sha256"]:
        raise SystemExit(f"archive sha256 mismatch for {item['id']}: {archive_sha}")

    model_path = cache / item["model_member"]
    if not model_path.exists():
        with tarfile.open(archive, "r:gz") as tar:
            try:
                member = tar.getmember(item["model_member"])
            except KeyError as exc:
                raise SystemExit(f"{item['model_member']} missing from {archive}") from exc
            source = tar.extractfile(member)
            if source is None:
                raise SystemExit(f"{item['model_member']} is not a regular file")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(source.read())
    return model_path


def prepare_direct_model(item: dict[str, Any], cache: Path, download: bool) -> Path:
    model_path = cache / item.get("model_name", Path(item["model_url"]).name)
    if not model_path.exists():
        if not download:
            raise SystemExit(f"missing model {model_path}; rerun with --download")
        download_file(item["model_url"], model_path)
    return model_path


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


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
