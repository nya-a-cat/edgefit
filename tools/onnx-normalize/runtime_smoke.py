from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from profile_reference_check import parse_profile, write_text
from real_world_corpus import sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "targets" / "ort-mobile-cpu.yaml"
DEFAULT_CORPUS_RESULT = ROOT / "tmp" / "real_world_corpus" / "corpus-result.json"
DEFAULT_OUT = ROOT / "tmp" / "real_world_corpus" / "runtime-smoke.json"
DEFAULT_PROVIDER = "CPUExecutionProvider"
COMPARISON_INPUT_PATTERN = "edgefit.deterministic_modulo.v1"

ORT_TYPE_TO_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

NUMPY_DTYPE_TO_EDGEFIT = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime smoke inference for verified corpus models.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--corpus-result", default=str(DEFAULT_CORPUS_RESULT))
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--reference-model", help="Reference ONNX model for exact output comparison.")
    parser.add_argument("--candidate-model", help="Candidate ONNX model for exact output comparison.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    comparison_requested = bool(args.reference_model or args.candidate_model)
    if comparison_requested and not (args.reference_model and args.candidate_model):
        raise SystemExit("runtime comparison requires both --reference-model and --candidate-model")
    if comparison_requested:
        summary = compare_runtime_models(
            Path(args.reference_model),
            Path(args.candidate_model),
            args.provider,
        )
    else:
        summary = build_summary(Path(args.profile), Path(args.corpus_result), args.provider)
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0 if summary["status"] == "pass" else 1


def build_summary(profile_path: Path, corpus_result_path: Path, provider: str) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime package is required for runtime smoke checks") from exc

    profile = parse_profile(profile_path)
    corpus = json.loads(corpus_result_path.read_text(encoding="utf-8"))
    if corpus.get("schema") != "edgefit.real_world_corpus.result.v1":
        raise SystemExit("expected schema edgefit.real_world_corpus.result.v1")

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise SystemExit(f"provider {provider} is not available: {available_providers}")

    models = []
    for model in corpus["results"]:
        if model.get("status") != "pass":
            continue
        models.append(run_model_smoke(ort, model, provider))

    status = "pass" if models and all(model["status"] == "pass" for model in models) else "fail"
    return {
        "schema": "edgefit.runtime_smoke.v1",
        "target_id": profile["target_id"],
        "profile": str(profile_path),
        "provider": provider,
        "runtime": {
            "name": "onnxruntime",
            "version": ort.__version__,
            "available_providers": available_providers,
        },
        "status": status,
        "model_count": len(models),
        "models": models,
    }


def run_model_smoke(ort: Any, model: dict[str, Any], provider: str) -> dict[str, Any]:
    model_path = Path(model["model_path"])
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])
    feeds = {item.name: make_input_array(item) for item in session.get_inputs()}
    outputs = session.run(None, feeds)
    actual_outputs = summarize_outputs(session.get_outputs(), outputs)
    expected_outputs = model.get("outputs", [])
    mismatches = compare_outputs(expected_outputs, actual_outputs)
    return {
        "id": model["id"],
        "status": "pass" if not mismatches else "fail",
        "model_path": str(model_path),
        "inputs": [summarize_input(item, feeds[item.name]) for item in session.get_inputs()],
        "outputs": actual_outputs,
        "mismatches": mismatches,
    }


def compare_runtime_models(reference_path: Path, candidate_path: Path, provider: str) -> dict[str, Any]:
    """在同一 ORT provider 和确定性输入下比较两个模型的实际输出。"""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime package is required for runtime comparison") from exc

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise SystemExit(f"provider {provider} is not available: {available_providers}")

    options = ort.SessionOptions()
    options.log_severity_level = 3
    reference = ort.InferenceSession(str(reference_path), sess_options=options, providers=[provider])
    candidate = ort.InferenceSession(str(candidate_path), sess_options=options, providers=[provider])
    reference_inputs = runtime_signature(reference.get_inputs())
    candidate_inputs = runtime_signature(candidate.get_inputs())
    if reference_inputs != candidate_inputs:
        raise SystemExit(f"runtime input signatures differ: {reference_inputs} != {candidate_inputs}")
    reference_outputs = runtime_signature(reference.get_outputs())
    candidate_outputs = runtime_signature(candidate.get_outputs())
    if reference_outputs != candidate_outputs:
        raise SystemExit(f"runtime output signatures differ: {reference_outputs} != {candidate_outputs}")

    feeds = {item.name: make_comparison_input(item) for item in reference.get_inputs()}
    reference_values = reference.run(None, feeds)
    candidate_values = candidate.run(None, feeds)
    outputs = compare_output_arrays(reference.get_outputs(), reference_values, candidate_values)
    status = "pass" if outputs and all(item["exact_match"] for item in outputs) else "fail"
    return {
        "schema": "edgefit.onnx_runtime_equivalence.v1",
        "status": status,
        "provider": provider,
        "runtime": {"name": "onnxruntime", "version": ort.__version__},
        "input_pattern": COMPARISON_INPUT_PATTERN,
        "reference_model": {
            "name": reference_path.name,
            "sha256": sha256(reference_path),
        },
        "candidate_model": {
            "name": candidate_path.name,
            "sha256": sha256(candidate_path),
        },
        "inputs": [summarize_comparison_input(item, feeds[item.name]) for item in reference.get_inputs()],
        "outputs": outputs,
    }


def runtime_signature(metadata: list[Any]) -> list[dict[str, Any]]:
    """提取运行时可见签名，确保比较对象的接口完全一致。"""
    return [
        {"name": item.name, "type": item.type, "shape": list(item.shape)}
        for item in metadata
    ]


def make_comparison_input(input_meta: Any) -> np.ndarray:
    """生成跨运行可复现的非零输入，避免随机种子或外部样本依赖。"""
    dtype = ORT_TYPE_TO_DTYPE.get(input_meta.type)
    if dtype is None:
        raise SystemExit(f"unsupported ONNX Runtime input dtype {input_meta.type} for {input_meta.name}")
    shape = strict_concrete_shape(input_meta.shape, input_meta.name)
    size = int(np.prod(shape, dtype=np.int64))
    sequence = np.arange(size, dtype=np.int64)
    if np.issubdtype(dtype, np.bool_):
        values = sequence % 2
    elif np.issubdtype(dtype, np.unsignedinteger):
        values = sequence % 251
    elif np.issubdtype(dtype, np.signedinteger):
        values = (sequence % 251) - 125
    else:
        values = ((sequence % 257) - 128) / 128.0
    return values.astype(dtype).reshape(shape)


def strict_concrete_shape(shape: list[Any], input_name: str) -> list[int]:
    """等价性证据拒绝动态维度，防止用任意替代尺寸掩盖接口差异。"""
    if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise SystemExit(f"runtime comparison requires a concrete shape for input {input_name}: {shape}")
    return list(shape)


def summarize_comparison_input(input_meta: Any, value: np.ndarray) -> dict[str, Any]:
    return {
        "name": input_meta.name,
        "type": input_meta.type,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def compare_output_arrays(
    output_meta: list[Any],
    reference_values: list[np.ndarray],
    candidate_values: list[np.ndarray],
) -> list[dict[str, Any]]:
    """要求输出数量、dtype、shape 和全部元素完全一致。"""
    if len(output_meta) != len(reference_values) or len(reference_values) != len(candidate_values):
        raise SystemExit("runtime output counts differ")
    comparisons = []
    for meta, reference, candidate in zip(output_meta, reference_values, candidate_values):
        exact = arrays_exact(reference, candidate)
        comparisons.append(
            {
                "name": meta.name,
                "dtype": edgefit_dtype(reference.dtype),
                "shape": list(reference.shape),
                "exact_match": exact,
                "max_abs_diff": max_abs_diff(reference, candidate),
            }
        )
    return comparisons


def arrays_exact(reference: np.ndarray, candidate: np.ndarray) -> bool:
    if reference.dtype != candidate.dtype or reference.shape != candidate.shape:
        return False
    equal_nan = np.issubdtype(reference.dtype, np.inexact)
    return bool(np.array_equal(reference, candidate, equal_nan=equal_nan))


def max_abs_diff(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    if reference.dtype != candidate.dtype or reference.shape != candidate.shape:
        return None
    if arrays_exact(reference, candidate):
        return 0.0
    if not np.issubdtype(reference.dtype, np.number) or reference.size == 0:
        return None
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    finite = difference[np.isfinite(difference)]
    return float(np.max(finite)) if finite.size else None


def make_input_array(input_meta: Any) -> np.ndarray:
    dtype = ORT_TYPE_TO_DTYPE.get(input_meta.type)
    if dtype is None:
        raise SystemExit(f"unsupported ONNX Runtime input dtype {input_meta.type} for {input_meta.name}")
    return np.zeros(concrete_shape(input_meta.shape), dtype=dtype)


def concrete_shape(shape: list[Any]) -> list[int]:
    return [dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape]


def summarize_input(input_meta: Any, value: np.ndarray) -> dict[str, Any]:
    return {
        "name": input_meta.name,
        "type": input_meta.type,
        "shape": list(value.shape),
    }


def summarize_outputs(output_meta: list[Any], values: list[np.ndarray]) -> list[dict[str, Any]]:
    result = []
    for meta, value in zip(output_meta, values):
        result.append(
            {
                "name": meta.name,
                "dtype": edgefit_dtype(value.dtype),
                "shape": list(value.shape),
            }
        )
    return result


def edgefit_dtype(dtype: np.dtype[Any]) -> str:
    return NUMPY_DTYPE_TO_EDGEFIT.get(np.dtype(dtype).name, np.dtype(dtype).name)


def compare_outputs(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[str]:
    mismatches = []
    actual_by_name = {item["name"]: item for item in actual}
    for expected_output in expected:
        actual_output = actual_by_name.get(expected_output["name"])
        if actual_output is None:
            mismatches.append(f"missing output {expected_output['name']}")
            continue
        if actual_output["dtype"] != expected_output["dtype"]:
            mismatches.append(f"{expected_output['name']} dtype {actual_output['dtype']} != {expected_output['dtype']}")
        if not shape_matches(expected_output["shape"], actual_output["shape"]):
            mismatches.append(f"{expected_output['name']} shape {actual_output['shape']} does not match {expected_output['shape']}")
    return mismatches


def shape_matches(expected: list[Any], actual: list[int]) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_dim, actual_dim in zip(expected, actual):
        if isinstance(expected_dim, int) and expected_dim != actual_dim:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
