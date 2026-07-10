from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from profile_reference_check import parse_profile, write_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "targets" / "ort-mobile-cpu.yaml"
DEFAULT_CORPUS_RESULT = ROOT / "tmp" / "real_world_corpus" / "corpus-result.json"
DEFAULT_OUT = ROOT / "tmp" / "real_world_corpus" / "runtime-smoke.json"
DEFAULT_PROVIDER = "CPUExecutionProvider"

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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

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