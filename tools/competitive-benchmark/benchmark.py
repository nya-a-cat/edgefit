#!/usr/bin/env python3
"""EdgeFit 第一阶段竞品基准编排器。

固定比较 EdgeFit、ORT Mobile Checker 和 onnx-tool。工具只保存原始证据并
提取语义明确的指标，不会把含义不同的内存数字合成分数或自动宣布胜负。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("benchmark_manifest.json")
DEFAULT_OUT_DIR = ROOT / "tmp" / "competitive_benchmark"
MANIFEST_SCHEMA = "edgefit.competitive_benchmark_manifest.v1"
RESULT_SCHEMA = "edgefit.competitive_benchmark.v1"
TOOLS = ("edgefit", "ort-mobile", "onnx-tool")
EVIDENCE_STATUSES = {"completed", "tool_rejected"}
MISSING_DEPENDENCY_MARKERS = (
    "modulenotfounderror",
    "no module named",
    "packagenotfounderror",
    "distributionnotfound",
)


class InputError(Exception):
    """基准输入无法形成可复现证据。"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EdgeFit, ORT Mobile Checker and onnx-tool on one corpus."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--corpus-cache")
    parser.add_argument("--edgefit", default=str(default_edgefit()))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--tools",
        default=",".join(TOOLS),
        help="Comma-separated tools to run: edgefit, ort-mobile, onnx-tool.",
    )
    parser.add_argument(
        "--edgefit-repetitions",
        type=int,
        default=1,
        help="Repeat EdgeFit per case and report the median end-to-end process time.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Only run this benchmark case ID; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--measure-peak-rss",
        action="store_true",
        help="Measure EdgeFit peak RSS through GNU time on Linux.",
    )
    args = parser.parse_args()

    try:
        summary = benchmark(args)
    except InputError as exc:
        print(f"competitive-benchmark: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    json_path = out_dir / "competitive-benchmark.json"
    markdown_path = out_dir / "competitive-benchmark.md"
    write_text(json_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(markdown_path, render_markdown(summary))
    print(display_path(json_path))
    print(display_path(markdown_path))
    return 0 if summary["status"] == "complete" else 1


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout_seconds <= 0:
        raise InputError("--timeout-seconds must be greater than zero")
    selected_tools = parse_tool_selection(args.tools)
    require(
        1 <= args.edgefit_repetitions <= 20,
        "--edgefit-repetitions must be between 1 and 20",
    )
    require(
        "edgefit" in selected_tools or args.edgefit_repetitions == 1,
        "--edgefit-repetitions requires the edgefit tool",
    )
    if args.measure_peak_rss:
        require(platform.system() == "Linux", "--measure-peak-rss requires Linux")
        require(Path("/usr/bin/time").is_file(), "--measure-peak-rss requires /usr/bin/time")

    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == MANIFEST_SCHEMA, f"expected {MANIFEST_SCHEMA}")
    corpus_path = declared_path(manifest_path, text_field(manifest, "corpus_manifest"))
    corpus = read_json(corpus_path)
    require(
        corpus.get("schema") == "edgefit.real_world_corpus.v1",
        "expected edgefit.real_world_corpus.v1 corpus",
    )
    models = index_models(corpus)
    cache = (
        Path(args.corpus_cache).resolve()
        if args.corpus_cache
        else declared_path(manifest_path, text_field(manifest, "corpus_cache"))
    )
    target = declared_path(manifest_path, text_field(manifest, "default_target"))
    require(target.is_file(), f"missing target profile {target}")
    cases = select_case_specs(manifest.get("cases"), args.case_id)
    require(isinstance(cases, list) and cases, "benchmark manifest requires cases")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_cases = []
    seen = set()
    for case_spec in cases:
        require(isinstance(case_spec, dict), "benchmark cases must be objects")
        case_id = text_field(case_spec, "id")
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id) is not None,
            f"invalid benchmark case id {case_id}",
        )
        require(case_id not in seen, f"duplicate benchmark case {case_id}")
        seen.add(case_id)
        model_id = case_spec.get("model_id")
        generated_model = case_spec.get("generated_model")
        require(
            (isinstance(model_id, str) and bool(model_id.strip()))
            != isinstance(generated_model, dict),
            f"case {case_id} requires exactly one of model_id or generated_model",
        )
        if isinstance(generated_model, dict):
            model, model_path = prepare_generated_model(
                case_id,
                generated_model,
                out_dir / "generated-models",
            )
        else:
            model_id = str(model_id).strip()
            require(model_id in models, f"unknown corpus model {model_id}")
            model = models[model_id]
            model_path = resolve_model_path(model, cache)
        verify_model(model, model_path)
        prepared_cases.append((case_spec, model, model_path))

    # 所有模型和清单先通过哈希预检，再启动任何被比较工具，避免留下半套证据。
    edgefit = Path(args.edgefit)
    started_at = utc_now()
    versions = probe_versions(edgefit, args.python, args.timeout_seconds, selected_tools)
    results = [
        run_case(
            case_spec,
            model,
            model_path,
            target,
            edgefit,
            args.python,
            out_dir,
            args.timeout_seconds,
            selected_tools,
            args.edgefit_repetitions,
            args.measure_peak_rss,
        )
        for case_spec, model, model_path in prepared_cases
    ]

    status_counts = Counter(
        run["status"] for case in results for run in case["tools"].values()
    )
    versions_complete = all(
        versions[tool]["status"] == "completed" for tool in selected_tools
    )
    runs_complete = all(
        run["status"] in EVIDENCE_STATUSES
        for case in results
        for run in case["tools"].values()
    )
    comparisons = build_comparisons(manifest.get("comparisons", []), results)
    comparisons_complete = all(
        comparison["status"] == "complete" for comparison in comparisons
    )
    expectations_complete = all(
        case["expectations"]["status"] == "pass" for case in results
    )
    return {
        "schema": RESULT_SCHEMA,
        "runner_version": "3",
        "status": (
            "complete"
            if versions_complete
            and runs_complete
            and comparisons_complete
            and expectations_complete
            else "incomplete"
        ),
        "started_at": started_at,
        "finished_at": utc_now(),
        "manifest": evidence_file(manifest_path),
        "corpus_manifest": evidence_file(corpus_path),
        "target_profile": evidence_file(target),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor_count": os.cpu_count(),
        },
        "tools": list(selected_tools),
        "edgefit_repetitions": args.edgefit_repetitions,
        "peak_rss_measured": bool(args.measure_peak_rss),
        "tool_versions": versions,
        "case_count": len(results),
        "run_count": sum(status_counts.values()),
        "run_status_counts": dict(sorted(status_counts.items())),
        "cases": results,
        "comparisons": comparisons,
        "metric_boundaries": {
            "edgefit_planned_activation_arena_bytes": (
                "target-relative deterministic arena high-water mark including alignment, "
                "declared operator workspace, fragmentation, and explicitly safe in-place reuse"
            ),
            "edgefit_estimated_peak_activation_bytes": (
                "logical tensor-liveness estimate before arena placement effects"
            ),
            "onnx_tool_summed_node_memory_bytes": (
                "sum of per-node output activation and static weight memory; not peak memory"
            ),
            "ort_mobile_partition_coverage": (
                "execution-provider estimate from the ORT Mobile checker"
            ),
        },
    }


def select_case_specs(value: Any, requested_ids: list[str]) -> list[dict[str, Any]]:
    """按清单顺序选择案例，避免局部 CI 与完整证据使用不同定义。"""
    require(isinstance(value, list) and value, "benchmark manifest requires cases")
    require(all(isinstance(item, dict) for item in value), "benchmark cases must be objects")
    if not requested_ids:
        return value
    require(len(requested_ids) == len(set(requested_ids)), "duplicate --case-id values")
    available = {text_field(item, "id") for item in value}
    unknown = sorted(set(requested_ids) - available)
    require(not unknown, f"unknown case IDs: {', '.join(unknown)}")
    selected = set(requested_ids)
    return [item for item in value if text_field(item, "id") in selected]


def prepare_generated_model(
    case_id: str,
    spec: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """生成确定性 Relu 链，用真实大图压测解析、分析、规划与报告全链。"""
    require(spec.get("kind") == "linear_relu_chain", f"unsupported generator for {case_id}")
    node_count = spec.get("node_count")
    tensor_elements = spec.get("tensor_elements")
    dtype = spec.get("dtype", "float32")
    require(
        isinstance(node_count, int) and not isinstance(node_count, bool) and 1 <= node_count <= 100_000,
        f"generated node_count for {case_id} must be between 1 and 100000",
    )
    require(
        isinstance(tensor_elements, int)
        and not isinstance(tensor_elements, bool)
        and 1 <= tensor_elements <= 65_536,
        f"generated tensor_elements for {case_id} must be between 1 and 65536",
    )
    require(dtype == "float32", f"generated dtype for {case_id} must be float32")

    generator_fingerprint = hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    values = [
        {"name": f"v{index}", "dtype": dtype, "shape": [1, tensor_elements]}
        for index in range(node_count - 1)
    ]
    nodes = []
    for index in range(node_count):
        nodes.append(
            {
                "name": f"relu_{index}",
                "domain": "ai.onnx",
                "op_type": "Relu",
                "inputs": ["input" if index == 0 else f"v{index - 1}"],
                "outputs": [f"v{index}"],
            }
        )
    data = {
        "schema": "edgefit.normalized_model.v1",
        "model": {
            "path": f"generated/{case_id}.onnx",
            "file_bytes": 0,
            # 生成案例没有原始 ONNX 文件；这里固定记录生成规格指纹，实际 JSON
            # 文件哈希由案例 evidence_file 单独记录并在运行前校验。
            "sha256": f"sha256:{generator_fingerprint}",
        },
        "graph": {
            "inputs": [{"name": "input", "dtype": dtype, "shape": [1, tensor_elements]}],
            "values": values,
            "outputs": [{"name": f"v{node_count - 1}", "dtype": dtype, "shape": [1, tensor_elements]}],
            "initializers": [],
            "nodes": nodes,
        },
    }
    # file_bytes 本身会改变 JSON 长度，迭代到位数稳定后再写入事实值。
    text = ""
    for _ in range(3):
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        size = len(text.encode("utf-8"))
        if data["model"]["file_bytes"] == size:
            break
        data["model"]["file_bytes"] = size
    output_path = output_dir / f"{case_id}.edgefit.json"
    write_text(output_path, text)
    model_hash = sha256(output_path)
    return (
        {
            "id": f"generated:{case_id}",
            "model_bytes": output_path.stat().st_size,
            "model_sha256": model_hash,
        },
        output_path,
    )


def run_case(
    case_spec: dict[str, Any],
    model: dict[str, Any],
    model_path: Path,
    target: Path,
    edgefit: Path,
    python: str,
    out_dir: Path,
    timeout: int,
    selected_tools: tuple[str, ...],
    edgefit_repetitions: int,
    measure_peak_rss: bool,
) -> dict[str, Any]:
    case_id = text_field(case_spec, "id")
    tags = case_spec.get("tags", [])
    require(
        isinstance(tags, list) and all(isinstance(tag, str) for tag in tags),
        f"invalid tags for {case_id}",
    )
    case_dir = out_dir / "artifacts" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    tools = {}
    if "edgefit" in selected_tools:
        tools["edgefit"] = run_tool(
            "edgefit",
            [
                str(edgefit),
                "check",
                str(model_path),
                "--target",
                str(target),
                "--format",
                "json",
                "--out",
                str(case_dir / "edgefit-report.json"),
            ],
            case_dir,
            timeout,
            {0, 1},
            parse_edgefit,
            case_dir / "edgefit-report.json",
            {**os.environ, "EDGEFIT_PYTHON": python},
            repetitions=edgefit_repetitions,
            measure_peak_rss=measure_peak_rss,
        )
    if "ort-mobile" in selected_tools:
        tools["ort-mobile"] = run_tool(
            "ort-mobile",
            [
                python,
                "-m",
                "onnxruntime.tools.check_onnx_model_mobile_usability",
                str(model_path),
            ],
            case_dir,
            timeout,
            {0},
            parse_ort_mobile,
        )
    if "onnx-tool" in selected_tools:
        tools["onnx-tool"] = run_tool(
            "onnx-tool",
            [
                python,
                "-m",
                "onnx_tool",
                "--mode",
                "profile",
                "--in",
                str(model_path),
                "--file",
                str(case_dir / "onnx-tool-profile.csv"),
            ],
            case_dir,
            timeout,
            {0},
            parse_onnx_tool,
            case_dir / "onnx-tool-profile.csv",
        )
    expectations = evaluate_case_expectations(case_spec, tools)
    return {
        "id": case_id,
        "model_id": model["id"],
        "model": evidence_file(model_path),
        "target": evidence_file(target),
        "purpose": str(case_spec.get("purpose", "")).strip(),
        "tags": tags,
        "generated_model": case_spec.get("generated_model"),
        "tools": tools,
        "expectations": expectations,
    }


def evaluate_case_expectations(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """把性能上限作为案例证据的一部分，缺失测量不能被当成零。"""
    spec = case_spec.get("expectations", {})
    require(isinstance(spec, dict), f"expectations for {case_spec.get('id')} must be an object")
    unknown = sorted(
        set(spec)
        - {
            "max_edgefit_duration_ms",
            "max_edgefit_peak_rss_bytes",
            "expected_edgefit_node_count",
            "require_deterministic_artifact",
        }
    )
    require(not unknown, f"unknown expectations for {case_spec.get('id')}: {', '.join(unknown)}")
    if not spec:
        return {"status": "pass", "checks": []}
    edgefit = tools.get("edgefit")
    require(edgefit is not None, f"expectations for {case_spec.get('id')} require edgefit")
    checks = []

    def add(name: str, actual: Any, limit: Any, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "expected": limit, "passed": passed})

    if "max_edgefit_duration_ms" in spec:
        limit = spec["max_edgefit_duration_ms"]
        require(
            isinstance(limit, int) and not isinstance(limit, bool) and limit > 0,
            "max_edgefit_duration_ms must be positive",
        )
        actual = edgefit.get("duration_ms")
        add("max_edgefit_duration_ms", actual, limit, isinstance(actual, int) and actual <= limit)
    if "max_edgefit_peak_rss_bytes" in spec:
        limit = spec["max_edgefit_peak_rss_bytes"]
        require(
            isinstance(limit, int) and not isinstance(limit, bool) and limit > 0,
            "max_edgefit_peak_rss_bytes must be positive",
        )
        actual = edgefit.get("peak_rss_bytes")
        add("max_edgefit_peak_rss_bytes", actual, limit, isinstance(actual, int) and actual <= limit)
    if "expected_edgefit_node_count" in spec:
        expected = spec["expected_edgefit_node_count"]
        require(
            isinstance(expected, int) and not isinstance(expected, bool) and expected > 0,
            "expected_edgefit_node_count must be positive",
        )
        actual = edgefit.get("observations", {}).get("node_count")
        add("expected_edgefit_node_count", actual, expected, actual == expected)
    if "require_deterministic_artifact" in spec:
        expected = spec["require_deterministic_artifact"]
        require(isinstance(expected, bool), "require_deterministic_artifact must be boolean")
        actual = edgefit.get("artifact_deterministic")
        add("require_deterministic_artifact", actual, expected, actual is expected)
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
    }


def run_tool(
    name: str,
    command: list[str],
    case_dir: Path,
    timeout: int,
    accepted_codes: set[int],
    parser: Callable[[str, Path | None], dict[str, Any]],
    artifact: Path | None = None,
    env: dict[str, str] | None = None,
    repetitions: int = 1,
    measure_peak_rss: bool = False,
) -> dict[str, Any]:
    processes = []
    artifact_hashes = []
    for _ in range(repetitions):
        if artifact and artifact.is_file():
            artifact.unlink()
        process = run_process(command, timeout, env, measure_peak_rss)
        processes.append(process)
        if artifact and artifact.is_file():
            artifact_hashes.append(sha256(artifact))
    statuses = [process_status(process, accepted_codes) for process in processes]
    first_failure = next(
        (index for index, status in enumerate(statuses) if status != "completed"),
        None,
    )
    process = processes[first_failure] if first_failure is not None else processes[-1]
    status = statuses[first_failure] if first_failure is not None else "completed"
    combined = "\n".join(
        part for part in [process["stdout"], process["stderr"]] if part
    )
    observations: dict[str, Any] = {}
    detail = ""
    if status == "completed":
        if artifact and len(artifact_hashes) != repetitions:
            status = "runner_error"
            detail = f"{name} completed without writing its requested artifact"
        elif len(set(artifact_hashes)) > 1:
            status = "runner_error"
            detail = f"{name} produced non-deterministic artifacts across repetitions"
        else:
            try:
                observations = parser(combined, artifact)
            except InputError as exc:
                status = "runner_error"
                detail = str(exc)
    elif name == "ort-mobile":
        # ORT 的失败日志仍可能包含有价值的兼容性证据。
        observations = parser(combined, None)

    stdout_path = case_dir / f"{name}.stdout.txt"
    stderr_path = case_dir / f"{name}.stderr.txt"
    stdout = sanitize(process["stdout"])
    stderr = sanitize(process["stderr"])
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    artifacts = [evidence_file(stdout_path), evidence_file(stderr_path)]
    if artifact and artifact.is_file():
        artifacts.insert(0, evidence_file(artifact))
    if not detail and status != "completed":
        detail = first_nonempty(stderr.strip(), stdout.strip())[:1000]
    result = {
        "status": status,
        "exit_code": process["exit_code"],
        "duration_ms": int(median(item["duration_ms"] for item in processes)),
        "duration_samples_ms": [item["duration_ms"] for item in processes],
        "peak_rss_bytes": max(
            (item["peak_rss_bytes"] for item in processes if item.get("peak_rss_bytes") is not None),
            default=None,
        ),
        "peak_rss_samples_bytes": [item.get("peak_rss_bytes") for item in processes],
        "artifact_sha256_samples": [f"sha256:{value}" for value in artifact_hashes],
        "artifact_deterministic": len(artifact_hashes) == repetitions
        and len(set(artifact_hashes)) <= 1,
        "command": [sanitize(arg) for arg in command],
        "observations": observations,
        "artifacts": artifacts,
    }
    if detail:
        result["detail"] = detail
    return result


def parse_edgefit(_: str, artifact: Path | None) -> dict[str, Any]:
    require(artifact is not None, "EdgeFit report path is missing")
    report = read_json(artifact)
    require(report.get("schema") == "edgefit.report.v1", "unsupported EdgeFit report schema")
    require(report.get("status") in {"pass", "fail"}, "invalid EdgeFit report status")
    metrics = report.get("metrics")
    diagnostics = report.get("diagnostics")
    require(isinstance(metrics, dict), "EdgeFit report has no metrics object")
    require(isinstance(diagnostics, list), "EdgeFit report has no diagnostics array")
    # 关键内存指标缺失时拒绝生成“完成”证据，避免旧报告被静默解释为 None。
    for field in (
        "estimated_peak_activation_bytes",
        "planned_activation_arena_bytes",
        "activation_tensor_alignment_bytes",
    ):
        value = metrics.get(field)
        require(
            isinstance(value, int) and not isinstance(value, bool),
            f"EdgeFit metric {field} must be an integer",
        )
    require(
        isinstance(metrics.get("activation_planner_algorithm"), str)
        and bool(metrics["activation_planner_algorithm"].strip()),
        "EdgeFit metric activation_planner_algorithm must be a non-empty string",
    )
    require(
        isinstance(metrics.get("activation_planning_overflowed"), bool),
        "EdgeFit metric activation_planning_overflowed must be a boolean",
    )
    severities = Counter(
        str(item.get("severity", "unknown"))
        for item in diagnostics
        if isinstance(item, dict)
    )
    return {
        "edgefit_version": report.get("edgefit_version"),
        "verdict": report["status"],
        "diagnostic_count": len(diagnostics),
        "error_count": severities.get("error", 0),
        "warning_count": severities.get("warning", 0),
        "diagnostic_ids": sorted(
            str(item["id"])
            for item in diagnostics
            if isinstance(item, dict) and item.get("id")
        ),
        "model_file_bytes": metrics.get("model_file_bytes"),
        "initializer_bytes": metrics.get("initializer_bytes"),
        "estimated_peak_activation_bytes": metrics.get("estimated_peak_activation_bytes"),
        "planned_activation_arena_bytes": metrics.get("planned_activation_arena_bytes"),
        "activation_tensor_alignment_bytes": metrics.get(
            "activation_tensor_alignment_bytes"
        ),
        "activation_planner_algorithm": metrics.get("activation_planner_algorithm"),
        "activation_planning_overflowed": metrics.get(
            "activation_planning_overflowed"
        ),
        "peak_activation_event": metrics.get("peak_activation_event"),
        "peak_activation_node_index": metrics.get("peak_activation_node_index"),
        "peak_activation_node_name": metrics.get("peak_activation_node_name"),
        "peak_activation_op_type": metrics.get("peak_activation_op_type"),
        "peak_activation_workspace_bytes": metrics.get(
            "peak_activation_workspace_bytes"
        ),
        "peak_activation_fragmentation_bytes": metrics.get(
            "peak_activation_fragmentation_bytes"
        ),
        "inplace_reuse_count": metrics.get("inplace_reuse_count"),
        "inplace_avoided_allocation_bytes": metrics.get(
            "inplace_avoided_allocation_bytes"
        ),
        "peak_activation_contributors": metrics.get(
            "peak_activation_contributors", []
        ),
        "peak_activation_confidence": metrics.get("peak_activation_confidence"),
        "dynamic_tensor_count": metrics.get("dynamic_tensor_count"),
        "unresolved_tensor_size_count": metrics.get("unresolved_tensor_size_count"),
        "unsupported_ops": metrics.get("unsupported_ops", []),
        "unknown_dtype_tensors": metrics.get("unknown_dtype_tensors", []),
        "quantization_representation": metrics.get("quantization_representation"),
        "quantization_operator_coverage": metrics.get("quantization_operator_coverage"),
        "node_count": metrics.get("node_count"),
        "tensor_count": metrics.get("tensor_count"),
    }


def parse_ort_mobile(output: str, _: Path | None) -> dict[str, Any]:
    recommendations = []
    partitions = []
    unsupported_ops = set()
    dynamic_shape_counts = []
    recommendation_re = re.compile(
        r"Model should perform well with (.+?):\s*(YES|MAYBE|NO)\s*$", re.I
    )
    partition_re = re.compile(
        r"(\d+) partitions? with a total of (\d+)/(\d+) nodes can be handled by the (.+?) EP",
        re.I,
    )
    dynamic_re = re.compile(r"dynamic shape\s*=\s*(\d+)", re.I)
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if match := recommendation_re.search(line):
            recommendations.append(
                {"scenario": match.group(1).strip(), "verdict": match.group(2).upper()}
            )
        if match := partition_re.search(line):
            partitions.append(
                {
                    "partition_count": int(match.group(1)),
                    "supported_node_count": int(match.group(2)),
                    "total_node_count": int(match.group(3)),
                    "execution_provider": match.group(4).strip(),
                }
            )
        if "Unsupported ops:" in line:
            unsupported_ops.update(
                value.strip()
                for value in line.split("Unsupported ops:", 1)[1].split(",")
                if value.strip()
            )
        if match := dynamic_re.search(line):
            dynamic_shape_counts.append(int(match.group(1)))
    return {
        "recommendations": recommendations,
        "partition_summaries": partitions,
        "unsupported_ops": sorted(unsupported_ops),
        "dynamic_shape_rejected_node_counts": dynamic_shape_counts,
    }


def parse_onnx_tool(_: str, artifact: Path | None) -> dict[str, Any]:
    require(artifact is not None, "onnx-tool profile path is missing")
    try:
        with artifact.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise InputError(f"failed to parse onnx-tool CSV: {exc}") from exc
    require("Name" in fields, "onnx-tool CSV has no Name column")
    total = next((row for row in rows if row.get("Name") == "Total"), None)
    require(total is not None, "onnx-tool CSV has no Total row")
    dedup = next((row for row in rows if row.get("Name") == "Dedup_Params"), None)
    forward_key = next((field for field in fields if field.startswith("Forward_")), None)
    return {
        "node_row_count": sum(
            row.get("Name") not in {"Total", "Dedup_Params"} for row in rows
        ),
        "forward_metric": forward_key,
        "forward_operations": scaled_integer(total.get(forward_key)) if forward_key else None,
        "summed_node_memory_bytes": scaled_integer(total.get("Memory")),
        "parameter_elements": scaled_integer(total.get("Params")),
        "deduplicated_parameter_elements": (
            scaled_integer(dedup.get("Params")) if dedup else scaled_integer(total.get("Params"))
        ),
        "memory_metric_semantics": (
            "sum of per-node output activation and static weight memory; not peak memory"
        ),
    }


def run_process(
    command: list[str],
    timeout: int,
    env: dict[str, str] | None = None,
    measure_peak_rss: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    actual_command = command
    if measure_peak_rss:
        actual_command = ["/usr/bin/time", "-f", "EDGEFIT_MAX_RSS_KB=%M", *command]
    try:
        completed = subprocess.run(
            actual_command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stderr = completed.stderr
        peak_rss_bytes = None
        if measure_peak_rss:
            matches = re.findall(r"^EDGEFIT_MAX_RSS_KB=(\d+)\s*$", stderr, re.MULTILINE)
            require(len(matches) == 1, "GNU time did not emit one peak RSS measurement")
            peak_rss_bytes = int(matches[0]) * 1024
            stderr = re.sub(r"^EDGEFIT_MAX_RSS_KB=\d+\s*$", "", stderr, flags=re.MULTILINE)
        return {
            "state": "finished",
            "exit_code": completed.returncode,
            "duration_ms": elapsed_ms(started),
            "stdout": completed.stdout,
            "stderr": stderr,
            "peak_rss_bytes": peak_rss_bytes,
        }
    except FileNotFoundError as exc:
        return process_failure("unavailable", started, str(exc))
    except subprocess.TimeoutExpired as exc:
        return {
            "state": "timed_out",
            "exit_code": None,
            "duration_ms": elapsed_ms(started),
            "stdout": process_text(exc.stdout),
            "stderr": process_text(exc.stderr),
            "peak_rss_bytes": None,
        }
    except OSError as exc:
        return process_failure("runner_error", started, str(exc))


def process_failure(state: str, started: int, detail: str) -> dict[str, Any]:
    return {
        "state": state,
        "exit_code": None,
        "duration_ms": elapsed_ms(started),
        "stdout": "",
        "stderr": detail,
        "peak_rss_bytes": None,
    }


def process_status(process: dict[str, Any], accepted_codes: set[int]) -> str:
    if process["state"] != "finished":
        return str(process["state"])
    if process["exit_code"] in accepted_codes:
        return "completed"
    output = f"{process['stdout']}\n{process['stderr']}".lower()
    if any(marker in output for marker in MISSING_DEPENDENCY_MARKERS):
        return "unavailable"
    return "tool_rejected"


def probe_versions(
    edgefit: Path,
    python: str,
    timeout: int,
    selected_tools: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    code = "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))"
    commands = {
        "edgefit": [str(edgefit), "--version"],
        "ort-mobile": [python, "-c", code, "onnxruntime"],
        "onnx-tool": [python, "-c", code, "onnx-tool"],
    }
    versions = {}
    for name in selected_tools:
        command = commands[name]
        process = run_process(command, timeout)
        status = process_status(process, {0})
        stdout = sanitize(process["stdout"]).strip()
        stderr = sanitize(process["stderr"]).strip()
        versions[name] = {
            "status": status,
            "version": stdout.splitlines()[0] if status == "completed" and stdout else None,
            "command": [sanitize(arg) for arg in command],
            "detail": first_nonempty(stderr, stdout)[:1000],
        }
    return versions


def parse_tool_selection(value: str) -> tuple[str, ...]:
    """解析工具选择并保持公共工具顺序，避免同一清单产生随机排序。"""
    requested = [item.strip() for item in value.split(",") if item.strip()]
    require(requested, "--tools must select at least one tool")
    require(len(requested) == len(set(requested)), "--tools contains duplicates")
    unknown = sorted(set(requested) - set(TOOLS))
    require(not unknown, f"unknown tools: {', '.join(unknown)}")
    return tuple(tool for tool in TOOLS if tool in requested)


def build_comparisons(
    specs: Any,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从同一次固定清单运行中提取前后对照，不跨环境拼接数字。"""
    require(isinstance(specs, list), "comparisons must be an array")
    case_index = {case["id"]: case for case in cases}
    comparisons = []
    seen = set()
    for spec in specs:
        require(isinstance(spec, dict), "comparison entries must be objects")
        comparison_id = text_field(spec, "id")
        baseline_id = text_field(spec, "baseline_case")
        candidate_id = text_field(spec, "candidate_case")
        require(comparison_id not in seen, f"duplicate comparison {comparison_id}")
        require(baseline_id != candidate_id, f"comparison {comparison_id} reuses one case")
        require(baseline_id in case_index, f"unknown baseline case {baseline_id}")
        require(candidate_id in case_index, f"unknown candidate case {candidate_id}")
        seen.add(comparison_id)
        baseline_run = case_index[baseline_id]["tools"].get("edgefit")
        candidate_run = case_index[candidate_id]["tools"].get("edgefit")
        require(
            baseline_run is not None and candidate_run is not None,
            f"comparison {comparison_id} requires edgefit evidence",
        )
        comparison = {
            "id": comparison_id,
            "baseline_case": baseline_id,
            "candidate_case": candidate_id,
            "hypothesis": str(spec.get("hypothesis", "")).strip(),
            "status": "complete",
            "metrics": {},
        }
        if baseline_run["status"] != "completed" or candidate_run["status"] != "completed":
            comparison["status"] = "incomplete"
            comparisons.append(comparison)
            continue
        before = baseline_run["observations"]
        after = candidate_run["observations"]
        for field in (
            "model_file_bytes",
            "initializer_bytes",
            "estimated_peak_activation_bytes",
            "planned_activation_arena_bytes",
        ):
            comparison["metrics"][field] = compare_integer(before.get(field), after.get(field))
        comparison["metrics"]["edgefit_process_duration_ms"] = compare_integer(
            baseline_run["duration_ms"], candidate_run["duration_ms"]
        )
        comparison["baseline"] = {
            "verdict": before.get("verdict"),
            "quantization_representation": before.get("quantization_representation"),
            "peak_node": before.get("peak_activation_node_name")
            or before.get("peak_activation_node_index"),
            "peak_op_type": before.get("peak_activation_op_type"),
        }
        comparison["candidate"] = {
            "verdict": after.get("verdict"),
            "quantization_representation": after.get("quantization_representation"),
            "peak_node": after.get("peak_activation_node_name")
            or after.get("peak_activation_node_index"),
            "peak_op_type": after.get("peak_activation_op_type"),
        }
        comparisons.append(comparison)
    return comparisons


def compare_integer(before: Any, after: Any) -> dict[str, Any]:
    """保留原值和差值；只有正基线才给出减少比例。"""
    if not isinstance(before, int) or isinstance(before, bool):
        return {"before": before, "after": after, "delta": None, "reduction_percent": None}
    if not isinstance(after, int) or isinstance(after, bool):
        return {"before": before, "after": after, "delta": None, "reduction_percent": None}
    reduction = round((before - after) * 10000 / before) / 100 if before > 0 else None
    return {
        "before": before,
        "after": after,
        "delta": after - before,
        "reduction_percent": reduction,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise InputError(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def index_models(corpus: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = corpus.get("models")
    require(isinstance(items, list) and items, "corpus requires models")
    index = {}
    for item in items:
        require(isinstance(item, dict), "corpus models must be objects")
        model_id = text_field(item, "id")
        require(model_id not in index, f"duplicate corpus model {model_id}")
        index[model_id] = item
    return index


def resolve_model_path(model: dict[str, Any], cache: Path) -> Path:
    if isinstance(model.get("model_member"), str):
        return (cache / model["model_member"]).resolve()
    if isinstance(model.get("model_name"), str):
        return (cache / model["model_name"]).resolve()
    if isinstance(model.get("model_url"), str):
        name = Path(urlparse(model["model_url"]).path).name
        if name:
            return (cache / name).resolve()
    raise InputError(f"corpus model {model.get('id')} has no model path")


def verify_model(model: dict[str, Any], path: Path) -> None:
    require(path.is_file(), f"missing corpus model {path}; prepare the corpus first")
    require(path.stat().st_size == model.get("model_bytes"), f"byte mismatch for {model['id']}")
    expected = str(model.get("model_sha256", "")).removeprefix("sha256:").lower()
    require(expected and sha256(path) == expected, f"sha256 mismatch for {model['id']}")


def evidence_file(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "bytes": path.stat().st_size,
        "sha256": "sha256:" + sha256(path),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    tools = tuple(summary.get("tools", TOOLS))
    lines = [
        "# EdgeFit Competitive Benchmark",
        "",
        f"**Status:** `{summary['status']}`",
        f"**Cases:** `{summary['case_count']}`",
        f"**Runs:** `{summary['run_count']}`",
        "",
        "## Tool versions",
        "",
        "| Tool | Probe | Version |",
        "| --- | --- | --- |",
    ]
    for tool in tools:
        probe = summary["tool_versions"][tool]
        lines.append(
            f"| `{tool}` | `{probe['status']}` | `{md(probe.get('version') or 'unknown')}` |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Case | Tool | Run | Exit | Duration | Peak RSS | Observation |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for case in summary["cases"]:
        for tool in tools:
            result = case["tools"][tool]
            exit_code = result["exit_code"] if result["exit_code"] is not None else ""
            peak_rss = result.get("peak_rss_bytes")
            peak_rss_text = str(peak_rss) if peak_rss is not None else ""
            lines.append(
                f"| `{md(case['id'])}` | `{tool}` | `{result['status']}` | {exit_code} | "
                f"{result['duration_ms']} ms | {peak_rss_text} | {md(observation(tool, result))} |"
            )
    expectation_cases = [case for case in summary["cases"] if case["expectations"]["checks"]]
    if expectation_cases:
        lines.extend(
            [
                "",
                "## Case expectations",
                "",
                "| Case | Status | Check | Actual | Expected |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for case in expectation_cases:
            for check in case["expectations"]["checks"]:
                lines.append(
                    f"| `{md(case['id'])}` | `{case['expectations']['status']}` | "
                    f"`{md(check['name'])}` | {md(check['actual'])} | {md(check['expected'])} |"
                )
    if summary.get("comparisons"):
        lines.extend(["", "## Before/after comparisons", ""])
        for comparison in summary["comparisons"]:
            lines.extend(
                [
                    f"### {md(comparison['id'])}",
                    "",
                    f"**Status:** `{comparison['status']}`  ",
                    f"**Baseline:** `{md(comparison['baseline_case'])}`  ",
                    f"**Candidate:** `{md(comparison['candidate_case'])}`  ",
                    f"**Hypothesis:** {md(comparison.get('hypothesis') or 'not specified')}",
                    "",
                ]
            )
            if comparison["status"] != "complete":
                lines.append("EdgeFit evidence is incomplete; no numeric comparison is reported.")
                continue
            lines.extend(
                [
                    "| Metric | Baseline | Candidate | Delta | Reduction |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for field, values in comparison["metrics"].items():
                reduction = values["reduction_percent"]
                reduction_text = f"{reduction:.2f}%" if reduction is not None else "n/a"
                lines.append(
                    f"| `{field}` | {values['before']} | {values['after']} | "
                    f"{values['delta']} | {reduction_text} |"
                )
            baseline = comparison["baseline"]
            candidate = comparison["candidate"]
            lines.extend(
                [
                    "",
                    f"- Verdict: `{md(baseline.get('verdict'))}` → `{md(candidate.get('verdict'))}`.",
                    f"- Quantization: `{md(baseline.get('quantization_representation'))}` → `{md(candidate.get('quantization_representation'))}`.",
                    f"- Peak location: `{md(baseline.get('peak_node'))}` / `{md(baseline.get('peak_op_type'))}` → "
                    f"`{md(candidate.get('peak_node'))}` / `{md(candidate.get('peak_op_type'))}`.",
                    "- Duration is the median end-to-end process time on this runner, not device inference latency.",
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "## Metric boundaries",
            "",
            "- EdgeFit reports both logical live tensor bytes and a target-relative planned arena high-water mark.",
            "- The planned arena includes declared alignment, workspace, fragmentation and explicitly safe in-place reuse.",
            "- onnx-tool `Total/Memory` is summed per-node memory, not peak activation memory.",
            "- ORT Mobile partition coverage applies only to the execution providers checked by ORT.",
            "- `tool_rejected` is valid evidence; missing tools, timeouts and runner errors make the suite incomplete.",
            "- The benchmark records evidence and does not automatically declare a winner.",
            "",
        ]
    )
    return "\n".join(lines)


def observation(tool: str, result: dict[str, Any]) -> str:
    data = result.get("observations", {})
    if tool == "edgefit" and data:
        return (
            f"verdict={data.get('verdict')}; planned_arena={data.get('planned_activation_arena_bytes')}; "
            f"peak_node={data.get('peak_activation_node_name') or data.get('peak_activation_node_index')}; "
            f"planner_overflowed={data.get('activation_planning_overflowed')}; "
            f"inplace_reuse={data.get('inplace_reuse_count')}; "
            f"unsupported_ops={len(data.get('unsupported_ops', []))}"
        )
    if tool == "ort-mobile" and data:
        values = ", ".join(
            f"{item['scenario']}={item['verdict']}" for item in data.get("recommendations", [])
        )
        return values or (
            f"partitions={len(data.get('partition_summaries', []))}; "
            f"unsupported_ops={len(data.get('unsupported_ops', []))}"
        )
    if tool == "onnx-tool" and data:
        return (
            f"forward_ops={data.get('forward_operations')}; "
            f"summed_node_memory={data.get('summed_node_memory_bytes')}; "
            f"params={data.get('parameter_elements')}"
        )
    return str(result.get("detail", result["status"]))


def scaled_integer(value: Any) -> int | None:
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace("_", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([kKmMgGtT]?)", value)
    if not match:
        return None
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000, "t": 1_000_000_000_000}
    return int(round(float(match.group(1)) * multiplier[match.group(2).lower()]))


def text_field(value: dict[str, Any], field: str) -> str:
    item = value.get(field)
    require(isinstance(item, str) and bool(item.strip()), f"{field} is required")
    return item.strip()


def declared_path(owner: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def default_edgefit() -> Path:
    executable = "edgefit.exe" if os.name == "nt" else "edgefit"
    candidates = [
        ROOT / "tmp" / "cargo-target" / "debug" / executable,
        ROOT / "target" / "debug" / executable,
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sanitize(value: str) -> str:
    value = process_text(value)
    for source, target in [(str(ROOT), "<repo>"), (str(Path.home()), "<home>")]:
        value = value.replace(source, target)
    return value


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return "<repo>/" + path.relative_to(ROOT).as_posix()
    except ValueError:
        return sanitize(str(path))


def process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def elapsed_ms(started: int) -> int:
    return max(0, (time.perf_counter_ns() - started) // 1_000_000)


def first_nonempty(*values: str) -> str:
    return next((value for value in values if value), "")


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
