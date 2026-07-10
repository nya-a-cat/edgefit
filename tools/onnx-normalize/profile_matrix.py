"""EdgeFit 真实模型与目标配置矩阵编排器。

本模块只汇总各目标下可比较的稳定指标；完整分配轨迹保留在原始 JSON 报告中。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import real_world_corpus


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("real_world_corpus.json")
DEFAULT_CACHE = ROOT / "tmp" / "real_world_corpus"
DEFAULT_REPORT_DIR = DEFAULT_CACHE / "profile_matrix"
DEFAULT_PROFILES = [
    ROOT / "targets" / "esp32s3.yaml",
    ROOT / "targets" / "ort-mobile-cpu.yaml",
    ROOT / "targets" / "tflm-micro.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EdgeFit corpus models across target profiles.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--download", action="store_true", help="Download missing corpus archives.")
    parser.add_argument("--edgefit", default=str(find_default_edgefit()))
    parser.add_argument("--profile", action="append", help="Target profile path. Repeat for multiple profiles.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--out", help="Write matrix JSON to this path.")
    parser.add_argument("--markdown-out", help="Write matrix Markdown to this path.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    cache = Path(args.cache)
    report_dir = Path(args.report_dir)
    edgefit = Path(args.edgefit)
    profiles = [Path(value) for value in args.profile] if args.profile else DEFAULT_PROFILES

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    normalize = real_world_corpus.load_normalize()
    models = [
        real_world_corpus.verify_model(item, cache, args.download, normalize)
        for item in manifest["models"]
    ]

    cells = []
    for model in models:
        for profile in profiles:
            cells.append(run_profile_check(edgefit, model, profile, report_dir))

    summary = {
        "schema": "edgefit.profile_matrix.v1",
        "edgefit": str(edgefit),
        "models": [compact_model(item) for item in models],
        "profiles": [str(profile) for profile in profiles],
        "matrix": cells,
    }

    if args.out:
        write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.markdown_out:
        write_text(Path(args.markdown_out), render_markdown(summary))

    return 0


def find_default_edgefit() -> Path:
    executable = "edgefit.exe" if os.name == "nt" else "edgefit"
    return ROOT / "target" / "debug" / executable


def run_profile_check(
    edgefit: Path, model: dict[str, Any], profile: Path, report_dir: Path
) -> dict[str, Any]:
    report_path = report_dir / f"{safe_name(model['id'])}__{safe_name(profile.stem)}.json"
    command = [
        str(edgefit),
        "check",
        model["model_path"],
        "--target",
        str(profile),
        "--format",
        "json",
        "--out",
        str(report_path),
    ]
    env = os.environ.copy()
    env["EDGEFIT_PYTHON"] = sys.executable
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"edgefit check failed for {model['id']} on {profile}: {detail}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = report.get("diagnostics", [])
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"edgefit report has no metrics object: {report_path}")
    # 矩阵必须绑定具体 planner 合同，不能把旧报告的缺失字段当作空结果。
    for field in (
        "estimated_peak_activation_bytes",
        "planned_activation_arena_bytes",
        "activation_tensor_alignment_bytes",
    ):
        value = metrics.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise SystemExit(f"edgefit report metric {field} must be an integer: {report_path}")
    planner_algorithm = metrics.get("activation_planner_algorithm")
    if not isinstance(planner_algorithm, str) or not planner_algorithm.strip():
        raise SystemExit(
            "edgefit report metric activation_planner_algorithm must be a non-empty "
            f"string: {report_path}"
        )
    planner_overflowed = metrics.get("activation_planning_overflowed")
    if not isinstance(planner_overflowed, bool):
        raise SystemExit(
            "edgefit report metric activation_planning_overflowed must be a boolean: "
            f"{report_path}"
        )
    severity_counts = count_severities(diagnostics)
    return {
        "model_id": model["id"],
        "target_id": report["target"]["id"],
        "profile": str(profile),
        "status": report["status"],
        "exit_code": completed.returncode,
        "diagnostic_count": len(diagnostics),
        "error_count": severity_counts.get("error", 0),
        "warning_count": severity_counts.get("warning", 0),
        "diagnostic_ids": sorted({item["id"] for item in diagnostics}),
        "estimated_peak_activation_bytes": metrics.get(
            "estimated_peak_activation_bytes"
        ),
        "planned_activation_arena_bytes": metrics.get(
            "planned_activation_arena_bytes"
        ),
        "activation_tensor_alignment_bytes": metrics.get(
            "activation_tensor_alignment_bytes"
        ),
        "activation_planner_algorithm": planner_algorithm,
        "activation_planning_overflowed": planner_overflowed,
        "peak_activation_event": metrics.get("peak_activation_event"),
        "peak_activation_node_index": metrics.get(
            "peak_activation_node_index"
        ),
        "peak_activation_node_name": metrics.get(
            "peak_activation_node_name"
        ),
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
        "peak_activation_confidence": metrics.get("peak_activation_confidence"),
        "dynamic_tensor_count": metrics.get("dynamic_tensor_count"),
        "unsupported_ops": metrics.get("unsupported_ops", []),
        "unsupported_dtypes": metrics.get("unsupported_dtypes", []),
        "report_path": str(report_path),
    }


def count_severities(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in diagnostics:
        severity = item.get("severity", "unknown")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def compact_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model["id"],
        "status": model["status"],
        "model_path": model["model_path"],
        "node_count": model["node_count"],
        "ops": model["ops"],
        "outputs": model["outputs"],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EdgeFit Profile Matrix",
        "",
        "| Model | Target | Status | Errors | Warnings | Diagnostics |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for cell in summary["matrix"]:
        diagnostics = ", ".join(cell["diagnostic_ids"]) if cell["diagnostic_ids"] else "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    code(cell["model_id"]),
                    code(cell["target_id"]),
                    code(cell["status"]),
                    str(cell["error_count"]),
                    str(cell["warning_count"]),
                    code(diagnostics),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def code(value: str) -> str:
    return "`" + value.replace("|", "\\|") + "`"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
