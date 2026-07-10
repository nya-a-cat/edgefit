from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_reference_check import parse_profile, split_op_key, write_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "targets" / "ort-mobile-cpu.yaml"
DEFAULT_CORPUS_RESULT = ROOT / "tmp" / "real_world_corpus" / "corpus-result.json"
DEFAULT_FIXTURE_RESULT = ROOT / "tmp" / "operator_fixtures" / "operator-fixtures-result.json"
DEFAULT_ORT_SOURCE = ROOT / "tmp" / "ort-src"
DEFAULT_OUT = ROOT / "tmp" / "ort-runtime-boundary" / "ort-runtime-boundary.json"
DEFAULT_MARKDOWN = ROOT / "tmp" / "ort-runtime-boundary" / "ort-runtime-boundary.md"
DEFAULT_CONFIG = ROOT / "tmp" / "ort-runtime-boundary" / "edgefit-ort-required-ops.config"
ORT_FORMAT_FIXTURE = Path("onnxruntime/test/testdata/required_ops.config")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify an ORT reduced-operator boundary for an EdgeFit profile.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--corpus-result", default=str(DEFAULT_CORPUS_RESULT))
    parser.add_argument("--fixture-result", default=str(DEFAULT_FIXTURE_RESULT))
    parser.add_argument("--ort-source", default=str(DEFAULT_ORT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    summary, config_text = build_summary(
        Path(args.profile),
        Path(args.corpus_result),
        Path(args.ort_source),
        Path(args.fixture_result),
    )
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.markdown_out), render_markdown(summary))
    write_text(Path(args.config_out), config_text)
    return 0 if summary["status"] == "pass" else 1


def build_summary(
    profile_path: Path,
    corpus_result_path: Path,
    ort_source_path: Path,
    fixture_result_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    profile = parse_profile(profile_path)
    corpus = read_json(corpus_result_path)
    corpus_entries = [item for item in corpus.get("results", []) if item.get("status") == "pass"]
    fixture_entries = load_optional_fixture_entries(fixture_result_path)
    evidence_entries = corpus_entries + fixture_entries
    required = collect_required_ops(evidence_entries)
    required_keys = {op_key(domain, op) for (domain, _version), ops in required.items() for op in ops}
    profile_keys = set(profile["ops"])
    format_fixture = ort_source_path / ORT_FORMAT_FIXTURE
    format_evidence = parse_reduced_config(format_fixture) if format_fixture.exists() else {}

    config_text = render_required_ops_config(required)
    parsed_generated = parse_reduced_config_text(config_text)
    parsed_keys = {op_key(domain, op) for (domain, _version), ops in parsed_generated.items() for op in ops}

    missing_from_profile = sorted(required_keys - profile_keys)
    profile_ops_not_required = sorted(profile_keys - required_keys)
    generated_roundtrip_missing = sorted(required_keys - parsed_keys)
    generated_roundtrip_extra = sorted(parsed_keys - required_keys)

    status = "pass"
    if (
        missing_from_profile
        or profile_ops_not_required
        or generated_roundtrip_missing
        or generated_roundtrip_extra
        or not corpus_entries
        or not format_evidence
    ):
        status = "fail"

    summary = {
        "schema": "edgefit.ort_runtime_boundary.v1",
        "status": status,
        "target_id": profile["target_id"],
        "profile": str(profile_path),
        "profile_operator_count": len(profile_keys),
        "corpus_result": str(corpus_result_path),
        "fixture_result": str(fixture_result_path) if fixture_result_path else None,
        "corpus_model_count": len(corpus_entries),
        "fixture_model_count": len(fixture_entries),
        "evidence_model_count": len(evidence_entries),
        "ort_source": str(ort_source_path),
        "ort_format_fixture": str(format_fixture),
        "ort_format_fixture_status": "pass" if format_evidence else "missing",
        "required_config_line_count": len(required),
        "required_operator_count": len(required_keys),
        "profile_coverage_status": "pass" if not missing_from_profile and not profile_ops_not_required else "fail",
        "generated_config_roundtrip_status": "pass" if not generated_roundtrip_missing and not generated_roundtrip_extra else "fail",
        "missing_from_profile": missing_from_profile,
        "profile_ops_not_required": profile_ops_not_required,
        "generated_roundtrip_missing": generated_roundtrip_missing,
        "generated_roundtrip_extra": generated_roundtrip_extra,
        "required_config": config_lines(required),
        "required_operator_keys": sorted(required_keys),
    }
    return summary, config_text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_fixture_entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    result = read_json(path)
    return [item for item in result.get("results", []) if item.get("status") == "pass"]


def collect_required_ops(entries: list[dict[str, Any]]) -> dict[tuple[str, int], set[str]]:
    required: dict[tuple[str, int], set[str]] = {}
    for entry in entries:
        imports = load_opset_imports(Path(entry["model_path"]))
        for key in entry.get("domain_ops", []):
            domain, op = split_op_key(key)
            version = imports.get(domain)
            if version is None:
                raise SystemExit(f"{entry['id']} uses {domain}::{op} without an opset import")
            required.setdefault((domain, version), set()).add(op)
    return required


def load_opset_imports(model_path: Path) -> dict[str, int]:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("official onnx package is required for ORT boundary evidence") from exc

    model = onnx.load(str(model_path), load_external_data=False)
    imports: dict[str, int] = {}
    for item in model.opset_import:
        imports[item.domain or "ai.onnx"] = int(item.version)
    return imports


def render_required_ops_config(required: dict[tuple[str, int], set[str]]) -> str:
    lines = ["# EdgeFit generated ORT reduced-operator boundary config"]
    lines.extend(config_lines(required))
    return "\n".join(lines) + "\n"


def config_lines(required: dict[tuple[str, int], set[str]]) -> list[str]:
    lines = []
    for domain, version in sorted(required):
        ops = ",".join(sorted(required[(domain, version)]))
        lines.append(f"{domain};{version};{ops}")
    return lines


def parse_reduced_config(path: Path) -> dict[tuple[str, int], set[str]]:
    return parse_reduced_config_text(path.read_text(encoding="utf-8"))


def parse_reduced_config_text(text: str) -> dict[tuple[str, int], set[str]]:
    parsed: dict[tuple[str, int], set[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("!"):
            continue
        parts = line.split(";", 2)
        if len(parts) != 3:
            raise SystemExit(f"invalid reduced-operator config line: {raw_line}")
        domain, version_text, op_text = parts
        version = int(version_text)
        ops = {strip_type_constraint(item) for item in split_op_specs(op_text)}
        parsed.setdefault((domain, version), set()).update(op for op in ops if op)
    return parsed


def split_op_specs(op_text: str) -> list[str]:
    items = []
    start = 0
    depth = 0
    for index, char in enumerate(op_text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(op_text[start:index].strip())
            start = index + 1
    items.append(op_text[start:].strip())
    return items


def strip_type_constraint(op_spec: str) -> str:
    brace = op_spec.find("{")
    if brace == -1:
        return op_spec.strip()
    return op_spec[:brace].strip()


def op_key(domain: str, op: str) -> str:
    return f"{domain}::{op}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EdgeFit ORT Runtime Boundary",
        "",
        f"**Target:** `{summary['target_id']}`",
        f"**Status:** `{summary['status']}`",
        f"**Corpus models:** `{summary['corpus_model_count']}`",
        f"**Fixture models:** `{summary['fixture_model_count']}`",
        f"**Evidence models:** `{summary['evidence_model_count']}`",
        f"**Profile operators:** `{summary['profile_operator_count']}`",
        f"**Required operators:** `{summary['required_operator_count']}`",
        f"**ORT format fixture:** `{summary['ort_format_fixture_status']}`",
        f"**Profile coverage:** `{summary['profile_coverage_status']}`",
        f"**Generated config roundtrip:** `{summary['generated_config_roundtrip_status']}`",
        "",
        "## Required Config Lines",
        "",
    ]
    for line in summary["required_config"]:
        lines.append(f"- `{line}`")
    lines.extend(
        [
            "",
            "## Boundary Gaps",
            "",
            f"- Missing from profile: `{', '.join(summary['missing_from_profile']) if summary['missing_from_profile'] else 'none'}`",
            f"- Profile ops not required by evidence models: `{', '.join(summary['profile_ops_not_required']) if summary['profile_ops_not_required'] else 'none'}`",
            f"- Generated config roundtrip missing: `{', '.join(summary['generated_roundtrip_missing']) if summary['generated_roundtrip_missing'] else 'none'}`",
            f"- Generated config roundtrip extra: `{', '.join(summary['generated_roundtrip_extra']) if summary['generated_roundtrip_extra'] else 'none'}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())