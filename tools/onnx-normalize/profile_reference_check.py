from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "targets" / "ort-mobile-cpu.yaml"
DEFAULT_MANIFEST = Path(__file__).with_name("real_world_corpus.json")
DEFAULT_FIXTURE_MANIFEST = Path(__file__).with_name("operator_fixtures.json")
DEFAULT_RUNTIME_EVIDENCE = Path(__file__).with_name("ort_runtime_evidence.json")
DEFAULT_REQUIREMENTS = Path(__file__).with_name("requirements.txt")
DEFAULT_OUT = ROOT / "tmp" / "real_world_corpus" / "profile-reference.json"
DEFAULT_MARKDOWN = ROOT / "tmp" / "real_world_corpus" / "profile-reference.md"
SOURCES = [
    {
        "id": "onnx-operator-schemas",
        "label": "ONNX operator schemas from the pinned official onnx package",
        "url": "https://onnx.ai/onnx/operators/",
    },
    {
        "id": "onnxruntime-operator-kernels",
        "label": "ONNX Runtime operator kernel documentation",
        "url": "https://onnxruntime.ai/docs/reference/operators/OperatorKernels.html",
    },
    {
        "id": "onnxruntime-reduced-operator-config",
        "label": "ONNX Runtime reduced operator config documentation",
        "url": "https://onnxruntime.ai/docs/reference/operators/reduced-operator-config-file.html",
    },
    {
        "id": "edgefit-real-world-corpus",
        "label": "EdgeFit verified real-world corpus manifest",
        "url": "tools/onnx-normalize/real_world_corpus.json",
    },
    {
        "id": "edgefit-operator-fixtures",
        "label": "EdgeFit generated ONNX operator fixture manifest",
        "url": "tools/onnx-normalize/operator_fixtures.json",
    },
    {
        "id": "edgefit-ort-runtime-evidence",
        "label": "EdgeFit pinned ONNX Runtime source evidence manifest",
        "url": "tools/onnx-normalize/ort_runtime_evidence.json",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit target profile operator references.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--fixture-manifest", default=str(DEFAULT_FIXTURE_MANIFEST))
    parser.add_argument("--runtime-evidence", default=str(DEFAULT_RUNTIME_EVIDENCE))
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    summary = build_summary(
        Path(args.profile),
        Path(args.manifest),
        Path(args.fixture_manifest),
        Path(args.runtime_evidence),
        Path(args.requirements),
    )
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.markdown_out), render_markdown(summary))
    version_status = summary["reference_versions"]["onnx_python_package"]["status"]
    return 1 if summary["missing_reference_count"] or version_status != "match" else 0


def build_summary(
    profile_path: Path,
    manifest_path: Path,
    fixture_manifest_path: Path = DEFAULT_FIXTURE_MANIFEST,
    runtime_evidence_path: Path = DEFAULT_RUNTIME_EVIDENCE,
    requirements_path: Path = DEFAULT_REQUIREMENTS,
) -> dict[str, Any]:
    profile = parse_profile(profile_path)
    evidence = merge_op_mappings(
        load_corpus_ops(manifest_path), load_fixture_ops(fixture_manifest_path)
    )
    runtime_evidence = load_runtime_ops(runtime_evidence_path)
    official_reference = load_official_onnx_reference()
    official = official_reference["ops"]
    pinned_version = load_pinned_package_version(requirements_path, "onnx")
    version_status = version_pin_status(official_reference["version"], pinned_version)

    operators = []
    counts = {
        "official_and_corpus": 0,
        "official_only": 0,
        "runtime_and_corpus": 0,
        "runtime_only": 0,
        "corpus_only": 0,
        "missing_reference": 0,
    }
    for key in profile["ops"]:
        domain, op = split_op_key(key)
        observed_models = evidence.get(key, [])
        runtime_items = runtime_evidence.get(key, [])
        if key in official and observed_models:
            status = "official_and_corpus"
        elif runtime_items and observed_models:
            status = "runtime_and_corpus"
        elif key in official:
            status = "official_only"
        elif runtime_items:
            status = "runtime_only"
        elif observed_models:
            status = "corpus_only"
        else:
            status = "missing_reference"
        counts[status] += 1
        operators.append(
            {
                "domain": domain,
                "op": op,
                "op_key": key,
                "status": status,
                "official_onnx_schema": key in official,
                "official_runtime_evidence": bool(runtime_items),
                "runtime_evidence_ids": runtime_items,
                "evidence_models": observed_models,
                "corpus_models": observed_models,
            }
        )

    return {
        "schema": "edgefit.profile_reference.v1",
        "target_id": profile["target_id"],
        "profile": str(profile_path),
        "profile_metadata": profile["metadata"],
        "sources": SOURCES,
        "evidence_manifests": {
            "real_world_corpus": str(manifest_path),
            "operator_fixtures": str(fixture_manifest_path),
            "ort_runtime_evidence": str(runtime_evidence_path),
        },
        "reference_versions": {
            "onnx_python_package": {
                "installed": official_reference["version"],
                "pinned": pinned_version,
                "status": version_status,
                "requirements": str(requirements_path),
                "official_operator_count": official_reference["operator_count"],
            }
        },
        "operator_count": len(operators),
        "official_and_corpus_count": counts["official_and_corpus"],
        "official_only_count": counts["official_only"],
        "runtime_and_corpus_count": counts["runtime_and_corpus"],
        "runtime_only_count": counts["runtime_only"],
        "corpus_only_count": counts["corpus_only"],
        "missing_reference_count": counts["missing_reference"],
        "operators": operators,
    }


def parse_profile(path: Path) -> dict[str, Any]:
    metadata = {"source": "", "confidence": "", "last_verified": ""}
    target_id = ""
    ops: list[str] = []
    section = ""
    in_ops_allow = False
    current_domain = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if indent == 0:
            section = key
            in_ops_allow = False
            current_domain = ""
            continue

        if section == "metadata" and indent == 2 and key in metadata:
            metadata[key] = clean_scalar(value)
        elif section == "target" and indent == 2 and key == "id":
            target_id = clean_scalar(value)
        elif section == "ops":
            if indent == 2 and key == "allow":
                in_ops_allow = True
            elif in_ops_allow and indent == 4:
                current_domain = key
            elif in_ops_allow and current_domain and indent == 6:
                ops.append(op_key(current_domain, key))

    if not target_id:
        raise SystemExit(f"target.id missing in {path}")
    if not ops:
        raise SystemExit(f"ops.allow missing in {path}")
    return {"target_id": target_id, "metadata": metadata, "ops": sorted(set(ops))}


def clean_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")


def load_corpus_ops(path: Path) -> dict[str, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    for model in manifest["models"]:
        for key in expected_domain_ops(model):
            mapping.setdefault(key, []).append(model["id"])
    return {key: sorted(models) for key, models in mapping.items()}


def load_fixture_ops(path: Path) -> dict[str, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    for model in manifest["models"]:
        model_id = f"fixture:{model['id']}"
        for key in expected_domain_ops(model):
            mapping.setdefault(key, []).append(model_id)
    return {key: sorted(models) for key, models in mapping.items()}


def expected_domain_ops(model: dict[str, Any]) -> list[str]:
    domain_overrides = model.get("expected_operator_domains", {})
    keys = []
    for op in model["expected_ops"]:
        for domain in domain_overrides.get(op, ["ai.onnx"]):
            keys.append(op_key(domain, op))
    return sorted(keys)

def load_runtime_ops(path: Path) -> dict[str, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    for operator in manifest["operators"]:
        mapping[operator["op_key"]] = [item["id"] for item in operator["runtime_evidence"]]
    return {key: sorted(items) for key, items in mapping.items()}

def merge_op_mappings(*mappings: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {}
    for mapping in mappings:
        for key, models in mapping.items():
            merged.setdefault(key, set()).update(models)
    return {key: sorted(models) for key, models in merged.items()}


def load_official_onnx_reference() -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("official onnx package is required for profile reference checks") from exc

    ops = set()
    for schema in onnx.defs.get_all_schemas_with_history():
        domain = schema.domain or "ai.onnx"
        if domain == "ai.onnx":
            ops.add(op_key(domain, schema.name))
    return {
        "version": onnx.__version__,
        "operator_count": len(ops),
        "ops": ops,
    }


def op_key(domain: str | None, op: str) -> str:
    normalized_domain = domain or "ai.onnx"
    return f"{normalized_domain}::{op}"


def split_op_key(key: str) -> tuple[str, str]:
    domain, op = key.split("::", 1)
    return domain, op


def load_pinned_package_version(path: Path, package: str) -> str | None:
    normalized_package = package.lower()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        if name.strip().lstrip("\ufeff").lower() == normalized_package:
            return version.strip()
    return None


def version_pin_status(installed: str, pinned: str | None) -> str:
    if pinned is None:
        return "unpinned"
    if installed == pinned:
        return "match"
    return "mismatch"


def render_markdown(summary: dict[str, Any]) -> str:
    onnx_version = summary["reference_versions"]["onnx_python_package"]
    metadata = summary["profile_metadata"]
    lines = [
        "# EdgeFit Profile Reference",
        "",
        f"**Target:** `{summary['target_id']}`",
        f"**Profile confidence:** `{metadata['confidence']}`",
        f"**Profile last verified:** `{metadata['last_verified']}`",
        f"**Profile source:** {metadata['source']}",
        f"**Operators:** `{summary['operator_count']}`",
        f"**Missing references:** `{summary['missing_reference_count']}`",
        "",
        "## Reference Versions",
        "",
        "| Reference | Installed | Pinned | Status | Official operators |",
        "| --- | --- | --- | --- | ---: |",
        f"| `onnx` Python package | `{onnx_version['installed']}` | `{onnx_version['pinned'] or 'none'}` | `{onnx_version['status']}` | `{onnx_version['official_operator_count']}` |",
        "",
        "## Operators",
        "",
        "| Operator | Status | Evidence Models |",
        "| --- | --- | --- |",
    ]
    for item in summary["operators"]:
        evidence_models = item.get("evidence_models", item.get("corpus_models", []))
        models = ", ".join(evidence_models) if evidence_models else "none"
        lines.append(f"| `{item['op_key']}` | `{item['status']}` | `{models}` |")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for source in summary["sources"]:
        lines.append(f"- `{source['id']}`: {source['url']}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())