from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("ort_runtime_evidence.json")
DEFAULT_SOURCE_ROOT = ROOT / "tmp" / "ort-src"
DEFAULT_OUT = ROOT / "tmp" / "ort-runtime-evidence" / "ort-runtime-evidence-result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ONNX Runtime source evidence for target-profile operators.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    summary = verify_manifest(Path(args.manifest), Path(args.source_root))
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


def verify_manifest(manifest_path: Path, source_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "edgefit.ort_runtime_evidence.v1":
        raise SystemExit("expected schema edgefit.ort_runtime_evidence.v1")
    if not source_root.exists():
        raise SystemExit(f"missing ONNX Runtime source root {source_root}")

    expected_commit = manifest["source"]["commit"]
    actual_commit = git_commit(source_root)
    if actual_commit != expected_commit:
        raise SystemExit(f"ONNX Runtime commit mismatch: {actual_commit}")

    domain_results = [verify_source_check(item, source_root) for item in manifest.get("domain_definitions", [])]
    operator_results = []
    for operator in manifest["operators"]:
        checks = [verify_source_check(item, source_root) for item in operator["runtime_evidence"]]
        operator_results.append(
            {
                "op_key": operator["op_key"],
                "status": "pass",
                "runtime_evidence_ids": [item["id"] for item in checks],
                "providers": sorted({item["provider"] for item in operator["runtime_evidence"]}),
            }
        )

    return {
        "schema": "edgefit.ort_runtime_evidence.result.v1",
        "source_root": str(source_root),
        "repo": manifest["source"]["repo"],
        "commit": actual_commit,
        "domain_definitions": domain_results,
        "operators": operator_results,
    }


def verify_source_check(item: dict[str, Any], source_root: Path) -> dict[str, Any]:
    source_file = source_root / item["source_file"]
    if not source_file.exists():
        raise SystemExit(f"missing source file {source_file}")
    text = source_file.read_text(encoding="utf-8", errors="replace")
    missing = [needle for needle in item["contains"] if needle not in text]
    if missing:
        raise SystemExit(f"source evidence {item['id']} missing strings: {missing}")
    return {
        "id": item["id"],
        "status": "pass",
        "source_file": item["source_file"],
        "source_url": item["source_url"],
    }


def git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())