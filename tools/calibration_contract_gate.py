from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError(
            f"expected exit {expected}, got {result.returncode}: {' '.join(command)}\n{result.stderr}"
        )
    return result


def write_evidence(
    path: Path,
    model: Path,
    target: Path,
    runtime: Path,
    latency_budget: int,
) -> None:
    evidence = {
        "schema": "edgefit.calibration_evidence.v1",
        "identity": {
            "target_id": "esp32s3_custom_v1",
            "device_id": "ci-calibration-device",
            "runtime_name": "edgefit-contract-runtime",
            "runtime_version": "1.0.0",
        },
        "environment": {
            "operating_system": "github-hosted",
            "architecture": "x86_64",
            "hardware": "synthetic-contract-fixture",
            "toolchain": "edgefit-ci",
        },
        "capture": {
            "captured_at": "2026-07-12T12:34:56Z",
            "command": "edgefit calibration contract fixture",
            "warmup_runs": "1",
            "measured_runs": "3",
        },
        "bindings": {
            "model_sha256": digest(model),
            "target_profile_sha256": digest(target),
            "runtime_binary_sha256": digest(runtime),
        },
        "runtime": {"accepted": True, "rejected_reason": None},
        "measurements": {
            "arena_high_water": {"unit": "bytes", "value": "300000"},
            "latency": {"unit": "ns", "samples": ["10", "20", "30"]},
        },
        "thresholds": {
            "arena_budget": {"unit": "bytes", "value": "350000"},
            "p95_latency_budget": {"unit": "ns", "value": str(latency_budget)},
        },
        "attachments": [
            {
                "name": "runtime-binary",
                "path": runtime.name,
                "media_type": "application/octet-stream",
                "bytes": str(runtime.stat().st_size),
                "sha256": digest(runtime),
            }
        ],
        "attestation": {"kind": "none"},
    }
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def verify_command(
    edgefit: Path,
    evidence: Path,
    model: Path,
    target: Path,
    format: str,
    out: Path,
) -> list[str]:
    return [
        str(edgefit),
        "calibration",
        "verify",
        str(evidence),
        "--model",
        str(model),
        "--target",
        str(target),
        "--format",
        format,
        "--out",
        str(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgefit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime = args.out_dir / "runtime.bin"
    runtime.write_bytes(b"edgefit calibration contract runtime\n")

    passing = args.out_dir / "pass-evidence.json"
    write_evidence(passing, args.model, args.target, runtime, 30)
    pass_json = args.out_dir / "pass.json"
    pass_json_repeat = args.out_dir / "pass-repeat.json"
    pass_markdown = args.out_dir / "pass.md"
    run(verify_command(args.edgefit, passing, args.model, args.target, "json", pass_json), 0)
    run(
        verify_command(
            args.edgefit, passing, args.model, args.target, "json", pass_json_repeat
        ),
        0,
    )
    run(
        verify_command(
            args.edgefit, passing, args.model, args.target, "markdown", pass_markdown
        ),
        0,
    )
    if pass_json.read_bytes() != pass_json_repeat.read_bytes():
        raise RuntimeError("calibration JSON output is not deterministic")
    parsed = json.loads(pass_json.read_text(encoding="utf-8"))
    if parsed.get("schema") != "edgefit.calibration_verification.v1":
        raise RuntimeError("unexpected calibration verification schema")
    if parsed.get("status") != "pass":
        raise RuntimeError("passing calibration evidence did not pass")

    failing = args.out_dir / "fail-evidence.json"
    write_evidence(failing, args.model, args.target, runtime, 29)
    fail_json = args.out_dir / "fail.json"
    run(verify_command(args.edgefit, failing, args.model, args.target, "json", fail_json), 1)
    if json.loads(fail_json.read_text(encoding="utf-8")).get("status") != "fail":
        raise RuntimeError("threshold failure did not emit a fail verification")

    runtime_before_alias_check = runtime.read_bytes()
    alias = run(
        verify_command(args.edgefit, passing, args.model, args.target, "json", runtime),
        2,
    )
    if "must not alias attachment" not in alias.stderr:
        raise RuntimeError("attachment output alias was not rejected")
    if runtime.read_bytes() != runtime_before_alias_check:
        raise RuntimeError("attachment output alias modified the attachment")

    runtime.write_bytes(b"tampered runtime\n")
    error_json = args.out_dir / "tampered.json"
    error_json.write_text("stale artifact\n", encoding="utf-8")
    tampered = run(
        verify_command(args.edgefit, passing, args.model, args.target, "json", error_json),
        2,
    )
    if "runtime binary SHA-256 binding mismatch" not in tampered.stderr:
        raise RuntimeError("tampered runtime did not report its identity failure")
    error = json.loads(error_json.read_text(encoding="utf-8"))
    if error.get("schema") != "edgefit.calibration_verification_error.v1":
        raise RuntimeError("tampered runtime did not replace the stale artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
