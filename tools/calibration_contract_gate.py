"""调度 Calibration 模拟器与外部证据打包器，核对失败、绑定和篡改契约。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILES = (
    "simulator-runtime.bin",
    "simulation-trace.json",
    "evidence.json",
    "verification.json",
    "verification.md",
)
PACK_FILES = (
    "capture.json",
    "external-runtime.example.bin",
    "external-runtime.example.log",
    "evidence.json",
    "verification.json",
    "verification.md",
)


def run(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError(
            f"expected exit {expected}, got {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def simulate(
    edgefit: Path,
    model: Path,
    target: Path,
    scenario: Path,
    out_dir: Path,
    expected: int,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            str(edgefit),
            "calibration",
            "simulate",
            str(model),
            "--target",
            str(target),
            "--scenario",
            str(scenario),
            "--out-dir",
            str(out_dir),
        ],
        expected,
    )


def verify(
    edgefit: Path,
    evidence: Path,
    model: Path,
    target: Path,
    out: Path,
    expected: int,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            str(edgefit),
            "calibration",
            "verify",
            str(evidence),
            "--model",
            str(model),
            "--target",
            str(target),
            "--format",
            "json",
            "--out",
            str(out),
        ],
        expected,
    )


def pack(
    edgefit: Path,
    capture: Path,
    model: Path,
    target: Path,
    out_dir: Path,
    expected: int,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            str(edgefit),
            "calibration",
            "pack",
            str(capture),
            "--model",
            str(model),
            "--target",
            str(target),
            "--out-dir",
            str(out_dir),
        ],
        expected,
    )


def require_simulated(directory: Path, expected_status: str) -> dict[str, object]:
    for name in OUTPUT_FILES:
        if not (directory / name).is_file():
            raise RuntimeError(f"simulation output is missing {name}")
    evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    trace = json.loads((directory / "simulation-trace.json").read_text(encoding="utf-8"))
    verification = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    if evidence.get("schema") != "edgefit.calibration_evidence.v1":
        raise RuntimeError("simulation emitted an unexpected evidence schema")
    if trace.get("schema") != "edgefit.calibration_simulation_trace.v1":
        raise RuntimeError("simulation emitted an unexpected trace schema")
    if verification.get("schema") != "edgefit.calibration_verification.v1":
        raise RuntimeError("simulation emitted an unexpected verification schema")
    if evidence["environment"]["operating_system"] != "simulated":
        raise RuntimeError("evidence is not explicitly simulated")
    if evidence["attestation"]["kind"] != "none":
        raise RuntimeError("simulation unexpectedly claims attestation")
    if trace.get("confidence") != "simulated":
        raise RuntimeError("trace is not explicitly simulated")
    if "not_real_hardware" not in trace.get("limitations", []):
        raise RuntimeError("trace omits its real-hardware limitation")
    if verification.get("status") != expected_status:
        raise RuntimeError(f"expected verification status {expected_status}")
    return trace


def require_failed_check(directory: Path, check_id: str) -> None:
    verification = json.loads(
        (directory / "verification.json").read_text(encoding="utf-8")
    )
    checks = {item["id"]: item["status"] for item in verification["checks"]}
    if checks.get(check_id) != "fail":
        raise RuntimeError(f"expected failed verification check {check_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgefit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ROOT / "examples/calibration"
    nominal_a = args.out_dir / "nominal-a"
    nominal_b = args.out_dir / "nominal-b"
    nominal_a_result = simulate(
        args.edgefit,
        args.model,
        args.target,
        scenarios / "nominal.simulation.json",
        nominal_a,
        0,
    )
    simulate(
        args.edgefit,
        args.model,
        args.target,
        scenarios / "nominal.simulation.json",
        nominal_b,
        0,
    )
    require_simulated(nominal_a, "pass")
    require_simulated(nominal_b, "pass")
    if nominal_a_result.stdout != (nominal_a / "verification.json").read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("simulation stdout does not match canonical verification JSON")
    for name in OUTPUT_FILES:
        if (nominal_a / name).read_bytes() != (nominal_b / name).read_bytes():
            raise RuntimeError(f"simulation output {name} is not deterministic")
    nominal_before = {name: (nominal_a / name).read_bytes() for name in OUTPUT_FILES}
    existing = simulate(
        args.edgefit,
        args.model,
        args.target,
        scenarios / "nominal.simulation.json",
        nominal_a,
        2,
    )
    if "already exists" not in existing.stderr:
        raise RuntimeError("existing simulation directory was not rejected")
    for name, expected in nominal_before.items():
        if (nominal_a / name).read_bytes() != expected:
            raise RuntimeError(f"existing simulation output {name} was modified")

    runtime = nominal_a / "simulator-runtime.bin"
    runtime_before_alias_check = runtime.read_bytes()
    alias = verify(
        args.edgefit,
        nominal_a / "evidence.json",
        args.model,
        args.target,
        runtime,
        2,
    )
    if "must not alias attachment" not in alias.stderr:
        raise RuntimeError("attachment output alias was not rejected")
    if runtime.read_bytes() != runtime_before_alias_check:
        raise RuntimeError("attachment output alias modified the attachment")

    latency_fail = args.out_dir / "latency-fail"
    simulate(
        args.edgefit,
        args.model,
        args.target,
        scenarios / "latency-fail.simulation.json",
        latency_fail,
        1,
    )
    require_simulated(latency_fail, "fail")
    require_failed_check(latency_fail, "evidence_latency_threshold")

    arena_fail = args.out_dir / "arena-fail"
    simulate(
        args.edgefit,
        args.model,
        args.target,
        scenarios / "arena-fail.simulation.json",
        arena_fail,
        1,
    )
    require_simulated(arena_fail, "fail")
    require_failed_check(arena_fail, "evidence_arena_threshold")

    spill = args.out_dir / "spill-reload"
    simulate(
        args.edgefit,
        ROOT / "examples/models/virtual_npu_spill.edgefit.json",
        ROOT / "targets/virtual-npu-small-scratchpad.yaml",
        scenarios / "nominal.simulation.json",
        spill,
        0,
    )
    spill_trace = require_simulated(spill, "pass")
    events = spill_trace["plan"]["events"]
    if int(events["spill"]) == 0 or int(events["reload"]) == 0:
        raise RuntimeError("spill scenario did not retain spill/reload plan evidence")

    blocked = args.out_dir / "blocked"
    blocked_result = simulate(
        args.edgefit,
        ROOT / "examples/models/virtual_npu_spill.edgefit.json",
        ROOT / "targets/virtual-npu-no-spill.yaml",
        scenarios / "nominal.simulation.json",
        blocked,
        2,
    )
    if blocked.exists() or "without blockers" not in blocked_result.stderr:
        raise RuntimeError("blocked plan created a formal simulation directory")

    tampered_runtime = args.out_dir / "tampered-runtime"
    shutil.copytree(nominal_a, tampered_runtime)
    (tampered_runtime / "simulator-runtime.bin").write_bytes(b"tampered runtime\n")
    tampered_runtime_verification = args.out_dir / "tampered-runtime-verification.json"
    tampered_runtime_verification.write_text("stale artifact\n", encoding="utf-8")
    tampered_runtime_result = verify(
        args.edgefit,
        tampered_runtime / "evidence.json",
        args.model,
        args.target,
        tampered_runtime_verification,
        2,
    )
    if "runtime binary SHA-256 binding mismatch" not in tampered_runtime_result.stderr:
        raise RuntimeError("tampered runtime did not report its binding failure")
    error = json.loads(tampered_runtime_verification.read_text(encoding="utf-8"))
    if error.get("schema") != "edgefit.calibration_verification_error.v1":
        raise RuntimeError("tampered runtime did not replace the stale artifact")

    tampered_trace = args.out_dir / "tampered-trace"
    shutil.copytree(nominal_a, tampered_trace)
    with (tampered_trace / "simulation-trace.json").open("ab") as file:
        file.write(b"tampered\n")
    tampered_trace_result = verify(
        args.edgefit,
        tampered_trace / "evidence.json",
        args.model,
        args.target,
        args.out_dir / "tampered-trace-verification.json",
        2,
    )
    if "simulation-trace.json" not in tampered_trace_result.stderr:
        raise RuntimeError("tampered trace did not report its attachment failure")

    binding_mismatch = verify(
        args.edgefit,
        nominal_a / "evidence.json",
        ROOT / "examples/models/virtual_npu_segmented.edgefit.json",
        args.target,
        args.out_dir / "binding-mismatch-verification.json",
        2,
    )
    if "model SHA-256 binding mismatch" not in binding_mismatch.stderr:
        raise RuntimeError("model binding mismatch was not reported")

    capture_fixture = scenarios / "external-capture.example.json"
    packed = args.out_dir / "external-pack"
    packed_result = pack(
        args.edgefit,
        capture_fixture,
        args.model,
        args.target,
        packed,
        0,
    )
    for name in PACK_FILES:
        if not (packed / name).is_file():
            raise RuntimeError(f"calibration pack is missing {name}")
    packed_verification = json.loads((packed / "verification.json").read_text(encoding="utf-8"))
    packed_evidence = json.loads((packed / "evidence.json").read_text(encoding="utf-8"))
    if packed_result.stdout != (packed / "verification.json").read_text(encoding="utf-8"):
        raise RuntimeError("calibration pack stdout is not canonical verification JSON")
    if packed_verification.get("status") != "pass":
        raise RuntimeError("calibration pack fixture did not pass")
    if packed_evidence.get("attestation", {}).get("kind") != "none":
        raise RuntimeError("calibration pack unexpectedly claims attestation")
    if {item["name"] for item in packed_evidence.get("attachments", [])} != {
        "capture-manifest",
        "runtime-binary",
        "raw-log",
    }:
        raise RuntimeError("calibration pack attachment bindings are incomplete")
    existing_pack = pack(
        args.edgefit,
        capture_fixture,
        args.model,
        args.target,
        packed,
        2,
    )
    if "already exists" not in existing_pack.stderr:
        raise RuntimeError("existing calibration pack directory was not rejected")

    tampered_pack = args.out_dir / "tampered-pack"
    shutil.copytree(packed, tampered_pack)
    (tampered_pack / "external-runtime.example.log").write_text(
        "tampered\n", encoding="utf-8"
    )
    tampered_pack_result = verify(
        args.edgefit,
        tampered_pack / "evidence.json",
        args.model,
        args.target,
        args.out_dir / "tampered-pack-verification.json",
        2,
    )
    if "external-runtime.example.log" not in tampered_pack_result.stderr:
        raise RuntimeError("tampered external attachment was not rejected")

    failing_source = args.out_dir / "capture-threshold-fixture"
    failing_source.mkdir()
    shutil.copy2(scenarios / "external-runtime.example.bin", failing_source)
    shutil.copy2(scenarios / "external-runtime.example.log", failing_source)
    failing_capture = json.loads(capture_fixture.read_text(encoding="utf-8"))
    failing_capture["thresholds"]["p95_latency_budget_ns"] = "7000"
    failing_capture_path = failing_source / "capture.json"
    failing_capture_path.write_text(
        json.dumps(failing_capture, indent=2) + "\n", encoding="utf-8"
    )
    failing_pack = args.out_dir / "external-pack-threshold-fail"
    pack(
        args.edgefit,
        failing_capture_path,
        args.model,
        args.target,
        failing_pack,
        1,
    )
    require_failed_check(failing_pack, "evidence_latency_threshold")

    unsafe_capture = json.loads(capture_fixture.read_text(encoding="utf-8"))
    unsafe_capture["runtime_binary"] = "../external-runtime.example.bin"
    unsafe_path = failing_source / "unsafe-capture.json"
    unsafe_path.write_text(json.dumps(unsafe_capture) + "\n", encoding="utf-8")
    unsafe_result = pack(
        args.edgefit,
        unsafe_path,
        args.model,
        args.target,
        args.out_dir / "unsafe-pack",
        2,
    )
    if "unsafe attachment path" not in unsafe_result.stderr:
        raise RuntimeError("capture directory traversal was not rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
