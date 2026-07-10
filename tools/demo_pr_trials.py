from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tmp" / "demo_pr_trials"
DEFAULT_EDGEFIT = ROOT / "tmp" / "cargo-target" / "debug" / "edgefit.exe"

TRIALS = [
    {
        "id": "esp32s3-good-tiny-pass",
        "model": "examples/models/good_tiny.edgefit.json",
        "target": "targets/esp32s3.yaml",
        "expected_status": "pass",
        "purpose": "baseline PR keeps the strict MCU deployment budget within limits",
    },
    {
        "id": "esp32s3-bad-detector-fail",
        "model": "examples/models/bad_detector.edgefit.json",
        "target": "targets/esp32s3.yaml",
        "expected_status": "fail",
        "purpose": "regression PR exceeds the strict MCU deployment budget and emits SARIF plus summary diagnostics",
    },
    {
        "id": "ort-mobile-bad-detector-pass",
        "model": "examples/models/bad_detector.edgefit.json",
        "target": "targets/ort-mobile-cpu.yaml",
        "expected_status": "pass",
        "purpose": "same detector model passes the wider ORT mobile-like target profile",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local demo PR trials for EdgeFit CI evidence.")
    parser.add_argument("--edgefit", default=str(DEFAULT_EDGEFIT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    summary = run_trials(Path(args.edgefit), Path(args.out_dir))
    write_text(Path(args.out_dir) / "demo-pr-trials.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.out_dir) / "job-summary.md", render_job_summary(summary))
    return 0 if summary["status"] == "pass" else 1


def run_trials(edgefit: Path, out_dir: Path) -> dict[str, Any]:
    if not edgefit.exists():
        raise SystemExit(f"missing edgefit binary {edgefit}")
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = [run_trial(edgefit, out_dir, trial) for trial in TRIALS]
    return {
        "schema": "edgefit.demo_pr_trials.v1",
        "edgefit": str(edgefit),
        "status": "pass" if all(trial["status"] == "pass" for trial in trials) else "fail",
        "trial_count": len(trials),
        "trials": trials,
    }


def run_trial(edgefit: Path, out_dir: Path, trial: dict[str, str]) -> dict[str, Any]:
    trial_dir = out_dir / trial["id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    sarif_path = trial_dir / "edgefit.sarif"
    summary_path = trial_dir / "edgefit-summary.md"
    command = [
        str(edgefit),
        "check",
        trial["model"],
        "--target",
        trial["target"],
        "--format",
        "sarif",
        "--out",
        str(sarif_path),
        "--summary",
        str(summary_path),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report_status = "pass" if result.returncode == 0 else "fail"
    expected_status = trial["expected_status"]
    sarif = inspect_sarif(sarif_path)
    summary = inspect_summary(summary_path)
    status = "pass" if report_status == expected_status and sarif["status"] == "pass" and summary["status"] == "pass" else "fail"
    return {
        "id": trial["id"],
        "purpose": trial["purpose"],
        "model": trial["model"],
        "target": trial["target"],
        "expected_status": expected_status,
        "report_status": report_status,
        "status": status,
        "exit_code": result.returncode,
        "sarif": str(sarif_path),
        "sarif_status": sarif["status"],
        "sarif_result_count": sarif["result_count"],
        "summary": str(summary_path),
        "summary_status": summary["status"],
        "summary_excerpt": summary["excerpt"],
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def inspect_sarif(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "fail", "result_count": 0}
    try:
        sarif = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "fail", "result_count": 0}
    runs = sarif.get("runs", [])
    result_count = sum(len(run.get("results", [])) for run in runs)
    status = "pass" if sarif.get("version") == "2.1.0" and runs else "fail"
    return {"status": status, "result_count": result_count}


def inspect_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "fail", "excerpt": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    excerpt = [line for line in lines if line.startswith("**Status:**") or line.startswith("| `EF") or line.startswith("| ID |")][:10]
    status = "pass" if any(line.startswith("**Status:**") for line in lines) else "fail"
    return {"status": status, "excerpt": excerpt}


def render_job_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# EdgeFit Demo PR Trials",
        "",
        f"**Status:** `{summary['status']}`",
        f"**Trials:** `{summary['trial_count']}`",
        "",
        "| Trial | Expected | Actual | Status | Purpose |",
        "| --- | --- | --- | --- | --- |",
    ]
    for trial in summary["trials"]:
        lines.append(
            f"| `{trial['id']}` | `{trial['expected_status']}` | `{trial['report_status']}` | `{trial['status']}` | {trial['purpose']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())