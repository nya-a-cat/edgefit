from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_reference_check import parse_profile, write_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "targets" / "ort-mobile-cpu.yaml"
DEFAULT_REFERENCE = ROOT / "tmp" / "real_world_corpus" / "profile-reference.json"
DEFAULT_MATRIX = ROOT / "tmp" / "real_world_corpus" / "profile-matrix.json"
DEFAULT_CORPUS_GATE = ROOT / "tmp" / "real_world_corpus" / "corpus-expansion-gate.json"
DEFAULT_OPERATOR_AUDIT = ROOT / "tmp" / "real_world_corpus" / "operator-support-audit.json"
DEFAULT_RUNTIME_EVIDENCE_RESULT = ROOT / "tmp" / "ort-runtime-evidence" / "ort-runtime-evidence-result.json"
DEFAULT_RUNTIME_SMOKE = ROOT / "tmp" / "real_world_corpus" / "runtime-smoke.json"
DEFAULT_RUNTIME_BOUNDARY = ROOT / "tmp" / "ort-runtime-boundary" / "ort-runtime-boundary.json"
DEFAULT_DIAGNOSTIC_POLICY = ROOT / "docs" / "DIAGNOSTIC_POLICY.md"
DEFAULT_PUBLIC_PR_TRIALS = ROOT / "tmp" / "public_pr_trials" / "public-pr-trial-gate.json"
DEFAULT_OUT = ROOT / "tmp" / "real_world_corpus" / "profile-confidence-gate.json"
DEFAULT_MARKDOWN = ROOT / "tmp" / "real_world_corpus" / "profile-confidence-gate.md"


REFERENCE_GAP_COUNTS = [
    "missing_reference_count",
    "official_only_count",
    "runtime_only_count",
    "corpus_only_count",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate profile evidence before confidence uplift.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--corpus-gate", default=str(DEFAULT_CORPUS_GATE))
    parser.add_argument("--operator-audit", default=str(DEFAULT_OPERATOR_AUDIT))
    parser.add_argument("--runtime-evidence-result", default=str(DEFAULT_RUNTIME_EVIDENCE_RESULT))
    parser.add_argument("--runtime-smoke", default=str(DEFAULT_RUNTIME_SMOKE))
    parser.add_argument("--runtime-boundary", default=str(DEFAULT_RUNTIME_BOUNDARY))
    parser.add_argument("--diagnostic-policy", default=str(DEFAULT_DIAGNOSTIC_POLICY))
    parser.add_argument("--public-pr-trials", default=str(DEFAULT_PUBLIC_PR_TRIALS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    summary = build_summary(
        Path(args.profile),
        Path(args.reference),
        Path(args.matrix),
        Path(args.corpus_gate),
        Path(args.operator_audit),
        Path(args.runtime_evidence_result),
        Path(args.runtime_smoke),
        Path(args.runtime_boundary),
        Path(args.diagnostic_policy),
        Path(args.public_pr_trials),
    )
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.markdown_out), render_markdown(summary))
    return exit_code(summary)


def build_summary(
    profile_path: Path,
    reference_path: Path,
    matrix_path: Path,
    corpus_gate_path: Path,
    operator_audit_path: Path,
    runtime_evidence_result_path: Path,
    runtime_smoke_path: Path,
    runtime_boundary_path: Path,
    diagnostic_policy_path: Path,
    public_pr_trials_path: Path,
) -> dict[str, Any]:
    profile = parse_profile(profile_path)
    reference = read_json(reference_path)
    matrix = read_json(matrix_path)
    corpus_gate = read_json(corpus_gate_path) if corpus_gate_path.exists() else None
    operator_audit = read_json(operator_audit_path) if operator_audit_path.exists() else None
    runtime_evidence = read_json(runtime_evidence_result_path) if runtime_evidence_result_path.exists() else None
    runtime_smoke = read_json(runtime_smoke_path) if runtime_smoke_path.exists() else None
    runtime_boundary = read_json(runtime_boundary_path) if runtime_boundary_path.exists() else None
    public_pr_trials = read_json(public_pr_trials_path) if public_pr_trials_path.exists() else None

    target_id = profile["target_id"]
    matrix_entries = [cell for cell in matrix["matrix"] if cell.get("target_id") == target_id]
    models = matrix.get("models", [])

    checks = [
        check(
            "profile_reference_target_match",
            reference.get("target_id") == target_id,
            f"reference target is {reference.get('target_id')}",
        ),
        check(
            "onnx_version_pin_match",
            reference["reference_versions"]["onnx_python_package"]["status"] == "match",
            f"onnx status is {reference['reference_versions']['onnx_python_package']['status']}",
        ),
        check(
            "reference_gap_counts_clear",
            all(reference.get(name, 0) == 0 for name in REFERENCE_GAP_COUNTS),
            count_detail(reference, REFERENCE_GAP_COUNTS),
        ),
        check(
            "matrix_target_coverage",
            len(matrix_entries) == len(models) and bool(models),
            f"{len(matrix_entries)} target entries for {len(models)} corpus models",
        ),
        check(
            "matrix_target_passes",
            bool(matrix_entries)
            and all(cell.get("status") == "pass" and cell.get("error_count") == 0 for cell in matrix_entries),
            matrix_detail(matrix_entries),
        ),
        check(
            "warning_diagnostic_policy_documented",
            diagnostic_policy_passes(diagnostic_policy_path),
            diagnostic_policy_detail(diagnostic_policy_path),
        ),
        check(
            "corpus_expansion_gate_verified",
            corpus_gate_passes(corpus_gate),
            corpus_gate_detail(corpus_gate_path, corpus_gate),
        ),
        check(
            "operator_support_audit_verified",
            operator_audit_passes(operator_audit),
            operator_audit_detail(operator_audit_path, operator_audit),
        ),
        check(
            "runtime_evidence_verified",
            runtime_evidence_passes(reference, runtime_evidence),
            runtime_evidence_detail(reference, runtime_evidence),
        ),
        check(
            "runtime_smoke_verified",
            runtime_smoke_passes(target_id, runtime_smoke),
            runtime_smoke_detail(target_id, runtime_smoke_path, runtime_smoke),
        ),
        check(
            "runtime_boundary_verified",
            runtime_boundary_passes(target_id, runtime_boundary),
            runtime_boundary_detail(target_id, runtime_boundary_path, runtime_boundary),
        ),
        check(
            "public_pr_trials_verified",
            public_pr_trials_passes(public_pr_trials),
            public_pr_trials_detail(public_pr_trials_path, public_pr_trials),
        ),
    ]

    confidence_uplift_ready = all(item["status"] == "pass" for item in checks)
    decision = decide(profile["metadata"].get("confidence", ""), confidence_uplift_ready)

    return {
        "schema": "edgefit.profile_confidence_gate.v1",
        "target_id": target_id,
        "profile": str(profile_path),
        "profile_metadata": profile["metadata"],
        "reference": str(reference_path),
        "matrix": str(matrix_path),
        "corpus_gate": str(corpus_gate_path),
        "operator_audit": str(operator_audit_path),
        "runtime_evidence_result": str(runtime_evidence_result_path),
        "runtime_smoke": str(runtime_smoke_path),
        "runtime_boundary": str(runtime_boundary_path),
        "diagnostic_policy": str(diagnostic_policy_path),
        "public_pr_trials": str(public_pr_trials_path),
        "decision": decision,
        "confidence_uplift_ready": confidence_uplift_ready,
        "checks": checks,
        "next_actions": next_actions(checks),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(check_id: str, passed: bool, detail: str) -> dict[str, str]:
    return {"id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def count_detail(summary: dict[str, Any], names: list[str]) -> str:
    return ", ".join(f"{name}={summary.get(name, 0)}" for name in names)


def matrix_detail(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "no target entries"
    failed = [cell["model_id"] for cell in entries if cell.get("status") != "pass" or cell.get("error_count") != 0]
    if failed:
        return "failing models: " + ", ".join(sorted(failed))
    warnings = sum(int(cell.get("warning_count", 0)) for cell in entries)
    return f"all target entries pass; warning_count={warnings}"


def diagnostic_policy_passes(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    return all(token in text for token in ["ef0104", "warning", "error", "pass", "sarif", "suppressed_diagnostics"])


def diagnostic_policy_detail(path: Path) -> str:
    if not path.exists():
        return f"diagnostic policy missing at {path}"
    text = path.read_text(encoding="utf-8").lower()
    ef0104 = "yes" if "ef0104" in text else "no"
    severity_rules = "yes" if all(token in text for token in ["warning", "error", "pass"]) else "no"
    reporting = "yes" if all(token in text for token in ["sarif", "suppressed_diagnostics"]) else "no"
    return f"policy={path}; ef0104={ef0104}; severity_rules={severity_rules}; reporting={reporting}"


def corpus_gate_passes(corpus_gate: dict[str, Any] | None) -> bool:
    if corpus_gate is None:
        return False
    return (
        corpus_gate.get("schema") == "edgefit.corpus_expansion_gate.v1"
        and corpus_gate.get("status") == "ready_for_profile_matrix"
        and corpus_gate.get("label_status") == "complete"
        and int(corpus_gate.get("models_needed", 1)) == 0
        and int(corpus_gate.get("missing_label_cell_count", 1)) == 0
        and int(corpus_gate.get("duplicate_label_cell_count", 1)) == 0
        and int(corpus_gate.get("unknown_label_cell_count", 1)) == 0
    )


def corpus_gate_detail(path: Path, corpus_gate: dict[str, Any] | None) -> str:
    if corpus_gate is None:
        return f"corpus expansion gate missing at {path}"
    return (
        f"status={corpus_gate.get('status')}; models={corpus_gate.get('model_count')}/{corpus_gate.get('sample_goal')}; "
        f"labels={corpus_gate.get('label_cell_count')}/{corpus_gate.get('expected_label_cell_count')}"
    )


def operator_audit_passes(operator_audit: dict[str, Any] | None) -> bool:
    if operator_audit is None:
        return False
    review = operator_audit.get("precision_recall_review", {})
    return (
        operator_audit.get("schema") == "edgefit.operator_support_audit.v1"
        and operator_audit.get("status") == "pass"
        and review.get("status") == "pass"
        and int(review.get("mismatch_count", 1)) == 0
        and int(operator_audit.get("sample_model_count", 0)) >= int(operator_audit.get("sample_goal", 1))
    )


def operator_audit_detail(path: Path, operator_audit: dict[str, Any] | None) -> str:
    if operator_audit is None:
        return f"operator support audit missing at {path}"
    review = operator_audit.get("precision_recall_review", {})
    return (
        f"status={operator_audit.get('status')}; models={operator_audit.get('sample_model_count')}/{operator_audit.get('sample_goal')}; "
        f"review={review.get('status')}; labels={review.get('labeled_cell_count')}"
    )


def runtime_evidence_passes(reference: dict[str, Any], runtime_evidence: dict[str, Any] | None) -> bool:
    if reference.get("runtime_and_corpus_count", 0) == 0 and reference.get("runtime_only_count", 0) == 0:
        return True
    if runtime_evidence is None:
        return False
    if runtime_evidence.get("schema") != "edgefit.ort_runtime_evidence.result.v1":
        return False
    operators = runtime_evidence.get("operators", [])
    return bool(operators) and all(item.get("status") == "pass" for item in operators)


def runtime_evidence_detail(reference: dict[str, Any], runtime_evidence: dict[str, Any] | None) -> str:
    runtime_count = reference.get("runtime_and_corpus_count", 0) + reference.get("runtime_only_count", 0)
    if runtime_count == 0:
        return "profile has no runtime-backed operators"
    if runtime_evidence is None:
        return "runtime evidence result file missing"
    providers = sorted({provider for item in runtime_evidence.get("operators", []) for provider in item.get("providers", [])})
    return f"{len(runtime_evidence.get('operators', []))} runtime-backed operators; providers={','.join(providers)}"


def runtime_smoke_passes(target_id: str, runtime_smoke: dict[str, Any] | None) -> bool:
    if runtime_smoke is None:
        return False
    if runtime_smoke.get("schema") != "edgefit.runtime_smoke.v1":
        return False
    if runtime_smoke.get("target_id") != target_id:
        return False
    return runtime_smoke.get("status") == "pass"


def runtime_smoke_detail(target_id: str, path: Path, runtime_smoke: dict[str, Any] | None) -> str:
    if runtime_smoke is None:
        return f"runtime smoke result missing at {path}"
    return f"target={runtime_smoke.get('target_id')}; status={runtime_smoke.get('status')}; expected={target_id}; models={runtime_smoke.get('model_count')}"


def runtime_boundary_passes(target_id: str, runtime_boundary: dict[str, Any] | None) -> bool:
    if runtime_boundary is None:
        return False
    return (
        runtime_boundary.get("schema") == "edgefit.ort_runtime_boundary.v1"
        and runtime_boundary.get("target_id") == target_id
        and runtime_boundary.get("status") == "pass"
        and runtime_boundary.get("profile_coverage_status") == "pass"
        and runtime_boundary.get("generated_config_roundtrip_status") == "pass"
        and int(runtime_boundary.get("required_operator_count", -1)) == int(runtime_boundary.get("profile_operator_count", -2))
        and not runtime_boundary.get("missing_from_profile")
        and not runtime_boundary.get("profile_ops_not_required")
    )


def runtime_boundary_detail(target_id: str, path: Path, runtime_boundary: dict[str, Any] | None) -> str:
    if runtime_boundary is None:
        return f"runtime boundary result missing at {path}"
    return (
        f"target={runtime_boundary.get('target_id')}; status={runtime_boundary.get('status')}; expected={target_id}; "
        f"operators={runtime_boundary.get('required_operator_count')}/{runtime_boundary.get('profile_operator_count')}; "
        f"coverage={runtime_boundary.get('profile_coverage_status')}; roundtrip={runtime_boundary.get('generated_config_roundtrip_status')}"
    )


def public_pr_trials_passes(public_pr_trials: dict[str, Any] | None) -> bool:
    if public_pr_trials is None:
        return False
    return (
        public_pr_trials.get("schema") == "edgefit.public_pr_trial_gate.v1"
        and public_pr_trials.get("status") == "ready_for_confidence_review"
        and int(public_pr_trials.get("verified_trial_count", 0)) >= int(public_pr_trials.get("sample_goal", 3))
        and int(public_pr_trials.get("distinct_repository_count", 0)) >= int(public_pr_trials.get("sample_goal", 3))
        and int(public_pr_trials.get("invalid_trial_count", 1)) == 0
    )


def public_pr_trials_detail(path: Path, public_pr_trials: dict[str, Any] | None) -> str:
    if public_pr_trials is None:
        return f"public PR trial gate result missing at {path}"
    return (
        f"status={public_pr_trials.get('status')}; "
        f"verified_trials={public_pr_trials.get('verified_trial_count')}/{public_pr_trials.get('sample_goal')}; "
        f"repositories={public_pr_trials.get('distinct_repository_count')}/{public_pr_trials.get('sample_goal')}; "
        f"trials_needed={public_pr_trials.get('trials_needed')}"
    )


def decide(confidence: str, confidence_uplift_ready: bool) -> str:
    if confidence_uplift_ready:
        return "uplift_ready"
    if confidence == "seed":
        return "hold_seed"
    return "fail"


def next_actions(checks: list[dict[str, str]]) -> list[str]:
    actions = []
    failed = {item["id"] for item in checks if item["status"] != "pass"}
    if "runtime_smoke_verified" in failed:
        actions.append("add runtime smoke result for the target profile")
    if "runtime_boundary_verified" in failed:
        actions.append("add ORT reduced-operator boundary evidence for the target profile")
    if "public_pr_trials_verified" in failed:
        actions.append("run 3 public repository PR trials through the GitHub Action path")
    if "warning_diagnostic_policy_documented" in failed:
        actions.append("document warning-only diagnostic policy before confidence review")
    if "operator_support_audit_verified" in failed:
        actions.append("refresh the 20-model operator support audit and labels")
    if "corpus_expansion_gate_verified" in failed:
        actions.append("complete corpus expansion gate coverage before confidence review")
    if "reference_gap_counts_clear" in failed:
        actions.append("close profile reference evidence gaps")
    if "matrix_target_passes" in failed or "matrix_target_coverage" in failed:
        actions.append("refresh the corpus-by-profile matrix and resolve target failures")
    if "runtime_evidence_verified" in failed:
        actions.append("refresh pinned ONNX Runtime source evidence verification")
    if "onnx_version_pin_match" in failed:
        actions.append("restore the pinned ONNX adapter dependency version")
    if "profile_reference_target_match" in failed:
        actions.append("regenerate profile reference output for the selected target")
    return actions


def exit_code(summary: dict[str, Any]) -> int:
    return 1 if summary["decision"] == "fail" else 0


def render_markdown(summary: dict[str, Any]) -> str:
    metadata = summary["profile_metadata"]
    lines = [
        "# EdgeFit Profile Confidence Gate",
        "",
        f"**Target:** `{summary['target_id']}`",
        f"**Current confidence:** `{metadata['confidence']}`",
        f"**Decision:** `{summary['decision']}`",
        f"**Confidence uplift ready:** `{str(summary['confidence_uplift_ready']).lower()}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in summary["checks"]:
        lines.append(f"| `{item['id']}` | `{item['status']}` | {item['detail']} |")
    lines.extend(["", "## Next Actions", ""])
    if summary["next_actions"]:
        for action in summary["next_actions"]:
            lines.append(f"- {action}")
    else:
        lines.append("- profile confidence can be reviewed for uplift")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())