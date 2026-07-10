from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "public_pr_trials.json"
DEFAULT_OUT = ROOT / "tmp" / "public_pr_trials" / "public-pr-trial-gate.json"
DEFAULT_MARKDOWN = ROOT / "tmp" / "public_pr_trials" / "public-pr-trial-gate.md"

GITHUB_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")
RESULTS = {"pass", "fail"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate public GitHub PR trial evidence.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    summary = build_summary(Path(args.manifest))
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.markdown_out), render_markdown(summary))
    return exit_code(summary)


def build_summary(manifest_path: Path) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    sample_goal = int(manifest.get("sample_goal", 3))
    schema_ok = manifest.get("schema") == "edgefit.public_pr_trials.v1"
    sample_goal_ok = 3 <= sample_goal <= 5
    trials = [evaluate_trial(item) for item in manifest.get("trials", [])]
    verified_trials = [trial for trial in trials if trial["status"] == "verified"]
    invalid_trials = [trial for trial in trials if trial["status"] == "invalid"]
    verified_repositories = sorted({trial["repository"] for trial in verified_trials})
    trials_needed = max(0, sample_goal - len(verified_trials))
    repositories_needed = max(0, sample_goal - len(verified_repositories))

    checks = [
        check("manifest_schema", schema_ok, f"schema={manifest.get('schema')}; expected edgefit.public_pr_trials.v1"),
        check("sample_goal_range", sample_goal_ok, f"sample_goal={sample_goal}; expected 3 to 5"),
        check("invalid_trial_count", len(invalid_trials) == 0, f"invalid_trial_count={len(invalid_trials)}"),
        check(
            "verified_trial_count",
            len(verified_trials) >= sample_goal,
            f"verified_trial_count={len(verified_trials)}; sample_goal={sample_goal}",
        ),
        check(
            "distinct_repository_count",
            len(verified_repositories) >= sample_goal,
            f"distinct_repository_count={len(verified_repositories)}; sample_goal={sample_goal}",
        ),
    ]
    if invalid_trials or not schema_ok or not sample_goal_ok:
        status = "invalid_evidence"
    elif all(item["status"] == "pass" for item in checks):
        status = "ready_for_confidence_review"
    else:
        status = "needs_public_trials"

    return {
        "schema": "edgefit.public_pr_trial_gate.v1",
        "manifest": str(manifest_path),
        "status": status,
        "sample_goal": sample_goal,
        "trial_count": len(trials),
        "verified_trial_count": len(verified_trials),
        "invalid_trial_count": len(invalid_trials),
        "distinct_repository_count": len(verified_repositories),
        "trials_needed": trials_needed,
        "repositories_needed": repositories_needed,
        "verified_repositories": verified_repositories,
        "checks": checks,
        "trials": trials,
    }


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_trial(trial: dict[str, Any]) -> dict[str, Any]:
    errors = trial_errors(trial)
    return {
        "id": str(trial.get("id", "")),
        "repository": str(trial.get("repository", "")),
        "pull_request_url": str(trial.get("pull_request_url", "")),
        "workflow_run_url": str(trial.get("workflow_run_url", "")),
        "target_profile": str(trial.get("target_profile", "")),
        "model_path": str(trial.get("model_path", "")),
        "expected_result": str(trial.get("expected_result", "")),
        "actual_result": str(trial.get("actual_result", "")),
        "status": "verified" if not errors else "invalid",
        "errors": errors,
    }


def trial_errors(trial: dict[str, Any]) -> list[str]:
    errors = []
    required_text = [
        "id",
        "repository",
        "pull_request_url",
        "workflow_run_url",
        "commit_sha",
        "target_profile",
        "model_path",
        "expected_result",
        "actual_result",
    ]
    for field in required_text:
        if not str(trial.get(field, "")).strip():
            errors.append(f"{field} is required")

    repository = str(trial.get("repository", ""))
    pull_request_url = str(trial.get("pull_request_url", ""))
    workflow_run_url = str(trial.get("workflow_run_url", ""))
    commit_sha = str(trial.get("commit_sha", ""))
    expected_result = str(trial.get("expected_result", ""))
    actual_result = str(trial.get("actual_result", ""))

    if repository and not GITHUB_REPO.fullmatch(repository):
        errors.append("repository must use owner/name form")
    if repository and pull_request_url and not pull_request_url.startswith(f"https://github.com/{repository}/pull/"):
        errors.append("pull_request_url must point at the repository PR")
    if workflow_run_url and "/actions/runs/" not in workflow_run_url:
        errors.append("workflow_run_url must point at a GitHub Actions run")
    if workflow_run_url and not workflow_run_url.startswith("https://github.com/"):
        errors.append("workflow_run_url must use a GitHub URL")
    if commit_sha and not COMMIT_SHA.fullmatch(commit_sha):
        errors.append("commit_sha must be a 7 to 40 character hex SHA")
    if expected_result and expected_result not in RESULTS:
        errors.append("expected_result must be pass or fail")
    if actual_result and actual_result not in RESULTS:
        errors.append("actual_result must be pass or fail")
    for field in ["public_repository", "sarif_uploaded", "job_summary_present", "outcome_clear"]:
        if trial.get(field) is not True:
            errors.append(f"{field} must be true")
    return errors


def check(check_id: str, passed: bool, detail: str) -> dict[str, str]:
    return {"id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def exit_code(summary: dict[str, Any]) -> int:
    return 1 if summary["status"] == "invalid_evidence" else 0


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EdgeFit Public PR Trial Gate",
        "",
        f"**Status:** `{summary['status']}`",
        f"**Verified trials:** `{summary['verified_trial_count']}/{summary['sample_goal']}`",
        f"**Distinct repositories:** `{summary['distinct_repository_count']}/{summary['sample_goal']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in summary["checks"]:
        lines.append(f"| `{item['id']}` | `{item['status']}` | {item['detail']} |")
    lines.extend(["", "## Trials", "", "| Trial | Repository | Status | Errors |", "| --- | --- | --- | --- |"])
    if summary["trials"]:
        for trial in summary["trials"]:
            errors = "; ".join(trial["errors"]) if trial["errors"] else ""
            lines.append(f"| `{trial['id']}` | `{trial['repository']}` | `{trial['status']}` | {errors} |")
    else:
        lines.append("|  |  | `missing` | add public GitHub PR trial evidence |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
