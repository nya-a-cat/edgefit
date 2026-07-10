from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_reference_check import (
    load_corpus_ops,
    load_fixture_ops,
    load_runtime_ops,
    merge_op_mappings,
    parse_profile,
    write_text,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("real_world_corpus.json")
DEFAULT_FIXTURE_MANIFEST = Path(__file__).with_name("operator_fixtures.json")
DEFAULT_RUNTIME_EVIDENCE = Path(__file__).with_name("ort_runtime_evidence.json")
DEFAULT_LABELS = Path(__file__).with_name("operator_support_labels.json")
DEFAULT_MATRIX = ROOT / "tmp" / "real_world_corpus" / "profile-matrix.json"
DEFAULT_OUT = ROOT / "tmp" / "real_world_corpus" / "operator-support-audit.json"
DEFAULT_MARKDOWN = ROOT / "tmp" / "real_world_corpus" / "operator-support-audit.md"
DEFAULT_PROFILES = [
    ROOT / "targets" / "esp32s3.yaml",
    ROOT / "targets" / "ort-mobile-cpu.yaml",
    ROOT / "targets" / "tflm-micro.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit profile operator support evidence.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--fixture-manifest", default=str(DEFAULT_FIXTURE_MANIFEST))
    parser.add_argument("--runtime-evidence", default=str(DEFAULT_RUNTIME_EVIDENCE))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--profile", action="append", help="Target profile path. Repeat for multiple profiles.")
    parser.add_argument("--sample-goal", type=int, default=20)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    profiles = [Path(item) for item in args.profile] if args.profile else DEFAULT_PROFILES
    summary = build_summary(
        Path(args.manifest),
        Path(args.fixture_manifest),
        Path(args.runtime_evidence),
        Path(args.matrix),
        profiles,
        args.sample_goal,
        Path(args.labels) if args.labels else None,
    )
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.markdown_out), render_markdown(summary))
    return 0


def build_summary(
    manifest_path: Path,
    fixture_manifest_path: Path,
    runtime_evidence_path: Path,
    matrix_path: Path,
    profile_paths: list[Path],
    sample_goal: int,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    corpus_ops = load_corpus_ops(manifest_path)
    fixture_ops = load_fixture_ops(fixture_manifest_path)
    runtime_ops = load_runtime_ops(runtime_evidence_path)
    observed_ops = merge_op_mappings(corpus_ops, fixture_ops)
    evidence_ops = merge_op_mappings(observed_ops, runtime_ops)
    matrix = read_json(matrix_path)
    models = matrix.get("models", [])
    matrix_cells = matrix.get("matrix", [])
    profiles = [profile_summary(path, observed_ops, evidence_ops, matrix_cells) for path in profile_paths]
    all_observed_ops = sorted(observed_ops)
    any_missing_evidence = any(profile["allowed_without_evidence_count"] for profile in profiles)
    labels = load_labels(labels_path)
    precision_recall_review = build_precision_recall_review(labels, matrix_cells, sample_goal)
    return {
        "schema": "edgefit.operator_support_audit.v1",
        "sample_goal": sample_goal,
        "sample_model_count": len(models),
        "sample_status": "below_goal" if len(models) < sample_goal else "meets_goal",
        "profile_count": len(profiles),
        "observed_operator_count": len(all_observed_ops),
        "observed_operators": all_observed_ops,
        "evidence_operator_count": len(evidence_ops),
        "status": "needs_more_models" if len(models) < sample_goal else ("needs_profile_evidence" if any_missing_evidence else "pass"),
        "precision_recall_review": precision_recall_review,
        "profiles": profiles,
    }


def profile_summary(
    path: Path,
    observed_ops: dict[str, list[str]],
    evidence_ops: dict[str, list[str]],
    matrix_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    profile = parse_profile(path)
    allowed = set(profile["ops"])
    cells = [cell for cell in matrix_cells if cell.get("target_id") == profile["target_id"]]
    unsupported_ops = sorted({op for cell in cells for op in cell.get("unsupported_ops", [])})
    unsupported_dtypes = sorted({dtype for cell in cells for dtype in cell.get("unsupported_dtypes", [])})
    observed_allowed = sorted(allowed & set(observed_ops))
    missing_evidence = sorted(key for key in allowed if key not in evidence_ops)
    observed_outside_allowlist = sorted(set(observed_ops) - allowed)
    return {
        "target_id": profile["target_id"],
        "profile": str(path),
        "confidence": profile["metadata"].get("confidence", ""),
        "allowed_operator_count": len(allowed),
        "allowed_with_observed_evidence_count": len(observed_allowed),
        "allowed_without_evidence_count": len(missing_evidence),
        "allowed_without_evidence": missing_evidence,
        "observed_outside_allowlist_count": len(observed_outside_allowlist),
        "observed_outside_allowlist": observed_outside_allowlist,
        "matrix_model_count": len(cells),
        "matrix_pass_count": sum(1 for cell in cells if cell.get("status") == "pass"),
        "matrix_fail_count": sum(1 for cell in cells if cell.get("status") == "fail"),
        "matrix_warning_count": sum(int(cell.get("warning_count", 0)) for cell in cells),
        "matrix_error_count": sum(int(cell.get("error_count", 0)) for cell in cells),
        "unsupported_ops": unsupported_ops,
        "unsupported_dtypes": unsupported_dtypes,
    }


def load_labels(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    data = read_json(path)
    return list(data.get("labels", []))


def build_precision_recall_review(
    labels: list[dict[str, Any]], matrix_cells: list[dict[str, Any]], sample_goal: int
) -> dict[str, Any]:
    if not labels:
        return {
            "status": "requires_manual_labels",
            "reason": "unsupported-op precision and recall require sampled model labels outside the current manifest",
            "labeled_cell_count": 0,
            "labeled_model_count": 0,
        }

    matrix_by_key = {(cell.get("model_id"), cell.get("target_id")): cell for cell in matrix_cells}
    status_checks = 0
    status_matches = 0
    missing_cells: list[dict[str, str]] = []
    op_tp = op_fp = op_fn = 0
    dtype_tp = dtype_fp = dtype_fn = 0
    mismatches: list[dict[str, Any]] = []

    for label in labels:
        model_id = str(label.get("model_id", ""))
        target_id = str(label.get("target_id", ""))
        cell = matrix_by_key.get((model_id, target_id))
        if not cell:
            missing_cells.append({"model_id": model_id, "target_id": target_id})
            continue

        expected_status = label.get("expected_status")
        if expected_status:
            status_checks += 1
            if cell.get("status") == expected_status:
                status_matches += 1
            else:
                mismatches.append(
                    {
                        "model_id": model_id,
                        "target_id": target_id,
                        "field": "status",
                        "expected": expected_status,
                        "actual": cell.get("status"),
                    }
                )

        expected_ops = set(label.get("unsupported_ops", []))
        actual_ops = set(cell.get("unsupported_ops", []))
        op_tp += len(expected_ops & actual_ops)
        op_fp += len(actual_ops - expected_ops)
        op_fn += len(expected_ops - actual_ops)
        append_set_mismatch(mismatches, model_id, target_id, "unsupported_ops", expected_ops, actual_ops)

        expected_dtypes = set(label.get("unsupported_dtypes", []))
        actual_dtypes = set(cell.get("unsupported_dtypes", []))
        dtype_tp += len(expected_dtypes & actual_dtypes)
        dtype_fp += len(actual_dtypes - expected_dtypes)
        dtype_fn += len(expected_dtypes - actual_dtypes)
        append_set_mismatch(mismatches, model_id, target_id, "unsupported_dtypes", expected_dtypes, actual_dtypes)

    labeled_model_count = len({str(label.get("model_id", "")) for label in labels if label.get("model_id")})
    if missing_cells:
        status = "missing_matrix_cells"
        reason = "one or more manual labels have no matching profile-matrix cell"
    elif mismatches:
        status = "label_mismatch"
        reason = "profile-matrix output differs from manual unsupported-op labels"
    elif labeled_model_count < sample_goal:
        status = "below_sample_goal"
        reason = f"{labeled_model_count} labeled models are below the {sample_goal}-model release-grade target"
    else:
        status = "pass"
        reason = "manual unsupported-op labels match the profile matrix and meet the sample target"

    return {
        "status": status,
        "reason": reason,
        "labeled_cell_count": len(labels),
        "labeled_model_count": labeled_model_count,
        "status_match_count": status_matches,
        "status_check_count": status_checks,
        "missing_matrix_cell_count": len(missing_cells),
        "missing_matrix_cells": missing_cells,
        "unsupported_op_true_positive": op_tp,
        "unsupported_op_false_positive": op_fp,
        "unsupported_op_false_negative": op_fn,
        "unsupported_op_precision": ratio(op_tp, op_fp),
        "unsupported_op_recall": ratio(op_tp, op_fn),
        "unsupported_dtype_true_positive": dtype_tp,
        "unsupported_dtype_false_positive": dtype_fp,
        "unsupported_dtype_false_negative": dtype_fn,
        "unsupported_dtype_precision": ratio(dtype_tp, dtype_fp),
        "unsupported_dtype_recall": ratio(dtype_tp, dtype_fn),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def append_set_mismatch(
    mismatches: list[dict[str, Any]],
    model_id: str,
    target_id: str,
    field: str,
    expected: set[str],
    actual: set[str],
) -> None:
    false_positive = sorted(actual - expected)
    false_negative = sorted(expected - actual)
    if false_positive or false_negative:
        mismatches.append(
            {
                "model_id": model_id,
                "target_id": target_id,
                "field": field,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )


def ratio(true_positive: int, false_count: int) -> float | None:
    denominator = true_positive + false_count
    if denominator == 0:
        return None
    return round(true_positive / denominator, 4)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(summary: dict[str, Any]) -> str:
    review = summary["precision_recall_review"]
    lines = [
        "# EdgeFit Operator Support Audit",
        "",
        f"**Status:** `{summary['status']}`",
        f"**Sample models:** `{summary['sample_model_count']}` of `{summary['sample_goal']}`",
        f"**Observed operators:** `{summary['observed_operator_count']}`",
        f"**Profiles:** `{summary['profile_count']}`",
        "",
        "## Profiles",
        "",
        "| Target | Confidence | Allowed ops | Evidence gaps | Matrix pass/fail | Unsupported ops |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for profile in summary["profiles"]:
        unsupported = ", ".join(profile["unsupported_ops"]) if profile["unsupported_ops"] else "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    code(profile["target_id"]),
                    code(profile["confidence"]),
                    str(profile["allowed_operator_count"]),
                    str(profile["allowed_without_evidence_count"]),
                    f"{profile['matrix_pass_count']}/{profile['matrix_fail_count']}",
                    code(unsupported),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Precision And Recall Review",
            "",
            f"**Status:** `{review['status']}`",
            f"**Labeled cells:** `{review['labeled_cell_count']}`",
            f"**Labeled models:** `{review['labeled_model_count']}` of `{summary['sample_goal']}`",
            f"**Status matches:** `{review.get('status_match_count', 0)}` of `{review.get('status_check_count', 0)}`",
            f"**Unsupported-op precision/recall:** `{format_metric(review.get('unsupported_op_precision'))}` / `{format_metric(review.get('unsupported_op_recall'))}`",
            f"**Unsupported-dtype precision/recall:** `{format_metric(review.get('unsupported_dtype_precision'))}` / `{format_metric(review.get('unsupported_dtype_recall'))}`",
            "",
            review["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def code(value: str) -> str:
    return "`" + value.replace("|", "\\|") + "`"


def format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())