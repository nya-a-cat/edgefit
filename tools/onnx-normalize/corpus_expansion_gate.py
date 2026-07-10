from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_reference_check import parse_profile, write_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("real_world_corpus.json")
DEFAULT_LABELS = Path(__file__).with_name("operator_support_labels.json")
DEFAULT_OUT = ROOT / "tmp" / "real_world_corpus" / "corpus-expansion-gate.json"
DEFAULT_MARKDOWN = ROOT / "tmp" / "real_world_corpus" / "corpus-expansion-gate.md"
DEFAULT_PROFILES = [
    ROOT / "targets" / "esp32s3.yaml",
    ROOT / "targets" / "ort-mobile-cpu.yaml",
    ROOT / "targets" / "tflm-micro.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check corpus and label coverage before profile-matrix calibration.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--profile", action="append", help="Target profile path. Repeat for multiple profiles.")
    parser.add_argument("--sample-goal", type=int, default=20)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    profiles = [Path(item) for item in args.profile] if args.profile else DEFAULT_PROFILES
    summary = build_summary(Path(args.manifest), Path(args.labels), profiles, args.sample_goal)
    write_text(Path(args.out), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_text(Path(args.markdown_out), render_markdown(summary))
    return 0


def build_summary(manifest_path: Path, labels_path: Path, profile_paths: list[Path], sample_goal: int) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    labels = read_json(labels_path)
    model_ids = [str(model.get("id", "")) for model in manifest.get("models", []) if model.get("id")]
    target_ids = [parse_profile(path)["target_id"] for path in profile_paths]
    expected_cells = {(model_id, target_id) for model_id in model_ids for target_id in target_ids}
    label_cells: list[tuple[str, str]] = [
        (str(label.get("model_id", "")), str(label.get("target_id", ""))) for label in labels.get("labels", [])
    ]
    label_cell_set = set(label_cells)
    duplicate_cells = sorted(cell for cell in label_cell_set if label_cells.count(cell) > 1)
    missing_cells = sorted(expected_cells - label_cell_set)
    unknown_cells = sorted(label_cell_set - expected_cells)
    label_status = "complete" if not missing_cells and not duplicate_cells and not unknown_cells else "incomplete"
    if label_status == "incomplete":
        status = "label_coverage_incomplete"
    elif len(model_ids) < sample_goal:
        status = "needs_more_models"
    else:
        status = "ready_for_profile_matrix"
    return {
        "schema": "edgefit.corpus_expansion_gate.v1",
        "status": status,
        "sample_goal": sample_goal,
        "model_count": len(model_ids),
        "models_needed": max(sample_goal - len(model_ids), 0),
        "target_count": len(target_ids),
        "target_ids": target_ids,
        "expected_label_cell_count": len(expected_cells),
        "label_cell_count": len(label_cells),
        "label_status": label_status,
        "missing_label_cell_count": len(missing_cells),
        "missing_label_cells": render_cells(missing_cells),
        "duplicate_label_cell_count": len(duplicate_cells),
        "duplicate_label_cells": render_cells(duplicate_cells),
        "unknown_label_cell_count": len(unknown_cells),
        "unknown_label_cells": render_cells(unknown_cells),
    }


def render_cells(cells: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"model_id": model_id, "target_id": target_id} for model_id, target_id in cells]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EdgeFit Corpus Expansion Gate",
        "",
        f"**Status:** `{summary['status']}`",
        f"**Models:** `{summary['model_count']}` of `{summary['sample_goal']}`",
        f"**Models needed:** `{summary['models_needed']}`",
        f"**Targets:** `{summary['target_count']}`",
        f"**Label cells:** `{summary['label_cell_count']}` of `{summary['expected_label_cell_count']}`",
        f"**Label status:** `{summary['label_status']}`",
        "",
        "## Label Coverage",
        "",
        "| Gap type | Count |",
        "| --- | ---: |",
        f"| Missing labels | {summary['missing_label_cell_count']} |",
        f"| Duplicate labels | {summary['duplicate_label_cell_count']} |",
        f"| Unknown labels | {summary['unknown_label_cell_count']} |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())