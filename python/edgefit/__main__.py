"""`python -m edgefit` 的单一维护入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .framework import (
    EdgeFitError,
    _run_calibration,
    check,
    optimize,
    render,
    render_optimization,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m edgefit")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("check")
    command.add_argument("model")
    command.add_argument("--target", required=True)
    command.add_argument("--format", choices=["text", "json", "markdown", "sarif"], default="text")
    command.add_argument("--out")
    command.add_argument("--suppress", action="append", default=[])
    optimize_command = subcommands.add_parser("optimize")
    optimize_command.add_argument("model")
    optimize_command.add_argument("--target", required=True)
    optimize_command.add_argument("--format", choices=["json", "markdown"], default="json")
    optimize_command.add_argument("--out")
    calibration_command = subcommands.add_parser("calibration")
    calibration_subcommands = calibration_command.add_subparsers(
        dest="calibration_command", required=True
    )
    verify_command = calibration_subcommands.add_parser("verify")
    verify_command.add_argument("evidence")
    verify_command.add_argument("--model", required=True)
    verify_command.add_argument("--target", required=True)
    verify_command.add_argument("--format", choices=["json", "markdown"], default="json")
    verify_command.add_argument("--out")
    args = parser.parse_args()

    try:
        if args.command == "calibration":
            status, output = _run_calibration(
                args.evidence,
                args.model,
                args.target,
                args.format,
            )
            report = {"status": status}
        elif args.command == "optimize":
            report = optimize(args.model, args.target)
            output = render_optimization(args.model, args.target, format=args.format)
        else:
            report = check(args.model, args.target, suppress=args.suppress)
            output = render(args.model, args.target, format=args.format, suppress=args.suppress)
    except (EdgeFitError, OSError) as exc:
        parser.exit(2, f"edgefit: {exc}\n")

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="" if output.endswith("\n") else "\n")
    return 1 if report.get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
