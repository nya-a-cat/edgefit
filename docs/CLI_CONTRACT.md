# EdgeFit Alpha CLI Contract

This document freezes the public command and machine-output boundary for the
`0.2` Alpha line. Internal analysis and report fields may grow, but automation
must be able to rely on the behavior below.

## Commands

```text
edgefit version
edgefit target validate <profile>
edgefit check <model.onnx|model.edgefit.json> --target <profile> [--format text|json|markdown|sarif] [--out path] [--summary path] [--suppress id[,id]]
edgefit optimize <model.onnx|model.edgefit.json> --target <profile> [--format json|markdown] [--out path]
edgefit snapshot <model.onnx|model.edgefit.json> --target <profile> --out path
edgefit diff --old path --new path [--format markdown|json] [--out path]
```

No command in this list may be removed or renamed within the `0.2` Alpha line.
New optional flags must preserve existing defaults.

## Exit Codes

| Code | Meaning |
| ---: | --- |
| `0` | The command completed and the model, profile, or diff gate passed. |
| `1` | Analysis completed, but an unsuppressed policy diagnostic or snapshot regression failed the gate. |
| `2` | The command could not produce a trustworthy gate result because arguments, input, dependencies, or execution failed. |

Exit code `1` is evidence, not a CLI crash. CI integrations should still retain
the generated report or plan. Exit code `2` means the requested result was not
produced and must not be interpreted as either pass or policy fail.

### Optimizer outcomes

`edgefit optimize` uses all three exit-code classes:

- `0`: planning completed and produced an `edgefit.optimization_plan.v1` document
  with `status: "pass"` and no unresolved blockers.
- `1`: planning completed and produced a canonical
  `edgefit.optimization_plan.v1` fail plan with `status: "fail"`. The plan is
  valid evidence, not an execution error: it retains the deterministic
  `plan_hash`, complete node assignments, exact NPU segments, transfer events,
  totals, and the blockers that prevented a passing plan. Consumers must retain
  and parse this plan rather than treating it as missing output.
- `2`: EdgeFit could not produce a trustworthy optimization plan because input
  preparation, target loading, a dependency, or planner execution failed. No
  pass or fail plan may be inferred from this result. When execution-error
  artifact emission applies, the requested output contains
  `edgefit.execution_error.v1` instead of an optimization plan.

For the same normalized model, target profile, and implementation, pass and fail
plans are deterministic. A fail plan remains canonical even when latency is
unknown or scratchpad pressure creates blockers: `status`, blocker counts,
assignments, segments, events, totals, and `plan_hash` must describe that one
completed planning result consistently.

## Stable Machine Schemas

- JSON report: `edgefit.report.v1`
- Snapshot: `edgefit.snapshot.v1`
- Optimization plan: `edgefit.optimization_plan.v1`; `plan_hash` is deterministic
  and binds the model hash, target profile fingerprint, assignments, segments,
  and transfer events
- Snapshot diff: `edgefit.diff.v1`
- Execution error: `edgefit.execution_error.v1`
- SARIF: SARIF `2.1.0` with stable EdgeFit diagnostic IDs and logical locations

Within a `v1` schema, fields may be added but existing fields cannot be removed,
renamed, or change meaning. Arrays may gain new entries. Consumers must ignore
unknown fields and use the schema identifier before reading a document.

Legacy `edgefit.report.v1` input remains accepted by the diff loader for
snapshots produced before the dedicated snapshot schema existed.

When direct ONNX normalization or adapter-backed analysis or planning cannot
produce a trustworthy result, `check`, `optimize`, and `snapshot` exit with code
`2`. If `--out` was provided, EdgeFit overwrites any stale file at that path with
an `edgefit.execution_error.v1` artifact in the requested machine or Markdown
format instead of leaving an earlier report, plan, or snapshot in place. For
`check --format text`, the output artifact retains the human-readable CLI error
form. A requested `--summary` is likewise overwritten with the corresponding
Markdown execution-error document. This replacement rule prevents artifacts from
a previous successful or policy-failing invocation from being mistaken for the
current result.

Argument parsing and validation failures do not create or replace execution
artifacts. Execution-error artifacts record an execution failure and must never
be interpreted as a normal report, optimization plan, or snapshot. Consumers
must inspect the schema or format-specific execution-error marker before reading
command-specific fields.

## Compatibility Gate

Hosted CI verifies command discovery, the three exit-code classes, report and
diff schema identifiers, parseable JSON/SARIF, and snapshot regression behavior
on Linux, Windows, and macOS. A change that intentionally breaks this contract
requires a new schema or a new minor command surface and an explicit migration
document.
