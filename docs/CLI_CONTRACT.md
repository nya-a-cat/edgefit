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
the generated report. Exit code `2` means the requested result was not produced
and must not be interpreted as either pass or policy fail.

## Stable Machine Schemas

- JSON report: `edgefit.report.v1`
- Snapshot: `edgefit.snapshot.v1`
- Optimization plan: `edgefit.optimization_plan.v1`
- Snapshot diff: `edgefit.diff.v1`
- Execution error: `edgefit.execution_error.v1`
- SARIF: SARIF `2.1.0` with stable EdgeFit diagnostic IDs and logical locations

Within a `v1` schema, fields may be added but existing fields cannot be removed,
renamed, or change meaning. Arrays may gain new entries. Consumers must ignore
unknown fields and use the schema identifier before reading a document.

Legacy `edgefit.report.v1` input remains accepted by the diff loader for
snapshots produced before the dedicated snapshot schema existed.

When direct ONNX normalization or adapter-backed analysis cannot produce a
trustworthy result, `check` and `snapshot` exit with code `2`. If `--out` was
provided with `--format json`, `--format markdown`, or `--format sarif`, EdgeFit
writes `edgefit.execution_error.v1` instead of leaving that evidence path empty.
Text output retains its human-readable CLI error form. A requested `--summary`
receives the corresponding Markdown execution-error document. Argument parsing
and validation failures do not create execution artifacts. These artifacts record
an execution failure and must never be interpreted as a normal report or snapshot.

## Compatibility Gate

Hosted CI verifies command discovery, the three exit-code classes, report and
diff schema identifiers, parseable JSON/SARIF, and snapshot regression behavior
on Linux, Windows, and macOS. A change that intentionally breaks this contract
requires a new schema or a new minor command surface and an explicit migration
document.
