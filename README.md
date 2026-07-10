# EdgeFit

[![CI](https://github.com/nya-a-cat/edgefit/actions/workflows/ci.yml/badge.svg)](https://github.com/nya-a-cat/edgefit/actions/workflows/ci.yml)

EdgeFit is an ONNX deployment-budget verifier and CI gate. It checks a model
against a target profile before runtime or firmware integration, then emits
stable diagnostics for humans, GitHub Actions, and agent workflows.

The public implementation boundary is:

- Rust core for static analysis, policy, diff, reporting, and the CLI.
- Replaceable Python ONNX adapter in `tools/onnx-normalize/` for
  `onnx.checker.check_model` and `onnx.shape_inference.infer_shapes`.
- `edgefit check`, `edgefit target validate`, `edgefit snapshot`, and
  `edgefit diff` as the first command surface.
- text, JSON, Markdown, and SARIF report output with stable diagnostic locations.

## Quick Start

```powershell
powershell -ExecutionPolicy Bypass -File tools\low_disk_verify.ps1
```

The verification script keeps Rust build output, Python temporary files, uv cache,
and reports under `$env:TEMP\edgefit-work`, then removes generated files by default.
Use `-KeepTemp` when you want to inspect generated reports after a run.

Manual commands are still available:

```powershell
cargo test --workspace
cargo run -p edgefit-cli -- target validate targets/esp32s3.yaml
cargo run -p edgefit-cli -- target validate targets/ort-mobile-cpu.yaml
cargo run -p edgefit-cli -- target validate targets/tflm-micro.yaml
cargo run -p edgefit-cli -- check examples/models/good_tiny.edgefit.json --target targets/esp32s3.yaml
cargo run -p edgefit-cli -- check examples/models/bad_detector.edgefit.json --target targets/esp32s3.yaml --format sarif --out reports/edgefit.sarif --summary reports/edgefit-summary.md
cargo run -p edgefit-cli -- check examples/models/rank_dynamic.edgefit.json --target targets/esp32s3.yaml --format json --suppress EF0101,EF0102
```

For real `.onnx` files, the CLI can invoke the Python adapter directly:

```powershell
cargo run -p edgefit-cli -- check model.onnx --target targets/esp32s3.yaml
```

You can also write a normalized JSON file explicitly:

```powershell
python tools/onnx-normalize/normalize_onnx.py model.onnx --out model.edgefit.json
cargo run -p edgefit-cli -- check model.edgefit.json --target targets/esp32s3.yaml
```

The Python ONNX adapter requires the pinned official `onnx` package. Prefer an external virtual environment under `$env:TEMP\edgefit-work\venv` when installing dependencies for local evidence refreshes. Runtime smoke verification additionally uses `tools/onnx-normalize/requirements-runtime.txt`. The Rust workspace has no external crate dependency in the MVP so it can run locally without network access.

Accepted diagnostics can be suppressed by stable ID with `--suppress EF0104` or
comma-separated IDs such as `--suppress EF0101,EF0203`. Reports keep suppressed
diagnostics in a separate section so accepted risk remains visible. Snapshot
comparison treats active and suppressed states separately: a newly suppressed
error remains visible and blocks the regression diff.

## GitHub Action

The composite Action in `action.yml` builds the CLI, validates the target
profile, runs `edgefit check`, writes SARIF, writes a Markdown summary, and
appends that summary to `$GITHUB_STEP_SUMMARY`. See
`docs/GITHUB_ACTION_USAGE.md` for a workflow that also uploads the SARIF file
with `github/codeql-action/upload-sarif@v3`.

## Current Source State

The verifier source now treats missing tensor dtype/size metadata, unresolved
activation sizes under a configured budget, malformed snapshots, and
cross-profile snapshot comparisons as non-provable rather than passing them.
Direct ONNX normalization records shape-inference fallback state, includes
external weight files and ONNX opset imports in model metadata, and rejects
unsupported nested subgraphs, local functions, and sparse initializers. A
pre-normalized JSON model is always marked as trusted input, even if it contains
adapter-looking metadata; only the CLI's direct `.onnx` path supplies adapter
provenance out of band. Every used ONNX domain must have a target-profile opset
cap, and snapshot diffs cannot hide a new error by moving it into suppression.
The memory planner separates logical live tensor bytes from a deterministic
arena high-water mark. The arena plan accounts for declared tensor alignment,
per-operator workspace, fragmentation, and only explicitly authorized safe
in-place reuse; JSON reports retain allocation lifetimes and peak contributors,
while snapshots keep stable summaries instead of the full trace.

The project is not compiled or tested locally. The GitHub Actions badge is the
current source-level build and test evidence across Linux, Windows, and macOS.
Passing hosted CI does not establish real-device latency, memory, runtime, or
firmware compatibility.

## Competitive Benchmark

`tools/competitive-benchmark/benchmark.py` defines the source for a fixed,
offline comparison across EdgeFit, the ONNX Runtime Mobile usability checker,
and onnx-tool. It reuses ten SHA-256-pinned models from the existing real-world
corpus, records tool versions, duration, raw evidence and stable extracted
metrics, including the logical activation peak and planned arena high-water
mark, and deliberately keeps unlike memory metrics separate. The benchmark
source has not been run. See `docs/COMPETITIVE_BENCHMARK.md` for the metric
boundaries and planned command.

## Workspace Layout

```text
crates/
  edgefit-cli      command-line entrypoint
  edgefit-core     orchestration layer
  edgefit-ir       normalized model IR and JSON parser
  edgefit-target   target profile parser and validator
  edgefit-analyze  static facts and deterministic activation-arena planning
  edgefit-policy   stable diagnostics, suppression, and pass/fail policy
  edgefit-diff     snapshot comparison
  edgefit-report   text, JSON, Markdown, and SARIF rendering
tools/
  onnx-normalize   Python ONNX adapter
  competitive-benchmark  fixed three-tool comparison runner and case manifest
```


## Profile Metadata

Target profiles must include `metadata.source`, `metadata.confidence`, and
`metadata.last_verified`. The checked-in profiles are seed templates derived
from the feasibility report and current public ONNX Runtime profile concepts.
See `docs/TARGET_PROFILES.md` for source boundaries, trust level, and
validation commands.

## Current MVP

The local MVP covers model/profile validation, unsupported operator and dtype
checks, static-shape checks, tensor-rank checks, op-level dtype/rank checks, initializer bytes, incremental activation-lifetime accounting, deterministic best-fit arena placement, target-declared alignment/workspace and safe in-place reuse, peak-location/contributor/allocation-trace evidence, QDQ/QOperator coverage metrics, initializer dtype distribution, explicit int8 boundary policy, no-int8-path target-op detection, SARIF output with logical locations, GitHub job-summary output, stable-ID diagnostic suppression, snapshot diff, toy ONNX adapter fixtures for Add, MatMul, Conv, Reshape, Transpose, and Softmax, a quantized QLinearMatMul ONNX fixture, manifest-based 20-model real-world ONNX corpus entries, three seed target profiles for strict MCU, generic TFLM-like, and ORT mobile-like gates, a corpus-by-profile calibration matrix, generated operator fixture evidence for `Gemm`, `Resize`, and `Softmax`, pinned ONNX Runtime source evidence for `com.microsoft::QLinearAdd`, `com.microsoft::QLinearConcat`, and `com.microsoft::QLinearGlobalAveragePool`, a domain-aware ORT seed profile reference check that records profile metadata, verifies the installed ONNX package against the pinned `onnx==1.22.0` schema source, records Microsoft-domain runtime evidence, runs ONNX Runtime CPU smoke inference on all 20 verified corpus models, emits a profile confidence gate result that holds seed confidence until public PR trials pass, records the ORT profile confidence review, runs three local demo PR trials that generate SARIF and Markdown summary artifacts, adds a public PR trial gate that currently records 0/3 verified public repository trials, records a passing three-profile operator-support audit with 60 reviewed label cells and a corpus expansion gate for the 20-model target, adds ORT reduced-operator boundary evidence with a generated required-ops config, and documents the warning-only diagnostic policy used by the confidence gate.

The current public PR trial gate needs three verified public repository trials before raising the ORT profile confidence label beyond `seed`.



