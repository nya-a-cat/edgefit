# EdgeFit

[![CI](https://github.com/nya-a-cat/edgefit/actions/workflows/ci.yml/badge.svg)](https://github.com/nya-a-cat/edgefit/actions/workflows/ci.yml)

EdgeFit statically verifies an ONNX model against an explicit deployment target
contract and can generate a deterministic, hardware-aware optimization plan. It
reports graph compatibility, activation-memory planning, policy violations,
regression evidence, and profile-driven CPU/NPU partition estimates through the
Rust CLI and Python API; the verifier is also available as a GitHub Action.

```text
model.onnx + target profile
            ↓
 graph facts · memory plan · policy ──→ pass / fail + JSON · Markdown · SARIF
            │
            └─ accelerator contract ──→ assignments · NPU segments · DMA/spill
                                        + estimated latency · plan hash
```

EdgeFit is not an inference runtime, compiler, or device benchmark. Optimization
latency is a profile-driven estimate, not measured hardware performance, and a
successful plan does not establish real-device deployment compatibility.

## Scope

- Operator, domain, dtype, opset, rank, shape, model-size, and activation-memory
  checks are defined by a target profile.
- Missing dtype, unresolved tensor size, unsupported graph structure, and
  unverifiable memory budgets fail closed.
- Memory reports separate logical live-tensor bytes from deterministic arena
  placement, alignment, workspace, fragmentation, and safe in-place reuse.
- Stable diagnostic IDs, suppressions, snapshots, diffs, JSON, Markdown, and
  SARIF support automated review.

## Quick Start — Linux and macOS

The included normalized fixture needs only a stable Rust toolchain:

```bash
cargo build -p edgefit-cli --release --locked

./target/release/edgefit check \
  examples/models/good_tiny.edgefit.json \
  --target targets/esp32s3.yaml

./target/release/edgefit optimize \
  examples/models/virtual_npu_tiny.edgefit.json \
  --target targets/virtual-npu.yaml \
  --format json \
  --out edgefit-plan.json
```

Exit codes are part of the public CLI contract:

| Code | Meaning |
| ---: | --- |
| `0` | Verification or planning completed without an unresolved blocker. |
| `1` | Verification or planning completed with a deployment-blocking decision. |
| `2` | Input, configuration, adapter, or invocation failure prevented a trustworthy result. |

### Check a real ONNX model

Direct `.onnx` input uses the pinned Python adapter. With
[uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
  -r requirements-onnx.txt

EDGEFIT_PYTHON="$PWD/.venv/bin/python" \
  ./target/release/edgefit check model.onnx \
  --target targets/ort-mobile-cpu.yaml \
  --format json \
  --out edgefit-report.json
```

The adapter delegates model validation and shape inference to the pinned
official ONNX package. It is replaceable; the Rust analysis and policy core do
not depend on Python packages.

### Python framework

Prebuilt Python 3.10+ wheels for Linux x86_64, Windows x86_64, and macOS
universal2 are attached to the
[`v0.4.0-alpha.1` GitHub Release](https://github.com/nya-a-cat/edgefit/releases/tag/v0.4.0-alpha.1).
Download the wheel matching the current platform, verify it with the release
`SHA256SUMS`, and install that local file; EdgeFit is not published on PyPI:

```bash
python -m pip install ./edgefit-0.4.0a1-cp310-abi3-<platform>.whl
```

The package uses a prebuilt PyO3 extension over the same Rust engine. It does
not compile Rust source during import:

```python
import edgefit

report = edgefit.check("model.onnx", "targets/device.yaml")
reports = edgefit.batch(["a.onnx", "b.onnx"], "targets/device.yaml")
plan = edgefit.optimize("model.onnx", "targets/virtual-npu.yaml")
```

### Deterministic calibration simulation

The source build can generate controlled, reproducible Calibration v1 evidence
without claiming a real device measurement:

```bash
./target/release/edgefit calibration simulate \
  examples/models/virtual_npu_tiny.edgefit.json \
  --target targets/virtual-npu.yaml \
  --scenario examples/calibration/nominal.simulation.json \
  --out-dir edgefit-simulation

./target/release/edgefit calibration verify \
  edgefit-simulation/evidence.json \
  --model examples/models/virtual_npu_tiny.edgefit.json \
  --target targets/virtual-npu.yaml
```

The simulator runs the normal analyzer and optimizer, then applies the exact
ppm perturbations declared by the scenario. Its evidence is SHA-256-bound and
explicitly marked `simulated`; it provides no device attestation, hardware
latency, or authority to modify a target profile.

`--out-dir` must not already exist. A completed directory contains
`simulator-runtime.bin`, `simulation-trace.json`, `evidence.json`,
`verification.json`, and `verification.md`.


## Use It as a Pull Request Gate

```yaml
name: EdgeFit

on: [pull_request]

jobs:
  deployment-budget:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v7
      - uses: nya-a-cat/edgefit@v0.4.0-alpha.1
        with:
          model: models/model.onnx
          target: targets/device.yaml
          report: edgefit.sarif
          summary: edgefit-summary.md
```

`v0.4.0-alpha.1` is a prerelease intended for reproducible evaluation. Until a
stable tag is published, pin long-lived or production workflows to a reviewed
full commit SHA. On Linux x86_64, the Action downloads the matching release
archive and verifies it against the published `SHA256SUMS`; it does not install
Rust or compile EdgeFit in the caller workflow. The Action validates the target,
runs the check, publishes the Markdown job summary, and then restores the
EdgeFit exit status. SARIF upload remains a caller-controlled step so the
consuming repository can apply its own fork-PR permissions policy.

## What Gets Verified

| Layer | Evidence produced |
| --- | --- |
| Model integrity | ONNX validation, shape-inference status, external data, opset imports, graph-boundary checks |
| Target compatibility | Operator/domain, dtype, rank, shape, model-file, and runtime-policy diagnostics |
| Activation memory | Logical peak, deterministic arena plan, peak node, workspace, fragmentation, and allocation trace |
| Quantization | QDQ/QOperator coverage, int8 boundary state, dtype distribution, and missing metadata |
| Change control | Stable diagnostic IDs, explicit suppressions, snapshots, and cross-run regression diffs |
| Automation | Text, JSON, Markdown, SARIF, GitHub job summaries, and defined exit codes |

The checked-in target profiles are **seed templates**, not verified hardware
claims:

| Profile | Intended starting point |
| --- | --- |
| `targets/esp32s3.yaml` | Strict MCU-style budget review |
| `targets/tflm-micro.yaml` | TensorFlow Lite Micro-like review |
| `targets/ort-mobile-cpu.yaml` | ONNX Runtime Mobile CPU-like review |
| `targets/virtual-npu.yaml` | Simulated CPU/NPU optimization seed; costs and latency are not hardware measurements |

Project-specific profiles should live with the consuming repository and record
their source, confidence, and last verification date.

## Hosted Evidence

The normal CI gate runs Rust, ONNX adapter, and Python binding tests on Linux,
Windows, and macOS, plus Composite Action, 10K-node activation-planner,
optimizer, and deterministic calibration-simulation contract gates.

The hosted maturity run processed deterministic linear graphs five times each:

| Nodes | Median process time | Maximum peak RSS | Deterministic report |
| ---: | ---: | ---: | --- |
| 1,000 | 7 ms | 6,275,072 B | 5/5 hashes identical |
| 10,000 | 70 ms | 35,991,552 B | 5/5 hashes identical |
| 100,000 | 854 ms | 336,494,592 B | 5/5 hashes identical |

The same run executed a fixed ten-model matrix with EdgeFit, ONNX Runtime Mobile
Checker, and onnx-tool:

| Tool | Completed analysis | Explicit rejection |
| --- | ---: | ---: |
| EdgeFit | 9/10 | 1/10 |
| ORT Mobile Checker | 9/10 | 1/10 |
| onnx-tool | 4/10 | 6/10 |

EdgeFit produced target-budget decisions for five models that onnx-tool rejected.
The tools cover different tasks: EdgeFit evaluates target contracts, onnx-tool
profiles compute and model structure, and ORT Mobile Checker estimates execution
provider usability.

These figures are hosted end-to-end process observations, not device inference
latency, throughput, power, firmware, or real-hardware memory measurements. The
[successful maturity run](https://github.com/nya-a-cat/edgefit/actions/runs/29103544134)
and fixed manifests under `tools/competitive-benchmark/` retain the evidence.

## Command Surface

```text
edgefit check             verify a model against a target
edgefit optimize          estimate and partition an accelerator execution plan
edgefit calibration verify    verify hash-bound Calibration v1 evidence
edgefit calibration simulate  generate controlled simulated evidence
edgefit target validate   validate a target profile
edgefit snapshot          freeze a reviewable result
edgefit diff              block deployment regressions
```

The exit codes above and `edgefit.*.v1` machine-output schemas are public
compatibility contracts. Incompatible changes require a new schema version.

## Architecture

- `crates/edgefit-*` — dependency-light Rust core for IR, target profiles,
  analysis, policy, hardware planning, calibration simulation/verification,
  reporting, diffs, and CLI orchestration.
- `tools/onnx-normalize/` — replaceable Python boundary for official ONNX
  checking and shape inference.
- `tools/competitive-benchmark/` — fixed-corpus evidence runner that preserves
  raw tool output and keeps unlike metrics separate.
- `action.yml` — Linux/bash composite Action for deployment-budget gating.

The Rust workspace forbids unsafe code and currently has no external crate
dependency.

## Limitations

- Checked-in targets are seed profiles and require project-specific validation;
  `targets/virtual-npu.yaml` is explicitly simulated.
- Hosted timings measure complete CLI processes, not device inference.
- Optimizer latency is derived from profile costs, not measured on hardware.
- Calibration simulation applies controlled perturbations to static estimates;
  it is not empirical calibration or device evidence.
- Passing verification or optimization planning does not establish firmware,
  runtime, power, or real-device memory compatibility.
- Direct ONNX normalization rejects nested subgraphs, local functions, and sparse
  initializers instead of partially analyzing them.

## License

MIT
