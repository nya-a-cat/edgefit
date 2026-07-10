# EdgeFit

[![CI](https://github.com/nya-a-cat/edgefit/actions/workflows/ci.yml/badge.svg)](https://github.com/nya-a-cat/edgefit/actions/workflows/ci.yml)

EdgeFit statically verifies an ONNX model against an explicit deployment target
contract. It reports graph compatibility, activation-memory planning, policy
violations, and regression evidence through a CLI and GitHub Action.

```text
model.onnx + target profile
            ↓
  graph facts · memory plan · policy
            ↓
 pass / fail + JSON · Markdown · SARIF
```

EdgeFit is not an inference runtime or a device benchmark.

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
```

Exit codes are part of the public CLI contract:

| Code | Meaning |
| ---: | --- |
| `0` | Analysis completed and the model fits the target contract. |
| `1` | Analysis completed and produced a deployment-blocking decision. |
| `2` | Input, configuration, adapter, or invocation failure. |

### Check a real ONNX model

Direct `.onnx` input uses the pinned Python adapter. With
[uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
  -r tools/onnx-normalize/requirements.txt

EDGEFIT_PYTHON="$PWD/.venv/bin/python" \
  ./target/release/edgefit check model.onnx \
  --target targets/ort-mobile-cpu.yaml \
  --format json \
  --out edgefit-report.json
```

The adapter delegates model validation and shape inference to the pinned
official ONNX package. It is replaceable; the Rust analysis and policy core do
not depend on Python packages.

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
      - uses: actions/checkout@v4
      - uses: nya-a-cat/edgefit@main
        with:
          model: models/model.onnx
          target: targets/device.yaml
          report: edgefit.sarif
          summary: edgefit-summary.md
```

Until the first stable tag is published, pin long-lived workflows to a reviewed
commit SHA rather than `main`. The Action validates the target, runs the check,
publishes the Markdown job summary, and then restores the EdgeFit exit status.
See [GitHub Action usage](docs/GITHUB_ACTION_USAGE.md) for SARIF upload,
suppression, and fork-PR handling.

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

Project-specific profiles should live with the consuming repository and record their
source, confidence, and last verification date. See
[Target profiles](docs/TARGET_PROFILES.md).

## Hosted Evidence

The normal CI gate runs Rust and ONNX adapter tests on Linux, Windows, and macOS,
plus the composite Action smoke test and a required 10K-node Release planning
check.

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
latency, throughput, power, firmware, or real-hardware memory measurements. See
the [successful maturity run](https://github.com/nya-a-cat/edgefit/actions/runs/29103544134)
and [benchmark methodology](docs/COMPETITIVE_BENCHMARK.md).

## Command Surface

```text
edgefit check             verify a model against a target
edgefit target validate   validate a target profile
edgefit snapshot          freeze a reviewable result
edgefit diff              block deployment regressions
```

The compatibility policy, machine-output schemas, and exit behavior are defined
in the [CLI contract](docs/CLI_CONTRACT.md).

## Architecture

- `crates/edgefit-*` — dependency-light Rust core for IR, target profiles,
  analysis, policy, reporting, diffs, and CLI orchestration.
- `tools/onnx-normalize/` — replaceable Python boundary for official ONNX
  checking and shape inference.
- `tools/competitive-benchmark/` — fixed-corpus evidence runner that preserves
  raw tool output and keeps unlike metrics separate.
- `action.yml` — Linux/bash composite Action for deployment-budget gating.

The Rust workspace forbids unsafe code and currently has no external crate
dependency. See [Architecture](docs/ARCHITECTURE.md) and
[MVP scope](docs/MVP_SCOPE.md) for the detailed implementation boundary.

## Limitations

- Checked-in targets are seed profiles and require project-specific validation.
- Hosted timings measure complete CLI processes, not device inference.
- Passing CI does not establish firmware, runtime, power, or real-device memory
  compatibility.
- Direct ONNX normalization rejects nested subgraphs, local functions, and sparse
  initializers instead of partially analyzing them.

## Documentation

- [CLI contract](docs/CLI_CONTRACT.md)
- [GitHub Action usage](docs/GITHUB_ACTION_USAGE.md)
- [Target profiles](docs/TARGET_PROFILES.md)
- [Real-world ONNX corpus](docs/REAL_WORLD_CORPUS.md)
- [Competitive benchmark](docs/COMPETITIVE_BENCHMARK.md)
- [ESP-DL / ESP32-S3 simulated deployment](docs/SIMULATED_DEPLOYMENT.md)
- [Publishing policy](docs/PUBLISHING.md)

## License

MIT
