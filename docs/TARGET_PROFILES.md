# Target Profiles

EdgeFit target profiles are CI contracts for deployment budgets. Early profiles
must carry `metadata.source`, `metadata.confidence`, and `metadata.last_verified`
so users can judge the trust level before wiring a profile into a gate.

## Included Profiles

| File | Target ID | Intended use | Confidence |
| --- | --- | --- | --- |
| `targets/esp32s3.yaml` | `esp32s3_custom_v1` | Strict static-shape int8 MCU gate for an ESP32-S3-style board | `seed` |
| `targets/ort-mobile-cpu.yaml` | `ort_mobile_cpu_seed_v1` | ORT mobile-like CPU gate for edge devices that allow fp32/int8 models, common ONNX vision operators, quantized operators, and detection post-processing operators | `seed` |
| `targets/tflm-micro.yaml` | `tflm_micro_seed_v1` | Generic TFLM-like MCU gate for int8 static-shape models | `seed` |
| `targets/virtual-npu.yaml` | `edgefit_virtual_npu_v1` | Simulated CPU/NPU optimization planning with declared kernel, DMA, alignment, and scratchpad costs | `seed`; accelerator `seed-simulated` |
| `targets/virtual-npu-segmented.yaml` | `edgefit_virtual_npu_segmented_v1` | Test-only simulated CPU/NPU segment boundary evidence | `seed`; accelerator `seed-simulated` |
| `targets/virtual-npu-small-scratchpad.yaml` | `edgefit_virtual_npu_small_scratchpad_v1` | Test-only simulated spill/reload evidence on a constrained scratchpad | `seed`; accelerator `seed-simulated` |
| `targets/virtual-npu-no-spill.yaml` | `edgefit_virtual_npu_no_spill_v1` | Test-only fail-closed scratchpad pressure with spilling disabled | `seed`; accelerator `seed-simulated` |

These profiles are seed templates. The verifier profiles make compatibility
behavior reproducible and show how a repository should encode source,
confidence, budgets, dtype rules, operator allowlists, and shape policy. The
virtual NPU profiles instead supply simulated inputs to `edgefit optimize`; their
latency output is a profile-driven estimate, not measured hardware performance.
The segmented, small-scratchpad, and no-spill variants are evidence fixtures
only and must not be used as deployable target contracts.

## Source Boundary

The feasibility report calls out a small set of early profiles and warns that
profile trust is a product risk. The checked-in profiles follow that boundary:
the strict MCU seed and generic TFLM-like seed keep narrow int8/static-shape
contracts, while the ORT mobile-like seed uses a broader operator, dtype, and
memory policy for CI experimentation. Their metadata keeps the confidence at
`seed`. The virtual NPU profile is an EdgeFit-authored simulation contract with
`accelerator.confidence: seed-simulated`; it is not evidence for a physical NPU,
compiler, runtime, or deployable device configuration.

The ONNX Runtime reduced operator config documentation shows that constrained
mobile and web builds can be represented by an operator set. EdgeFit uses that
idea as a YAML profile contract for pre-deployment checks. The current ORT
mobile-like seed includes common ONNX vision operators plus Microsoft-domain
quantized operators verified against pinned ONNX Runtime source evidence.

The `quantization.require_int8` field is an explicit graph-boundary contract.
It may be enabled only with `quantization.required: true` and with `int8` or
`uint8` in `dtype.allowed`. The strict MCU and TFLM-like profiles enable it;
the ORT mobile-like profile leaves it disabled because that runtime profile
allows floating-point model boundaries.

## Profile Contract Fields

The parser requires target identity, source metadata, physical memory values and
budgets, runtime allocation/external-memory flags, dtype policy, shape rank,
operator allowlist, and both quantization minimums. It rejects unknown policy
keys instead of silently ignoring a typo.

Memory Planner v2 adds three optional, backward-compatible contract fields:

```yaml
memory:
  tensor_alignment_bytes: 16
ops:
  allow:
    ai.onnx:
      Conv:
        dtypes: [int8]
        workspace_bytes: 32768
        first_output_inplace_input_index: 0
```

- `memory.tensor_alignment_bytes` is the global arena alignment. It must be a
  non-zero power of two and defaults to `1` for an older profile.
- `workspace_bytes` is temporary memory simultaneously needed by one operator
  event. It defaults to `0` and cannot exceed known `memory.ram_bytes`.
- `first_output_inplace_input_index` authorizes one candidate input for the
  first output. It does not force reuse: the planner still requires last use,
  exclusive slot ownership, sufficient capacity, and a non-graph-boundary
  source tensor.

The three verifier seed profiles intentionally omit these fields. Their
effective values remain alignment `1`, workspace `0`, and no in-place reuse
because no target-specific runtime evidence has been reviewed yet. The virtual
NPU profile declares simulated alignment and cost inputs only for optimization
planning. Add deployable values only from a concrete runtime/kernel contract;
guessed values would make a RAM budget or latency estimate look more certain
than it is.

### Accelerator contract strictness

The entire `accelerator` section is optional, but it is all-or-nothing. Once
present, it must explicitly declare non-empty `id` and `confidence` values plus
`scratchpad_bytes`, `tensor_alignment_bytes`, `dma_burst_bytes`, `dma_setup_ns`,
`dma_read_bytes_per_second`, `dma_write_bytes_per_second`, and `spill_allowed`.
There are no inferred accelerator defaults. Unknown accelerator fields are
rejected. `scratchpad_bytes` and both DMA bandwidths must be greater than zero;
tensor alignment and DMA burst size must be non-zero powers of two.

An `accelerator` section alone does not make an operator eligible for NPU
assignment. A direct NPU candidate also needs an `ops.allow` rule whose
`npu_cost` is complete and whose dtype, port, rank, and attribute contracts all
match the normalized node. Conversely, any `npu_cost` or `recipes` declaration
requires an accelerator section. If a required tensor dtype or shape is missing,
the optimizer does not assume compatibility. `cpu_cost` is independent: when it
is absent or cannot be evaluated, EdgeFit records a CPU-baseline blocker rather
than inventing latency.

The cost contract is intentionally narrow. `kind` must be `fixed`, `element`,
`bytes`, or `mac`; non-`fixed` costs require a positive
`throughput_per_second`. These values are planning inputs, not measurements,
unless the profile source and confidence metadata explicitly establish measured
hardware evidence.

### NPU scratchpad and workspace accounting

`accelerator.scratchpad_bytes` is the capacity of the simulated NPU-local
scratchpad. It is distinct from `memory.ram_bytes` and
`memory.peak_activation_budget_bytes`: the latter govern the model-wide CPU-side
memory contract, while the accelerator value constrains an individual NPU
segment's resident tensors and temporary kernel workspace.

For NPU execution, tensor allocations and `workspace_bytes` are rounded up to
`accelerator.tensor_alignment_bytes`. Workspace is allocated for the duration of
one node, contributes to `peak_scratchpad_bytes`, and is released after that
node. It therefore competes with all tensors that must remain resident at the
same point. Transfers are separately rounded to `accelerator.dma_burst_bytes`.
`spill_allowed: true` permits eligible resident tensors to be moved out and
later reloaded, with DMA-rounded transfer latency, transfer bytes, and spill
bytes included in the plan. Protected current inputs cannot be spilled. With
`spill_allowed: false`, or when no eligible spill victim exists, insufficient
capacity produces a `scratchpad_unavailable` blocker; EdgeFit does not
overcommit the scratchpad.

The profile parser bounds each operator's `workspace_bytes` against known
`memory.ram_bytes`. The tighter NPU feasibility test happens during optimization
against `accelerator.scratchpad_bytes`, after accelerator alignment and
simultaneous tensor residency are included. Thus a workspace value can be valid
profile syntax yet still make a particular NPU assignment infeasible.

For a direct NPU kernel, the node uses that operator rule's `workspace_bytes`.
For a trusted replacement recipe, the node-level workspace requirement is the
**maximum** `workspace_bytes` among its replacement operators, not their sum.
The recipe models a sequential replacement expansion, so this maximum is the
largest temporary requirement of any one replacement step; resident tensors
still share the scratchpad with it.

### Operator port dtype and rank rules

Operator rules may also narrow dtype by zero-based port, constrain tensor rank,
and constrain captured ONNX attributes:

```yaml
ops:
  allow:
    ai.onnx:
      Softmax:
        dtypes: [float32]
        max_rank: 4
        input_dtypes:
          0: [float32]
        output_dtypes:
          0: [float32]
        attributes:
          axis: [int:-1, int:1]
```

`input_dtypes` and `output_dtypes` keys are canonical zero-based integers:
`0`, `1`, and so on. Signed values, leading-zero spellings such as `00`,
duplicate ports, and empty dtype lists are rejected while parsing or validating
the profile. Dtype names are normalized before comparison.

A listed port replaces the aggregate `dtypes` rule for that port only. Every
other non-empty input and output remains subject to `dtypes`; declaring one port
does not exempt the rest of the node. A listed port that is absent or an empty
optional input, a missing tensor record, an unknown dtype, or a dtype outside its
list fails closed with `EF0207`. Aggregate mismatches on ports without an
override are reported as `EF0202`.

`shape.max_rank` is the required target-wide ceiling. An operator-level
`max_rank` is optional and narrows the contract for every non-empty input and
output of that operator. It never widens the global ceiling: the effective NPU
eligibility limit is the smaller of the global and operator values. Known rank
violations are reported as `EF0102` for the global rule and `EF0203` for the
operator rule. When the optimizer cannot establish a required tensor rank, it
does not treat the node as accelerator-compatible.

Attribute values use typed canonical forms: `int:`, `float:`, `string:`, and
their `ints:`, `floats:`, or `strings:` array forms (array elements are joined
with `;`). Missing, mismatched, or unmodeled evidence for an explicitly
constrained attribute fails with `EF0206`. Unconstrained attributes remain
recorded without changing legacy profile behavior. Older profiles omit the
port, operator-rank, and attribute maps and retain aggregate dtype plus global
rank behavior. The checked-in seed profiles intentionally add no such claims
without reviewed runtime-kernel evidence.

### Replacement recipe maximums

A recipe is not a general graph-rewrite language. It is a trusted,
profile-local planning claim that one source operator can be represented by a
listed sequence of already-declared NPU operators:

```yaml
recipes:
  ai.onnx:
    HardSwish:
      id: recipe.hardswish.v1
      trusted: true
      source: EdgeFit simulated semantic contract
      version: 1
      replacement_ops: [HardSigmoid, Mul]
```

Each recipe must have `trusted: true`, non-empty `id`, `source`, and `version`,
and at least one `replacement_ops` entry. Every replacement name is resolved in
the same operator domain and must have a valid `npu_cost`; otherwise profile
validation fails. At optimization time, every replacement rule must also satisfy
the original node's known dtype, port, rank, and attribute evidence. This is a
conservative compatibility check, not proof that the replacement is
semantically equivalent; that assurance belongs in the recipe's reviewed
source evidence.

Recipe launch and compute estimates are summed across all replacement entries,
including repeated entries. There is no hidden expansion, recursion, or
unbounded application: one source node selects at most its one declared recipe,
and the finite `replacement_ops` list is the complete expansion considered for
that node. As described above, recipe workspace uses the maximum replacement
workspace rather than the sum. This finite list is the recipe maximum: a profile
must enumerate every replacement step it wants EdgeFit to cost, and EdgeFit does
not append or recursively expand additional steps. Direct-kernel and recipe
candidates compete on their declared estimated cost; the recipe does not grant
support to other operators or alter the target-wide shape and dtype contracts.

Older profiles may omit both `accelerator` and `recipes`. They remain verifier
contracts but cannot be passed to `edgefit optimize`, which requires an explicit
accelerator contract.

For ONNX version compatibility, a profile declares per-domain caps:

```yaml
opsets:
  ai.onnx: 13
  com.microsoft: 1
```

An adapter-generated model that exceeds a declared cap fails with `EF0204`.
When no cap is declared for a used domain, `EF0205` fails closed so the profile
cannot silently claim runtime-version compatibility. The verifier profiles use
conservative caps matching the current reviewed corpus boundary (`ai.onnx: 13`,
plus `com.microsoft: 1` for the ORT seed). The virtual NPU seed uses
`ai.onnx: 18` as a simulated optimizer-fixture boundary. None of these values
claims a runtime's full capability; raise a deployable profile cap only after
reviewing the actual deployment runtime.

## Local Validation

```powershell
cargo run -p edgefit-cli -- target validate targets/esp32s3.yaml
cargo run -p edgefit-cli -- target validate targets/ort-mobile-cpu.yaml
cargo run -p edgefit-cli -- target validate targets/tflm-micro.yaml
cargo run -p edgefit-cli -- target validate targets/virtual-npu.yaml
cargo run -p edgefit-cli -- check examples/models/bad_detector.edgefit.json --target targets/ort-mobile-cpu.yaml
cargo run -p edgefit-cli -- optimize examples/models/virtual_npu_tiny.edgefit.json --target targets/virtual-npu.yaml --format json
```

The detector fixture fails under strict MCU-style seed profiles and passes under
the ORT mobile-like seed profile. That difference is expected because the ORT
profile allows fp32 tensors, `Resize`, and a larger model and activation budget.

## Calibration Matrix

Run the matrix after the Rust CLI is built and the real-world corpus cache is
available:

```powershell
cargo build -p edgefit-cli
python tools/onnx-normalize/profile_matrix.py --edgefit tmp/cargo-target/debug/edgefit.exe --cache tmp/real_world_corpus --out tmp/real_world_corpus/profile-matrix.json --markdown-out tmp/real_world_corpus/profile-matrix.md
```

Last recorded matrix result (2026-07-09, before the 2026-07-10 profile metadata
and opset-cap change):

| Target | Matrix result | Notes |
| --- | --- | --- |
| `esp32s3_custom_v1` | `0/20` pass | Strict MCU constraints reject the verified corpus as expected. |
| `ort_mobile_cpu_seed_v1` | `20/20` pass | All ORT target entries pass with 4 warning-only dynamic-shape diagnostics. |
| `tflm_micro_seed_v1` | `0/20` pass | Generic TFLM-like MCU constraints reject the verified corpus as expected. |

This historical matrix must be refreshed before it is used as evidence for the
current profile fingerprints. Use the refreshed result with the reference check
and operator-support audit below when deciding whether an operator rule has
enough evidence for a confidence uplift.

## Operator Support Audit

Run the corpus expansion gate before interpreting the profile matrix:

```powershell
python tools/onnx-normalize/corpus_expansion_gate.py --out tmp/real_world_corpus/corpus-expansion-gate.json --markdown-out tmp/real_world_corpus/corpus-expansion-gate.md
```

Then run the operator-support audit after the profile matrix is available:

```powershell
python tools/onnx-normalize/operator_support_audit.py --matrix tmp/real_world_corpus/profile-matrix.json --labels tools/onnx-normalize/operator_support_labels.json --out tmp/real_world_corpus/operator-support-audit.json --markdown-out tmp/real_world_corpus/operator-support-audit.md
```

Current audit result:

| Field | Value |
| --- | --- |
| `status` | `pass` |
| `sample_model_count` | `20` |
| `sample_goal` | `20` |
| `profile_count` | `3` |
| `observed_operator_count` | `42` |
| `evidence_operator_count` | `42` |
| `corpus_expansion_gate.status` | `ready_for_profile_matrix` |
| `corpus_expansion_gate.label_status` | `complete` |
| `corpus_expansion_gate.models_needed` | `0` |
| `precision_recall_review.status` | `pass` |
| `precision_recall_review.labeled_cell_count` | `60` |
| `precision_recall_review.status_match_count` | `60 / 60` |
| `precision_recall_review.unsupported_op_precision` | `1.0` |
| `precision_recall_review.unsupported_op_recall` | `1.0` |

## Reference Check

Run generated operator fixture verification before the reference check:

```powershell
python tools/onnx-normalize/operator_fixture_corpus.py --cache tmp/operator_fixtures --out tmp/operator_fixtures/operator-fixtures-result.json
```

Verify pinned ONNX Runtime source evidence after sparse-checking out the official ONNX Runtime repository under `tmp/ort-src`:

```powershell
python tools/onnx-normalize/ort_runtime_evidence.py --manifest tools/onnx-normalize/ort_runtime_evidence.json --source-root tmp/ort-src --out tmp/ort-runtime-evidence/ort-runtime-evidence-result.json
```

The runtime evidence manifest covers `com.microsoft::QLinearAdd`,
`com.microsoft::QLinearConcat`, and `com.microsoft::QLinearGlobalAveragePool` at
ONNX Runtime commit `c57e0e50ad068905a4140d361b0b3fd8c251e540`. It verifies the
`com.microsoft` domain definition, CPU provider kernel registrations, CPU
implementations, and DirectML supported registrations from pinned source files.

Run the reference check for the ORT mobile-like seed profile:

```powershell
python tools/onnx-normalize/profile_reference_check.py --profile targets/ort-mobile-cpu.yaml --manifest tools/onnx-normalize/real_world_corpus.json --fixture-manifest tools/onnx-normalize/operator_fixtures.json --runtime-evidence tools/onnx-normalize/ort_runtime_evidence.json --out tmp/real_world_corpus/profile-reference.json --markdown-out tmp/real_world_corpus/profile-reference.md
```

Current reference result:

| Status | Count | Meaning |
| --- | ---: | --- |
| `official_and_corpus` | 39 | The operator appears in the official ONNX operator schemas and in real-world or generated fixture evidence. |
| `runtime_and_corpus` | 3 | The operator appears in pinned ONNX Runtime source evidence and verified real-world corpus evidence. |
| `official_only` | 0 | No operator is only schema-backed. |
| `runtime_only` | 0 | No operator is only runtime-source-backed. |
| `corpus_only` | 0 | No operator is only corpus-backed. |
| `missing_reference` | 0 | No allowlist operator lacks schema, fixture, corpus, or runtime-source evidence. |

Current reference version result:

| Reference | Installed | Pinned | Status | Official operators |
| --- | --- | --- | --- | ---: |
| `onnx` Python package | `1.22.0` | `1.22.0` | `match` | 202 |

## Confidence Gate

Run runtime inference verification before the confidence gate:

```powershell
python tools/onnx-normalize/runtime_smoke.py --profile targets/ort-mobile-cpu.yaml --corpus-result tmp/real_world_corpus/corpus-result.json --provider CPUExecutionProvider --out tmp/real_world_corpus/runtime-verification.json
```

The current runtime verification result uses ONNX Runtime `1.22.0` with `CPUExecutionProvider`,
runs all 20 verified corpus models, and records `status = pass` with no output
dtype or fixed-shape mismatches.

Run the ORT runtime-boundary check after corpus and fixture evidence are
available:

```powershell
python tools/onnx-normalize/ort_runtime_boundary.py --profile targets/ort-mobile-cpu.yaml --corpus-result tmp/real_world_corpus/corpus-result.json --fixture-result tmp/operator_fixtures/operator-fixtures-result.json --ort-source tmp/ort-src --out tmp/ort-runtime-boundary/ort-runtime-boundary.json --markdown-out tmp/ort-runtime-boundary/ort-runtime-boundary.md --config-out tmp/ort-runtime-boundary/edgefit-ort-required-ops.config
```

The current runtime-boundary result has `status = pass`, 23 evidence models, 42
profile operators, 42 required operators, `profile_coverage_status = pass`, and
`generated_config_roundtrip_status = pass`.

Run the public PR trial gate before the confidence gate:

```powershell
cp examples/public_pr_trials.example.json docs/public_pr_trials.json
python tools/public_pr_trial_gate.py --manifest docs/public_pr_trials.json --out tmp/public_pr_trials/public-pr-trial-gate.json --markdown-out tmp/public_pr_trials/public-pr-trial-gate.md
```

Run the confidence gate after the reference check, matrix, corpus gate,
operator-support audit, runtime evidence, runtime inference verification, ORT
runtime-boundary check, warning-only diagnostic policy check, and public PR trial
gate:

```powershell
python tools/onnx-normalize/profile_confidence_gate.py --profile targets/ort-mobile-cpu.yaml --reference tmp/real_world_corpus/profile-reference.json --matrix tmp/real_world_corpus/profile-matrix.json --corpus-gate tmp/real_world_corpus/corpus-expansion-gate.json --operator-audit tmp/real_world_corpus/operator-support-audit.json --runtime-evidence-result tmp/ort-runtime-evidence/ort-runtime-evidence-result.json --runtime-smoke tmp/real_world_corpus/runtime-verification.json --runtime-boundary tmp/ort-runtime-boundary/ort-runtime-boundary.json --diagnostic-policy docs/DIAGNOSTIC_POLICY.md --public-pr-trials tmp/public_pr_trials/public-pr-trial-gate.json --out tmp/real_world_corpus/profile-confidence-gate.json --markdown-out tmp/real_world_corpus/profile-confidence-gate.md
```

Current gate result:

| Field | Value |
| --- | --- |
| `decision` | `hold_seed` |
| `confidence_uplift_ready` | `false` |
| `corpus_expansion_gate_verified` | `pass` |
| `operator_support_audit_verified` | `pass` |
| `warning_diagnostic_policy_documented` | `pass` |
| `runtime_boundary_verified` | `pass` |
| `public_pr_trials_verified` | `fail`, 0/3 verified public repository trials |
| `matrix_target_coverage` | `pass` |
| `matrix_target_passes` | `pass` |
| `runtime_evidence_verified` | `pass` |
| Runtime inference check (`runtime_smoke_verified`) | `pass` |

## Confidence Label Review

The gate decision remains `hold_seed` until the local public PR manifest records
three verified repositories. Public policy is documented in
`docs/DIAGNOSTIC_POLICY.md`; detailed confidence and runtime review artifacts
are maintained locally and are not part of the public repository.
