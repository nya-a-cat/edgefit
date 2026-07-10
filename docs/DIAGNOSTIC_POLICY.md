# Diagnostic Severity and Gate Policy

This policy defines how EdgeFit diagnostics affect local checks, GitHub Action
runs, JSON reports, Markdown summaries, and SARIF output.

## Gate Outcome

| Severity | Report status | CI effect | Required user action |
| --- | --- | --- | --- |
| `error` | `fail` | The EdgeFit command exits with failure. | Fix the model or target profile, or use an explicit suppression for an accepted risk. |
| `warning` | `pass` | The EdgeFit command exits successfully. | Review the diagnostic and keep the evidence visible in reports. |

The Rust policy layer computes status from unsuppressed diagnostics. Any
unsuppressed `error` diagnostic sets the report status to `fail`; warning-only
results keep the status at `pass`.

## Diagnostic Severity Rationale

`EF0001` marks every externally supplied `*.edgefit.json` as trusted metadata.
The file cannot remove this warning by adding a `normalization` object; only the
CLI's direct `.onnx` path supplies adapter provenance out of band after the
adapter succeeds.

`EF0103` reports a failed ONNX shape-inference attempt. It is an `error` for
profiles that require static/resolved shapes and a `warning` otherwise; the
checker-approved original graph remains available for analysis.

`EF0104` is warning-only when the target profile allows unknown dimensions and
activation memory is an estimate with reduced confidence. This is the expected
behavior for `targets/ort-mobile-cpu.yaml`, whose runtime section sets
`static_shapes_required: false` and whose shape policy allows unknown
dimensions.

`EF0104` becomes an `error` when a target profile requires static shapes. This
keeps strict MCU profiles conservative while allowing ORT-style profiles to
surface dynamic-shape uncertainty during review.

`EF0502` is always an `error` when a profile declares an activation-memory
budget but EdgeFit cannot calculate a concrete or symbol-bound size for every
relevant activation. A budget cannot be treated as satisfied without a safe
upper bound; an alignment, workspace, or cumulative arena value that cannot be
represented in `u64` sets `activation_planning_overflowed=true` and follows the
same fail-closed path without pretending that another tensor size is missing. `EF0501` compares the budget with the deterministic planned arena
high-water mark, not only the logical sum of live tensors; alignment,
fragmentation, declared workspace, and verified in-place reuse therefore affect
the same decision. `EF0503` likewise adds RAM-resident initializer bytes to that
planned arena value. `EF0302` is also an `error`: missing dtype metadata cannot
prove target compatibility.

`EF0205` is an error because an adapter-generated model's used domain cannot be
proven runtime-compatible without a target-profile opset cap. `EF0204` is an
error for a missing domain import or a version above a declared cap. `EF0105`,
`EF0402`, and `EF0503` are errors because a target that forbids dynamic
allocation/external data or requires RAM-resident weights cannot be proven
compatible without those constraints.

`EF0602` is warning-only because target-eligible QDQ/QOperator coverage is
static graph evidence, not proof that a concrete runtime will or will not
execute the model with quantized kernels. `EF0603` is also warning-only: floating-point
initializer bytes may include legal scale constants, while `EF0601` remains the
profile-threshold error for insufficient quantized initializer coverage.

## Strict int8 Diagnostics

`EF0604` is an error only when the profile explicitly sets
`quantization.require_int8: true` and a graph input or output has a non-int8
dtype. `EF0605` is an error when such a profile uses an operator whose target
rule exposes no int8 or uint8 dtype path. These rules describe the checked-in
target contract; they do not infer undocumented runtime kernel support.

## Reporting Rules

- Text, Markdown, and JSON reports keep warning diagnostics in the normal
  `diagnostics` section.
- SARIF maps `warning` severity to SARIF level `warning`, so GitHub code scanning
  and summaries can show the evidence without blocking the check.
- Suppressed diagnostics are removed from active `diagnostics` and preserved in
  `suppressed_diagnostics` for auditability.
- Suppressions are for accepted risks with stable IDs, such as
  `--suppress EF0104`; they remain visible in text, Markdown, and JSON reports.
- Snapshots retain `suppressed_diagnostics`. Diff compares active and suppressed
  states separately: a new suppressed error or an active error newly moved into
  suppression is visible and blocks the diff, while an unchanged accepted risk
  does not create a new regression.

## ORT Profile Confidence Boundary

For `ort_mobile_cpu_seed_v1`, warning-only `EF0104` diagnostics are acceptable
profile-confidence evidence when these conditions hold:

- The model has zero unsuppressed `error` diagnostics.
- The profile matrix records the warning count for the affected corpus cells.
- Runtime smoke passes for the verified corpus.
- The profile remains labeled `seed` until the public PR trial gate reaches `ready_for_confidence_review`.

This policy lets EdgeFit expose uncertain activation-memory estimates during CI
while keeping the ORT mobile-like seed profile usable for early adoption trials.
