# MVP Scope

The MVP implements the first loop from the feasibility report:

1. Validate target profiles with source, confidence, and last-verified metadata, including strict MCU, TFLM-like, and ORT mobile-like seed templates, a 20-model corpus-by-profile calibration matrix, generated operator fixture evidence, pinned ONNX Runtime source evidence for Microsoft-domain quantized operators, a domain-aware ORT seed profile reference check with pinned ONNX `1.22.0` operator-schema evidence, ONNX Runtime CPU smoke inference, a profile confidence gate that includes the public PR trial gate, a documented ORT profile confidence decision, local demo PR trials for the GitHub Action adoption path, a public PR trial gate with current 0/3 status, ORT reduced-operator boundary evidence, a passing three-profile operator-support audit with current manual-label precision/recall checks plus a corpus expansion gate for the 20-model target, and a documented warning-only diagnostic policy.
2. Load a normalized model produced by the Python ONNX adapter, including direct `.onnx` input through the CLI adapter path, toy ONNX coverage for Add, MatMul, Conv, Reshape, Transpose, and Softmax, quantized QLinearMatMul fixture coverage, and manifest-based 20-model real-world ONNX corpus entries.
3. Check memory, dtype, shape, operator, and quantization constraints.
4. Emit stable diagnostics.
5. Produce text, JSON, Markdown, SARIF, and GitHub job-summary reports, including visible suppressed diagnostics and stable diagnostic locations.
6. Store and compare snapshots.

## Commands

- `edgefit target validate <profile>`
- `edgefit check <model.edgefit.json> --target <profile> [--suppress EF0104[,EF0203]]`
- `edgefit snapshot <model.edgefit.json> --target <profile> --out <path>`
- `edgefit diff --old <snapshot> --new <snapshot>`

## Initial Diagnostics

- `EF0001`: a pre-normalized JSON model is treated as reviewed/trusted metadata rather than adapter-generated evidence.
- `EF0101`: static shapes are required by the target profile.
- `EF0102`: tensor rank exceeds `shape.max_rank`.
- `EF0103`: ONNX shape inference failed and analysis fell back to checker-approved graph metadata.
- `EF0104`: activation-memory estimate has reduced confidence.
- `EF0105`: target disallows dynamic allocation for dynamic tensors.
- `EF0201`: model uses an operator outside the target profile.
- `EF0202`: operator tensor dtype is outside the op rule.
- `EF0203`: operator tensor rank exceeds the op rule.
- `EF0204`: a used domain has no imported opset or exceeds a declared target maximum.
- `EF0205`: an adapter-generated model uses a domain with no target opset maximum; this is fail-closed.
- `EF0301`: tensor dtype is outside the target profile.
- `EF0302`: tensor dtype metadata is missing.
- `EF0401`: model file exceeds the target flash budget.
- `EF0402`: external ONNX data files are outside the target memory contract.
- `EF0501`: the deterministic planned activation arena exceeds the target RAM budget.
- `EF0502`: an activation-memory budget exists but a safe upper bound cannot be proven.
- `EF0503`: RAM-resident weights plus activations cannot be proven within RAM.
- `EF0601`: quantized initializer fraction is below the target requirement.
- `EF0602`: target-eligible quantized operator coverage is below the target requirement while non-int8 state remains.
- `EF0603`: significant floating-point initializer state remains visible in the dtype distribution.
- `EF0604`: graph input or output dtype violates `quantization.require_int8`.
- `EF0605`: a used target operator rule exposes no int8 or uint8 kernel path.

## Source-Only Status

The current implementation also fails closed for missing normalized graph
arrays, missing dtype metadata, unknown tensor sizes under an activation budget,
unsupported nested ONNX subgraphs/local functions/sparse initializers, malformed
snapshots, cross-profile snapshot comparisons, and new suppressed-error
regressions. Symbol bounds participate in activation estimation; external ONNX
weight files participate in model-file budget and model hash calculation.
Adapter-generated ONNX analysis records domain opset imports, and every used
domain requires `opsets.<domain>: <max>`. A direct pre-normalized JSON is always
marked as trusted input because its own metadata cannot establish adapter
provenance or bind it cryptographically to an original ONNX artifact.

The activation-memory path now keeps logical liveness and physical arena
placement separate. Budget policy uses a deterministic indexed best-fit plan
that includes profile-declared alignment, operator workspace, fragmentation,
and only explicitly safe first-output in-place reuse. Reports expose the peak
event, contributors, reuse summary, explicit planner-overflow state, and JSON allocation intervals; snapshots
exclude the full trace and retain only stable summary metrics.

These source changes have not been compiled, formatted, unit-tested, integration-tested,
or run on real models in this change because the user explicitly requested a
source-only pass. They must not be presented as runtime-proven until a later
verification cycle is authorized.

## Next Milestones

- Fill and verify three public repository PR trials before raising the ORT profile confidence label beyond `seed`.


