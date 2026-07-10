# Architecture

EdgeFit follows the verifier boundary described in the feasibility report:
Rust core + replaceable ONNX adapter + independent report layer.

## Layers

- `edgefit-ir`: normalized model structures and a small JSON parser for
  `edgefit.normalized_model.v1`.
- `edgefit-target`: target profile parsing and validation.
- `edgefit-analyze`: factual analysis such as tensor bytes, indexed lifetimes,
  deterministic activation-arena placement, dynamic tensors, symbol-bound
  activation estimates, unsupported ops, unsupported and unknown dtypes,
  initializer dtype distribution, and target-relative QDQ/QOperator coverage.
- `edgefit-policy`: stable diagnostic IDs, suppression, and pass/fail policy.
- `edgefit-report`: text, JSON, Markdown, and SARIF renderers with diagnostic locations.
- `edgefit-diff`: snapshot comparison for PR regression checks, including
  active-versus-suppressed diagnostic state changes.
- `edgefit-cli`: command-line entrypoint.
- `tools/onnx-normalize`: Python adapter that delegates ONNX checking and shape
  inference to the official `onnx` package. It includes external weight files
  in model-budget metadata and rejects nested subgraphs, local functions, and
  sparse initializers until those structures have dedicated analysis support.

The direct ONNX path is the evidence-producing path: it records package bytes,
external-data count, opset imports, and normalized tensor facts in a temporary
IR. Adapter provenance is carried out of band from the CLI's original `.onnx`
branch; a JSON `normalization` object cannot grant that identity. A checked-in
`*.edgefit.json` remains supported for reviewed fixtures and offline workflows,
but is labelled as trusted metadata because the MVP cannot bind it
cryptographically to an original ONNX artifact.

## Confidence Boundary

Activation memory has two separate metrics. `estimated_peak_activation_bytes`
is the logical live-tensor peak, while `planned_activation_arena_bytes` is the
deterministic arena high-water mark used by budget policy. The arena planner is
a linear-scan best-fit allocator with offset and size indexes. It applies the
profile's tensor alignment, allocates per-node workspace in the same event,
accounts for fragmentation, and permits first-output in-place reuse only when
the profile explicitly declares it and the input is at its last use, owns its
slot exclusively, and has enough capacity.

Planner work is expected
`O(P + T + D log(S + 2) + B + N + E + O + A log A + U log T + R)` for `P`
profile rules, `T` tensors, `D` total shape dimensions, `S` symbol bounds, `B`
graph-boundary occurrences, `N` nodes, `E` input uses, `O` output occurrences,
`A` arena allocation/free events, `U` bounded or unresolved facts, and `R` trace records. Working space is
`O(P + T + A + R + delta)`, where `delta` is the maximum node arity.
Trace follows execution order, and Top contributors retain only the best eight; live
bytes are updated incrementally instead of rescanning all live tensors at every node. Outputs are allocated before last-use inputs are released, graph outputs
remain pinned through graph end, and zero-consumer intermediate outputs are
released after their producer event.

Fully shaped graphs produce high confidence; profile symbol bounds can produce
a conservative bounded estimate. If a configured activation budget still has
unresolved tensor sizes, the policy fails rather than treating an incomplete
plan as proof of compliance. An arena/alignment sum that cannot be represented
in `u64` sets `activation_planning_overflowed=true` and drives the same
fail-closed budget path without changing the unresolved-tensor count. For RAM-resident weights, initializer bytes are
added to the planned arena high-water mark; external ONNX data and dynamic
allocation flags are also enforced from the profile contract.

JSON reports expose the allocation intervals, peak event, top contributors,
workspace, fragmentation, and verified reuse. Text and Markdown keep only the
decision-oriented summary. Snapshots deliberately omit the full allocation
trace so a small placement shift cannot create an unreadable regression diff.

For adapter-generated ONNX facts, each used operator domain must have an
explicit target-profile opset cap. Missing caps fail closed instead of silently
claiming runtime compatibility.

When ONNX checking succeeds but shape inference fails, the adapter emits the
checker-approved graph with `normalization.shape_inference.status = failed`.
The policy surfaces that separately from incomplete activation metadata.

## Product Boundary

The MVP is a deployment-budget verifier and CI gate. It does not implement a
runtime, compiler, latency predictor, web dashboard, or automatic fixer.
