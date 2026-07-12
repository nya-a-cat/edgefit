# Hardware Optimizer

`edgefit optimize` generates a deterministic, auditable execution plan for a
normalized ONNX graph and an accelerator-aware target profile:

```bash
edgefit optimize examples/models/virtual_npu_tiny.edgefit.json \
  --target targets/virtual-npu.yaml \
  --format json \
  --out edgefit-plan.json
```

The optimizer does not rewrite ONNX, invoke a vendor compiler, or execute the
model. It compares profile-declared CPU and NPU costs, chooses eligible kernels
or trusted replacement recipes, forms contiguous NPU segments, and simulates
scratchpad residency and DMA transfers. The result is an
`edgefit.optimization_plan.v1` document.

Checked-in accelerator costs are simulated seed assumptions. They are not
physical-device measurements, benchmark results, compiler output, runtime
compatibility evidence, or deployment guarantees.

## Planning contract

A cost entry alone does not make a node executable. Before EdgeFit uses a CPU
cost, NPU cost, or replacement operation, the node must satisfy the applicable
operator contract:

- every present input and output tensor must have a known dtype accepted by the
  aggregate or zero-based port-specific dtype rule;
- every explicitly constrained input or output port must exist;
- tensor rank must be known and must satisfy the stricter of the operator and
  target-wide rank limits;
- every explicitly constrained attribute must be present with an allowed typed
  value;
- dimensions needed by an `element`, `bytes`, or `mac` cost must be known or
  bounded by the target profile;
- convolution MAC estimation requires a positive group that does not exceed the
  input channel count and divides it exactly.

Missing or incompatible evidence fails closed. EdgeFit does not infer a dtype,
shape, attribute, throughput, or tensor size merely to keep a candidate
available.

A replacement recipe is eligible only when it is trusted, has non-empty
identity, source, and version metadata, lists at least one replacement
operation, and every listed operation has a compatible, evaluable NPU contract.
The listed operations are the complete finite expansion for that source node;
recipes are not recursively expanded. When both a direct NPU kernel and a
recipe are eligible, EdgeFit chooses the lower declared launch-plus-compute
cost. A tie keeps the direct kernel.

## Deterministic, conservative selection

The current planner is a deterministic conservative heuristic. It does **not**
claim to find a global optimum.

Nodes are considered in graph order. For each node:

1. EdgeFit derives an eligible CPU candidate, if one exists.
2. It derives the best eligible direct-NPU or recipe candidate, if one exists.
3. If only the NPU candidate exists, the node is assigned to the NPU.
4. If both exist, the NPU must be strictly cheaper than the CPU after adding the
   boundary-transfer cost that can be determined from the current graph prefix
   and graph outputs. Unknown boundary size keeps the node on the CPU.
5. Equal estimated cost keeps the node on the CPU.
6. If neither candidate is available, the assignment is `unsupported` and the
   completed plan contains a blocker.

After this ordered placement pass, EdgeFit builds and simulates both the mixed
CPU/NPU proposal and an all-CPU fallback plan. It chooses the proposal only
when it has fewer blockers, or when blocker counts are equal and it has a known,
strictly lower total latency. Equal latency, two unknown latencies, or any other
tie keeps the CPU fallback.

This fallback prevents a locally attractive NPU assignment from replacing a
cleaner or no-slower CPU plan after scratchpad and transfer effects are known.
It does not manufacture CPU support: a node without a compatible, evaluable CPU
kernel remains unsupported in the CPU baseline and fallback plan.

The planner does not enumerate every partition, backtrack over earlier node
choices, run dynamic programming over segment boundaries, or jointly search all
placement and spill schedules. A different legal assignment can therefore have
lower modeled latency than the emitted plan. The guarantee is reproducible,
conservative selection under the declared contract, not global optimality.

## CPU baseline and latency comparison

The `baseline` object represents the graph evaluated entirely against declared
CPU kernel contracts:

- `blockers` counts nodes without an eligible CPU candidate;
- `latency_ns` is the checked sum of CPU launch and compute latency when every
  node is evaluable, otherwise it is `null`.

The selected plan's `proposed` latency is:

```text
sum(assignment launch_ns)
+ sum(assignment compute_ns)
+ sum(transfer event latency_ns)
```

A missing assignment latency makes proposed latency unknown and adds the
`latency_unknown` blocker. Profile latency is a comparison inside the declared
cost model; it is not an inference-time prediction for real hardware.

## Scratchpad, workspace, and transfers

Each maximal contiguous run of NPU assignments becomes one NPU segment. The
scratchpad simulation starts with an empty resident set for each segment and
tracks aligned tensor allocations, temporary workspace, dead-tensor release,
spills, reloads, and final stores.

### Tensor residency

- Tensor allocations are rounded up to
  `accelerator.tensor_alignment_bytes`.
- An input not already resident produces a `load`, or a `reload` if it was
  previously spilled in the same segment.
- Inputs required by the current node are protected from spilling.
- Outputs are allocated while the node's workspace is still live and become
  protected for the remainder of that node.
- Tensors with no later use in the segment, no graph-output role, and no
  consumer beyond the segment are released after the node.
- Remaining resident tensors are stored at the end of the segment.

Transfer sizes are separately rounded up to
`accelerator.dma_burst_bytes`. Every `load`, `store`, `spill`, and `reload`
records its aligned byte count, node index, and profile-derived DMA latency.
Transfer and spill totals are derived from these events.

### Operator workspace

`workspace_bytes` is accelerator scratchpad required temporarily by one NPU
node. CPU assignments do not consume accelerator scratchpad.

For a direct NPU kernel, EdgeFit uses that operator rule's `workspace_bytes`.
For a replacement recipe, it uses the maximum workspace declared by any listed
replacement operation, not their sum. This is the profile's modeled peak for
the finite replacement sequence.

Workspace is aligned to `accelerator.tensor_alignment_bytes`, allocated while
the current inputs are resident, held through output allocation, included in
`peak_scratchpad_bytes`, and released after the node. It therefore competes with
both live inputs and outputs; it is not an off-plan allowance.

### Spill policy

If an allocation does not fit and `spill_allowed: true`, EdgeFit deterministically
selects eligible resident tensors by latest next use, then larger allocation,
then stable tensor-name ordering. It first identifies a complete victim set. If
no complete set can make enough room, the attempted allocation fails without
emitting a partial sequence of spill events.

When spilling is disabled, or every possible victim is protected, EdgeFit
records a `scratchpad_unavailable` blocker. A disabled-spill blocker also states
that spilling was disabled. The planner never resolves pressure by exceeding
`accelerator.scratchpad_bytes`, and the final invariant check rejects a plan
whose reported peak is above that capacity.

## Checked arithmetic

All capacity, size, MAC, and latency accounting that can affect plan status or
metrics uses checked arithmetic:

- tensor element and byte products use checked multiplication;
- matrix and convolution MAC counts use checked multiplication;
- launch, compute, boundary, transfer, spill, blocker, and scratchpad totals use
  checked addition;
- scratchpad releases use checked subtraction;
- alignment uses checked addition and rejects zero alignment;
- DMA latency uses checked ceiling multiplication and division with a wider
  intermediate, rejects zero throughput, and checks conversion back to `u64`.

An overflow, underflow, invalid divisor, or failed integer conversion is a
planner execution error. EdgeFit does not wrap, saturate, clamp, or substitute a
smaller number and then emit a passing or policy-failing plan. Through the CLI,
such an error belongs to exit-code class `2`; when `--out` applies, the requested
artifact is an `edgefit.execution_error.v1` document rather than an optimization
plan.

The FNV-1a state used for `plan_hash` intentionally uses wrapping hash
multiplication. That operation is only part of the deterministic fingerprint;
it is not used for memory, capacity, cost, or latency accounting.

## Plan invariants

Before returning a plan, the optimizer validates the completed result against
the normalized model and target profile. The validation requires, among other
properties:

- model and target identities in the plan match the inputs;
- there is exactly one contiguous, correctly indexed assignment per model node;
- every CPU assignment matches its declared CPU kernel contract and evaluated
  latency;
- every NPU assignment matches either one direct NPU kernel or one trusted
  recipe contract and its evaluated latency;
- segments are exactly the maximal contiguous NPU assignment runs;
- transfer events reference valid NPU nodes and known tensors, use allowed event
  kinds, and are DMA-burst aligned;
- events do not move backward across NPU segments;
- every reload has a preceding unmatched spill in the same segment;
- no spill is present when spilling is disabled;
- event-derived transfer latency, transfer bytes, and spill bytes equal the
  reported totals;
- assignment-derived launch and compute totals equal the reported totals;
- proposed latency equals launch plus compute plus transfer latency when known;
- the CPU baseline recomputes to the reported blocker count and latency;
- `status`, blocker count, blocker list, and `latency_unknown` state agree;
- `plan_hash` recomputes from the recorded model hash, target fingerprint,
  assignments, segments, and events.

An invariant failure is an execution error, not a normal fail plan.

## Output and exit codes

The JSON form uses schema `edgefit.optimization_plan.v1` and records:

- model hash, target identity and fingerprint, accelerator identity, and
  confidence;
- CPU baseline blockers and latency;
- selected-plan blockers, launch, compute, transfer, and total latency;
- transfer bytes, spill bytes, and peak scratchpad bytes;
- one assignment per node, including device and kernel or recipe identity;
- exact NPU segment boundaries;
- ordered transfer events;
- unresolved blockers;
- deterministic `plan_hash`.

Exit codes distinguish a completed plan from an inability to produce one:

| Code | Meaning |
| ---: | --- |
| `0` | Planning completed with `status: "pass"` and no blockers. |
| `1` | Planning completed with `status: "fail"`; the canonical plan and blockers are valid evidence. |
| `2` | Input preparation, target loading, dependency execution, checked arithmetic, or invariant validation prevented a trustworthy plan. |

A code-`1` plan must be retained and parsed. It still contains complete
assignments, segments, events, totals, blockers, and a deterministic plan hash.
A code-`2` result must not be interpreted as either a passing plan or a
policy-failing plan.

For the same normalized model, target profile, and EdgeFit implementation, both
pass and fail plans are deterministic. `plan_hash` is a compact reproducibility
fingerprint, not a cryptographic integrity or authenticity proof.

## Full contract evidence

A successful process launch or parseable JSON file is not, by itself, full
contract evidence. For optimizer evidence to be considered complete, all of the
following boundaries must agree:

1. the CLI exit code matches the artifact schema and internal status;
2. the artifact passes strict structural and cross-field parsing;
3. assignments, segments, events, blockers, and latency totals are internally
   consistent;
4. the case-specific manifest expectations pass;
5. repeated runs have stable exit codes and identical full-artifact hashes when
   determinism is required;
6. the generated models, plans, raw stdout/stderr, summaries, and recorded file
   hashes remain traceable in the uploaded evidence.

`tools/competitive-benchmark/benchmark.py` independently parses serialized
plans. It checks required identity fields, status, assignment indexing and
device states, exact contiguous NPU segments, legal transfer-event structure,
spill/reload pairing, blocker/status agreement, event-derived totals, and
assignment-derived latency totals. The manifest can additionally require exact
or bounded assignment counts, segment counts, event kinds, spill and transfer
bytes, peak scratchpad, blockers, recipes, latency improvement, duration, and
peak RSS.

The evidence manifest is
`tools/competitive-benchmark/optimizer_evidence_manifest.json`. Its cases cover:

- deterministic 1K, 10K, and 100K-node contiguous NPU plans;
- direct-kernel contract acceptance;
- fail-closed behavior when the operator contract is unavailable;
- trusted replacement recipes at single-node and 1K-node scale;
- a deterministic CPU boundary producing multiple NPU segments;
- constrained-scratchpad spill and reload;
- fail-closed capacity pressure when spilling is disabled.

The required `optimizer-contract-gate` job in `.github/workflows/ci.yml` runs the
10K passing case and the no-spill failing case three times and uploads its
artifacts. The manually dispatched `.github/workflows/optimizer-evidence.yml`
runs the complete manifest five times and uploads generated models, canonical
plans, raw process output, JSON and Markdown summaries, and hash records.

Hosted duration and peak RSS are observations of the complete CLI process on a
GitHub-hosted runner. They do not measure device inference latency, accelerator
throughput, DMA behavior, power, firmware integration, numerical accuracy, or
production readiness. Likewise, full contract evidence demonstrates that the
selected simulated cases satisfy the documented machine and planning contract;
it does not establish a global optimum or real-hardware deployability.
