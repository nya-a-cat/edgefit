# Hardware Optimizer

`edgefit optimize` generates a deterministic, auditable execution plan for a
normalized ONNX graph and an accelerator-aware target profile:

```bash
edgefit optimize examples/models/virtual_npu_tiny.edgefit.json \
  --target targets/virtual-npu.yaml \
  --format json
```

The planner compares declared CPU and NPU kernel costs, applies only trusted and
versioned replacement recipes, groups consecutive NPU nodes, and simulates
aligned scratchpad allocation, DMA loads/stores, and deterministic
spill/reload. Exit code `0` means the plan has no unresolved blocker, `1` means
planning completed with blockers, and `2` means trustworthy planning could not
be performed.

The `edgefit.optimization_plan.v1` output records target identity, confidence,
baseline and proposed latency, assignments, segment boundaries, transfer
events, scratchpad peak, spill bytes, blockers, and a deterministic plan hash.

Checked-in accelerator costs are simulated seed assumptions. They are not
device measurements, benchmark results, compiler output, or runtime guarantees.

## Hosted simulated evidence

`.github/workflows/optimizer-evidence.yml` runs the optimizer only on GitHub-hosted
Ubuntu runners. It uses deterministic generated int8 graphs and checked-in
simulated profiles; it does not contact a physical device or an external
repository.

The evidence suite covers:

- 1K, 10K, and 100K-node contiguous NPU planning;
- repeated full-plan artifact hashing and stable `plan_hash` output bound to the
  model and target profile fingerprints;
- trusted `HardSwish` replacement recipes;
- deterministic CPU boundaries and multiple NPU segments;
- constrained scratchpad spill attempts with explicit unresolved-capacity blockers;
- fail-closed planning when spill is disabled;
- hosted end-to-end CLI duration and peak RSS.

The required pull-request gate runs the 10K case three times. The manual workflow
runs the complete suite five times and uploads the generated models, canonical
plans, raw stdout/stderr, JSON summary, and Markdown summary. Duration and RSS
are GitHub runner process observations. Plan latency remains a profile-driven
simulation and must not be interpreted as hardware inference performance.

The manifest is
`tools/competitive-benchmark/optimizer_evidence_manifest.json`. The test
profiles exist only to create reproducible partition and pressure cases:

- `targets/virtual-npu-segmented.yaml` creates a deterministic CPU boundary;
- `targets/virtual-npu-small-scratchpad.yaml` exercises deterministic spill attempts
  and unresolved-capacity blockers;
- `targets/virtual-npu-no-spill.yaml` requires the same pressure to fail closed.
