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
