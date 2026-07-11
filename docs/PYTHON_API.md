# Python API

EdgeFit keeps IR parsing, policy, memory planning, and future optimization
algorithms in Rust. The Python package is an orchestration layer for ONNX
normalization, explicit profile loading, batch execution, and report handling.

The supported development and packaging path is PyO3 with maturin. EdgeFit does
not compile Rust source during Python import and does not require `rustimport`.
Published Rust CLI archives remain independent from the Python package.

```python
import edgefit

report = edgefit.check("model.onnx", "targets/device.yaml")
reports = edgefit.batch(["a.onnx", "b.onnx"], "targets/device.yaml")
```

The module CLI uses one entry point:

```bash
python -m edgefit check model.onnx --target targets/device.yaml --format json
```

Single-model JSON is rendered by the Rust engine and is therefore the canonical
contract shared with the native CLI. Python batch order follows input order and
does not enable implicit concurrency.
