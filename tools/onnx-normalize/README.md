# ONNX Normalize Tool

This adapter uses the pinned official Python `onnx` package to run
`onnx.checker.check_model`, apply `onnx.shape_inference.infer_shapes`, and emit
`edgefit.normalized_model.v1` JSON for the Rust analyzer.

If checker validation succeeds but shape inference fails, the adapter emits the
checker-approved graph with an explicit failed shape-inference status so the
Rust policy can lower confidence or fail a strict target. It includes external
ONNX data files in package bytes/hash metadata, records ONNX domain-opset
imports, and rejects nested subgraphs, local functions, and sparse initializers
until EdgeFit has complete support for them.

Install the adapter dependency:

```powershell
uv pip install --python .venv\Scripts\python.exe -r tools/onnx-normalize/requirements.txt
```

Install the optional runtime-smoke dependency when running ONNX Runtime smoke verification:

```powershell
uv pip install --python .venv\Scripts\python.exe -r tools/onnx-normalize/requirements-runtime.txt
```

Normalize explicitly:

```powershell
python tools/onnx-normalize/normalize_onnx.py model.onnx --out model.edgefit.json
cargo run -p edgefit-cli -- check model.edgefit.json --target targets/esp32s3.yaml
```

The CLI also accepts `.onnx` directly and invokes this adapter:

```powershell
cargo run -p edgefit-cli -- check model.onnx --target targets/esp32s3.yaml
```

The release CLI embeds this same adapter source and still requires Python plus
the pinned `onnx` package for direct ONNX input. Set `EDGEFIT_PYTHON` to choose
the interpreter. A pre-normalized `*.edgefit.json` is supported for reviewed
offline input but is reported as trusted metadata regardless of its
`normalization` fields. Only the CLI's original `.onnx` branch supplies adapter
provenance out of band; use that path for stronger CI evidence.

Run adapter tests:

```powershell
python -m unittest discover -s tools/onnx-normalize -p "test_*.py"
```

The toy operator coverage currently includes Add, MatMul, Conv, Reshape,
Transpose, and Softmax.
Run the real-world corpus verifier after downloading a manifest entry into the
cache directory:

```powershell
python tools/onnx-normalize/real_world_corpus.py --cache tmp/real_world_corpus --out tmp/real_world_corpus/corpus-result.json
```

Use `--download` to fetch missing archives from manifest URLs when network
access is available.

Run the profile calibration matrix after building the Rust CLI and verifying the corpus cache:

```powershell
cargo build -p edgefit-cli
python tools/onnx-normalize/profile_matrix.py --cache tmp/real_world_corpus --out tmp/real_world_corpus/profile-matrix.json --markdown-out tmp/real_world_corpus/profile-matrix.md
```

The matrix runs each real-world corpus model against the checked-in target
profiles and records pass/fail status, diagnostic IDs, and warning/error counts.

Run the corpus expansion gate before interpreting the profile matrix:

```powershell
python tools/onnx-normalize/corpus_expansion_gate.py --out tmp/real_world_corpus/corpus-expansion-gate.json --markdown-out tmp/real_world_corpus/corpus-expansion-gate.md
```

The gate checks model count against the 20-model release-grade target and verifies
that every current model/profile cell has one reviewed label.

Run the operator-support audit after the matrix is available:

```powershell
python tools/onnx-normalize/operator_support_audit.py --matrix tmp/real_world_corpus/profile-matrix.json --labels tools/onnx-normalize/operator_support_labels.json --out tmp/real_world_corpus/operator-support-audit.json --markdown-out tmp/real_world_corpus/operator-support-audit.md
```

The audit compares each checked-in profile allowlist with corpus, fixture,
runtime-source evidence, and reviewed labels from
`tools/onnx-normalize/operator_support_labels.json`. It records the current
sample count against the 20-model release-grade target and computes
unsupported-op and unsupported-dtype precision/recall for labeled cells. The
current gate reports complete coverage for 20 models and 60 label cells.

Run generated operator fixture verification for profile-evidence gaps:

```powershell
python tools/onnx-normalize/operator_fixture_corpus.py --cache tmp/operator_fixtures --out tmp/operator_fixtures/operator-fixtures-result.json
```

The current generated fixture manifest covers `Gemm`, `Resize`, and `Softmax`.
Each fixture is saved as ONNX, normalized through the adapter, and checked
against expected operators, operator domains, and outputs.

Verify pinned ONNX Runtime source evidence after sparse-checking out the official ONNX Runtime repository under `tmp/ort-src`:

```powershell
python tools/onnx-normalize/ort_runtime_evidence.py --manifest tools/onnx-normalize/ort_runtime_evidence.json --source-root tmp/ort-src --out tmp/ort-runtime-evidence/ort-runtime-evidence-result.json
```

The current runtime evidence manifest covers `com.microsoft::QLinearAdd`,
`com.microsoft::QLinearConcat`, and `com.microsoft::QLinearGlobalAveragePool` at ONNX Runtime commit
`c57e0e50ad068905a4140d361b0b3fd8c251e540`.

Run the profile reference check after the adapter dependency is installed:

```powershell
python tools/onnx-normalize/profile_reference_check.py --profile targets/ort-mobile-cpu.yaml --manifest tools/onnx-normalize/real_world_corpus.json --fixture-manifest tools/onnx-normalize/operator_fixtures.json --runtime-evidence tools/onnx-normalize/ort_runtime_evidence.json --out tmp/real_world_corpus/profile-reference.json --markdown-out tmp/real_world_corpus/profile-reference.md
```

The check compares target-profile allowed operators, including their domains,
against the official ONNX operator schemas from the installed `onnx` package,
the verified real-world corpus, generated operator fixtures, and pinned ONNX Runtime source evidence. It also records the profile
metadata and verifies that the installed `onnx` version matches the exact pin in
`requirements.txt`. It fails when the version status is not `match` or when an
allowlist operator has neither official schema coverage nor generated, real-world, or runtime-source evidence for the same domain and operator.

Run runtime smoke inference with ONNX Runtime after the corpus verifier passes:

```powershell
python tools/onnx-normalize/runtime_smoke.py --profile targets/ort-mobile-cpu.yaml --corpus-result tmp/real_world_corpus/corpus-result.json --provider CPUExecutionProvider --out tmp/real_world_corpus/runtime-smoke.json
```

Run the ORT runtime-boundary check after corpus and fixture evidence are available:

```powershell
python tools/onnx-normalize/ort_runtime_boundary.py --profile targets/ort-mobile-cpu.yaml --corpus-result tmp/real_world_corpus/corpus-result.json --fixture-result tmp/operator_fixtures/operator-fixtures-result.json --ort-source tmp/ort-src --out tmp/ort-runtime-boundary/ort-runtime-boundary.json --markdown-out tmp/ort-runtime-boundary/ort-runtime-boundary.md --config-out tmp/ort-runtime-boundary/edgefit-ort-required-ops.config
```

Run the public PR trial gate before the profile confidence gate:

```powershell
cp examples/public_pr_trials.example.json docs/public_pr_trials.json
python tools/public_pr_trial_gate.py --manifest docs/public_pr_trials.json --out tmp/public_pr_trials/public-pr-trial-gate.json --markdown-out tmp/public_pr_trials/public-pr-trial-gate.md
```

Run the profile confidence gate after the reference check, matrix, corpus gate,
operator-support audit, runtime evidence, runtime smoke verification, ORT
runtime-boundary check, diagnostic policy check, and public PR trial gate:

```powershell
python tools/onnx-normalize/profile_confidence_gate.py --profile targets/ort-mobile-cpu.yaml --reference tmp/real_world_corpus/profile-reference.json --matrix tmp/real_world_corpus/profile-matrix.json --corpus-gate tmp/real_world_corpus/corpus-expansion-gate.json --operator-audit tmp/real_world_corpus/operator-support-audit.json --runtime-evidence-result tmp/ort-runtime-evidence/ort-runtime-evidence-result.json --runtime-smoke tmp/real_world_corpus/runtime-smoke.json --runtime-boundary tmp/ort-runtime-boundary/ort-runtime-boundary.json --diagnostic-policy docs/DIAGNOSTIC_POLICY.md --public-pr-trials tmp/public_pr_trials/public-pr-trial-gate.json --out tmp/real_world_corpus/profile-confidence-gate.json --markdown-out tmp/real_world_corpus/profile-confidence-gate.md
```

The current gate decision is `hold_seed` because the local public PR trial
manifest has 0/3 verified repositories. The warning-only diagnostic policy is
public in `docs/DIAGNOSTIC_POLICY.md`; detailed confidence, runtime-boundary,
and trial review artifacts remain local.
