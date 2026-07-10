# Real-World ONNX Corpus

The feasibility report recommends moving beyond toy fixtures with a real-world
ONNX corpus. EdgeFit keeps large model artifacts out of the repository and
tracks real samples through a manifest plus local verification scripts.

## Source Boundary

The historical ONNX Model Zoo repository now points model downloads to the ONNX
Model Zoo organization on Hugging Face. The corpus uses 20 validated legacy
models from that source: 4 archived `tar.gz` packages and 16 direct `.onnx`
files. The manifest records source pages, download URLs, model sizes, SHA-256
hashes, expected operators, operator domains, outputs, and verification date.

- Manifest: `tools/onnx-normalize/real_world_corpus.json`
- Verification script: `tools/onnx-normalize/real_world_corpus.py`
- Cache directory: `tmp/real_world_corpus`

## Current Entries

| ID | Model | Source form | Model bytes | Operator count |
| --- | --- | --- | ---: | ---: |
| `onnx-model-zoo-mnist-8` | MNIST-Handwritten Digit Recognition | `tar.gz` | 26,454 | 6 |
| `onnx-model-zoo-super-resolution-10` | Super Resolution | `tar.gz` | 240,078 | 5 |
| `onnx-model-zoo-mobilenetv2-12-int8` | MobileNetV2 int8 | `tar.gz` | 3,655,033 | 11 |
| `onnx-model-zoo-ssd-mobilenet-v1-12-int8` | SSD MobileNetV1 int8 | `tar.gz` | 9,540,809 | 22 |
| `onnx-model-zoo-mnist-12-int8` | mnist-12-int8 | `direct .onnx` | 10,969 | 7 |
| `onnx-model-zoo-mnist-12` | mnist-12 | `direct .onnx` | 26,143 | 6 |
| `onnx-model-zoo-mnist-7` | mnist-7 | `direct .onnx` | 26,454 | 6 |
| `onnx-model-zoo-version-rfb-320-int8` | version-RFB-320-int8 | `direct .onnx` | 458,144 | 15 |
| `onnx-model-zoo-version-rfb-320` | version-RFB-320 | `direct .onnx` | 1,270,727 | 17 |
| `onnx-model-zoo-squeezenet1.0-12-int8` | squeezenet1.0-12-int8 | `direct .onnx` | 1,293,388 | 7 |
| `onnx-model-zoo-squeezenet1.0-13-qdq` | squeezenet1.0-13-qdq | `direct .onnx` | 1,345,213 | 9 |
| `onnx-model-zoo-version-rfb-640` | version-RFB-640 | `direct .onnx` | 1,588,012 | 17 |
| `onnx-model-zoo-shufflenet-v2-12-int8` | shufflenet-v2-12-int8 | `direct .onnx` | 2,388,912 | 11 |
| `onnx-model-zoo-shufflenet-v2-12-qdq` | shufflenet-v2-12-qdq | `direct .onnx` | 2,415,805 | 11 |
| `onnx-model-zoo-mobilenetv2-12-qdq` | mobilenetv2-12-qdq | `direct .onnx` | 3,593,903 | 11 |
| `onnx-model-zoo-squeezenet1.0-7` | squeezenet1.0-7 | `direct .onnx` | 4,952,222 | 7 |
| `onnx-model-zoo-squeezenet1.0-12` | squeezenet1.0-12 | `direct .onnx` | 4,952,956 | 7 |
| `onnx-model-zoo-squeezenet1.1-7` | squeezenet1.1-7 | `direct .onnx` | 4,956,208 | 7 |
| `onnx-model-zoo-shufflenet-7` | shufflenet-7 | `direct .onnx` | 5,723,770 | 11 |
| `onnx-model-zoo-shufflenet-8` | shufflenet-8 | `direct .onnx` | 5,723,770 | 11 |

## Domain Notes

- Microsoft-domain quantized operators in the corpus are
  `com.microsoft::QLinearAdd`, `com.microsoft::QLinearConcat`, and
  `com.microsoft::QLinearGlobalAveragePool`.
- These Microsoft-domain operators are tied to pinned ONNX Runtime source
  evidence through `tools/onnx-normalize/ort_runtime_evidence.json`.
- Other listed operators in the current corpus normalize to `ai.onnx::<op>`.

## Verification

Local verification command:

```powershell
.venv\Scripts\python.exe tools\onnx-normalize\real_world_corpus.py --cache tmp\real_world_corpus --out tmp\real_world_corpus\corpus-result.json
```

On a fresh cache, add `--download` to fetch missing manifest artifacts before
validation. The script validates artifact size and SHA-256, model size and
SHA-256, operator set, operator domain set, output dtype/shape, and adapter JSON
serializability. It supports both `archive_url` entries and direct `model_url`
entries.

Current verification result: 20 corpus entries pass.

## Focused value_info repair

The corpus CLI can repair one missing intermediate declaration produced by
`com.microsoft::QLinearGlobalAveragePool` while keeping the downloaded source
model unchanged:

```bash
python tools/onnx-normalize/real_world_corpus.py \
  --cache tmp/real_world_corpus \
  --model-id onnx-model-zoo-squeezenet1.0-12-int8 \
  --repair-qlinear-global-average-pool pool10_1_quantized \
  --repair-out tmp/real_world_corpus/squeezenet1.0-12-int8-value-info.onnx \
  --out tmp/alpha-case/value-info-repair.json
```

This is not a free-form metadata override. The repair refuses to write unless
there is exactly one matching Microsoft-domain producer, a concrete int8/uint8
input, a matching output zero-point type, a consumer, and a separate output
path. Shape and type follow the pinned ONNX Runtime v1.22.0 schema: output type
inherits the input, N/C are retained, and all spatial dimensions become 1.
The repaired model is checked, reloaded, normalized, and hashed before evidence
is accepted. The generated model remains a temporary workflow artifact and is
not committed or uploaded.

The hosted closure run derived `uint8 [1, 1000, 1, 1]` from the concrete
`conv10_1_quantized` input `[1, 1000, 13, 13]` and matching uint8 output zero
point. The added declaration increased the file by 47 bytes, removed the sole
unknown dtype and unresolved activation, raised memory confidence from medium
to high, and changed EdgeFit from fail to pass without suppressions. Evidence:
<https://github.com/nya-a-cat/edgefit/actions/runs/29094249434>.

The Alpha workflow also compares the original and repaired models through the
existing runtime smoke CLI:

```bash
python tools/onnx-normalize/runtime_smoke.py \
  --reference-model tmp/real_world_corpus/squeezenet1.0-12-int8.onnx \
  --candidate-model tmp/real_world_corpus/squeezenet1.0-12-int8-value-info.onnx \
  --provider CPUExecutionProvider \
  --out tmp/alpha-case/runtime-equivalence.json
```

This comparison requires identical runtime input/output signatures, rejects
dynamic input dimensions, feeds both models the same deterministic non-zero
input, and requires every output element to match exactly. The evidence records
the model and input hashes, ONNX Runtime version, provider, output dtype/shape,
exact-match result, and maximum absolute difference. It proves that this
metadata-only repair preserves the result for the recorded input; it is not a
dataset-level accuracy or real-device test.

## Profile Matrix Consumer

`tools/onnx-normalize/profile_matrix.py` consumes this corpus cache and runs each
model against the checked-in target profiles. The current matrix has 60 cells:
20 strict MCU failures, 20 ORT mobile-like passes, and 20 generic TFLM-like
failures. The strict failures are expected deployment-boundary results.

## Runtime Smoke Consumer

`tools/onnx-normalize/runtime_smoke.py` consumes the verified corpus result and
runs each passing corpus model with ONNX Runtime 1.22.0 on
`CPUExecutionProvider`. The current smoke result covers all 20 corpus entries
and records `status = pass` with no output dtype or fixed-shape mismatches.
