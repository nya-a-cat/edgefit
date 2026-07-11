#!/usr/bin/env bash
# ============================================================================
# EdgeFit ESP-DL / ESP32-S3 QEMU 证据入口
#
# 下载并校验固定的 Espressif 示例模型，构建固件，在官方 QEMU 中验证 ESP-DL
# 模型加载与内存规划，并生成明确标注为 simulated 的 JSON/Markdown 证据。
# ============================================================================

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FIRMWARE_DIR="$ROOT/tools/espdl-qemu/firmware"
OUTPUT_DIR=${EDGEFIT_ESPDL_QEMU_OUT:-"$ROOT/tmp/espdl-qemu"}
BUILD_DIR="$OUTPUT_DIR/build"
MODEL_DIR="$FIRMWARE_DIR/main/models"
MODEL_PATH="$MODEL_DIR/model.espdl"
RAW_LOG="$OUTPUT_DIR/qemu.raw.log"
PUBLIC_LOG="$OUTPUT_DIR/qemu.log"
REPORT_JSON="$OUTPUT_DIR/evidence.json"
REPORT_MD="$OUTPUT_DIR/evidence.md"
PROJECT_ELF="$BUILD_DIR/edgefit_espdl_qemu.elf"

ESPIDF_VERSION="v5.5.4"
ESPDL_VERSION="3.3.7"
ESPDL_COMMIT="7a3d4c02e8b978b5d4b7ddb23dc68f42e56e83c7"
MODEL_BYTES="7664"
MODEL_SHA256="877fc69afcb00dc0682a765f33031c6c78d53bdecdd0e6613387db07ab023537"
MODEL_SOURCE_PATH="examples/tutorial/how_to_load_test_profile_model/model_in_flash_rodata/main/models/s3/model.espdl"
MODEL_URL="https://raw.githubusercontent.com/espressif/esp-dl/${ESPDL_COMMIT}/${MODEL_SOURCE_PATH}"
QEMU_VERSION="esp-develop-9.2.2-20260417"
QEMU_TIMEOUT_SECONDS=${EDGEFIT_QEMU_TIMEOUT_SECONDS:-90}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "espdl-qemu: missing command: $1" >&2
        exit 2
    }
}

require_command curl
require_command idf.py
require_command qemu-system-xtensa
require_command sha256sum
require_command timeout

if [[ ${IDF_PATH:-} == "" ]]; then
    echo "espdl-qemu: IDF_PATH is required" >&2
    exit 2
fi

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"
rm -f "$MODEL_PATH" "$RAW_LOG" "$PUBLIC_LOG" "$REPORT_JSON" "$REPORT_MD"

curl --fail --location --silent --show-error "$MODEL_URL" --output "$MODEL_PATH"
actual_model_bytes=$(stat -c '%s' "$MODEL_PATH")
actual_model_sha256=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
if [[ "$actual_model_bytes" != "$MODEL_BYTES" || "$actual_model_sha256" != "$MODEL_SHA256" ]]; then
    echo "espdl-qemu: upstream model integrity mismatch" >&2
    exit 2
fi

idf.py -C "$FIRMWARE_DIR" -B "$BUILD_DIR" set-target esp32s3
idf.py -C "$FIRMWARE_DIR" -B "$BUILD_DIR" build

set +e
timeout --signal=TERM "${QEMU_TIMEOUT_SECONDS}s" \
    idf.py -C "$FIRMWARE_DIR" -B "$BUILD_DIR" qemu >"$RAW_LOG" 2>&1
qemu_exit=$?
set -e

sed "s#${ROOT}#<repo>#g" "$RAW_LOG" >"$PUBLIC_LOG"
if ! grep -Fq "EDGEFIT_SIMULATION_START soc=esp32s3 scope=boot_model_load evidence=simulated" "$PUBLIC_LOG"; then
    echo "espdl-qemu: firmware start marker is missing" >&2
    exit 1
fi
if ! grep -Fq "EDGEFIT_MODEL_LOAD_PASS soc=esp32s3 signature=pass evidence=simulated" "$PUBLIC_LOG"; then
    echo "espdl-qemu: ESP-DL model-load marker is missing" >&2
    exit 1
fi
if ! grep -Fq "EDGEFIT_SIMULATION_PASS soc=esp32s3 model_load=pass memory_profile=pass numeric_inference=not_evaluated evidence=simulated" "$PUBLIC_LOG"; then
    echo "espdl-qemu: simulation pass marker is missing" >&2
    exit 1
fi
if grep -Fq "EDGEFIT_SIMULATION_FAIL" "$PUBLIC_LOG"; then
    echo "espdl-qemu: firmware emitted a failure marker" >&2
    exit 1
fi
if [[ "$qemu_exit" != "0" && "$qemu_exit" != "124" ]]; then
    echo "espdl-qemu: QEMU exited unexpectedly with status $qemu_exit" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_ELF" ]]; then
    echo "espdl-qemu: firmware ELF is missing" >&2
    exit 1
fi

firmware_bytes=$(stat -c '%s' "$PROJECT_ELF")
firmware_sha256=$(sha256sum "$PROJECT_ELF" | awk '{print $1}')

{
    printf '%s\n' '{'
    printf '  "schema": "edgefit.simulated_deployment.v1",\n'
    printf '  "status": "pass",\n'
    printf '  "confidence": "simulated",\n'
    printf '  "soc": "esp32s3",\n'
    printf '  "esp_idf_version": "%s",\n' "$ESPIDF_VERSION"
    printf '  "esp_dl_version": "%s",\n' "$ESPDL_VERSION"
    printf '  "emulator": "espressif-qemu",\n'
    printf '  "emulator_version": "%s",\n' "$QEMU_VERSION"
    printf '  "scope": "firmware_boot_model_load",\n'
    printf '  "numeric_inference": "not_evaluated",\n'
    printf '  "optimized_esp32s3_pie_validated": false,\n'
    printf '  "qemu_exit_code": %s,\n' "$qemu_exit"
    printf '  "model": {"bytes": %s, "sha256": "sha256:%s", "upstream_commit": "%s", "source_path": "%s"},\n' \
        "$actual_model_bytes" "$actual_model_sha256" "$ESPDL_COMMIT" "$MODEL_SOURCE_PATH"
    printf '  "firmware": {"bytes": %s, "sha256": "sha256:%s"},\n' \
        "$firmware_bytes" "$firmware_sha256"
    printf '  "assertions": {"firmware_started": true, "model_loaded": true, "model_signature_validated": true, "memory_profile_completed": true, "failure_marker_absent": true},\n'
    printf '  "limitations": ["not_real_hardware", "numeric_inference_not_evaluated", "no_optimized_pie_kernel_claim", "no_device_latency_claim", "no_power_claim", "no_psram_claim"]\n'
    printf '%s\n' '}'
} >"$REPORT_JSON"

{
    printf '%s\n\n' '# ESP-DL / ESP32-S3 QEMU Evidence'
    printf '**Status:** `pass`  \n'
    printf '**Confidence:** `simulated`  \n'
    printf '**ESP-IDF:** `%s`  \n' "$ESPIDF_VERSION"
    printf '**ESP-DL:** `%s`  \n' "$ESPDL_VERSION"
    printf '**QEMU:** `%s`  \n' "$QEMU_VERSION"
    printf '**Scope:** `firmware_boot_model_load`  \n'
    printf '**Numeric inference:** `not_evaluated`  \n'
    printf '**Model SHA-256:** `sha256:%s`  \n' "$actual_model_sha256"
    printf '**Firmware SHA-256:** `sha256:%s`\n\n' "$firmware_sha256"
    printf '%s\n' '- ESP32-S3 firmware booted in Espressif QEMU.'
    printf '%s\n' '- ESP-DL parsed and loaded the aligned ESP32-S3 rodata model with its expected INT8 input/output signature.'
    printf '%s\n' '- ESP-DL completed its memory-profile path.'
    printf '%s\n' '- No firmware failure marker was emitted.'
    printf '\n%s\n' 'QEMU evidence does not evaluate numeric inference or validate optimized ESP32-S3 PIE/TIE kernels, real-hardware latency, throughput, power, cache, PSRAM, or firmware compatibility.'
} >"$REPORT_MD"

printf '%s\n' "$REPORT_JSON" "$REPORT_MD" "$PUBLIC_LOG"
