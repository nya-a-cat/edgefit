/**
 * @file app_main.cpp
 * @brief 在 ESP32-S3 QEMU 中加载 ESP-DL 模型并验证内存规划链路。
 *
 * 该程序只形成模拟器证据。QEMU 日志不得解释为真实芯片时延、功耗、缓存、
 * PSRAM 或固件兼容性结论。
 */

#include <cstddef>
#include <cstdint>

#include "dl_model_base.hpp"
#include "esp_log.h"

namespace {

constexpr char kTag[] = "edgefit-sim";

extern const uint8_t kModelStart[] asm("_binary_model_espdl_start");
extern const uint8_t kModelEnd[] asm("_binary_model_espdl_end");

bool is_expected_int8_scalar(dl::TensorBase *tensor)
{
    if (tensor == nullptr || tensor->get_dtype() != dl::DATA_TYPE_INT8) {
        return false;
    }
    const auto shape = tensor->get_shape();
    return shape.size() == 2 && shape[0] == 1 && shape[1] == 1;
}

}  // namespace

extern "C" void app_main(void)
{
    const auto model_bytes = static_cast<size_t>(kModelEnd - kModelStart);
    ESP_LOGI(kTag, "EDGEFIT_SIMULATION_START soc=esp32s3 scope=boot_model_load "
                  "evidence=simulated model_bytes=%u",
             static_cast<unsigned>(model_bytes));

    auto *model = new dl::Model(reinterpret_cast<const char *>(kModelStart),
                                fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    auto &inputs = model->get_inputs();
    auto &outputs = model->get_outputs();
    const auto input = inputs.find("onnx::Gemm_0");
    const auto output = outputs.find("11");
    if (inputs.size() != 1 || outputs.size() != 1 || input == inputs.end() ||
        output == outputs.end() || !is_expected_int8_scalar(input->second) ||
        !is_expected_int8_scalar(output->second)) {
        ESP_LOGE(kTag, "EDGEFIT_SIMULATION_FAIL model_signature=invalid");
        delete model;
        return;
    }
    ESP_LOGI(kTag, "EDGEFIT_MODEL_LOAD_PASS soc=esp32s3 signature=pass evidence=simulated");

    // 只输出内存分类，不使用 QEMU 层耗时作为芯片性能证据。
    model->profile_memory();
    ESP_LOGI(kTag,
             "EDGEFIT_SIMULATION_PASS soc=esp32s3 model_load=pass memory_profile=pass "
             "numeric_inference=not_evaluated evidence=simulated");
    delete model;
}
