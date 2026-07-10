/**
 * @file app_main.cpp
 * @brief 在 ESP32-S3 QEMU 中加载 ESP-DL 模型并校验其内置测试向量。
 *
 * 该程序只形成模拟器证据。QEMU 日志不得解释为真实芯片时延、功耗、缓存、
 * PSRAM 或固件兼容性结论。
 */

#include <cstddef>
#include <cstdint>

#include "dl_model_base.hpp"
#include "esp_err.h"
#include "esp_log.h"

namespace {

constexpr char kTag[] = "edgefit-sim";

extern const uint8_t kModelStart[] asm("_binary_model_espdl_start");
extern const uint8_t kModelEnd[] asm("_binary_model_espdl_end");

}  // namespace

extern "C" void app_main(void)
{
    const auto model_bytes = static_cast<size_t>(kModelEnd - kModelStart);
    ESP_LOGI(kTag, "EDGEFIT_SIMULATION_START soc=esp32s3 evidence=simulated model_bytes=%u",
             static_cast<unsigned>(model_bytes));

    auto *model = new dl::Model(reinterpret_cast<const char *>(kModelStart),
                                fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    const esp_err_t test_status = model->test();
    if (test_status != ESP_OK) {
        ESP_LOGE(kTag, "EDGEFIT_SIMULATION_FAIL espdl_test=%s", esp_err_to_name(test_status));
        delete model;
        return;
    }

    // 只输出内存分类，不使用 QEMU 层耗时作为芯片性能证据。
    model->profile_memory();
    ESP_LOGI(kTag, "EDGEFIT_SIMULATION_PASS soc=esp32s3 espdl_test=pass evidence=simulated");
    delete model;
}
