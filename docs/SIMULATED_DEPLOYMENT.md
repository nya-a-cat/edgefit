# ESP-DL / ESP32-S3 模拟部署证据

`.github/workflows/espdl-qemu-evidence.yml` 在 Espressif 官方 ESP-IDF 容器中构建
一个最小 ESP-DL 固件，并用 Espressif QEMU 的 ESP32-S3 machine 执行。该工作流
只形成 `simulated` 证据，不提升 `targets/esp32s3.yaml` 的 `seed` 可信度。

## 固定依赖

| 依赖 | 固定值 | 用途 |
| --- | --- | --- |
| ESP-IDF | `v5.5.4` | 构建固件并提供 `idf.py qemu` |
| ESP-DL | `3.3.7` | 加载和校验 `.espdl` 模型 |
| ESP-DL source commit | `7a3d4c02e8b978b5d4b7ddb23dc68f42e56e83c7` | 固定官方示例模型来源 |
| 示例模型 | 7,664 B / `sha256:877fc69a...b023537` | 内含 ESP-PPQ 导出的测试输入和输出 |

模型二进制不提交到仓库。证据脚本从固定提交下载模型，在构建前同时校验字节数
和完整 SHA-256；ESP-DL 组件由 IDF Component Manager 以精确版本解析。

## 通过条件

1. ESP32-S3 固件成功构建，且产生可哈希的 ELF。
2. 官方 QEMU 启动固件并输出 `evidence=simulated` 起始标记。
3. ESP-DL 从对齐的 rodata 加载模型，`dl::Model::test()` 对内置输入/输出返回
   `ESP_OK`。
4. 日志出现唯一 pass 标记，且没有 firmware failure 标记。
5. JSON 证据固定记录 `confidence: simulated` 和禁止推导的结论。

`idf.py qemu` 在固件任务返回后仍保持 SoC 运行，因此证据入口允许 QEMU 被宿主
`timeout` 正常终止。超时退出本身不构成通过；全部固件标记仍是强制条件。

## 不能由此证明

- 真实 ESP32-S3 时延、吞吐或功耗。
- cache、PSRAM、flash 总线或双核调度的真实行为。
- 任意用户模型的量化精度或部署兼容性。
- `targets/esp32s3.yaml` 的预算和算子规则已经由硬件校准。

工作流刻意不调用 `profile_module()`，因为 QEMU 层级耗时没有芯片性能含义；只
调用 `profile_memory()` 保留 ESP-DL 的内存分类日志，且不把它升级为硬件测量。

## 上游依据

- [ESP-IDF ESP32-S3 QEMU guide](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/api-guides/tools/qemu.html)
- [ESP-DL 3.3.7 component](https://components.espressif.com/components/espressif/esp-dl/versions/3.3.7)
- [ESP-DL load, test and profile guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html)
