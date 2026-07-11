# ESP-DL / ESP32-S3 模拟部署证据

`.github/workflows/espdl-qemu-evidence.yml` 在 Espressif 官方 ESP-IDF 容器中构建
一个最小 ESP-DL 固件，并用 Espressif QEMU 的 ESP32-S3 machine 执行。工作流
验证固件启动、模型解析/加载和内存规划，不把 QEMU 当作 ESP32-S3 PIE/TIE 数值
等价实现。该工作流只形成 `simulated` 证据，不提升 `targets/esp32s3.yaml` 的
`seed` 可信度。

## 固定依赖

| 依赖 | 固定值 | 用途 |
| --- | --- | --- |
| ESP-IDF | `v5.5.4` | 构建固件并提供 `idf.py qemu` |
| ESP-DL | `3.3.7` | 解析和加载 `.espdl` 模型 |
| Espressif QEMU | `esp-develop-9.2.2-20260417` | 执行 ESP32-S3 启动与模型加载 smoke |
| ESP-DL source commit | `7a3d4c02e8b978b5d4b7ddb23dc68f42e56e83c7` | 固定官方示例模型来源 |
| 示例模型 | 7,664 B / `sha256:877fc69a...b023537` | ESP32-S3 三层 INT8 Gemm/ReLU 模型 |

模型二进制不提交到仓库。证据脚本从固定提交下载模型，在构建前同时校验字节数
和完整 SHA-256；ESP-DL 组件由 IDF Component Manager 以精确版本解析。

## 通过条件

1. ESP32-S3 固件成功构建，且产生可哈希的 ELF。
2. 官方 QEMU 启动固件并输出 `scope=boot_model_load`、`evidence=simulated`
   起始标记。
3. ESP-DL 从对齐的 rodata 解析并加载 ESP32-S3 模型，确认唯一输入
   `onnx::Gemm_0` 与唯一输出 `11` 均为 INT8 `[1,1]`，随后完成内存分类路径。
4. 日志出现唯一 pass 标记，且没有 firmware failure 标记。
5. JSON 证据固定记录 `confidence: simulated`、
   `numeric_inference: not_evaluated` 和禁止推导的结论。

`idf.py qemu` 在固件任务返回后仍保持 SoC 运行，因此证据入口允许 QEMU 被宿主
`timeout` 正常终止。超时退出本身不构成通过；全部启动、加载和内存规划标记仍
是强制条件。

## 数值验证边界

诊断运行 [29112232002](https://github.com/nya-a-cat/edgefit/actions/runs/29112232002)
使用当前官方 QEMU 执行 `dl::Model::test()` 时，最终 Gemm 输出的内置真值为
`101`，模拟结果为 `0`；日志没有 illegal/unknown instruction。Espressif 当前
QEMU 文档没有承诺 ESP-DL PIE/TIE 数值等价，因此该差异作为模拟器边界保留，
不得通过修改真值、接受错误输出或切换非生产内核来消除。数值正确性必须由真实
ESP32-S3 证据关闭。

## 不能由此证明

- 真实 ESP32-S3 时延、吞吐或功耗。
- cache、PSRAM、flash 总线或双核调度的真实行为。
- ESP32-S3 PIE/TIE 优化内核的数值正确性或性能。
- 模型推理输出的数值正确性。
- 任意用户模型的量化精度或部署兼容性。
- `targets/esp32s3.yaml` 的预算和算子规则已经由硬件校准。

工作流刻意不调用 `profile_module()`，因为 QEMU 层级耗时没有芯片性能含义；只
调用 `profile_memory()` 保留 ESP-DL 的内存分类日志，且不把它升级为硬件测量。

## 上游依据

- [Espressif QEMU `esp-develop-9.2.2-20260417`](https://github.com/espressif/qemu/releases/tag/esp-develop-9.2.2-20260417)
- [该版本包含的 ESP32-S3 TIE 指令修复](https://github.com/espressif/qemu/commit/ba5950398f1e7f48cd31bca8a850f987ed0cf787)
- [ESP-IDF ESP32-S3 QEMU guide](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/api-guides/tools/qemu.html)
- [ESP-DL 3.3.7 component](https://components.espressif.com/components/espressif/esp-dl/versions/3.3.7)
- [ESP-DL load, test and profile guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html)
