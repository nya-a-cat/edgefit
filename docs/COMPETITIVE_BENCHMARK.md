# EdgeFit 竞品基准

## 目的

这套基准用于回答“EdgeFit 相比现有 ONNX 分析工具多解决了什么问题”，而不是通过自定义总分宣布 EdgeFit 获胜。第一阶段固定比较三项本地工具：

- EdgeFit：目标约束、稳定诊断、预算判定和 CI 工件。
- ONNX Runtime Mobile Model Usability Checker：ORT Mobile、NNAPI 与 CoreML 的模型适用性和分区估计。
- onnx-tool：shape inference、MACs、参数和逐节点内存统计。

源码入口为 `tools/competitive-benchmark/benchmark.py`，案例清单为
`tools/competitive-benchmark/benchmark_manifest.json`。聚焦 EdgeFit 的 Alpha
案例和完整三工具矩阵均已在 GitHub 托管 runner 上执行。

Prototype → Alpha 的第一条聚焦证据使用同一 CLI 和
`tools/competitive-benchmark/alpha_case_manifest.json`，对比 ONNX Model Zoo
SqueezeNet 1.0 的 FP32 与 INT8 QOperator 文件。工作流只下载清单中这两个
SHA-256 固定模型，并重复运行 EdgeFit 7 次，以同一托管 runner 上的端到端
进程耗时中位数降低偶发启动噪声。

## 首条托管 Alpha 证据

[GitHub Actions run 29093503757](https://github.com/nya-a-cat/edgefit/actions/runs/29093503757)
在同一个 Ubuntu runner 上完成模型哈希/结构校验、Release 构建和 FP32/INT8
各 7 次 EdgeFit 分析。结果如下：

| 指标 | FP32 基线 | INT8 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 模型文件 | 4,952,956 B | 1,293,388 B | -73.89% |
| initializer | 4,941,988 B | 1,267,228 B | -74.36% |
| 逻辑 activation 峰值 | 6,308,352 B | 3,097,600 B | -50.90% |
| 计划 arena 高水位 | 6,910,464 B | 3,864,384 B | -44.08% |
| EdgeFit 进程耗时中位数 | 226 ms | 218 ms | -3.54% |
| 部署判定 | pass | fail | 被证据完整性门槛阻断 |

INT8 候选不是因为内存超预算或算子不受支持而失败。自定义量化链末端的
`pool10_1_quantized` 缺少可证明的 dtype 和 size，触发 `EF0302` 与
`EF0502`；同时 `EF0104` 明确把峰值可信度降为 medium。这个结果说明模型
压缩率不能替代部署验证，也是 EdgeFit 当前最直接的商业价值：在集成运行时
或固件前阻止“模型更小，所以一定能部署”的错误判断。

这仍不是设备实测。耗时只是托管 runner 上包含进程和 Python adapter 启动的
端到端时间，arena 是 target profile 下的确定性静态规划值。

### value_info 修复闭环

[GitHub Actions run 29094249434](https://github.com/nya-a-cat/edgefit/actions/runs/29094249434)
完成了同一模型的 fail-to-pass 验证。工作流先断言原始 INT8 模型必须失败，
再依据模型自身 producer 输入、output zero-point 和固定 ORT v1.22.0 schema
补充 `pool10_1_quantized`，最后要求修复模型通过 EdgeFit。

| 证据 | 原始模型 | 修复模型 |
| --- | --- | --- |
| SHA-256 | `3da17dfa...c0a972b` | `61edbf7d...b89f595` |
| 文件字节 | 1,293,388 | 1,293,435 |
| `pool10_1_quantized` | dtype/shape 缺失 | `uint8 [1,1000,1,1]` |
| 未解析 activation | 1 | 0 |
| 未知 dtype tensor | 1 | 0 |
| 峰值可信度 | medium | high |
| 计划 arena | 3,864,384 B | 3,864,384 B |
| EdgeFit 判定 | fail | pass |
| diagnostics | `EF0502`, `EF0104`, `EF0302` | 空 |
| suppressed diagnostics | 空 | 空 |

修复只增加 47 字节的 value_info，不改变 initializer、节点、arena 或预算。
原始和修复报告会作为 Artifact 上传，修复后的 ONNX 二进制不会上传。这证明
pass 来自证据补全，而不是放宽 target、扩大预算或 suppression。Alpha 工作流
还会在同一 `CPUExecutionProvider` 上给原始和修复模型输入确定性的非零张量，
并要求输出接口、dtype、shape 与全部元素完全相等。该单输入等价性检查仍不能
替代数据集精度评估或设备 runtime/hardware 验证。

[GitHub Actions run 29095301929](https://github.com/nya-a-cat/edgefit/actions/runs/29095301929)
已通过这条运行时门槛：ONNX Runtime 1.22.0 对固定的
`data_0 float32 [1,3,224,224]` 输入运行两个模型，得到的
`softmaxout_1 float32 [1,1000,1,1]` 输出逐元素完全一致，最大绝对差为 `0.0`。
因此当前证据链是“原始模型 EdgeFit fail → 元数据修复 → EdgeFit pass → 托管
ORT 单输入结果不变”，而不是只依赖静态报告自证。

## 完整托管成熟度证据

[GitHub Actions run 29103544134](https://github.com/nya-a-cat/edgefit/actions/runs/29103544134)
在提交 `ba324bc` 上完成 Release 构建、三档规模门禁、十个模型的文件完整性
校验和三工具矩阵。工作流状态为 success，规模结果和竞品结果分别作为 Artifact
保存；模型二进制没有进入 Artifact。

### 规划器规模结果

每档案例在同一个 Ubuntu runner 上重复运行 EdgeFit 5 次。耗时取端到端进程
中位数，RSS 取五次子进程峰值中的最大值：

| 线性 Relu 节点 | Release 耗时中位数 | 五次耗时样本 | 最大峰值 RSS | 报告一致性 |
| ---: | ---: | --- | ---: | --- |
| 1,000 | 7 ms | 15, 7, 7, 7, 7 ms | 6,275,072 B | 5/5 SHA-256 相同 |
| 10,000 | 70 ms | 70, 69, 70, 69, 70 ms | 35,991,552 B | 5/5 SHA-256 相同 |
| 100,000 | 854 ms | 850, 859, 843, 854, 869 ms | 336,494,592 B | 5/5 SHA-256 相同 |

三档的节点数、耗时上限、RSS 上限、报告确定性和完整证据状态全部通过。它证明
当前 Release CLI 可以在托管环境处理 100K 节点线性图，并给出稳定报告；它不
证明任意图拓扑的最坏复杂度，也不是设备推理延迟、吞吐、功耗或 MCU 内存实测。

### 十模型三工具结果

矩阵固定使用 EdgeFit `0.1.0`、ONNX Runtime `1.22.0` 和 onnx-tool `1.0.1`，
共形成 30 条工具运行证据：

| 工具 | 完成分析 | 明确拒绝 | 结果边界 |
| --- | ---: | ---: | --- |
| EdgeFit | 9/10 | 1/10 | 3 个 target pass、6 个带稳定诊断的 target fail；控制流模型在 adapter 边界拒绝 |
| ORT Mobile Checker | 9/10 | 1/10 | 输出 NNAPI/CoreML/ORT Mobile 适用性建议；控制流模型拒绝 |
| onnx-tool | 4/10 | 6/10 | 对可处理模型输出 MACs、参数和逐节点内存；对动态 shape、部分量化算子和控制流拒绝 |

EdgeFit 对 onnx-tool 拒绝的六个模型中的五个仍完成了分析，并输出具体预算或
元数据缺口，例如未解析 activation、未知 dtype 和 arena 超预算。这是当前最可
销售的差异：不是替代 MACs 工具或 ORT EP 检查器，而是把“能否在指定 target
contract 下继续集成”变成稳定的 CI 判定和修复入口。

三工具的正常模型端到端耗时处于相近量级，不能据此宣称 EdgeFit 普遍更快。
EdgeFit 的 9 个已完成案例为 209–246 ms，onnx-tool 的案例为 198–264 ms，ORT
Mobile Checker 除控制流失败案例外为 241–315 ms；三者工作内容不同，这些数字
只用于发现数量级回退，不用于产品排名。

## 为什么不直接比较“谁通过了更多模型”

三个工具回答的问题不同：

| 指标 | 含义 | 能否直接横向比较 |
| --- | --- | --- |
| EdgeFit `planned_activation_arena_bytes` | 确定性 best-fit arena 高水位，包含 profile 声明的对齐、workspace、碎片与安全 in-place 复用 | 只能和相同 target contract 或运行时 arena 测量比较 |
| EdgeFit `estimated_peak_activation_bytes` | 同一生命周期扫描中的逻辑 live tensor 峰值，不含物理 arena 放置影响 | 用于解释 allocator 开销，不能代替实际 arena 高水位 |
| onnx-tool `Total/Memory` | 各节点输出 activation 与静态权重内存的求和；共享权重可能重复计数 | 不能当作峰值 activation |
| ORT Mobile partition coverage | 指定 ORT Execution Provider 可覆盖的节点和分区估计 | 只适用于该 ORT 场景 |

因此基准保留原始 stdout、stderr 和工具原始报告，并对每个文件记录 SHA-256。统一 JSON 只提取有明确语义的字段，不把这些内存数字合并成一个排名。

## 固定案例

第一阶段从现有 20 模型语料清单中选取 10 个案例，覆盖：

- 小型与中型静态 fp32 模型。
- QOperator 与 QDQ 两种 int8 表示。
- `com.microsoft` 扩展域。
- symbolic shape、目标检测图和控制流失败边界。
- 同一模型家族的 fp32、QOperator、QDQ 对照。

案例只引用 `tools/onnx-normalize/real_world_corpus.json` 中已有的模型 ID、字节数和 SHA-256，不复制下载地址或另建一套模型事实源。

## 执行方式

运行前需要满足以下条件：

1. 现有 real-world corpus 已按清单下载、解包并校验到 `tmp/real_world_corpus/`。
2. EdgeFit CLI 已构建，并通过 `--edgefit` 指向对应二进制。
3. `--python` 指向同时安装官方 `onnxruntime` 与上游 `onnx-tool` 的 Python 环境。
4. 所有网络下载应在运行基准前单独完成；基准 CLI 本身不联网、不安装依赖。

本地复现命令如下；公开结论以托管工作流结果为准：

```bash
uv run python tools/competitive-benchmark/benchmark.py \
  --edgefit tmp/cargo-target/debug/edgefit \
  --out-dir tmp/competitive_benchmark
```

Windows 二进制路径应改为 `tmp/cargo-target/debug/edgefit.exe`。

聚焦 Alpha 案例由 `.github/workflows/alpha-evidence.yml` 手动触发。它输出
两个完整 EdgeFit JSON 报告以及一张 before/after 表，覆盖模型文件、权重、
逻辑 activation 峰值、计划 arena 高水位、峰值节点和进程耗时。模型二进制
只在 runner 临时目录使用，不上传为公开 Artifact。

完整成熟度证据由 `.github/workflows/maturity-evidence.yml` 手动触发，分成两个
互不争抢资源的 job：一个运行 1K、10K、100K 节点规划器规模证据，另一个运行
固定 10 模型的三工具矩阵。竞品依赖固定为 `onnx==1.22.0`、
`onnxruntime==1.22.0` 和 `onnx-tool==1.0.1`。模型仍从现有 corpus 清单下载并
按字节数和 SHA-256 校验，不进入上传 Artifact。下载阶段使用
`real_world_corpus.py --file-integrity-only`，不会提前规范化并过滤故意保留的
不支持图结构；EdgeFit、ORT Mobile Checker 与 onnx-tool 必须各自留下接受或
拒绝证据。

规模案例仍复用同一个 `benchmark.py`。清单声明确定性的 float32 Relu 线性链，
runner 生成规范化 JSON 后运行 Release EdgeFit；Linux 使用 GNU time 记录子进程
峰值 RSS。每个案例保留全部时延样本和报告哈希，重复运行报告不一致、缺少 RSS、
节点数不符或超过案例上限都会使证据不完整。10K 案例进入普通 PR `ci-gate`，
100K 案例只在托管成熟度工作流运行，避免三平台 PR 重复消耗资源。

生成案例的报告模型哈希是生成规格的 SHA-256 指纹；实际规范化 JSON 的字节数与
SHA-256 由案例文件证据单独记录，并在启动 EdgeFit 前校验。二者不混用，避免把
不存在的原始 ONNX 哈希包装成真实来源证据。

```bash
python tools/competitive-benchmark/benchmark.py \
  --manifest tools/competitive-benchmark/planner_scale_manifest.json \
  --edgefit target/release/edgefit \
  --tools edgefit \
  --edgefit-repetitions 5 \
  --measure-peak-rss \
  --out-dir tmp/performance-evidence
```

## 输出

```text
tmp/competitive_benchmark/
├── competitive-benchmark.json
├── competitive-benchmark.md
└── artifacts/
    └── <case-id>/
        ├── edgefit-report.json
        ├── edgefit.stdout.txt
        ├── edgefit.stderr.txt
        ├── ort-mobile.stdout.txt
        ├── ort-mobile.stderr.txt
        ├── onnx-tool-profile.csv
        ├── onnx-tool.stdout.txt
        └── onnx-tool.stderr.txt
```

结果状态含义：

- `completed`：工具以该适配器允许的退出码完成，所需报告也可解析。
- `tool_rejected`：工具已实际运行，但拒绝或无法分析该模型；这是有效的竞品边界证据。
- `unavailable`：二进制或 Python 包不可用。
- `timed_out`：超过单工具单案例超时。
- `runner_error`：工具声称完成，但缺失或损坏了约定输出。

只有三个工具的版本探针都有结果，且所有案例都得到 `completed` 或 `tool_rejected` 证据时，整套结果才标记为 `complete`。

## 复杂度

设案例数为 `C`，单个模型文件大小为 `S`，三个上游工具的执行成本分别为 `E`、`O`、`T`：

- 基准前置 SHA-256 校验为 `O(C × S)`。
- 编排成本为 `O(C × (E + O + T))`，三个工具按固定顺序串行执行，避免并发争抢 CPU 和内存影响结果。
- 证据磁盘占用为模型外的 `O(C × 工具输出大小)`；模型本身复用既有 corpus cache。
- 默认计时是一次独立进程的端到端 wall time；Alpha 案例固定报告 7 次运行的中位数。两者都包括 Python 启动和模型读取，不是微基准，更不是设备推理延迟。
- 普通竞品矩阵不跨平台比较 RSS；规模证据只在 Linux 上通过 GNU time 采集 EdgeFit 子进程峰值 RSS，其他平台的缺失值保持为空，不能伪装成零。

EdgeFit 的 activation planner 使用增量 live-byte 计数和按 offset/size
建立双索引的 best-fit free list。设 profile 规则数为 `P`、张量数为 `T`、
shape 总维数为 `D`、symbol bound 数为 `S`、graph boundary 出现次数为 `B`、节点数为 `N`、
输入/输出出现次数为 `E/O`、arena 事件数为 `A`、bounded/unresolved 事实数为 `U`、trace 记录数为 `R`，
规划目标复杂度为期望
`O(P + T + D log(S + 2) + B + N + E + O + A log A + U log T + R)`，空间为
`O(P + T + A + R + delta)`，其中 `delta` 是单节点最大输入输出数；
trace 按执行顺序生成，Top contributors 只保留前八项。热路径不再在每个节点
重扫全部存活张量。该复杂度仍是源码结构结论；100K 线性图结果只提供一个托管
环境观测点，不能代替任意拓扑的渐近证明，更不能表述成设备吞吐或推理时延。

## 判定标准

这一阶段不设置“EdgeFit 总分”，而输出四类结论：

1. **已被上游做得更好**：例如 ORT 专属 EP 覆盖或 onnx-tool MACs，应复用或明确不竞争。
2. **EdgeFit 独立价值**：target profile、稳定诊断 ID、预算 pass/fail、snapshot/diff、SARIF。
3. **需要校准**：EdgeFit activation estimate 与后续设备或厂商分析器之间的误差。
4. **应删除或后置**：没有可验证差异、维护成本却持续增长的功能。

## 第二阶段边界

ST Edge AI CLI 和 Edge Impulse 更接近设备级真值，但需要供应商工具、账户、格式转换或云端流程。第二阶段应将其输出作为人工核验依据，不在第一版编排器里加入未经验证的自动适配器。

上游依据：

- ORT Mobile Model Usability Checker：<https://onnxruntime.ai/docs/tutorials/mobile/helpers/model-usability-checker.html>
- onnx-tool CLI 与 profiling：<https://github.com/ThanatosShinji/onnx-tool>
- ST Edge AI `analyze`：<https://stedgeai-dc.st.com/assets/embedded-docs/command_line_interface.html>
- Edge Impulse target budget：<https://docs.edgeimpulse.com/studio/projects/dashboard/target-device>
