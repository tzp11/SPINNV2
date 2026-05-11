# SPINNV2 执行计划

## 1. 计划目标

本文档把 `architecture.md` 中的架构设计拆解为可执行任务、依赖关系、验证指标、测试集和集成流程。它的目标不是描述“想做什么”，而是回答：

1. 先做什么，后做什么。
2. 每个阶段完成到什么程度才算通过。
3. 每个任务的输入、输出、依赖是什么。
4. 哪些指标必须采集。
5. 用哪些模型、数据和测试验证正确性。
6. 如何把 compiler、format、runtime、kernel、codegen 集成成稳定闭环。

## 2. 总体路线

SPINNV2 的工程主线是：

```text
M0 工程骨架
  -> M1 最小推理闭环
  -> M2 静态内存规划
  -> M3 图优化
  -> M4 优化 kernel 与 backend
  -> M5 Codegen 与星载部署特性
  -> M6 论文实验与冻结
```

核心依赖关系：

```text
SIR 数据结构
  -> ONNX Importer
  -> Pass Pipeline
  -> SPK Writer
  -> Runtime Loader
  -> Reference Kernels
  -> E2E Numerical Validation
  -> Memory Planner
  -> KernelSpec / Target Profile
  -> Optimized Kernels
  -> Codegen
  -> Full Benchmark
```

任何性能优化都必须晚于数值正确性闭环。任何复杂部署特性都必须晚于 SPK 格式和 Runtime loader 稳定。

## 3. 阶段总览

| 阶段 | 目标 | 关键产物 | 阻塞后续内容 |
|---|---|---|---|
| M0 | 工程骨架与规格冻结初版 | 目录、CLI、CMake、SIR/SPK 草案、target profile schema | 全部 |
| M1 | ONNX 到 Runtime 的最小闭环 | MNIST/LeNet 可跑，fp32 reference kernels | M2/M3/M4 |
| M2 | 静态内存规划 | lifetime、naive/best-fit、memory_plan section、malloc 统计 | M4/M5/论文实验 |
| M3 | 图优化与转译优化 | Conv+BN、Conv+Relu、constant fold、dead elimination | M4 性能对比 |
| M4 | KernelSpec 与优化 kernel | target profile、backend hint、GEMM/Conv 优化、fallback | M5 codegen |
| M5 | 静态部署与星载特性 | SPK->C、checksum、external IO bind、platform abstraction | 论文系统展示 |
| M6 | 实验、文档、冻结 | benchmark 表、消融实验、论文图表、复现实验脚本 | 毕业材料 |

## 4. 任务拆分原则

### 4.1 模块边界

任务按以下模块拆分：

```text
compiler.ir        SIR 数据结构和图操作
compiler.frontend  ONNX 导入
compiler.passes    图优化和 lowering
compiler.planner   lifetime、memory plan、KernelSpec、backend binding
compiler.packager  SPK writer、debug JSON、CSV
runtime.core       loader、verifier、context、executor
runtime.memory     arena、scratch、malloc 统计
runtime.kernels    reference 与 optimized kernels
runtime.backend    kernel registry、fallback
codegen            SPK/SIR 到 C
tests              单元、数值、集成、系统测试
benchmarks         性能和内存实验
```

### 4.2 每个任务的完成定义

每个任务必须包含：

```text
输入: 依赖哪些文件、模型或接口
输出: 生成哪些代码、文档、测试或数据
验证: 通过哪些测试
指标: 需要采集哪些数值
```

### 4.3 不允许的推进方式

1. 不允许在没有 reference 数值闭环前写 SIMD kernel。
2. 不允许在 SPK loader 未稳定前反复改二进制格式而不保留版本号。
3. 不允许 Runtime 偷偷解析 ONNX 或执行图优化。
4. 不允许只测单个模型就宣称框架可用。
5. 不允许只报告平均时间，不报告内存和误差。

## 5. M0：工程骨架与规格初版

### 5.1 任务列表

| ID | 任务 | 依赖 | 输出 | 验收 |
|---|---|---|---|---|
| M0.1 | 建立目录结构 | 无 | `compiler/ runtime/ format/ tests/ benchmarks/ examples/ docs/` | 目录存在 |
| M0.2 | Python package 骨架 | M0.1 | `spinnv2.compiler` CLI | `python -m spinnv2.compiler --help` |
| M0.3 | Runtime CMake 骨架 | M0.1 | `runtime/CMakeLists.txt` | `cmake -S runtime -B build` |
| M0.4 | SIR spec 初版 | architecture.md | `format/sir_spec.md` | 覆盖 tensor/node/graph |
| M0.5 | SPK spec 初版 | architecture.md | `format/spk_format.md` | 覆盖 header/section/table |
| M0.6 | target profile schema | architecture.md | `compiler/target/profiles/cpu_ref.json` | `--print-target cpu_ref` |
| M0.7 | 测试基础设施 | M0.2/M0.3 | pytest + CTest | 空测试能跑通 |

### 5.2 关键决策

M0 必须冻结以下初版接口：

1. SIR tensor/node/graph 字段名。
2. SPK section ID 和 header magic。
3. Runtime 最小 API。
4. target profile JSON 字段。
5. debug JSON 基本结构。

允许后续扩展字段，但不应频繁重命名核心字段。

### 5.3 验收命令

```bash
python -m spinnv2.compiler --help
python -m spinnv2.compiler --print-target cpu_ref
pytest tests/compiler
cmake -S runtime -B build
cmake --build build
ctest --test-dir build
```

## 6. M1：最小推理闭环

### 6.1 目标

完成最小链路：

```text
ONNX -> SIR -> SPK -> Runtime -> Output
```

只支持 fp32 和 reference kernel。

M1 的 Runtime 可以采用顺序 arena 分配所有非 weight tensor。M2 已完成生命周期复用和 Memory Plan Section 后，Runtime 必须改为按 compiler 生成的 offset 绑定 tensor。

### 6.2 算子范围

必须支持：

```text
Conv
Relu
MaxPool
Flatten
Gemm
Softmax
```

建议同时支持：

```text
Add
Mul
Reshape
Transpose
```

### 6.3 任务列表

| ID | 任务 | 依赖 | 输出 | 验收 |
|---|---|---|---|---|
| M1.1 | SIR Python 数据结构 | M0.4 | `compiler/ir/*.py` | 构造小图单测 |
| M1.2 | ONNX importer 最小实现 | M1.1 | ONNX -> SIR | MNIST ONNX 导入成功 |
| M1.3 | 静态 shape 推断 | M1.2 | tensor shape 完整 | 无 unknown shape |
| M1.4 | SPK writer 最小实现 | M0.5/M1.1 | `.spk` | section bounds 单测 |
| M1.5 | SPK debug JSON | M1.4 | `.spk.json` | 包含 tensors/nodes |
| M1.6 | Runtime loader/verifier | M1.4 | C loader | 错误 header/section 单测 |
| M1.7 | Runtime executor | M1.6 | 顺序执行器 | 单节点 Add/Gemm 测试 |
| M1.8 | Reference kernels | M1.7 | Conv/Relu/Pool/Gemm/Softmax | kernel 单测 |
| M1.9 | 数值对齐工具 | M1.2/M1.8 | ORT vs SPINNV2 脚本 | 输出误差报告 |
| M1.10 | E2E MNIST | M1.1-M1.9 | 完整闭环 | ORT 数值对齐 |

### 6.4 M1 验收指标

| 指标 | 目标 |
|---|---|
| `max_abs_error` | fp32 下小模型通常 `< 1e-4`，Softmax 后可放宽到 `< 1e-3` |
| `mean_abs_error` | `< 1e-5` 或按模型设置 |
| `top1_equal` | 分类模型应一致 |
| SPK section 越界检查 | 100% 覆盖异常路径 |
| 不支持 op 行为 | compiler 报错，不进入 runtime |

### 6.5 M1 测试模型

| 模型 | 目的 | 输入 |
|---|---|---|
| `linear_gemm.onnx` | Gemm / Softmax | 固定随机向量 |
| `tiny_conv.onnx` | Conv / Relu | `1x1x8x8` |
| `mnist_cnn.onnx` | 最小完整 CNN | MNIST 样本或随机输入 |
| `lenet.onnx` | 稍复杂 CNN | MNIST 样本 |

M1 不需要真实数据集全量评测，优先使用固定随机输入和少量样本做数值一致性。

## 7. M2：静态内存规划

### 7.1 目标

把 activation tensor 的内存分配从 runtime 动态行为变成 compiler 静态产物。

核心结果：

```text
memory_plan.csv
Memory Plan Section
activation_arena_bytes
scratch_arena_bytes
runtime activation malloc count = 0
```

### 7.2 任务列表

| ID | 任务 | 依赖 | 输出 | 验收 |
|---|---|---|---|---|
| M2.1 | producer/consumer 分析 | M1.1 | tensor users | 小图单测 |
| M2.2 | lifetime analysis | M2.1 | first_use/last_use | 输出 tensor 延长到 graph end |
| M2.3 | naive planner | M2.2 | naive bytes | baseline 正确 |
| M2.4 | best-fit planner | M2.2 | offset plan | 无生命周期重叠冲突 |
| M2.5 | IO allocation policy | M2.4 | alloc_input/output 配置 | external IO 不占 arena |
| M2.6 | memory_plan section 写入 | M2.4 | SPK 内存表 | Runtime 可读取 |
| M2.7 | Runtime arena bind | M2.6 | tensor data pointer | 不越界 |
| M2.8 | malloc 统计 | M2.7 | malloc count | run 阶段 activation malloc = 0 |
| M2.9 | memory_plan.csv/debug JSON | M2.4 | CSV/JSON | 可生成论文表 |
| M2.10 | memory budget check | M2.4/M0.6 | 超限拒绝编译 | target 限制测试通过 |

### 7.3 内存正确性测试

必须实现以下测试：

1. 两个生命周期不重叠 tensor 可以复用同一 offset。
2. 两个生命周期重叠 tensor 不能复用同一 offset。
3. 模型输出 tensor 不得被后续 tensor 覆盖。
4. external input/output 不进入 activation arena。
5. `activation_arena_bytes` 小于 target profile 上限。
6. Runtime prepare 时 arena 不足必须失败。
7. Run 阶段不发生 activation malloc。

### 7.4 M2 验收指标

| 指标 | 目标 |
|---|---|
| `planned_activation_bytes` | 小于 `naive_activation_bytes` |
| `runtime_activation_malloc_count` | 0 |
| `memory_plan_conflict_count` | 0 |
| `arena_oob_count` | 0 |
| `target_budget_violation` | compiler 阶段发现 |

M2 实现备注：

```text
Memory Plan Section 用于审计和调试。
Runtime 快路径直接使用 TensorRecord.data_offset 作为 arena offset。
权重 tensor 的 data_offset 仍表示 Weight section offset。
```

### 7.5 消融实验

M2 开始准备论文消融实验：

```text
Naive memory
BestFit memory
BestFit + external IO
BestFit + scratch arena
```

输出表：

```text
model, naive_bytes, bestfit_bytes, external_io_bytes, reduction_ratio
```

## 8. M3：图优化与转译优化

### 8.1 目标

通过 compiler pass 减少 runtime 工作量，体现“地面端复杂、星上端简单”的设计。

### 8.2 任务列表

| ID | 任务 | 依赖 | 输出 | 验收 |
|---|---|---|---|---|
| M3.1 | Pass manager | M1.1 | pass pipeline | pass 顺序可配置 |
| M3.2 | Constant folding | M3.1 | 常量子图替换 | 数值一致 |
| M3.3 | Identity/Dropout elimination | M3.1 | 删除无用节点 | 图连边正确 |
| M3.4 | Conv+BN fusion | M3.1/M1.8 | 融合 Conv 权重偏置 | 与 ORT 对齐 |
| M3.5 | Conv+Relu fusion | M3.1/M1.8 | fused activation attr | 节点减少 |
| M3.6 | Dead tensor/node elimination | M3.1 | 删除死节点 | 输出不变 |
| M3.7 | Pass report | M3.1 | pass_stats.json | 记录节点变化 |
| M3.8 | 优化前后对比脚本 | M3.2-M3.7 | benchmark/accuracy report | 可复现实验 |

### 8.3 Pass 单元测试图

| 测试图 | 验证内容 |
|---|---|
| `conv_bn.onnx` | BN 参数吸收到 Conv |
| `conv_relu.onnx` | ReLU 融合为 Conv attr |
| `const_add.onnx` | 常量折叠 |
| `identity_chain.onnx` | Identity 删除 |
| `dead_branch.onnx` | 无输出贡献分支删除 |

### 8.4 M3 验收指标

| 指标 | 目标 |
|---|---|
| `node_count_before/after` | 优化后不增加，特定测试应减少 |
| `max_abs_error` | 相对优化前 `< 1e-4` |
| `spk_size_before/after` | 不应异常增大 |
| `runtime_latency_before/after` | 不退化，融合场景应改善 |

### 8.5 M3 当前实现状态

M3 已实现为 compiler 默认开启的图优化 pipeline：

```text
EliminateIdentityDropout
  -> ConstantFold
  -> FuseConvBatchNorm
  -> FuseConvRelu
  -> EliminateDead
```

当前落地产物：

1. `compiler/passes/manager.py` 提供 pass registry、默认顺序和可配置 pipeline。
2. `compiler/passes/transforms.py` 实现 Identity/Dropout 删除、常量折叠、
   Conv+BN 融合、Conv+Relu 融合和 dead-node/tensor elimination。
3. `python -m spinnv2.compiler compile` 默认运行 M3 pipeline，并提供
   `--disable-passes`、`--pass-pipeline`、`--pass-stats-json`。
4. SPK debug JSON 写入 `metadata.pass_stats`，可用于节点数变化和 pass
   命中统计。
5. `benchmarks/compare_passes.py` 可生成优化前后对比报告；提供输入二进制时，
   还会通过 `spkv2_run` 采集 latency 和输出误差。
6. `tests/compiler/unit/test_passes.py` 覆盖 M3 pass 结构正确性。
7. `tests/e2e/test_m1_e2e.py` 增加 Conv+BN+Relu 融合后的 ORT 数值对齐。

边界说明：

1. M3 仍限定 fp32、固定 shape、当前 M1 runtime op 集。
2. compiler-only op 必须被 pass 消除后才能写入 SPK；否则 compiler 报错，
   不进入 runtime。
3. 大模型 benchmark 和论文级图表自动化放到 M6 继续扩展。

## 9. M4：KernelSpec、Target Profile 与优化 kernel

### 9.1 目标

建立从目标平台能力到 kernel 实现选择的路径，并引入第一批优化 kernel。

### 9.2 任务列表

| ID | 任务 | 依赖 | 输出 | 验收 |
|---|---|---|---|---|
| M4.1 | target profile parser | M0.6 | profile 对象 | schema 校验 |
| M4.2 | KernelSpec 生成 | M4.1/M1.1 | kernelspec section | SPK 可读 |
| M4.3 | Kernel registry | M1.7 | registry | hint 命中 |
| M4.4 | fallback 机制 | M4.3 | ref fallback | debug JSON 记录 |
| M4.5 | basic GEMM 优化 | M1.8 | cpu GEMM | 快于 reference |
| M4.6 | im2col Conv | M1.8/M4.5 | Conv kernel | 数值一致 |
| M4.7 | packed weight 可选 | M4.2/M4.5 | weight layout transform | 加载/运行正确 |
| M4.8 | SIMD kernel 可选 | M4.5 | avx2/neon 之一 | target profile 控制 |
| M4.9 | scratch 估算 | M4.2/M4.6 | scratch_arena_bytes | 不越界 |

### 9.3 Target Profile 测试

必须至少准备：

```text
cpu_ref.json
cpu_generic.json
cpu_x86_avx2.json 可选
cpu_arm_neon.json 可选
memory_limited_1mb.json
```

测试项：

1. 不支持 op 时 compiler 报错或 fallback。
2. 不支持 SIMD 时不得选择 SIMD kernel。
3. memory budget 不足时拒绝编译。
4. fallback 次数写入 debug JSON。

### 9.4 M4 验收指标

| 指标 | 目标 |
|---|---|
| `optimized_latency` | 小于 reference latency |
| `fallback_count` | 可解释，不能静默发生 |
| `scratch_oob_count` | 0 |
| `target_mismatch_count` | 0 |
| `numerical_error` | 满足 M1 阈值 |

### 9.5 M4 当前实现状态

M4 已实现 target profile -> KernelSpec -> runtime registry 的最小闭环。

当前落地产物：

1. `compiler/planner/kernel_spec.py` 生成 KernelSpec、scratch 估算和 fallback
   关系，并写入 `graph.metadata.kernel_specs`、`kernel_fallback_count` 和
   `scratch_arena_bytes`。
2. `cpu_ref.json` 保持全 reference；`cpu_generic.json` 选择 Conv
   `im2col_gemm` 和 Gemm `direct`；`memory_limited_1mb.json` 用于 memory/scratch
   budget 验证。
3. SPK 写入 KernelSpec Section，Node Table 写入 `kernel_spec_id` 和
   `scratch_bytes`，Header 写入 `scratch_arena_bytes`。
4. Runtime loader 解析 KernelSpec Section；prepare 阶段分配共享 scratch arena；
   executor 通过 registry 命中 selected kernel，无法命中时按
   `fallback_kernel_spec_id` 回退 reference。
5. 当前 CPU 优化 kernel 包括 direct Gemm 和 Conv im2col-style scratch kernel。
6. `benchmarks/compare_kernels.py` 可比较 `cpu_ref` 与 `cpu_generic` 的 SPK 大小、
   activation/scratch 内存、fallback 统计、可选 latency 和输出误差。
7. `tests/compiler/unit/test_kernel_spec.py` 覆盖 KernelSpec 选择、fallback 和
   scratch budget；`tests/e2e/test_m1_e2e.py` 覆盖 `cpu_generic` runtime 数值对齐。

边界说明：

1. M4 仍限定 fp32、NCHW、固定 shape、当前 M1/M3 runtime op 集。
2. packed weight 和 SIMD 属于 M4 可选项，当前保留为后续增强。
3. `optimized_latency < reference_latency` 通过 `benchmarks/compare_kernels.py`
   采集，具体论文级表格放到 M6 汇总。

## 10. M5：Codegen 与星载部署特性

### 10.1 目标

让模型不依赖文件加载即可部署，形成更贴近星载环境的静态执行形式。

### 10.2 任务列表

| ID | 任务 | 依赖 | 输出 | 验收 |
|---|---|---|---|---|
| M5.1 | SPK -> C codegen | M1/M2 | `model.c/model.h` | 可编译 |
| M5.2 | static arena | M2/M5.1 | 静态数组 | 无 malloc run |
| M5.3 | static const weights | M5.1 | `.rodata` 权重 | 输出一致 |
| M5.4 | generated main_test | M5.1 | 测试入口 | 可运行 |
| M5.5 | checksum | M1.4 | SPK 校验 | 损坏文件失败 |
| M5.6 | external IO bind | M2.5/M10 API | bind API | 无拷贝路径正确 |
| M5.7 | platform abstraction | M1.6 | posix/baremetal stub | 构建可切换 |
| M5.8 | int8 字段预留 | M13 | quant section | fp32 路径不受影响 |

### 10.3 验收指标

| 指标 | 目标 |
|---|---|
| generated model 编译 | 通过 |
| generated output error | 与 SPK Runtime 一致 |
| file IO dependency | codegen 模式无文件加载 |
| run 阶段 malloc | 0 |
| checksum corruption test | 必须失败 |

### 10.4 M5 当前实现状态

M5 已实现 SPK -> static C deployment 的最小闭环。

当前落地产物：

1. `python -m spinnv2.compiler codegen model.spk --out-dir ... --name ...`
   生成 `model.c`、`model.h`、`main_test.c` 和 `CMakeLists.txt`。
2. generated `model.c` 将完整 SPK package 嵌入为 `static const unsigned char[]`；
   权重随 SPK Weight Section 进入 `.rodata`。
3. generated `model.c` 使用静态 activation arena 和静态 scratch arena，并调用
   `spkv2_prepare_with_scratch`。
4. generated `model_run` 使用 `spkv2_bind_input` 和 `spkv2_bind_output`，验证
   external IO bind 的无模型文件加载路径。
5. SPK writer 写入 Checksum Section，Header `checksum_type=1`；runtime loader
   校验 checksum，损坏 SPK 会加载失败。
6. generated wrapper 也对 embedded SPK 做 checksum 校验，`model_init` 在校验失败
   时返回错误。
7. Runtime allocation 经过 `spkv2_platform_malloc/calloc/free`；默认
   `runtime/platform/platform.c` 是 libc 实现，CMake `SPKV2_PLATFORM_SOURCE`
   可替换为 bare-metal stub。
8. `tests/codegen/test_c_codegen.py` 覆盖 generated model 编译、运行、输出数值和
   checksum 损坏拒绝；`tests/e2e/test_m1_e2e.py` 覆盖普通 SPK runtime checksum
   损坏拒绝。

边界说明：

1. 当前 codegen 仍复用 runtime loader/executor/kernel registry，不生成每个 op 的
   fully inlined C 函数。
2. Persistent context/tensor state 仍由 runtime load 阶段分配；论文实验中的
   no-malloc 重点是 model file-free、static activation/scratch 和 run 阶段无动态
   allocation。
3. Quantization/INT8 仍为格式和字段预留，fp32 路径保持不受影响。

## 11. M6：论文实验与冻结

### 11.1 目标

把工程结果转成论文可用证据。M6 不再只停留在 toy/tiny 模型；必须把
ResNet101 和 YOLOv10n 纳入固定 shape fp32 大模型验证集，确保 compiler、
SPK、runtime reference kernel、静态内存规划和 ORT 对齐工具能承受真实模型。

### 11.2 实验矩阵

| 实验 | 对比项 | 输出 |
|---|---|---|
| 数值正确性 | ORT vs SPINNV2 | error table |
| 内存规划 | naive vs best-fit vs external IO | memory table / figure |
| 图优化 | no-pass vs pass | node count / latency / size |
| kernel 优化 | reference vs optimized | latency table |
| target profile | cpu_ref vs memory_limited | deploy success/fail report |
| codegen | SPK runtime vs generated C | latency / binary size / malloc |

### 11.3 当前 M6 执行状态

已实现：

1. 扩展 SIR/SPK/runtime reference 算子集合，覆盖 ResNet101 和 YOLOv10n
   所需的分类、张量搬运、reduce、resize、TopK/Gather 后处理算子。
2. SPK minor version 更新到 `0.2`，Attribute Record 增加通用 extra 字段，
   用于保存 `Transpose.perm`、`Reduce*.axes`、`keepdims`、`TopK` 和 `Cast`
   参数。
3. `benchmarks/run_m6_models.py` 固化大模型 op 统计、编译、ORT baseline、
   runtime 运行和误差指标导出。
4. `scripts/export_paper_tables.py` 从 `m6_report.json` 导出 correctness、
   memory、op count 和 freeze 表格的 CSV/Markdown。
5. `scripts/check_reproducibility.py` 检查 M6 report 的模型、target、编译、
   runtime、内存收益和数值阈值。
6. `benchmarks/run_all.py` 串联 runtime build、pytest、CTest、M6 benchmark、
   论文表格导出和复现检查。

当前结果：

| 模型 | 编译 | runtime | 内存规划 | ORT 对齐 |
|---|---|---|---|---|
| ResNet101 | 成功，241 nodes | 60.91s | 137803680 -> 14249984 bytes | top1 一致，max_abs=0.0546875，mean_abs=0.00544044 |
| YOLOv10n | 成功，308 nodes | 30.44s | 307286000 -> 24576000 bytes | score max_abs=5.46e-08；top10 max_abs=6.26e-04，class 10/10 |

YOLOv10n 全量 300 行输出中，低置信度候选的 TopK 排序会因 fp32 reference
累积误差出现候选框/类别分叉；M6 论文实验应同时报告 score 误差、topN 高置信
行误差、class match count 和全量 max/mean error，避免只用单个全量 max_abs
掩盖实际行为。

正式 M6 结果来源：

```text
build/m6_final/m6_report.json
build/m6_paper_tables/correctness.csv
build/m6_paper_tables/memory.csv
build/m6_paper_tables/ops.csv
build/m6_paper_tables/freeze.csv
```

### 11.4 冻结标准

进入论文写作前，必须冻结：

1. SIR/SPK 版本号。
2. Runtime API。
3. 支持算子列表。
4. 实验模型列表。
5. benchmark 脚本。
6. 误差阈值。
7. 所有论文图表数据来源。

## 12. 测试验证集设计

### 12.1 测试分层

```text
Unit Tests:
    单个函数、单个 pass、单个 kernel。

Numerical Tests:
    ONNX Runtime 与 SPINNV2 输出对齐。

Format Tests:
    SPK header、section、越界、版本、checksum。

Memory Tests:
    lifetime、offset、arena、scratch、malloc count。

Integration Tests:
    ONNX -> SIR -> SPK -> Runtime。

System Tests:
    多模型 benchmark、codegen、target profile。
```

### 12.2 模型验证集

| 层级 | 模型 | 用途 | 阶段 |
|---|---|---|---|
| Toy | `add.onnx` | loader/executor 基础 | M1 |
| Toy | `gemm_softmax.onnx` | Gemm/Softmax | M1 |
| Toy | `conv_relu.onnx` | Conv/Relu/kernel | M1/M3 |
| Toy | `conv_bn_relu.onnx` | 融合 pass | M3 |
| Small | `mnist_cnn.onnx` | 最小端到端 | M1 |
| Small | `lenet.onnx` | CNN 完整链路 | M1/M2 |
| Medium | `resnet18.onnx` | 内存和性能评估 | M2-M4 |
| Medium | `mobilenetv2.onnx` | depthwise/轻量模型扩展 | M4 可选 |
| Detection | `yolo_tiny_prenms.onnx` | 检测前处理/后处理前子图 | M5/M6 可选 |

### 12.3 输入数据集

| 数据 | 用途 |
|---|---|
| 固定随机输入 | 每次测试可复现，覆盖所有模型 |
| MNIST 小样本 100 张 | 分类正确性和 top1 |
| ImageNet 样本 50-100 张 | ResNet/MobileNet smoke test |
| 合成边界输入 | NaN/Inf/极大极小值可选，用于 kernel 鲁棒性 |

第一阶段不需要完整训练集或完整精度评估。论文中如果要报告分类准确率，应明确是 smoke test 还是全量验证。

### 12.4 数值阈值

| 场景 | 阈值建议 |
|---|---|
| fp32 elementwise | `max_abs_error < 1e-5` |
| fp32 Conv/Gemm | `max_abs_error < 1e-4` |
| Softmax 输出 | `max_abs_error < 1e-3` |
| 融合前后 | `max_abs_error < 1e-4` |
| int8 可选 | 按量化误差单独设阈值 |

所有阈值必须写进测试配置，不能散落在脚本里。

### 12.5 当前测试验证集执行状态

M6 已将 12.2 中的模型验证集落成可执行生成器和脚本：

```text
tests/e2e/model_zoo.py
tests/e2e/run_e2e.py
```

当前 `tests/e2e/run_e2e.py --all` 已覆盖：

| 层级 | 当前执行模型 | 说明 |
|---|---|---|
| Toy | `add`, `gemm_softmax`, `conv_relu`, `conv_bn_relu` | 生成固定 shape ONNX，覆盖基础 executor、Gemm/Softmax、Conv/Relu、Conv+BN+Relu fusion |
| Small | `mnist`, `lenet` | 生成固定 shape CNN，覆盖最小端到端与稍复杂 CNN |
| Medium | `resnet18`, `mobilenetv2` | 生成 ResNet-like residual block 与 MobileNet-like depthwise block，用于内存和 depthwise smoke |
| Detection | `yolo_tiny_prenms` | 生成检测前 NMS 风格子图，覆盖 Sigmoid/Mul/Concat/Transpose |

这些是可复现的固定输入 smoke/e2e 模型，不是 MNIST/ImageNet 真实数据集准确率评估。
真实 MNIST 100 张和 ImageNet 50-100 张仍属于后续数据集级精度实验，不作为当前
M6 冻结门槛。

最新执行结果：

```text
python tests/e2e/run_e2e.py --all --out-dir build/e2e_all 通过。
build/e2e_all/e2e_report.json 记录每个模型的 SPK 大小、内存规划和 ORT 误差。
```

## 13. 指标体系

### 13.1 正确性指标

```text
max_abs_error
mean_abs_error
max_rel_error
cosine_similarity
top1_equal
unsupported_op_count
```

### 13.2 内存指标

```text
naive_activation_bytes
planned_activation_bytes
memory_reduction_ratio
scratch_arena_bytes
persistent_bytes
weight_bytes
runtime_malloc_count_load
runtime_malloc_count_prepare
runtime_malloc_count_run
arena_oob_count
```

### 13.3 性能指标

```text
load_time_ms
prepare_time_ms
run_time_ms_avg
run_time_ms_p50
run_time_ms_p90
run_time_ms_p99
throughput_fps
kernel_time_breakdown
```

### 13.4 部署指标

```text
spk_size_bytes
generated_c_size_bytes
runtime_binary_size_bytes
target_profile_name
kernel_fallback_count
checksum_enabled
external_io_enabled
```

## 14. 集成测试设计

### 14.1 每次提交必须跑的快速测试

```text
pytest tests/compiler/unit
pytest tests/compiler/format
ctest --test-dir build -R runtime_unit
python tests/e2e/run_e2e.py --model toy
```

目标耗时应控制在 1-3 分钟内。

### 14.2 每日或阶段性完整测试

```text
python tests/e2e/run_e2e.py --all-small
python benchmarks/run_memory.py --models mnist,lenet,resnet18
python benchmarks/run_latency.py --models mnist,lenet
python tests/codegen/run_codegen_test.py --models mnist,lenet
```

### 14.3 Release 前测试

```text
python tests/e2e/run_e2e.py --all
python benchmarks/run_all.py
python scripts/export_paper_tables.py
python scripts/check_reproducibility.py
```

Release 前必须重新生成论文表格，避免手工数据和脚本结果不一致。

### 14.4 当前集成测试执行状态

快速测试已执行：

```text
pytest tests/compiler/unit tests/compiler/format
ctest --test-dir build/runtime -R runtime_unit
python tests/e2e/run_e2e.py --model toy --out-dir build/e2e_toy
```

阶段性完整测试已执行：

```text
python tests/e2e/run_e2e.py --all-small --out-dir build/e2e_all_small
python benchmarks/run_memory.py --models mnist,lenet,resnet18 --out-dir build/memory_benchmark
python benchmarks/run_latency.py --models mnist,lenet --out-dir build/latency_benchmark --runs 3
python tests/codegen/run_codegen_test.py --models mnist,lenet --out-dir build/codegen_validation
```

Release 前测试已执行：

```text
python tests/e2e/run_e2e.py --all --out-dir build/e2e_all
python benchmarks/run_all.py --out-dir build/m6_release_final --tables-dir build/m6_release_tables
python scripts/export_paper_tables.py build/m6_release_final/m6_report.json --out-dir build/m6_release_tables
python scripts/check_reproducibility.py build/m6_release_final/m6_report.json
```

`benchmarks/run_all.py` 当前会串联 runtime build、pytest、CTest、small e2e、
memory benchmark、latency benchmark、codegen validation、ResNet101/YOLOv10n
M6 benchmark、论文表格导出和复现检查。

## 15. 任务依赖图

```text
M0.1 Directory
  -> M0.2 CLI
  -> M1.1 SIR
      -> M1.2 ONNX Importer
      -> M1.3 Shape Infer
      -> M1.4 SPK Writer
          -> M1.6 Runtime Loader
              -> M1.7 Executor
                  -> M1.8 Reference Kernels
                      -> M1.10 E2E MNIST
                          -> M2 Memory Planning
                          -> M3 Graph Passes
                              -> M4 KernelSpec / Optimized Kernels
                                  -> M5 Codegen
                                      -> M6 Paper Experiments
```

并行可做的任务：

```text
SIR spec 与 Runtime CMake 可并行。
Reference kernel 单测可与 SPK writer 并行。
target profile schema 可与 M1 闭环并行。
benchmark 脚本可从 M1 后逐步完善。
docs 和论文图表脚本可从 M2 开始维护。
```

不建议并行的任务：

```text
SPK 格式未稳定前，不应大规模写 codegen。
Reference kernel 未正确前，不应写 SIMD kernel。
Memory planner 未验证前，不应做 external IO bind。
Target profile 未稳定前，不应做复杂 backend fallback。
```

## 16. 风险与降级方案

| 风险 | 影响 | 降级方案 |
|---|---|---|
| ONNX importer 复杂度过高 | M1 延迟 | 只支持固定模型导出的有限 op |
| Conv 实现性能差 | M4 指标差 | 先强调内存/确定性，性能只与 reference 对比 |
| YOLO 后处理 TopK 对微小误差敏感 | 全量输出 max_abs 偏大 | 同时报告 score/topN/class match，并保留 ORT 中间输出定位脚本 |
| int8 来不及 | 低比特内容不足 | 只保留格式预留和相关工作分析 |
| Codegen 复杂 | M5 延迟 | 只生成单模型静态 C，不支持多模型 |
| SIMD 不稳定 | 数值风险 | SIMD 作为可选扩展，论文主线不依赖 |

## 17. 推荐实际推进顺序

最推荐的开发顺序：

```text
1. 建 M0 骨架。
2. 写 SIR 数据结构和 toy graph 单测。
3. 写 SPK 最小 writer 和 loader。
4. 写 Add/Gemm/Relu reference kernel，先跑 toy。
5. 写 Conv/Pool/Softmax，跑 MNIST。
6. 补 ORT 数值对齐工具。
7. 做 lifetime + naive planner。
8. 做 best-fit planner 和 arena bind。
9. 做 Conv+BN/Relu fusion。
10. 做 target profile + KernelSpec。
11. 做 GEMM/Conv 优化。
12. 做 codegen。
13. 跑完整 benchmark 和论文图表。
```

这条路径的优点是每一步都有可运行结果，不会长期停留在抽象设计阶段。
