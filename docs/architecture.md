# SPINNV2 设计文档

## 1. 项目定位

SPINNV2 是面向星载平台的轻量可移植推理框架与模型转译工具链。它不以替代 ONNX Runtime、OpenVINO、MNN 等通用推理框架为目标，而是面向星载任务的约束条件，设计一条从通用模型到确定性部署包的转译与执行路径。

核心思想是：

```text
地面端做复杂工作，星上端做确定执行。
```

也就是说，模型解析、图优化、算子规范化、静态 shape 推断、内存规划、后端选择和部署包生成都在地面端完成；星上运行时只负责加载、校验、绑定内存和按执行计划调用 kernel。

### 1.1 参考范式

SPINNV2 的设计参考多个成熟框架的成功范式，但不 fork 任何一个框架：

| 参考对象 | 可借鉴范式 | SPINNV2 中的落点 |
|---|---|---|
| ONNX Runtime | Execution Provider 能力查询与子图分配 | compiler 读取 target profile，提前生成 backend hint |
| ExecuTorch | AOT export、delegate lowering、fixed memory arenas | SIR -> SPK 前执行 memory planning 和 backend binding |
| TensorFlow Lite Micro | 单一 tensor arena、有限内存推理 | 星上 Runtime 使用静态 activation/scratch arena |
| TVM | 高层图 IR + 低层 tensor/kernel 表示 | SIR Graph IR + KernelSpec 双层描述 |
| ncnn / MNN / Tengine | 轻量格式、离线优化、端侧 kernel 注册 | SPK 容器、pass pipeline、kernel registry |
| ggml | 运行期零动态分配、模型容器与张量底座分离 | Runtime 和 kernel library 分层，推理阶段避免 activation malloc |

因此 SPINNV2 的核心路线不是“做一个小型 ONNX Runtime”，而是“做一个面向星载部署的 AOT 推理编译与轻量执行系统”。

## 2. 设计目标

### 2.1 功能目标

1. 支持从 ONNX 模型转译为 SPINNV2 自定义中间表示 SIR。
2. 支持将 SIR 编译为星载部署包 SPK。
3. 支持一个轻量 C Runtime 加载 SPK 并执行推理。
4. 支持静态 shape CNN / 轻量检测网络的最小闭环。
5. 支持地面端生成内存规划结果，使运行时避免复杂动态分配。
6. 支持 Reference C kernel 和优化 kernel 共存。
7. 支持将模型生成独立 C 代码，便于无文件系统或 ROM 化部署。
8. 支持目标平台描述文件，使转译结果与平台内存、指令集和后端能力绑定。

### 2.2 非功能目标

1. 运行时小型化：核心 Runtime 应保持低依赖、低代码量。
2. 可移植性：Runtime 优先使用 C99/C11，平台相关能力放入 platform 层。
3. 确定性：推理过程的执行顺序、内存占用、kernel 选择尽量在编译期确定。
4. 可验证性：转译结果应能和 ONNX Runtime 做数值对齐。
5. 可裁剪性：不同平台可只编译所需 kernel 和后端。
6. 可解释性：模型包应可导出 JSON debug 信息，便于论文实验和问题定位。

### 2.3 非目标

1. 不追求完整支持 ONNX 全算子。
2. 不在运行时实现复杂图优化 pass。
3. 不做动态 shape 通用推理框架。
4. 不做训练。
5. 不在第一阶段支持复杂异步调度、多模型并发或跨设备自动切分。
6. 不把大框架源码 fork 为主体。
7. 不在第一阶段实现 int4/fp8/fp4 等激进低比特路径，只在格式和 IR 中预留量化扩展。

## 3. 约束场景

SPINNV2 面向的星载部署场景有以下假设：

1. 模型通常在地面端训练和转译，星上端只执行推理。
2. 星上计算资源、内存资源和功耗预算受限。
3. 运行时环境可能缺少完整操作系统能力。
4. 模型输入 shape 多数可固定。
5. 部署前可以进行充分验证。
6. 推理过程需要可预测的内存峰值和运行行为。
7. 相比动态灵活性，更看重稳定性、可移植性和可审查性。

## 4. 总体架构

```text
              ┌────────────────────────────┐
              │  PyTorch / TensorFlow      │
              │  Exported ONNX Model       │
              └──────────────┬─────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                    SPINNV2 Compiler                       │
│                                                          │
│  Frontend  ->  SIR Builder  ->  Pass Manager             │
│      │             │              │                      │
│      │             │              ├─ Normalize           │
│      │             │              ├─ Shape Infer         │
│      │             │              ├─ Constant Fold       │
│      │             │              ├─ Op Fusion           │
│      │             │              ├─ Op Lowering         │
│      │             │              ├─ Layout Lowering     │
│      │             │              └─ Dead Tensor Remove  │
│      │             │                                     │
│      └─────────────┴─> Target Profile Loader             │
│                         Memory Planner                    │
│                         KernelSpec Selector               │
│                         Backend Binder                    │
│                         Package Writer                    │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │   SPK Model Pack   │
                    └─────────┬──────────┘
                              │
            ┌─────────────────┴────────────────┐
            ▼                                  ▼
┌────────────────────────────┐      ┌────────────────────────────┐
│      SPINNV2 Runtime       │      │       SPINNV2 Codegen      │
│                            │      │                            │
│  Load / Verify             │      │  SPK / SIR -> C source     │
│  Bind Arena                │      │  Static weights            │
│  Execute Plan              │      │  Static arena              │
│  Dispatch Kernel           │      │  model_run()               │
└────────────────────────────┘      └────────────────────────────┘
```

## 5. 目录规划

```text
SPINNV2/
├── compiler/
│   ├── frontend/
│   │   └── onnx_importer.py
│   ├── ir/
│   │   ├── graph.py
│   │   ├── node.py
│   │   ├── tensor.py
│   │   └── types.py
│   ├── passes/
│   │   ├── normalize.py
│   │   ├── shape_infer.py
│   │   ├── constant_fold.py
│   │   ├── fuse_conv_bn.py
│   │   ├── fuse_activation.py
│   │   ├── lower_ops.py
│   │   └── eliminate_dead.py
│   ├── planner/
│   │   ├── lifetime.py
│   │   ├── memory_plan.py
│   │   ├── kernel_spec.py
│   │   └── backend_bind.py
│   ├── target/
│   │   ├── profile.py
│   │   └── profiles/
│   │       ├── cpu_ref.json
│   │       ├── cpu_x86_avx2.json
│   │       └── cpu_arm_neon.json
│   ├── packager/
│   │   ├── spk_writer.py
│   │   └── debug_export.py
│   └── codegen/
│       └── c_codegen.py
│
├── runtime/
│   ├── include/
│   │   ├── spkv2_runtime.h
│   │   ├── spkv2_format.h
│   │   ├── spkv2_kernel.h
│   │   └── spkv2_platform.h
│   ├── core/
│   │   ├── loader.c
│   │   ├── verifier.c
│   │   ├── context.c
│   │   └── executor.c
│   ├── memory/
│   │   ├── arena.c
│   │   └── scratch.c
│   ├── kernels/
│   │   ├── reference/
│   │   └── optimized/
│   ├── backend/
│   │   ├── cpu_ref.c
│   │   ├── cpu_simd.c
│   │   └── delegate.h
│   └── platform/
│       ├── posix.c
│       └── baremetal_stub.c
│
├── format/
│   ├── spk_format.md
│   └── sir_spec.md
│
├── tests/
│   ├── compiler/
│   ├── runtime/
│   └── numerical/
│
├── benchmarks/
├── examples/
└── docs/
```

第一阶段可以不完整建立所有目录，但最终论文工程应尽量保持这个边界。

## 6. SIR 中间表示设计

SIR 是 Satellite Inference IR 的简称，是 SPINNV2 的核心中间表示。它不是完整 ONNX，而是面向部署的受限静态图 IR。

### 6.1 SIR 设计原则

1. 静态 shape 优先。
2. 显式 dtype。
3. 显式 layout。
4. 权重和激活分离。
5. op attribute 结构化。
6. producer / consumer 关系可直接分析。
7. 允许记录优化和后端信息，但不污染 op 语义。

### 6.2 Tensor 描述

每个 tensor 至少包含：

```text
id              全局唯一 ID
name            调试名
dtype           f32 / f16 / i8 / u8 / i32 等
shape           静态维度
layout          NCHW / NHWC / NC4HW4 / PACKED 等
role            input / output / weight / activation / constant
size_bytes      字节数
producer        生产该 tensor 的 node id
consumers       消费该 tensor 的 node id 列表
quant           可选量化参数
memory          编译后填充的内存规划结果
```

### 6.3 Node 描述

每个 node 至少包含：

```text
id
op_type
inputs
outputs
attrs
domain
execution_index
backend_hint
flags
```

`attrs` 必须转成 SPINNV2 自己的结构，不直接保存 ONNX AttributeProto。这样 Runtime 不需要理解 ONNX。

### 6.4 Graph 描述

Graph 至少包含：

```text
model_name
opset_source
inputs
outputs
tensors
nodes
initializers
metadata
```

Graph 在 compiler 内存中可以用 Python 对象表示，写入 SPK 时转为紧凑二进制表。

### 6.5 KernelSpec 描述

SIR 表达图语义，KernelSpec 表达某个 node 在目标平台上的实现策略。二者需要分离，避免把平台特化逻辑写进通用 IR。

每个 KernelSpec 至少包含：

```text
node_id
op_type
kernel_kind       reference / direct / im2col_gemm / packed_gemm / delegate
backend           ref / cpu / simd / delegate
dtype
layout
weight_layout
scratch_bytes
workspace_policy
required_features
fallback_kernel
```

示例：

```text
Conv node:
    SIR op_type      = Conv
    KernelSpec       = im2col_gemm
    backend          = cpu_simd
    weight_layout    = OIHW_PACKED_OC8
    scratch_bytes    = im2col_tile_bytes
```

第一阶段可以只生成 `reference` 和 `direct` 两类 KernelSpec；第二阶段再加入 `im2col_gemm`、`packed_gemm` 和 SIMD 变体。

### 6.6 Target Profile

Target Profile 是 SPINNV2 的目标平台描述文件，用于把“可移植”具体化。Compiler 不应该只根据宿主机能力做决定，而应该根据显式目标平台文件生成 SPK。

建议使用 JSON：

```json
{
  "name": "cpu_x86_avx2",
  "word_size": 64,
  "endianness": "little",
  "alignment": 32,
  "memory": {
    "activation_arena_max": 67108864,
    "scratch_arena_max": 16777216,
    "allow_runtime_malloc": false
  },
  "features": ["fp32", "int8", "avx2", "fma"],
  "layouts": ["NCHW", "PACKED_OC8"],
  "backends": ["ref", "cpu", "cpu_simd"],
  "ops": {
    "Conv": ["ref", "direct", "im2col_gemm"],
    "Gemm": ["ref", "packed_gemm"],
    "Relu": ["ref", "simd"]
  }
}
```

Compiler 根据 Target Profile 完成：

1. 判断模型是否可部署到目标平台。
2. 选择 kernel kind 和 backend hint。
3. 检查 activation/scratch 内存上限。
4. 决定权重 layout 预转换策略。
5. 决定是否允许 external IO bind、mmap 权重、SIMD kernel。

Runtime 只校验 SPK 中记录的 target 信息和当前编译进来的 kernel 是否匹配，不做复杂平台推断。

## 7. 支持算子范围

### 7.1 第一阶段算子

第一阶段目标是跑通 MNIST、小型 CNN、ResNet 子集。

```text
Conv
Relu
MaxPool
AveragePool
Flatten
Gemm
MatMul
Add
Mul
Softmax
Reshape
Transpose
```

### 7.2 第二阶段算子

第二阶段目标是支持 ResNet101 和 YOLOv10n 这类固定 shape fp32
分类/检测模型。当前实现已把以下算子纳入 SIR/SPK/runtime reference
路径：

```text
Cast
Concat
Div
GatherElements
Mod
ReduceMax
ReduceMean
Resize
Sigmoid
Split
Sub
Tile
TopK
Unsqueeze
```

其中 YOLOv10n 后处理中的整数索引张量在当前 fp32-only SIR 内以 float
承载整数值，并在 `TopK`、`GatherElements`、`Div`、`Mod` 等 runtime kernel
中按整数语义解释。后续如果扩展 dtype table，应把这一路径替换为显式 int64
tensor。

### 7.3 算子支持策略

对每个 ONNX op，转译器必须给出三种结果之一：

```text
SUPPORTED       可直接转为 SIR op
LOWERED         可分解为多个 SIR primitive op
REJECTED        不支持，编译期报错
```

禁止在 Runtime 中临时解释未知 op。

## 8. Compiler 流程设计

### 8.1 总流程

```text
load_onnx()
  -> load_target_profile()
  -> import_to_raw_graph()
  -> build_sir()
  -> run_pass_pipeline()
  -> plan_memory()
  -> select_kernel_spec()
  -> bind_backend()
  -> write_spk()
  -> export_debug_json()
```

### 8.2 Pass Pipeline

建议初始 pipeline：

```text
1. NormalizeNames
2. InferStaticShape
3. ExtractInitializers
4. NormalizeConvAttrs
5. FoldConstant
6. FuseConvBatchNorm
7. FuseConvActivation
8. LowerGemmToMatMulAdd 或保留 Gemm
9. LowerGlobalPool
10. EliminateIdentity
11. EliminateDeadTensor
12. AssignExecutionOrder
13. AnalyzeLifetime
14. PlanActivationMemory
15. EstimateScratchMemory
16. SelectKernelSpec
17. BindBackendKernel
18. CheckTargetMemoryBudget
```

M3 当前实现采用一个较小但可验证的默认 pipeline：

```text
EliminateIdentityDropout
  -> ConstantFold
  -> FuseConvBatchNorm
  -> FuseConvRelu
  -> EliminateDead
```

`python -m spinnv2.compiler compile` 默认启用该 pipeline，并提供
`--disable-passes`、`--pass-pipeline` 和 `--pass-stats-json` 用于关闭、
调整顺序和导出 pass 统计。SPK debug JSON 中也会包含 `pass_stats`。
未被 pass 消除的 compiler-only op 不允许进入 runtime packaging，compiler
会在写 SPK 前报错。

### 8.3 图优化边界

Compiler 可以做：

1. Conv + BN 融合。
2. Conv + Relu / Clip 融合。
3. 常量折叠。
4. Identity / Dropout 删除。
5. 静态 Reshape / Flatten 规范化。
6. 权重 layout 预转换。
7. 量化参数固化。
8. 根据目标平台能力选择 kernel 实现策略。
9. 生成内存规划和部署检查报告。

Compiler 第一阶段不做：

1. 自动混合精度搜索。
2. 动态 shape 优化。
3. 大规模子图调度搜索。
4. JIT kernel 生成。
5. 复杂算子自动代数化简。
6. 运行时自动分图和动态后端搜索。

## 9. SPK 模型包设计

SPK 是 SPINNV2 的部署格式。它应当是一个 section 化二进制容器。

### 9.1 Section 布局

```text
SPK Header
Section Directory
Model Metadata Section
Target Profile Section
Tensor Table Section
Node Table Section
Attribute Section
Weight Section
Memory Plan Section
KernelSpec Section
Backend Hint Section
Quantization Section 可选
String Table Section
Debug Section 可选
Checksum Section
```

### 9.2 Header 字段

Header 建议包含：

```text
magic
version_major
version_minor
endianness
header_size
section_count
model_flags
num_tensors
num_nodes
num_inputs
num_outputs
weight_bytes
activation_arena_bytes
scratch_arena_bytes
target_profile_hash
checksum_type
```

### 9.3 Tensor Table

Tensor table 记录：

```text
tensor_id
name_offset
dtype
layout
rank
shape
role
size_bytes
memory_class
memory_offset
quant_offset
```

### 9.4 Node Table

Node table 记录：

```text
node_id
op_type
flags
input_count
output_count
input_offset
output_offset
attr_offset
backend_hint
kernel_spec_id
scratch_bytes
```

### 9.5 Memory Plan Section

Memory plan section 记录：

```text
tensor_id
memory_class
offset
size
alignment
first_use
last_use
```

`memory_class` 包括：

```text
INPUT
OUTPUT
WEIGHT
ACTIVATION_ARENA
SCRATCH_ARENA
EXTERNAL
```

### 9.6 KernelSpec Section

KernelSpec section 记录 node 到 kernel 实现策略的编译期绑定结果：

```text
kernel_spec_id
node_id
kernel_kind
backend
dtype
layout
weight_layout
scratch_offset
scratch_bytes
fallback_kernel_spec_id
required_feature_mask
```

Runtime 根据 `kernel_spec_id` 找到已编译进来的 kernel。如果目标 kernel 不存在，可以按 `fallback_kernel_spec_id` 回退到 reference；如果 fallback 也不存在，则加载或 prepare 失败。

### 9.7 Quantization Section

第一阶段只实现 fp32，但 SPK 需要预留量化扩展。Quantization section 建议记录：

```text
tensor_id
quant_type        none / affine / per_channel / blockwise
storage_dtype     i8 / u8 / i4 / u4 / f8 / f4
compute_dtype     f32 / i32 / f16
scale_offset
zero_point_offset
axis
block_size
```

阶段策略：

```text
M1-M3: fp32 only
M4:    int8 weight / activation 可选
M5+:   int4 weight-only 或其他低比特预研
```

### 9.8 Debug JSON

除了二进制 SPK，compiler 应输出一个 debug JSON：

```text
model.spk.json
```

用于记录：

1. 原始 ONNX op 和 SIR op 映射。
2. pass 前后节点数量变化。
3. 每个 tensor 的生命周期。
4. 每个 tensor 的内存 offset。
5. 每个 node 的 backend 选择。
6. 编译期 warning。
7. Target profile 与模型需求的匹配情况。
8. KernelSpec 选择和 fallback 情况。

这对论文实验和错误定位很重要。

## 10. Runtime 设计

### 10.1 Runtime API

建议第一阶段 API：

```c
typedef struct Spkv2Context Spkv2Context;

int spkv2_load_file(const char *path, Spkv2Context **out_ctx);
int spkv2_load_memory(const void *data, size_t size, Spkv2Context **out_ctx);
int spkv2_verify(Spkv2Context *ctx);
int spkv2_prepare(Spkv2Context *ctx, void *arena, size_t arena_size);
int spkv2_set_input(Spkv2Context *ctx, int index, const void *data, size_t size);
int spkv2_bind_input(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_bind_output(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_get_output(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_run(Spkv2Context *ctx);
void spkv2_free(Spkv2Context *ctx);
```

`spkv2_set_input()` 表示 copy mode；`spkv2_bind_input()` 和 `spkv2_bind_output()` 表示 external bind mode。External bind mode 用于 DMA、共享内存、上下游模型串联等场景。

### 10.2 执行流程

```text
load
  -> parse header
  -> parse section directory
  -> verify section bounds
  -> bind tensor table
  -> bind node table
  -> bind weights

prepare
  -> allocate or accept activation arena
  -> allocate or accept scratch arena
  -> bind tensor data pointers
  -> initialize kernel registry
  -> verify backend hints
  -> verify target profile compatibility

run
  -> copy or bind input
  -> for node in execution order:
         collect input tensor pointers
         collect output tensor pointers
         call selected kernel
  -> output remains in output tensor memory
```

M1 实现说明：M1 Runtime 已实现二进制 SPK loader、顺序 executor 和 fp32 reference kernels。M2 已将 `prepare` 阶段切换为使用 compiler 写入的 arena offset，并通过 `activation_arena_bytes` 校验用户提供的 arena 大小。

### 10.3 Runtime 不应该做的事

Runtime 不应该：

1. 解析 ONNX。
2. 修改图结构。
3. 做复杂 shape inference。
4. 重新进行全图内存规划。
5. 搜索最优 kernel。
6. 动态注册未知算子。

这些事情都属于 compiler。

## 11. Backend 与 Kernel 设计

### 11.1 Kernel 注册项

```c
typedef int (*Spkv2KernelFn)(
    const Spkv2Tensor *inputs,
    int input_count,
    Spkv2Tensor *outputs,
    int output_count,
    const void *attrs,
    void *scratch);

typedef struct {
    uint16_t op_type;
    uint16_t backend;
    uint16_t dtype;
    uint16_t layout;
    uint16_t flags;
    int priority;
    Spkv2KernelFn fn;
} Spkv2KernelReg;
```

### 11.2 后端层次

```text
SPKV2_BACKEND_REF      纯 C reference
SPKV2_BACKEND_CPU      普通 CPU 优化
SPKV2_BACKEND_SIMD     SIMD 优化
SPKV2_BACKEND_DELEGATE 外部硬件委托
```

### 11.3 Kernel 选择原则

Compiler 负责优先选择：

```text
op_type + dtype + layout + target_platform -> backend_hint
```

Runtime 只做校验：

1. 如果 hint 对应 kernel 存在，使用 hint。
2. 如果不存在，fallback 到 reference kernel。
3. 如果 reference 也不存在，返回错误。

这能兼顾确定性和可移植性。

### 11.4 Runtime 与 Kernel Library 分层

SPINNV2 必须避免把执行器和算子实现耦合在一起。两者职责如下：

```text
Runtime:
    解析 SPK、校验 section、绑定 tensor、管理 arena、执行 node 顺序。

Kernel Library:
    提供 Conv/Gemm/Pool/Softmax 等具体 kernel。

Backend:
    把 KernelSpec 映射到具体 KernelFn，并处理 fallback。
```

这样可以达到三个目的：

1. Runtime 可以保持很小，适合星载移植和审查。
2. Kernel 可以独立替换，例如从 reference 切换到 SIMD。
3. 后续接入 DSP / FPGA / NPU delegate 时，不需要改模型包和执行器主体。

### 11.5 Target Capability 检查

Compiler 对每个 node 做能力检查：

```text
required = op_type + dtype + layout + kernel_kind
target_profile 是否支持 required
```

结果分三类：

```text
NATIVE      目标后端直接支持
FALLBACK    使用 reference 或更通用 kernel
REJECTED    目标平台无法部署，编译期报错
```

这相当于把 ORT EP / ExecuTorch delegate 的能力查询前移到地面端，避免星上 Runtime 动态分图。

## 12. 内存管理设计

### 12.1 是否需要自定义内存管理

SPINNV2 应该做自定义内存管理，但不应该一开始做复杂通用内存池。

推荐策略是：

```text
不要做一个替代 malloc 的通用 allocator。
要做一个面向推理图的静态 activation arena + scratch arena。
```

原因：

1. 星载推理更关心内存峰值可预测，而不是通用分配灵活性。
2. 模型图和 tensor 生命周期在编译期已知，适合静态规划。
3. 通用内存池会引入碎片、锁、调试复杂度，不一定符合论文主线。
4. 静态 arena 更容易实验验证：峰值内存、malloc 次数、运行稳定性都能量化。

这个设计与 ExecuTorch 的 fixed memory arenas 和 TensorFlow Lite Micro 的 tensor arena 思路一致：面向嵌入式部署时，关键不是实现一个通用 malloc 替代品，而是把模型执行所需的可变内存压缩到少量可验证的连续区域。

### 12.2 内存区域划分

Runtime 内存分为 6 类：

```text
1. Weight Memory
   只读权重区。来自 SPK weight section，或 codegen 后的 .rodata。

2. Persistent Memory
   Runtime 元数据、节点表、tensor 表、delegate 持久对象。

3. Input Memory
   用户输入区。可以拷贝进 arena，也可以外部绑定。

4. Output Memory
   模型输出区。默认保持到 spkv2_get_output 后。

5. Activation Arena
   中间激活 tensor 复用区。由 compiler 静态规划 offset。

6. Scratch Arena
   kernel 临时工作区，例如 im2col buffer、GEMM pack buffer。
```

第一阶段可以让 Persistent Memory 由普通 `malloc` 分配，因为它只发生在 load/prepare 阶段；论文实验中的“零动态分配”主要指 run 阶段 activation/scratch 不再 malloc。Codegen 模式下，Persistent Memory 也可以进一步静态化。

### 12.3 Activation Arena

Activation arena 是 SPINNV2 内存管理的核心。

Compiler 在地面端做：

```text
1. 根据拓扑序确定每个 tensor 的 first_use 和 last_use。
2. 对非 weight、非 external tensor 计算 size 和 alignment。
3. 根据 alloc_input / alloc_output 策略决定 IO 是否进入规划。
4. 使用区间分配算法为 tensor 分配 offset。
5. 得到 activation_arena_bytes。
6. 将 offset 写入 SPK Memory Plan Section。
```

Runtime 在星上端做：

```text
1. 申请一整块 activation arena，或使用用户提供的静态内存。
2. 对每个 tensor 设置 data = arena_base + offset。
3. 推理过程中不对 activation tensor malloc/free。
```

M2 实现状态：Compiler 已实现 naive baseline、best-fit planner、Memory Plan Section、memory_plan.csv 和 target memory budget check。Runtime 已按 Tensor Table 中的 offset 绑定非 weight tensor，并在 arena 不足时失败。

### 12.4 Tensor 生命周期

对每个 tensor：

```text
first_use = 生产它的 node index
last_use  = 最后一个消费它的 node index
```

特殊规则：

1. 模型输入的 `first_use = 0`，`last_use = 最后消费它的节点`。
2. 模型输出的生命周期必须延长到 `graph_end`。
3. 权重不进入 activation arena。
4. 常量小 tensor 可以进入 constant section，也可以内联进 attrs。
5. inplace tensor 需要单独标记 alias 关系，第一阶段可以先不支持 inplace。
6. 如果输入或输出采用 external bind，则其 memory_class 为 `EXTERNAL`，不占用 activation arena。

### 12.5 内存规划算法

第一阶段建议实现 naive 和 best-fit 两种算法：

```text
Naive:
    所有 activation tensor 顺序拼接，不复用。
    用作内存上界和论文 baseline。

BestFit:
    根据生命周期复用不重叠 tensor 的内存区间。
    用作默认部署算法。
```

Best-fit interval packing 流程：

```text
active_blocks = []
free_blocks = [(0, INF)]

for node in execution_order:
    release tensors whose last_use < node.index
    allocate node outputs by best-fit free block
    update peak
```

注意：模型输出不能提前释放。

第二阶段可以补充 FirstFit 用于论文对比：

```text
FirstFit   首个可用空闲块
```

论文实验可以比较：

1. peak memory。
2. planning time。
3. fragmentation ratio。
4. runtime malloc count。
5. IO external bind 对峰值内存的影响。

### 12.6 Scratch Arena

Scratch arena 用于 kernel 临时空间。典型例子：

1. Conv 的 im2col buffer。
2. GEMM 的 packed B buffer。
3. Reduce / Softmax 的临时 workspace。

Scratch 有两种设计：

第一阶段推荐：

```text
每个 node 在 compiler 阶段估算 scratch_bytes。
scratch_arena_bytes = max(node.scratch_bytes)。
Runtime 所有 node 共享同一块 scratch arena。
```

优点是简单、确定、无碎片。

第二阶段可以优化：

```text
把 scratch 也纳入生命周期规划。
```

但第一阶段没必要。

Scratch 估算应记录在 KernelSpec 中，因为不同 kernel kind 的 scratch 需求不同。例如 direct conv 可能几乎不需要 scratch，而 im2col conv 需要较大的临时 buffer。

### 12.7 Input / Output 处理

提供两种模式：

```text
Copy Mode:
    spkv2_set_input() 把用户输入复制到 input tensor 内存。

External Bind Mode:
    用户直接绑定输入输出指针，Runtime 不复制。
```

第一阶段优先实现 Copy Mode，简单可靠。

后续支持 External Bind Mode，可以减少拷贝，也适合星载系统中 DMA 或共享内存场景。

### 12.8 Weight Memory

权重应只读。

加载 SPK 时有两种策略：

```text
File-backed:
    权重指针直接指向 SPK buffer 或 mmap 区域。

Copied:
    把权重复制到一块对齐内存。
```

Codegen 模式下：

```text
weights -> static const uint8_t g_weights[] -> .rodata
```

第一阶段可以先使用 copied 或 memory-buffer 模式，后续再做 mmap。

### 12.9 是否需要 PoolAllocator

不建议第一阶段实现 ncnn / MNN 那种通用 PoolAllocator。

SPINNV2 的优先级应是：

```text
1. 静态 activation arena
2. 最大 scratch arena
3. 外部输入输出绑定
4. 权重只读映射
5. 最后才考虑通用 pool
```

只有当出现以下需求时，才考虑通用 pool：

1. 动态 shape。
2. 多模型并发。
3. 多 batch 可变输入。
4. 后端 delegate 需要运行时临时对象。
5. LLM KV Cache 这类长生命周期状态内存。

当前论文方向下，自定义通用内存池不是必要主线。

### 12.10 内存管理验证指标

必须在实验中输出：

```text
model_name
num_tensors
num_activation_tensors
naive_activation_bytes
planned_activation_bytes
memory_reduction_ratio
scratch_arena_bytes
runtime_malloc_count
load_time_ms
run_time_ms_avg
run_time_ms_p99
```

建议 compiler 输出：

```text
memory_plan.csv
```

字段：

```text
tensor_id,name,size,first_use,last_use,offset,memory_class
```

## 13. 低比特与量化规划

近年推理框架的低比特趋势很明显，但 SPINNV2 第一阶段不应直接追逐 int4/fp8/fp4。正确策略是先把格式和编译链路设计到位，再逐步加入量化实现。

### 13.1 阶段策略

```text
M1-M3:
    fp32 only。先保证转译、运行时、内存规划和数值验证闭环。

M4:
    int8 权重量化或 int8 activation 可选。
    重点验证 quant params 在 SIR/SPK 中的表达和 kernel dispatch。

M5+:
    int4 weight-only、blockwise quant、混合精度作为扩展研究。
```

### 13.2 IR 预留字段

Tensor quant 字段应能表达：

```text
quant_type
storage_dtype
compute_dtype
scale
zero_point
axis
block_size
```

Node / KernelSpec 应能表达：

```text
input_dtype
weight_dtype
accumulator_dtype
output_dtype
requant_policy
```

### 13.3 不提前实现复杂低比特的原因

1. 星载论文主线是可移植转译、确定性执行和内存规划，不是低比特算法论文。
2. 低比特 kernel 会显著增加调试成本。
3. 没有稳定 fp32 reference，量化误差无法可靠定位。
4. 第一阶段预留格式即可支撑后续扩展。

## 14. Codegen 设计

Codegen 把 SPK 或 SIR 生成独立 C 代码。

输出：

```text
model.h
model.c
weights.c 或 weights.bin
Makefile / CMakeLists.txt
main_test.c
```

生成后的接口：

```c
void model_init(void);
int model_run(const void *input, void *output);
```

Codegen 模式适合：

1. 无文件系统平台。
2. ROM 化部署。
3. 交叉编译。
4. 论文展示星载静态部署能力。

## 15. 验证与测试

### 15.1 数值验证

每个模型转译后都应进行：

```text
ONNX Runtime 输出
SPINNV2 Runtime 输出
误差统计
```

指标：

```text
max_abs_error
mean_abs_error
max_rel_error
cosine_similarity
top1_equal 可选
```

### 15.2 Pass 验证

每个 pass 都应有单元测试：

1. 输入一个小图。
2. 执行 pass。
3. 检查节点数、边关系、tensor shape、输出数值。

### 15.3 Runtime 验证

Runtime 测试包括：

1. header 校验失败。
2. section 越界。
3. tensor offset 越界。
4. backend hint 不存在时 fallback。
5. arena size 不足。
6. 不支持 op 返回错误。

### 15.4 内存验证

必须验证：

1. tensor offset 不重叠，除非生命周期不重叠。
2. 模型输出不会被提前覆盖。
3. arena 边界不越界。
4. scratch 不越界。
5. runtime 推理阶段不发生 activation malloc。
6. external IO bind 不覆盖仍在使用的 tensor。
7. target memory budget 超限时 compiler 必须报错。

## 16. Benchmark 设计

### 16.1 对比对象

建议对比：

```text
ONNX Runtime CPU
SPINNV2 Reference
SPINNV2 Optimized
旧 SPINN 可选
```

旧 SPINN 可以作为历史基线，但不要作为论文主体。

### 16.2 模型选择

第一阶段：

```text
MNIST CNN
LeNet
Small ResNet
```

第二阶段：

```text
ResNet18
MobileNetV2
YOLO tiny / prenms 子图
```

### 16.3 指标

```text
模型大小
加载时间
平均推理时间
P50 / P90 / P99
峰值 activation 内存
总运行时内存
malloc 次数
数值误差
Runtime 二进制大小
target profile
kernel fallback 次数
```

## 17. 阶段计划

### 17.1 M0：工程骨架

目标：

1. 建立 compiler/runtime/format/tests 目录。
2. 写 SIR 和 SPK spec。
3. 建立 CMake 和 Python CLI。
4. 定义 target_profile.json schema。
5. 准备 `cpu_ref.json` 作为默认目标平台。

验收：

```text
python -m spinnv2.compiler --help
cmake -S runtime -B build
python -m spinnv2.compiler --print-target cpu_ref
```

### 17.2 M1：最小推理闭环

支持：

```text
Conv, Relu, MaxPool, Flatten, Gemm, Softmax
```

验收：

```text
ONNX MNIST -> SIR -> SPK -> Runtime -> output
```

### 17.3 M2：静态内存规划

实现：

1. lifetime analysis。
2. best-fit activation arena。
3. memory_plan section。
4. memory_plan.csv。
5. naive baseline。
6. alloc_input / alloc_output 策略。

验收：

```text
planned_activation_bytes < naive_activation_bytes
runtime activation malloc count = 0
compiler 在 target memory budget 超限时拒绝生成 SPK
```

### 17.4 M3：图优化

实现：

1. Conv+BN fusion。
2. Conv+Relu fusion。
3. constant folding。
4. dead tensor elimination。
5. Identity/Dropout elimination。
6. CLI 默认 pass pipeline、可配置 pass 顺序、`pass_stats.json`。
7. `benchmarks/compare_passes.py` 用于比较 pass 前后节点数、SPK 大小、
   activation arena、可选 runtime latency 和输出误差。

验收：

```text
节点数减少
输出误差在阈值内
推理时间或内存有改善
```

当前 M3 验证状态：

```text
tests/compiler/unit/test_passes.py 覆盖各 pass 的结构正确性。
tests/e2e/test_m1_e2e.py 覆盖 Conv+BN+Relu 融合后的 ORT 数值对齐。
pytest tests/compiler tests/e2e 通过。
ctest --test-dir build/runtime 通过。
```

### 17.5 M4：KernelSpec、Backend 与优化 Kernel

当前实现：

1. `compiler/planner/kernel_spec.py` 根据 target profile 生成 KernelSpec，
   并估算每个 node 的 scratch 需求。
2. `cpu_ref` 选择全 reference kernel；`cpu_generic` 为 Conv 选择
   `im2col_gemm`，为 Gemm 选择 `direct`，并为二者记录 reference fallback。
3. SPK 写入 KernelSpec Section，Node Table 写入 `kernel_spec_id` 和
   `scratch_bytes`，Header 写入 `scratch_arena_bytes`。
4. Runtime loader 解析 KernelSpec Section，prepare 阶段分配共享 scratch arena。
5. Runtime 通过 kernel registry 按 `op_type + backend + kernel_kind` dispatch；
   如果 selected kernel 不存在，按 `fallback_kernel_spec_id` 回退到 reference。
6. 当前优化 kernel 包括 CPU direct Gemm 和 Conv im2col-style scratch kernel。
7. `benchmarks/compare_kernels.py` 用于比较 reference/optimized target profile
   的 SPK 大小、activation/scratch 内存、fallback 统计、可选 latency 和输出误差。

当前 M4 验证状态：

```text
tests/compiler/unit/test_kernel_spec.py 覆盖 KernelSpec 选择、fallback 和 scratch budget。
tests/e2e/test_m1_e2e.py 覆盖 cpu_generic KernelSpec runtime 数值对齐。
pytest tests/compiler tests/e2e 通过。
ctest --test-dir build/runtime 通过。
```

M4 后续增强：

1. packed weight transform。
2. SIMD kernel。
3. 更完整的大模型 latency benchmark。

### 17.6 M5：Codegen 与星载部署特性

当前实现：

1. `compiler/codegen/c_codegen.py` 将 SPK 生成 `model.c`、`model.h`、
   `main_test.c` 和 `CMakeLists.txt`。
2. generated `model.c` 把 SPK bytes 编译为 `static const` 数据；权重仍位于
   embedded SPK 的 Weight Section，因此进入 `.rodata`。
3. generated `model.c` 使用静态 activation arena 和静态 scratch arena，并通过
   `spkv2_prepare_with_scratch` 避免 prepare 阶段为 scratch 动态分配。
4. generated `model_run` 通过 external input/output bind API 运行，避免输入输出
   额外拷贝。
5. SPK writer 写入 Checksum Section；runtime loader 在 `checksum_type == 1` 时
   校验损坏 SPK 并失败。
6. generated wrapper 也保存 embedded SPK 的 checksum，并在 `model_init` 前校验。
7. Runtime persistent allocation 走 `spkv2_platform_malloc/calloc/free`；默认
   `runtime/platform/platform.c` 使用 libc，CMake 可替换 `SPKV2_PLATFORM_SOURCE`
   接入 bare-metal stub。
8. Quantization section ID 与 dtype/KernelSpec 字段继续保留，fp32 路径不受影响。

当前 M5 验证状态：

```text
tests/codegen/test_c_codegen.py 覆盖 generated C 编译、运行、checksum 损坏拒绝。
tests/e2e/test_m1_e2e.py 覆盖 SPK runtime checksum 损坏拒绝。
pytest tests/compiler tests/e2e tests/codegen 通过。
ctest --test-dir build/runtime 通过。
```

### 17.7 M6：大模型实验与冻结

当前实现：

1. SPK minor version 更新到 `0.2`，Attribute Record 扩展了 `perm`、`axes`、
   `keepdims`、`TopK` 参数和 cast 参数字段，用于承载 ResNet101/YOLOv10n
   需要的 shape、reduce、transpose 和后处理算子属性。
2. Compiler、target profile 和 runtime reference registry 支持：
   `Add`、`Cast`、`Concat`、`Conv`、`Div`、`Flatten`、`GatherElements`、
   `Gemm`、`MatMul`、`MaxPool`、`Mod`、`Mul`、`ReduceMax`、`ReduceMean`、
   `Relu`、`Reshape`、`Resize`、`Sigmoid`、`Softmax`、`Split`、`Sub`、`Tile`、
   `TopK`、`Transpose`、`Unsqueeze`。
3. Reference Conv 支持 grouped/depthwise convolution；`cpu_generic` 的
   im2col Conv 在 group != 1 时返回 missing-kernel，由 runtime fallback 到
   reference Conv。
4. `benchmarks/run_m6_models.py` 固化 ResNet101/YOLOv10n 的 op 统计、编译、
   ORT baseline、runtime 执行和误差指标采集。
5. `scripts/export_paper_tables.py` 和 `scripts/check_reproducibility.py` 固化
   论文表格数据源和复现实验阈值；`benchmarks/run_all.py` 串联 M6 全流程。

当前 M6 验证状态：

```text
ResNet101:
  compile 成功，241 nodes。
  FuseConvRelu: 67。
  naive activation: 137803680 bytes。
  planned activation: 14249984 bytes。
  runtime: 60.91s。
  ORT 对齐: top1_equal=true，max_abs_error=0.0546875，mean_abs_error=0.00544044。

YOLOv10n:
  compile 成功，308 nodes。
  naive activation: 307286000 bytes。
  planned activation: 24576000 bytes。
  runtime: 30.44s。
  ORT 对齐: score_max_abs_error=5.46e-08，score_mean_abs_error=1.08e-08。
  top10 rows: max_abs_error=6.26e-04，mean_abs_error=9.86e-05，class 10/10。
  top20+ 低置信度行受 TopK 微小数值差影响，候选框/类别顺序会分叉；
  论文实验应分别报告 score 误差、topN 高置信行误差和全量输出误差。

pytest tests/compiler tests/e2e tests/codegen 通过。
ctest --test-dir build/runtime 通过。
python benchmarks/run_all.py --skip-tests --out-dir build/m6_final --tables-dir build/m6_paper_tables 通过。
```

## 18. 论文贡献表述建议

SPINNV2 可以在论文中表述为四项主要贡献：

1. 提出一种面向星载推理部署的受限中间表示 SIR 和 section 化模型包 SPK，实现从通用模型到星载部署包的转译。
2. 设计 Target Profile 与 KernelSpec 机制，将目标平台能力、kernel 实现策略和 fallback 关系前移到地面端编译阶段。
3. 设计一种编译期静态内存规划方法，将 tensor 生命周期、arena offset、IO 外部绑定策略和 scratch 需求写入模型包，使星上运行时实现低动态分配和可预测内存峰值。
4. 实现一个可移植轻量 Runtime 和 kernel 后端机制，支持 reference 与优化 kernel 共存，并通过 codegen 支持静态 C 部署。

## 19. 关键取舍总结

SPINNV2 应坚持以下取舍：

```text
通用性让给 ONNX Runtime，部署确定性留给 SPINNV2。
复杂优化放在地面端，星上端只做简单执行。
支持少量关键算子，但把转译、内存、验证链路做完整。
先做静态 arena，不急着做通用内存池。
先用 target profile 固化部署约束，不在运行时动态猜测平台能力。
先做 reference 正确性，再做 SIMD 性能。
先预留量化格式，再实现 int8 和更低比特 kernel。
```

这个方向更符合硕士论文工程，也更符合星载可移植推理框架的题目。
