# 如何分析一个推理框架

本文档提供一套系统性方法，用于分析和解读开源推理框架项目，以便为 SPINNV2 的设计改进和性能优化提供参考。

## 1. 分析目标

分析一个推理框架不是为了通读全部代码，而是快速定位以下问题的答案：

1. 这个框架在哪些设计点上做了和我们不同的选择？
2. 它的某个具体实现（如 Winograd kernel）能否直接借鉴？
3. 它踩过哪些坑，我们可以避免？

## 2. 快速定位关键文件

拿到一个框架项目后，先执行以下命令建立全局地图：

```bash
# 找最大的源文件（热点代码通常在最大的文件里）
find . -name "*.c" -o -name "*.cpp" | xargs wc -l | sort -rn | head -20

# 找 SIMD kernel 实现
grep -rl "winograd\|Winograd\|im2col\|GEMM\|sgemm" . --include="*.c" --include="*.cpp"

# 找内存布局定义
grep -rl "NC4HW4\|NHWC\|pack\|reorder\|layout" . --include="*.h" --include="*.cpp"

# 找 INT8 量化实现
grep -rl "int8\|quantize\|vdotq\|requantize\|scale.*zero" . --include="*.c" --include="*.cpp"

# 找图优化 pass
grep -rl "pass\|Pass\|fuse\|Fuse\|fold\|Fold" . --include="*.cpp" --include="*.py"

# 找线程调度
grep -rl "parallel\|thread\|omp\|dispatch_apply\|ThreadPool" . --include="*.c" --include="*.cpp"

# 找算子注册表
grep -rl "registry\|Registry\|register_op\|OpTable" . --include="*.c" --include="*.cpp" --include="*.h"
```

## 3. 三个分析视角

### 3.1 内部实现视角（怎么做的）

#### 3.1.1 核心 SIMD Kernel

这是推理框架性能的决定性因素，也是最值得借鉴的部分。

**需要回答的问题：**

- 3×3 stride-1 Conv 用什么算法？Winograd F(2,3) / F(4,3)、im2col+GEMM、还是 Direct？选择阈值是什么（按 channel 数、spatial 大小还是 FLOP 数）？
- GEMM 微内核的 tile 大小是多少（MR×NR）？针对哪款 CPU 的寄存器数量设计？有没有 K 方向展开（unroll factor）和 prefetch？
- 1×1 pointwise Conv 是 reshape 为 GEMM 还是有专用路径？
- Depthwise Conv 怎么向量化？stride-2 的 depthwise 有没有用 `vld2q`（ARM）或 gather（x86）？
- 权重预打包（weight packing）的格式是什么？打包发生在加载时还是编译时？

**定位方法：** 找 `conv` 目录或最大的 kernel 文件，搜索 `micro_kernel`、`MR`、`NR`、`winograd`。

#### 3.1.2 内存布局与管理

Tensor 存储格式直接影响 SIMD 利用率和缓存命中率。

**需要回答的问题：**

- Tensor 存储格式是 NCHW、NHWC 还是 packed format（NC4HW4、NC8HW8）？
- 什么时候做 layout convert（加载时、编译期还是运行时）？有没有 layout propagation pass 自动推断最优布局？
- Activation arena 用什么分配算法（bump、best-fit、first-fit）？有没有对齐要求？
- 不同 kernel 的 scratch/workspace 怎么共享？是取 max 还是分时复用？
- 权重是 mmap 到模型文件还是加载时拷贝？加载后对齐到多少字节？

**定位方法：** 搜索 `arena`、`allocator`、`NC4HW4`、`pack`。

#### 3.1.3 图优化 Pass

编译期优化减少运行时工作量，是 AOT 框架的核心竞争力。

**需要回答的问题：**

- 有哪些 fusion pattern？除了 Conv+BN+ReLU/SiLU，还有什么（Add+Mul、Concat 内存优化、常量传播、shape 折叠）？
- Pass 执行顺序是固定的还是可配置的？有没有迭代直到不动点（fixpoint iteration）？
- 有没有 layout propagation pass（根据 kernel 偏好自动插入 layout convert 节点）？
- 有没有 subgraph fusion（将多个节点合成一个复合节点，如 attention block）？

**定位方法：** 搜索 `fuse`、`fold`、`eliminate`、`pass`、`transform`。

#### 3.1.4 算子分发与 Fallback

Kernel 的选择和回退机制决定了框架的可移植性和鲁棒性。

**需要回答的问题：**

- Kernel registry 是静态表、函数指针链还是虚函数表？
- 一个算子有几个 backend 实现（reference / CPU / SIMD / delegate）？选择逻辑是什么？
- Fallback 链是几级？fallback 决策在编译期还是运行时做？
- 对于新增算子，开发者需要实现哪些接口（最小工作量）？
- 有没有 subgraph delegation（将子图委托给外部 backend，如 CoreML、GPU）？

**定位方法：** 搜索 `registry`、`dispatch`、`fallback`、`delegate`。

#### 3.1.5 量化支持

INT8/INT4 量化是移动端推理性能的最大杠杆。

**需要回答的问题：**

- 支持哪些量化格式（INT8 weight-only、INT8 全量化、INT4、mixed-precision）？
- 量化参数（scale、zero_point）的粒度：per-tensor、per-channel 还是 per-block？
- 量化 Conv kernel 的实现：用 `vdotq_s32`（ARM DOT 指令）还是 `vmlal_s16`？
- 每层之后的 requantize 怎么做（scale 乘法 + clamp + round）？是否融合进下一层？
- PTQ 校准方法：min/max、percentile 还是 KL 散度？

**定位方法：** 搜索 `int8`、`quantize`、`scale`、`zero_point`、`vdotq`。

#### 3.1.6 线程调度

并行粒度的选择直接影响多核利用率和缓存效率。

**需要回答的问题：**

- 并行粒度：按 batch、output channel、spatial row 还是 output tile 并行？
- 启用并行的阈值是多少？小任务有没有跳过并行的逻辑？
- im2col 和 SGEMM 是串行执行还是可以 pipeline（im2col 一块、SGEMM 算一块交替）？
- 线程池的实现：OpenMP、pthread pool、GCD 还是自定义？

**定位方法：** 搜索 `parallel`、`thread`、`omp`、`dispatch`。

### 3.2 外部接口视角（提供什么保证）

#### 3.2.1 错误处理与诊断

框架怎么报错揭示了它对哪些问题认真对待。

**需要回答的问题：**

- 不支持的算子：编译期报错还是运行时返回错误码？
- 形状推断失败：是 crash 还是有明确的错误信息？
- 数值异常（NaN、Inf）：有没有运行时检测？可以开启吗？
- 用户如何定位"推理结果不对"：有没有逐层输出 dump 工具？

**定位方法：** 搜索 `error`、`assert`、`NaN`、`dump`、`debug`。

#### 3.2.2 算子扩展机制

用户如何添加框架不支持的自定义算子。

**需要回答的问题：**

- 注册接口是什么形式（函数指针、宏、继承）？
- 编译器端如何识别自定义算子的输出形状？
- 自定义算子能否参与图优化 pass？
- 有没有示例代码演示如何添加一个自定义算子？

**定位方法：** 搜索 `custom_op`、`register`、`plugin`。

#### 3.2.3 格式版本演化策略

二进制模型格式一旦发布就很难改，这个策略很重要。

**需要回答的问题：**

- 版本号字段在哪里？major/minor 的语义是什么？
- 新增字段时向前兼容怎么保证（旧 runtime 能读新格式吗）？
- 有没有格式迁移工具（旧版本模型转新版本）？

**定位方法：** 搜索 `version`、`magic`、`compatible`。

#### 3.2.4 部署集成方式

框架如何嵌入到更大的系统中。

**需要回答的问题：**

- API 是 C 还是 C++（C 更容易跨语言绑定）？
- 有没有 Python binding（方便调试和快速验证）？
- 多模型并发时资源怎么隔离（context 是 thread-safe 的吗）？
- 流水线场景：上一个模型的输出能否不经拷贝直接作为下一个模型的输入？

**定位方法：** 看 `include/` 目录下的公共头文件。

### 3.3 系统边界视角（假设什么、不做什么）

#### 3.3.1 数值精度保证

推理框架的数值误差是核心承诺。

**需要回答的问题：**

- fp32 推理与 ONNX Runtime 的误差容忍是多少？这个阈值是理论推导的还是实验确定的？
- 不同平台（x86 vs ARM）的输出是否保证一致？如果不一致，差异来源是什么（FMA 舍入、exp 近似算法）？
- Softmax/Sigmoid 这类数值敏感算子有没有特殊的精度保障策略（如 log-sum-exp 技巧）？
- 量化后的精度损失有没有官方基准数据？

**定位方法：** 搜索 `tolerance`、`atol`、`rtol`、`epsilon`、`numerical`。

#### 3.3.2 模型准备流水线

从训练到部署的完整链路，哪些步骤是框架的责任，哪些是用户的。

**需要回答的问题：**

- 模型导入：支持 ONNX 的哪些 opset？对模型有什么前提要求（固定 shape？无控制流？）？
- 模型简化：框架依赖 ONNX simplifier 吗？还是自己做 shape inference？
- 量化校准：数据集怎么准备？校准过程多久？需要多少样本？
- 模型验证：框架是否提供精度验证工具？

**定位方法：** 看 README 的"Getting Started"和"Model Preparation"部分。

#### 3.3.3 二进制体积管理

嵌入式部署时 runtime 大小是硬约束。

**需要回答的问题：**

- 如何只编译需要的算子（算子裁剪）？裁剪是 CMake 选项还是代码生成？
- 不同配置下 runtime 库的体积数据（全量 vs 最小 vs 特定模型）？
- 权重有没有压缩（如 LZ4、zstd）？解压发生在加载时还是运行时？

**定位方法：** 看 CMakeLists.txt 中的 `option()` 和条件编译。

## 4. 分析优先级

对于 SPINNV2 当前阶段，按以下优先级分析参考框架：

| 优先级 | 分析内容 | 理由 |
|--------|---------|------|
| P0 | INT8 Conv kernel 实现 | 最大的性能增长点（2-4× 提速），我们格式已预留但 kernel 未实现 |
| P1 | Winograd F(2,3) 的正确实现 | 我们的 Winograd 目前被禁用（实测比 im2col 慢），需要找出问题 |
| P2 | NC4HW4 packed layout | NCHW 在 NEON 上不是最优的，NC4HW4 天然适配 `float32x4_t` |
| P3 | 更多图融合 pattern | 减少节点数、降低调度开销 |
| P4 | 算子扩展机制 | 论文里展示框架可扩展性 |
| P5 | 数值精度的理论分析 | 论文核心指标，当前只有实测数据没有理论分析 |

## 5. 典型参考框架的关键目录

以下是几个常见开源推理框架中，上述各部分对应的关键文件位置：

### ncnn

```
src/layer/arm/          → NEON kernel（Conv、GEMM、Winograd）
src/layer/              → 算子实现和注册
src/mat.h               → Tensor 存储结构（NC4HW4）
src/allocator.h         → 内存分配器
src/net.cpp             → 模型加载和推理入口
tools/quantize/         → INT8 量化工具
```

### MNN

```
source/backend/cpu/compute/  → GEMM 微内核、Winograd、im2col
source/backend/cpu/          → CPU 算子实现
source/core/                 → 图优化 pass、调度器
source/shape/                → 形状推断
tools/quantization/          → 量化工具链
schema/                      → FlatBuffer 模型格式定义
```

### TensorFlow Lite Micro

```
tensorflow/lite/micro/kernels/  → 嵌入式算子实现
tensorflow/lite/micro/          → 微控制器 runtime
tensorflow/lite/kernels/        → 标准 kernel（含 INT8）
tensorflow/lite/schema/         → FlatBuffer schema
tensorflow/lite/tools/optimize/ → 量化工具
```

### ONNX Runtime (MLAS)

```
onnxruntime/core/mlas/lib/      → GEMM 微内核（手写汇编）
onnxruntime/core/providers/cpu/ → CPU 算子
onnxruntime/core/graph/         → 图优化 pass
onnxruntime/core/quantization/  → 量化工具
```

## 6. 分析产出模板

对每个参考框架，产出一份简短的分析笔记，格式如下：

```
框架名称：
版本/commit：

1. Conv 策略
   - 3x3 s1: [Winograd F(2,3) / im2col+GEMM / Direct]
   - 3x3 s2: [Direct with stride load / im2col / fallback to s1+downsample]
   - 1x1:    [reshape to GEMM / 专用 kernel]
   - GEMM tile: MR=?, NR=?

2. 内存布局
   - 默认 layout: [NCHW / NHWC / NC4HW4]
   - layout convert: [编译期 / 加载时 / 运行时]
   - arena 算法: [best-fit / bump / custom]

3. 量化
   - 支持格式: [INT8 per-channel / INT8 per-tensor / INT4 / 无]
   - 校准方法: [min-max / KL divergence / percentile]
   - kernel 指令: [vdotq_s32 / vmlal_s16 / 纯标量]

4. 值得借鉴的点
   - [具体描述，附文件路径和行号]

5. 我们做得更好的点
   - [具体描述]

6. 明确放弃的方案（如有）
   - [从 commit history 或 issue 中发现的]
```

## 7. 注意事项

- **不要通读代码**。用 grep 定位关键函数，只读那一个函数和它的调用者。
- **先跑 benchmark 再读代码**。知道哪个模型上快、哪个模型上慢，再去看对应的 kernel，效率最高。
- **关注被删掉的代码**。`git log --diff-filter=D` 可以找到被删除的文件，这些往往是失败的实验。
- **关注 TODO 和 FIXME**。这些是作者自己知道但没时间修的问题，可能正是我们能做得更好的地方。
- **关注测试里的 magic number**。比如某个测试用 `atol=0.01` 而不是 `1e-5`，说明这个 kernel 精度不高，可能是有意的 trade-off。
