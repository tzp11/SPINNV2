# SPINNV2

SPINNV2 is a clean-room prototype for a satellite-oriented portable inference framework and model translation toolchain.

The project follows an ahead-of-time deployment route:

```text
ONNX -> SIR -> SPK -> lightweight Runtime / generated C
```

## Current Status

SPINNV2 is currently a working M6 prototype: the M0 project skeleton, M1
ONNX-to-runtime path, M2 static activation memory planning, M3 graph
optimization pipeline, M4 KernelSpec/backend path, and M5 static C deployment
path are in place. M6 large-model validation now covers ResNet101 and YOLOv10n
compile/runtime smoke tests in addition to unit, tiny end-to-end, and codegen
tests.

Completed and validated:

- Compiler CLI can import fixed-shape fp32 ONNX models and write SPK packages.
- Supported runtime ops include `Add`, `Cast`, `Concat`, `Conv`, `Div`,
  `Flatten`, `GatherElements`, `Gemm`, `MatMul`, `MaxPool`, `Mod`, `Mul`,
  `ReduceMax`, `ReduceMean`, `Relu`, `Reshape`, `Resize`, `Sigmoid`, `Softmax`,
  `Split`, `Sub`, `Tile`, `TopK`, `Transpose`, and `Unsqueeze` through
  reference kernels.
- SPK writer emits tensor/node/attribute/weight tables, debug JSON, and an M2
  Memory Plan section.
- Memory planner performs lifetime analysis, best-fit activation arena reuse,
  external IO policy handling, target arena budget checks, and optional
  `memory_plan.csv` output.
- C runtime loads SPK files, prepares tensor pointers from compiler-provided
  arena offsets, checks arena bounds, runs the reference executor, and supports
  input/output copy or bind APIs.
- Compiler M3 passes run by default and can be disabled or reordered from the
  CLI. The pipeline includes identity/dropout elimination, constant folding,
  Conv+BatchNorm fusion, Conv+Relu fusion, and dead-node elimination.
- Pass statistics are emitted into SPK debug JSON and can also be written as a
  standalone `pass_stats.json`.
- `benchmarks/compare_passes.py` compares compile output with and without M3
  passes, and can collect runtime latency and output error when an input binary
  is provided.
- KernelSpec selection writes a SPK KernelSpec section, fills node
  `kernel_spec_id`/`scratch_bytes`, records fallback metadata in debug JSON, and
  checks target scratch budget.
- Runtime parses KernelSpec, dispatches through a kernel registry, uses
  reference fallback when needed, and allocates a shared scratch arena during
  prepare.
- `cpu_generic` enables the first optimized CPU paths: Gemm `direct` and Conv
  `im2col_gemm`; `cpu_ref` remains fully reference, and `memory_limited_1mb`
  exercises M4 memory-budget checks.
- `benchmarks/compare_kernels.py` compares reference and optimized target
  profiles, with optional runtime latency and output error collection.
- `benchmarks/run_m6_models.py` runs the ResNet101/YOLOv10n M6 large-model
  benchmark, `scripts/export_paper_tables.py` exports paper-ready CSV/Markdown
  tables, and `scripts/check_reproducibility.py` checks the frozen M6 thresholds.
- `tests/e2e/model_zoo.py` and `tests/e2e/run_e2e.py` provide the generated
  validation set from the execution plan: toy ops, MNIST/LeNet-style CNNs,
  ResNet/MobileNet-style medium models, and a YOLO pre-NMS-style detection
  subgraph.
- `python -m spinnv2.compiler codegen` turns an SPK package into generated
  `model.c`, `model.h`, `main_test.c`, and `CMakeLists.txt` files.
- Generated C embeds the SPK package as `static const` data, uses static
  activation and scratch arenas, verifies a generated checksum, and runs through
  external input/output bind APIs without loading a model file.
- SPK packages include a checksum section, and the runtime rejects corrupted
  packages when checksum validation is enabled.
- Runtime allocation now goes through `spkv2_platform_*`, with a default libc
  platform implementation that can be replaced for bare-metal builds.

Not started or not yet complete:

- SIMD kernels, packed-weight transforms, and full paper figure automation.
- Broad model coverage beyond fixed-shape fp32 CNN/detection paths.
- Dynamic shapes, quantization kernels, and production-level SPK compatibility
  guarantees.

## Smoke Checks

```bash
python -m spinnv2.compiler --help
python -m spinnv2.compiler --print-target cpu_ref
python -m spinnv2.compiler compile --help
python -m spinnv2.compiler codegen --help
pytest tests/compiler
pytest tests/e2e
pytest tests/codegen
cmake -S runtime -B build/runtime
cmake --build build/runtime
ctest --test-dir build/runtime
python benchmarks/run_all.py --skip-large-run
python tests/e2e/run_e2e.py --all
python benchmarks/run_memory.py --models mnist,lenet,resnet18
python benchmarks/run_latency.py --models mnist,lenet
python tests/codegen/run_codegen_test.py --models mnist,lenet
```

The compiler can also emit a memory-plan CSV:

```bash
python -m spinnv2.compiler compile model.onnx -o build/model.spk --memory-plan-csv build/memory_plan.csv
```

If `pytest` is unavailable, the compiler unit tests can still run with:

```bash
python -m unittest discover -s tests/compiler -p 'test*.py' -v
```
