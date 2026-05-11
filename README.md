# SPINNV2

SPINNV2 is a clean-room prototype for a satellite-oriented portable inference framework and model translation toolchain.

The project follows an ahead-of-time deployment route:

```text
ONNX -> SIR -> SPK -> lightweight Runtime / generated C
```

## Current Status

SPINNV2 is currently a working M4 prototype: the M0 project skeleton, M1
ONNX-to-runtime path, M2 static activation memory planning, M3 graph
optimization pipeline, and M4 KernelSpec/backend path are in place and covered
by smoke, unit, and tiny end-to-end tests.

Completed and validated:

- Compiler CLI can import fixed-shape fp32 ONNX models and write SPK packages.
- Supported runtime ops are `Add`, `Conv`, `Flatten`, `Gemm`, `MaxPool`, `Relu`,
  and `Softmax` through reference kernels.
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

Not started or not yet complete:

- SIMD kernels, packed-weight transforms, code generation, full benchmark
  suites, and paper experiment automation.
- Broad model coverage beyond the current fixed-shape fp32 toy/tiny CNN and
  Conv+BN+Relu validation paths.
- Dynamic shapes, quantization, checksum enforcement, and production-level SPK
  compatibility guarantees.

## Smoke Checks

```bash
python -m spinnv2.compiler --help
python -m spinnv2.compiler --print-target cpu_ref
python -m spinnv2.compiler compile --help
pytest tests/compiler
pytest tests/e2e
cmake -S runtime -B build/runtime
cmake --build build/runtime
ctest --test-dir build/runtime
```

The compiler can also emit a memory-plan CSV:

```bash
python -m spinnv2.compiler compile model.onnx -o build/model.spk --memory-plan-csv build/memory_plan.csv
```

If `pytest` is unavailable, the compiler unit tests can still run with:

```bash
python -m unittest discover -s tests/compiler -p 'test*.py' -v
```
