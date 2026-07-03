# Reliable Inference Research Workflow

This directory implements the thesis-specific reliability extension on top of
SPINNV2. Framework performance work outside required model compatibility is not
treated as a research contribution.

## Environment

### macOS (current, Apple Silicon)

- Platform: macOS on Apple M-series (ARM64).
- Python 3.11 with PyTorch, Ultralytics, PuLP, ONNX, onnxruntime.
- Toolchain: Apple Clang for the C runtime; the `cpu_generic` target uses
  BNNS/AMX acceleration via the Accelerate framework.
- Tasks: `ResNet50 + EuroSAT` (classification) and `YOLOv10n + DIOR` (detection).
- Fault scope: one transient single-bit flip in an FP32 runtime output tensor.
- YOLO export uses `simplify=True`; the fixed-shape graph requires the added
  `Gather` and `Slice` reference kernels after simplification.

### Windows (historical, archived)

- Environment: `E:\conda_envs\graduatepaper_reliable` with Python 3.11.
- GPU workload environment: `E:\conda_envs\graduatepaper_gpu_prevalidation`
  cloned from a locally verified CUDA 11.6/PyTorch 1.12.1 toolchain and
  extended only with Ultralytics 8.2.100.
- Toolchain: Visual Studio 2022/MSVC for the C runtime; the Release
  `cpu_generic` target uses AVX2/FMA on this Windows host.

## Commands

Run from the repository root with the Python environment active.

### Data Preparation

```bash
python3 -m research.reliability.prepare_data --dataset eurosat
python3 -m research.reliability.download_dior --source hf_parquet --splits train val test
python3 -m research.reliability.prepare_dior_hf_subset --train-samples 128 --val-samples 64 --test-samples 64 --workers 4
```

### Model Training and Export

```bash
python3 -m research.reliability.models.train_resnet50 --train-mode head --device auto
python3 -m research.reliability.models.export_models \
    --model resnet50 \
    --checkpoint artifacts/experiments/macos_reliability/resnet50/best.pt \
    --output artifacts/models/resnet50_eurosat.onnx

python3 -m research.reliability.models.train_yolov10n
python3 -m research.reliability.models.export_models \
    --model yolov10n \
    --weights artifacts/experiments/macos_reliability/yolov10n/train/weights/best.pt \
    --output artifacts/models/yolov10n_dior.onnx
```

### Compatibility Audit

```bash
python3 -m research.reliability.audit_spinn_compat artifacts/models/resnet50_eurosat.onnx
python3 -m research.reliability.audit_spinn_compat artifacts/models/yolov10n_dior.onnx
```

### Compile Baseline SPK (cpu_generic, best performance)

```bash
python3 -m compiler.cli compile artifacts/models/resnet50_eurosat.onnx \
    -o artifacts/models/resnet50_eurosat.spk --target cpu_generic -O2

python3 -m compiler.cli compile artifacts/models/yolov10n_dior.onnx \
    -o artifacts/models/yolov10n_dior.spk --target cpu_generic -O2
```

### Build Runtime

```bash
cmake -S runtime -B runtime/build
cmake --build runtime/build
```

### PyTorch Fault Screening (Stage 1-2)

```bash
python3 -m research.reliability.inject_faults \
    --checkpoint artifacts/experiments/macos_reliability/resnet50/best.pt \
    --output artifacts/injections/resnet50_screen.jsonl

python3 -m research.reliability.build_risk_profile \
    artifacts/injections/resnet50_screen.jsonl \
    --output artifacts/reports/resnet50_risk.json

python3 -m research.reliability.map_runtime_candidates \
    --candidate-map artifacts/injections/resnet50_screen.candidates.json \
    --onnx artifacts/models/resnet50_eurosat.onnx \
    --spk-debug artifacts/models/resnet50_eurosat.spk.json \
    --output artifacts/reports/resnet50_runtime_map.json
```

### Runtime Cost Profiling (Stage 3)

```bash
# ResNet50
python3 -m research.reliability.profile_protection_costs \
    --ranked-csv artifacts/experiments/macos_reliability/resnet50/ranked_candidates.csv \
    --spk artifacts/models/resnet50_eurosat.spk \
    --input artifacts/data/eurosat/test_input.bin \
    --runtime-bench runtime/build/spkv2_bench \
    --primitive-bench runtime/build/spkv2_protection_bench \
    --output-candidates artifacts/experiments/macos_reliability/resnet50/candidates_dmr.json \
    --output-profile artifacts/experiments/macos_reliability/resnet50/protection_profile.json

# YOLOv10n
python3 -m research.reliability.profile_protection_costs \
    --ranked-csv artifacts/experiments/macos_reliability/yolov10n_dior/ranked_candidates.csv \
    --spk artifacts/models/yolov10n_dior.spk \
    --input artifacts/data/dior/test_input.bin \
    --runtime-bench runtime/build/spkv2_bench \
    --primitive-bench runtime/build/spkv2_protection_bench \
    --output-candidates artifacts/experiments/macos_reliability/yolov10n_dior/candidates_dmr.json \
    --output-profile artifacts/experiments/macos_reliability/yolov10n_dior/protection_profile.json
```

### ILP Optimization (Stage 3)

```bash
for budget in 0.5 1.0 2.0 4.0 8.0; do
    python3 -m research.reliability.optimize_plan \
        artifacts/experiments/macos_reliability/resnet50/candidates_dmr.json \
        --model-id resnet50_eurosat \
        --platform-profile macos_arm64_bnns \
        --latency-budget-ms $budget \
        --memory-budget-bytes 8388608 \
        --method ilp \
        --output artifacts/experiments/macos_reliability/resnet50/plans/ilp_${budget}ms.json
done

for budget in 0.5 1.0 2.0 4.0 8.0; do
    python3 -m research.reliability.optimize_plan \
        artifacts/experiments/macos_reliability/yolov10n_dior/candidates_dmr.json \
        --model-id yolov10n_dior \
        --platform-profile macos_arm64_bnns \
        --latency-budget-ms $budget \
        --memory-budget-bytes 8388608 \
        --method ilp \
        --output artifacts/experiments/macos_reliability/yolov10n_dior/plans/ilp_${budget}ms.json
done
```

### Compile Protected SPK

```bash
for budget in 0.5 1.0 2.0 4.0 8.0; do
    python3 -m compiler.cli compile artifacts/models/resnet50_eurosat.onnx \
        -o artifacts/models/resnet50_eurosat_ilp_${budget}ms.spk \
        --target cpu_generic -O2 \
        --protection-plan artifacts/experiments/macos_reliability/resnet50/plans/ilp_${budget}ms.json
done

for budget in 0.5 1.0 2.0 4.0 8.0; do
    python3 -m compiler.cli compile artifacts/models/yolov10n_dior.onnx \
        -o artifacts/models/yolov10n_dior_ilp_${budget}ms.spk \
        --target cpu_generic -O2 \
        --protection-plan artifacts/experiments/macos_reliability/yolov10n_dior/plans/ilp_${budget}ms.json
done
```

### Monte Carlo Fault Evaluation (Stage 4)

```bash
# ResNet50 (classification)
for budget in 0.5 1.0 2.0 4.0 8.0; do
    python3 -m research.reliability.evaluate_runtime_faults \
        --config research/reliability/configs/macos_reliability.yaml \
        --library runtime/build/libspkv2_runtime.dylib \
        --baseline-spk artifacts/models/resnet50_eurosat.spk \
        --protected-spk artifacts/models/resnet50_eurosat_ilp_${budget}ms.spk \
        --plan artifacts/experiments/macos_reliability/resnet50/plans/ilp_${budget}ms.json \
        --ranked-csv artifacts/experiments/macos_reliability/resnet50/ranked_candidates.csv \
        --events 1000 --seed 2026 \
        --output artifacts/experiments/macos_reliability/resnet50/mc_results/mc1000_ilp_${budget}ms.json
done

# YOLOv10n (detection -- uses evaluate_detection_faults, not evaluate_runtime_faults)
for budget in 0.5 1.0 2.0 4.0 8.0; do
    python3 -m research.reliability.evaluate_detection_faults \
        --library runtime/build/libspkv2_runtime.dylib \
        --baseline-spk artifacts/models/yolov10n_dior.spk \
        --protected-spk artifacts/models/yolov10n_dior_ilp_${budget}ms.spk \
        --plan artifacts/experiments/macos_reliability/yolov10n_dior/plans/ilp_${budget}ms.json \
        --ranked-csv artifacts/experiments/macos_reliability/yolov10n_dior/ranked_candidates.csv \
        --test-images artifacts/data/dior/images/test \
        --imgsz 640 --output-shape "1,300,6" \
        --test-samples 128 --events 1000 --seed 2026 \
        --output artifacts/experiments/macos_reliability/yolov10n_dior/mc_results/mc1000_ilp_${budget}ms.json
done
```

### Correctness Verification

```bash
python3 -m research.reliability.compare_onnx_runtime_spk \
    --onnx artifacts/models/resnet50_eurosat.onnx \
    --spk artifacts/models/resnet50_eurosat.spk \
    --library runtime/build/libspkv2_runtime.dylib \
    --input artifacts/data/eurosat/test_input.bin \
    --input-shape 1 3 224 224 \
    --output artifacts/reports/resnet50_correctness.json

python3 -m research.reliability.compare_onnx_runtime_spk \
    --onnx artifacts/models/yolov10n_dior.onnx \
    --spk artifacts/models/yolov10n_dior.spk \
    --library runtime/build/libspkv2_runtime.dylib \
    --input artifacts/data/dior/test_input.bin \
    --input-shape 1 3 640 640 --task detection \
    --output artifacts/reports/yolov10n_correctness.json
```

### Performance Benchmark

```bash
runtime/build/spkv2_bench artifacts/models/resnet50_eurosat.spk \
    artifacts/data/eurosat/test_input.bin --warmup 5 --runs 20

runtime/build/spkv2_bench artifacts/models/yolov10n_dior.spk \
    artifacts/data/dior/test_input.bin --warmup 5 --runs 20
```

### Smoke Run (CPU-only, quick validation)

```bash
python3 -m research.reliability.models.train_resnet50 \
    --run-name resnet50_smoke --train-mode head --device cpu \
    --epochs 1 --max-train-samples 256 --max-val-samples 128 --max-test-samples 128
```

`spkv2_fault_run` accepts `model input output node tensor element bit invocation`
and prints structured injection/detection/recovery counters.

## Optimizer Notes

The budget optimizer minimizes expected critical task-failure probability under
an activation-byte-weighted runtime-object prior. DMR scratch is peak reused
storage, so the memory budget constrains the maximum selected scratch
requirement rather than summing buffers for sequential nodes. Range-guard
benefits use Wilson lower confidence bounds and false-positive rerun cost uses
the corresponding upper bound.

Range thresholds calibrated in PyTorch are screening thresholds only. Before a
deployable plan is reported, compile the plan, collect clean runtime range
observations with `evaluate_runtime_plan`, recalibrate with
`recalibrate_range_plan`, and evaluate on disjoint runtime test inputs.

## Target Profiles

The `--target` flag controls kernel selection and dominates performance:

| Profile | Convolution Kernels | Platform |
|---------|-------------------|----------|
| `cpu_generic` | BNNS/AMX (highest priority), im2col_gemm, pointwise_1x1, depthwise_direct | macOS ARM64 (recommended) |
| `cpu_arm_neon` | Winograd F(4,3), im2col_gemm, pointwise_1x1, depthwise_direct | ARM NEON (no BNNS) |
| `cpu_ref` | Reference-only kernels | Any (slowest, correctness baseline) |

Always use `cpu_generic` for experiments and benchmarks. The `cpu_arm_neon`
profile uses Winograd F(4,3) which is 2-3x slower than BNNS on Apple Silicon
despite being a theoretically efficient algorithm. Node IDs and graph structure
are identical across profiles; only kernel selection differs, so DMR plans
generated on one profile apply to any other.

## macOS Results (Apple Silicon, BNNS/AMX)

### Baseline Performance

| Model | Nodes | avg_ms | ONNX Alignment |
|-------|-------|--------|----------------|
| ResNet50 | 57 | 15.6 | max_abs = 3.74e-5, classification match |
| YOLOv10n | 167 | 17.4 | task-level identical (0 missed/0 FP/0 class change) |

### ResNet50 ILP Budget Sweep (1000 MC events, seed=2026)

| Budget | Nodes | Predicted | avg_ms | Overhead | Unprot Fail | Prot Fail | Risk Reduction | Coverage | Recovery |
|--------|-------|-----------|--------|----------|-------------|-----------|----------------|----------|----------|
| 0.5 ms | 4/57 | 14.2% | 16.3 | +0.7 ms | 12 | 9 | 25.0% | 14.4% | 100% |
| 1.0 ms | 8/57 | 26.6% | 18.3 | +2.7 ms | 12 | 8 | 33.3% | 28.1% | 100% |
| 2.0 ms | 13/57 | 46.1% | 16.3 | +0.7 ms | 12 | 6 | 50.0% | 48.5% | 100% |
| 4.0 ms | 25/57 | 72.7% | 19.5 | +3.9 ms | 12 | 2 | 83.3% | 73.7% | 100% |
| 8.0 ms | 44/57 | 94.5% | 25.3 | +9.7 ms | 12 | 0 | **100%** | 94.6% | 100% |

At 8 ms budget, all 12 critical failures are eliminated. Zero unrecovered
faults across all budget levels; every fault hitting a protected node is
detected and recovered via DMR re-execution.

### YOLOv10n ILP Budget Sweep (1000 MC events, seed=2026)

| Budget | Nodes | Predicted | avg_ms | Overhead | Unprot Fail | Prot Fail | Risk Reduction | Coverage | Recovery |
|--------|-------|-----------|--------|----------|-------------|-----------|----------------|----------|----------|
| 0.5 ms | 14/104 | 11.2% | 16.9 | -0.6 ms* | 25 | 24 | 4.0% | 11.3% | 100% |
| 1.0 ms | 20/104 | 16.0% | 19.2 | +1.8 ms | 25 | 22 | 12.0% | 16.1% | 100% |
| 2.0 ms | 27/104 | 25.0% | 17.1 | -0.3 ms* | 25 | 19 | 24.0% | 24.1% | 100% |
| 4.0 ms | 33/104 | 41.3% | 19.1 | +1.7 ms | 25 | 16 | 36.0% | 40.5% | 100% |
| 8.0 ms | 50/104 | 62.6% | 23.3 | +5.9 ms | 25 | 9 | 64.0% | 61.0% | 100% |

*Baseline avg_ms has a 35 ms outlier; min_ms trend is monotonic (15.4 -> 22.5 ms).

YOLOv10n has 167 runtime nodes vs ResNet50's 57, so the same budget covers
fewer nodes proportionally. Zero unrecovered faults; 100% recovery on protected
nodes at all budget levels.

### Key Findings

1. **DMR is 100% effective**: every fault hitting a protected node is detected
   and recovered, regardless of kernel type (BNNS, im2col_gemm, pointwise, depthwise).
2. **Zero introduced failures**: `new_protected_failures = 0` across all
   experiments; protection never causes a correct inference to become incorrect.
3. **ResNet50 achieves full elimination** at 8 ms budget (100% risk reduction).
4. **YOLOv10n requires higher budget** due to larger graph; 8 ms achieves 64%.

## Windows Results (Historical, AVX2/FMA)

### ResNet50

- Release AVX2 `cpu_generic`: 10-run mean baseline latency `131.65 ms`;
  maximum absolute difference from ONNX Runtime `5.53e-05`, top-1 unchanged.
- Final feedback-calibrated ILP plan: 30 protected modes, predicted critical
  risk reduction `66.14%`, peak extra memory `3,211,264` bytes.
- Clean disjoint runtime test: 512 samples, accuracy unchanged at `97.27%`,
  prediction agreement `100%`, false alarms `2/512`, measured overhead
  `19.53 ms` under the `20 ms` budget.
- Runtime fault Monte Carlo: 1024 activation-byte-weighted, uniformly sampled
  FP32 bit flips; critical failures reduced from `16` to `4` (`75.00%`
  observed reduction; paired bootstrap 95% interval `50.00%` to `94.74%`).

### YOLOv10n

- Full materialized DIOR workload: `18000/2000/3463` train/validation/test
  images. Test `mAP@0.5=0.84279`, `mAP@0.5:0.95=0.62100`, `mAP@0.75=0.67921`.
- The exported ONNX graph contains `308` nodes and compiles without unsupported
  operators. Activation planning `146,338,400 -> 11,468,800` bytes; a real
  DIOR image differs from ONNX Runtime by at most `6.10e-05`.
- Mean Release AVX2 runtime `172.10 ms`.
- Formal injection screen: 125 evaluable images, `170 x 16 = 2720` stratified
  events, `77` critical failures (`65` task-output, `12` controlled execution
  errors). `2000` activation-byte-prior events: `56` critical failures.
- Control-path DMR plan (5 nodes, `48,000` bytes, `+1.22 ms`): removes all 12
  controlled execution errors.
- Full ILP DMR plan (40 nodes, `3,276,800` bytes, `+30.31 ms` / `18.1%`):
  stratified failures `77 -> 13` (`83.12%`), activation-prior `56 -> 43` (`23.21%`).

### Cross-Platform Comparison

| Metric | Windows (AVX2) | macOS (BNNS/AMX) |
|--------|---------------|-----------------|
| ResNet50 baseline | 131.65 ms | 15.6 ms (8.4x faster) |
| YOLOv10n baseline | 172.10 ms | 17.4 ms (9.9x faster) |
| ResNet50 risk reduction (best) | 75% (20 ms budget) | 100% (8 ms budget) |
| DMR recovery rate | 100% | 100% |

Apple Silicon BNNS/AMX acceleration provides an order-of-magnitude speedup,
which means the same latency budget covers more nodes proportionally. The
DMR mechanism itself is platform-independent and achieves 100% recovery on
both platforms.

## Role Of Training

Training is not part of the proposed inference optimization method. It only
adapts public pretrained models into valid remote-sensing workloads so that a
fault can be evaluated as a classification error, missed detection, false
positive, or mAP degradation. ResNet50 defaults to frozen-backbone,
classification-head-only calibration and caches frozen backbone features once
per split; full fine-tuning is used only if the resulting baseline accuracy is
inadequate. YOLOv10n requires DIOR adaptation
because its pretrained COCO task does not define the DIOR evaluation target.

## Interpretation Boundary

The runtime injection experiment measures relative robustness under the stated
software fault model. It is not a radiation-rate estimate and is not evidence
of flight qualification.
