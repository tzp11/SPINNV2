#!/usr/bin/env bash
# Run calibrated full-method MC comparison for ResNet50.
#
# Prereqs: candidates_calibrated.json already generated via calibrate_risk_model.py
#
# Usage: bash research/reliability/scripts/run_resnet50_calibrated_mc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RUNTIME_LIB=runtime/build/libspkv2_runtime.dylib
BASELINE_SPK=artifacts/models/resnet50_eurosat.spk
ONNX=artifacts/models/resnet50_eurosat.onnx
CANDIDATES=artifacts/experiments/macos_reliability/resnet50/candidates_calibrated.json
RANKED_CSV=artifacts/experiments/macos_reliability/resnet50/ranked_candidates.csv
PLANS_DIR=artifacts/experiments/macos_reliability/resnet50/plans
MODELS_DIR=artifacts/models
MC_DIR=artifacts/experiments/macos_reliability/resnet50/mc_results
MODEL_ID=resnet50_eurosat
MEMORY_BUDGET=8388608
MC_SEED=2027
MC_EVENTS=1000
TARGET=cpu_arm_neon

METHODS=(ilp filr aspis ruospo ahmadilivani ranger uniform-dmr)
BUDGETS=(0.5 1.0 2.0 4.0 8.0)

echo "=== ResNet50 Calibrated MC Comparison ==="
echo "  Methods: ${METHODS[*]}"
echo "  Budgets: ${BUDGETS[*]}"
echo "  MC seed: $MC_SEED, events: $MC_EVENTS"
echo ""

for method in "${METHODS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    tag="${method}_cal_${budget}ms"
    plan="$PLANS_DIR/${method}_cal_${budget}ms.json"
    spk="$MODELS_DIR/${MODEL_ID}_${tag}.spk"
    mc_out="$MC_DIR/mc${MC_EVENTS}_cal_${method}_${budget}ms.json"

    # Skip if MC result already exists
    if [ -f "$mc_out" ]; then
      echo "SKIP $tag (already exists)"
      continue
    fi

    echo "--- $tag ---"

    # 1. Generate plan
    echo "  [1/3] Optimizing plan..."
    python3 -m research.reliability.optimize_plan \
      "$CANDIDATES" \
      --method "$method" \
      --latency-budget-ms "$budget" \
      --memory-budget-bytes "$MEMORY_BUDGET" \
      --model-id "$MODEL_ID" \
      --output "$plan"

    # 2. Compile protected SPK
    echo "  [2/3] Compiling SPK..."
    python3 -m compiler.cli compile "$ONNX" \
      --target "$TARGET" \
      -o "$spk" \
      --protection-plan "$plan"

    # 3. Run MC evaluation
    echo "  [3/3] Running MC ($MC_EVENTS events, seed=$MC_SEED)..."
    python3 -m research.reliability.evaluate_runtime_faults \
      --library "$RUNTIME_LIB" \
      --baseline-spk "$BASELINE_SPK" \
      --protected-spk "$spk" \
      --plan "$plan" \
      --ranked-csv "$RANKED_CSV" \
      --test-samples 128 \
      --events "$MC_EVENTS" \
      --seed "$MC_SEED" \
      --output "$mc_out"

    echo ""
  done
done

echo "=== All done. Results in $MC_DIR ==="
