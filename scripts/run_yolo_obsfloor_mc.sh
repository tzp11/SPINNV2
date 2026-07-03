#!/bin/bash
# YOLOv10n ObsFloor MC evaluation — 35 runs (7 methods × 5 budgets)
# Run after Wilson+Observed batch completes

set -e
cd /Users/tzp/work/c/SPINNV2

RUNTIME_LIB=runtime/build/libspkv2_runtime.dylib
BASELINE_SPK=artifacts/models/yolov10n_dior.spk
PLANS_DIR=artifacts/experiments/macos_reliability/yolov10n_dior/plans
MODELS_DIR=artifacts/models
MC_DIR=artifacts/experiments/macos_reliability/yolov10n_dior/mc_results
RANKED_CSV=artifacts/experiments/macos_reliability/yolov10n_dior/ranked_candidates.csv
TEST_IMAGES=artifacts/data/dior/images/test
MC_SEED=2027
MC_EVENTS=1000

total=0
done=0
skip=0

for method in ilp filr aspis ruospo ahmadilivani ranger uniform-dmr; do
  for budget in 0.5 1.0 2.0 4.0 8.0; do
    total=$((total + 1))
    tag="${method}_obsfloor_${budget}ms"
    mc_out="$MC_DIR/mc${MC_EVENTS}_obsfloor_${method}_${budget}ms.json"
    plan="$PLANS_DIR/${tag}.json"
    spk="$MODELS_DIR/yolov10n_dior_${tag}.spk"

    if [ -f "$mc_out" ]; then
      skip=$((skip + 1))
      continue
    fi

    echo "[$((done+skip+1))/$total] MC $tag ..."
    python3 -m research.reliability.evaluate_detection_faults \
      --library "$RUNTIME_LIB" \
      --baseline-spk "$BASELINE_SPK" \
      --protected-spk "$spk" \
      --plan "$plan" \
      --ranked-csv "$RANKED_CSV" \
      --test-images "$TEST_IMAGES" \
      --test-samples 128 \
      --events "$MC_EVENTS" \
      --seed "$MC_SEED" \
      --output "$mc_out" 2>/dev/null

    done=$((done + 1))
  done
done
echo "=== Complete: $done new MC runs, $skip skipped, $total total ==="
