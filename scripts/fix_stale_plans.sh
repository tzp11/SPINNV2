#!/bin/bash
# Fix stale ASPIS and Ahmadilivani plans that used old candidates (without Taylor)
# These plans at 0.5/1.0/2.0/4.0ms were generated Jun 21 with old candidates_heterogeneous.json
# New candidates (with Taylor) were generated Jun 22. Plans at 3.0/5.0/6.0ms are correct.
set -e
cd /Users/tzp/work/c/SPINNV2

RUNTIME_LIB=runtime/build/libspkv2_runtime.dylib
TARGET=cpu_arm_neon
MODELS_DIR=artifacts/models
MC_SEED=2027
MC_EVENTS=5000
MEMORY_BUDGET=8388608

YOLO_ONNX=$MODELS_DIR/yolov10n_dior.onnx
YOLO_SPK=$MODELS_DIR/yolov10n_dior.spk
YOLO_DIR=artifacts/experiments/macos_reliability/yolov10n_dior
YOLO_CANDIDATES=$YOLO_DIR/candidates_heterogeneous.json
YOLO_PLANS=$YOLO_DIR/plans
YOLO_MC=$YOLO_DIR/mc_results
YOLO_RANKED=$YOLO_DIR/ranked_candidates.csv
YOLO_TEST_IMAGES=artifacts/data/dior/images/test

STALE_METHODS="aspis ahmadilivani"
STALE_BUDGETS="0.5 1.0 2.0 4.0"

echo "=== Deleting stale plans, SPKs, and MC results ==="
for method in $STALE_METHODS; do
    for budget in $STALE_BUDGETS; do
        tag="${method}_obsfloor_${budget}ms"
        plan="$YOLO_PLANS/${tag}.json"
        spk="$MODELS_DIR/yolov10n_dior_${tag}.spk"
        mc_out="$YOLO_MC/mc${MC_EVENTS}_obsfloor_${method}_${budget}ms.json"

        echo "  Removing: $tag"
        rm -f "$plan" "$spk" "$mc_out"
    done
done

echo ""
echo "=== Re-running 8 experiments with correct candidates ==="
completed=0
for method in $STALE_METHODS; do
    for budget in $STALE_BUDGETS; do
        completed=$((completed + 1))
        tag="${method}_obsfloor_${budget}ms"
        plan="$YOLO_PLANS/${tag}.json"
        spk="$MODELS_DIR/yolov10n_dior_${tag}.spk"
        mc_out="$YOLO_MC/mc${MC_EVENTS}_obsfloor_${method}_${budget}ms.json"

        # Optimize
        echo "[$completed/8] Optimize $tag"
        python3 -m research.reliability.optimize_plan \
            "$YOLO_CANDIDATES" \
            --method "$method" \
            --latency-budget-ms "$budget" \
            --memory-budget-bytes "$MEMORY_BUDGET" \
            --model-id yolov10n_dior \
            --output "$plan"

        # Compile
        echo "  Compile: $tag"
        python3 -m compiler.cli compile "$YOLO_ONNX" \
            --target "$TARGET" -o "$spk" \
            --protection-plan "$plan" 2>/dev/null

        # MC
        echo "  MC: $tag ($MC_EVENTS events)"
        python3 -m research.reliability.evaluate_detection_faults \
            --library "$RUNTIME_LIB" \
            --baseline-spk "$YOLO_SPK" \
            --protected-spk "$spk" \
            --plan "$plan" \
            --ranked-csv "$YOLO_RANKED" \
            --test-images "$YOLO_TEST_IMAGES" \
            --test-samples 128 \
            --events "$MC_EVENTS" \
            --seed "$MC_SEED" \
            --output "$mc_out" 2>/dev/null

        # Quick verify
        cf=$(python3 -c "import json; d=json.load(open('$mc_out')); print(d['totals']['protected_critical_failures'])")
        echo "  Result: CF=$cf"
    done
done
echo "=== Fix complete ==="
