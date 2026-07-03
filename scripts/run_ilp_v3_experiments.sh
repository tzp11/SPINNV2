#!/bin/bash
# ILP-v3 (Coverage-First ILP) experiments: 7 budgets × 2 models, 5000 events
set -e
cd /Users/tzp/work/c/SPINNV2

RUNTIME_LIB=runtime/build/libspkv2_runtime.dylib
TARGET=cpu_arm_neon
MODELS_DIR=artifacts/models
MC_SEED=2027
MC_EVENTS=5000
MEMORY_BUDGET=8388608
METHOD=ilp-v3

# ── ResNet50 ─────────────────────────────────────────────────────────────────
R50_ONNX=$MODELS_DIR/resnet50_eurosat.onnx
R50_SPK=$MODELS_DIR/resnet50_eurosat.spk
R50_DIR=artifacts/experiments/macos_reliability/resnet50
R50_CAL=$R50_DIR/candidates_calibrated_obsfloor.json
R50_META=$R50_DIR/calibration_metadata_obsfloor.json
R50_PLANS=$R50_DIR/plans
R50_MC=$R50_DIR/mc_results
R50_RANKED=$R50_DIR/ranked_candidates.csv
R50_BUDGETS="0.5 1.0 1.5 2.0 2.5 3.0 3.5"

# ── YOLOv10n ─────────────────────────────────────────────────────────────────
YOLO_ONNX=$MODELS_DIR/yolov10n_dior.onnx
YOLO_SPK=$MODELS_DIR/yolov10n_dior.spk
YOLO_DIR=artifacts/experiments/macos_reliability/yolov10n_dior
YOLO_CAL=$YOLO_DIR/candidates_calibrated_obsfloor.json
YOLO_META=$YOLO_DIR/calibration_metadata_obsfloor.json
YOLO_PLANS=$YOLO_DIR/plans
YOLO_MC=$YOLO_DIR/mc_results
YOLO_RANKED=$YOLO_DIR/ranked_candidates.csv
YOLO_BUDGETS="0.5 1.0 2.0 3.0 4.0 5.0 6.0"
YOLO_TEST_IMAGES=artifacts/data/dior/images/test

echo "========================================"
echo " ILP-v3 experiments (Coverage-First ILP)"
echo "========================================"

completed=0
total=14

# ── ResNet50 ──
mkdir -p "$R50_PLANS" "$R50_MC"
for budget in $R50_BUDGETS; do
    completed=$((completed + 1))
    tag="${METHOD}_obsfloor_${budget}ms"
    plan="$R50_PLANS/${tag}.json"
    spk="$MODELS_DIR/resnet50_eurosat_${tag}.spk"
    mc_out="$R50_MC/mc${MC_EVENTS}_obsfloor_${METHOD}_${budget}ms.json"

    if [ -f "$mc_out" ]; then
        echo "[$completed/$total] SKIP ResNet50 $budget ms (exists)"
        continue
    fi

    echo "[$completed/$total] ResNet50 $budget ms"

    # Optimize
    OPT_ARGS=(
        "$R50_CAL"
        --method "$METHOD"
        --latency-budget-ms "$budget"
        --memory-budget-bytes "$MEMORY_BUDGET"
        --model-id resnet50_eurosat
        --output "$plan"
    )
    if [ -f "$R50_META" ]; then
        OPT_ARGS+=(--calibration-metadata "$R50_META")
    fi
    python3 -m research.reliability.optimize_plan "${OPT_ARGS[@]}"

    # Compile
    python3 -m compiler.cli compile "$R50_ONNX" \
        --target "$TARGET" -o "$spk" \
        --protection-plan "$plan" 2>/dev/null

    # MC
    python3 -m research.reliability.evaluate_runtime_faults \
        --library "$RUNTIME_LIB" \
        --baseline-spk "$R50_SPK" \
        --protected-spk "$spk" \
        --plan "$plan" \
        --ranked-csv "$R50_RANKED" \
        --events "$MC_EVENTS" \
        --seed "$MC_SEED" \
        --output "$mc_out" 2>/dev/null

    cf=$(python3 -c "import json; d=json.load(open('$mc_out')); print(d['totals']['protected_critical_failures'])")
    echo "  CF=$cf"
done

# ── YOLOv10n ──
mkdir -p "$YOLO_PLANS" "$YOLO_MC"
for budget in $YOLO_BUDGETS; do
    completed=$((completed + 1))
    tag="${METHOD}_obsfloor_${budget}ms"
    plan="$YOLO_PLANS/${tag}.json"
    spk="$MODELS_DIR/yolov10n_dior_${tag}.spk"
    mc_out="$YOLO_MC/mc${MC_EVENTS}_obsfloor_${METHOD}_${budget}ms.json"

    if [ -f "$mc_out" ]; then
        echo "[$completed/$total] SKIP YOLOv10n $budget ms (exists)"
        continue
    fi

    echo "[$completed/$total] YOLOv10n $budget ms"

    # Optimize
    OPT_ARGS=(
        "$YOLO_CAL"
        --method "$METHOD"
        --latency-budget-ms "$budget"
        --memory-budget-bytes "$MEMORY_BUDGET"
        --model-id yolov10n_dior
        --output "$plan"
    )
    if [ -f "$YOLO_META" ]; then
        OPT_ARGS+=(--calibration-metadata "$YOLO_META")
    fi
    python3 -m research.reliability.optimize_plan "${OPT_ARGS[@]}"

    # Compile
    python3 -m compiler.cli compile "$YOLO_ONNX" \
        --target "$TARGET" -o "$spk" \
        --protection-plan "$plan" 2>/dev/null

    # MC
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

    cf=$(python3 -c "import json; d=json.load(open('$mc_out')); print(d['totals']['protected_critical_failures'])")
    echo "  CF=$cf"
done

echo ""
echo "=== ILP-v3 experiments complete ==="
