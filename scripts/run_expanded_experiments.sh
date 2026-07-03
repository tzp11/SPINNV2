#!/bin/bash
# Expanded experiments: 9 methods × multiple budgets × 2 models
# Calibration: ObsFloor
# Validation: 5000 events, seed=2027
#
# Prerequisites:
#   - ResNet50: pilot MC, calibrated candidates, candidates_heterogeneous.json with Taylor
#   - YOLOv10n: pilot MC, calibrated candidates, candidates_heterogeneous.json with Taylor
#   - Compiled runtime library

set -e
cd /Users/tzp/work/c/SPINNV2

RUNTIME_LIB=runtime/build/libspkv2_runtime.dylib
TARGET=cpu_arm_neon
MODELS_DIR=artifacts/models
MC_SEED=2027
MC_EVENTS=5000
MEMORY_BUDGET=8388608

METHODS="ilp ilp-static ilp-v2 ranger filr aspis ruospo ahmadilivani uniform-dmr"

# ── ResNet50 ─────────────────────────────────────────────────────────────────
R50_ONNX=$MODELS_DIR/resnet50_eurosat.onnx
R50_SPK=$MODELS_DIR/resnet50_eurosat.spk
R50_DIR=artifacts/experiments/macos_reliability/resnet50
R50_CANDIDATES=$R50_DIR/candidates_heterogeneous.json
R50_PLANS=$R50_DIR/plans
R50_MC=$R50_DIR/mc_results
R50_META=$R50_DIR/calibration_metadata_obsfloor.json
R50_CAL=$R50_DIR/candidates_calibrated_obsfloor.json
R50_RANKED=$R50_DIR/ranked_candidates.csv
R50_BUDGETS="0.5 1.0 1.5 2.0 2.5 3.0 3.5"

# ── YOLOv10n ─────────────────────────────────────────────────────────────────
YOLO_ONNX=$MODELS_DIR/yolov10n_dior.onnx
YOLO_SPK=$MODELS_DIR/yolov10n_dior.spk
YOLO_DIR=artifacts/experiments/macos_reliability/yolov10n_dior
YOLO_CANDIDATES=$YOLO_DIR/candidates_heterogeneous.json
YOLO_PLANS=$YOLO_DIR/plans
YOLO_MC=$YOLO_DIR/mc_results
YOLO_META=$YOLO_DIR/calibration_metadata_obsfloor.json
YOLO_CAL=$YOLO_DIR/candidates_calibrated_obsfloor.json
YOLO_RANKED=$YOLO_DIR/ranked_candidates.csv
YOLO_BUDGETS="0.5 1.0 2.0 3.0 4.0 5.0 6.0"
YOLO_TEST_IMAGES=artifacts/data/dior/images/test

run_resnet50() {
    echo ""
    echo "========================================"
    echo " ResNet50 EuroSAT expanded experiments"
    echo "========================================"
    mkdir -p "$R50_PLANS" "$R50_MC"

    total=0; done=0; skip=0

    for method in $METHODS; do
        for budget in $R50_BUDGETS; do
            total=$((total + 1))
            tag="${method}_obsfloor_${budget}ms"
            plan="$R50_PLANS/${tag}.json"
            spk="$MODELS_DIR/resnet50_eurosat_${tag}.spk"
            mc_out="$R50_MC/mc${MC_EVENTS}_obsfloor_${method}_${budget}ms.json"

            if [ -f "$mc_out" ]; then
                skip=$((skip + 1))
                continue
            fi

            # Step 1: Optimize plan
            if [ ! -f "$plan" ]; then
                echo "[$((done+skip+1))/$total] Optimize $tag"
                case "$method" in
                    ilp|ilp-static|ilp-v2)
                        CANDS="$R50_CAL"
                        ;;
                    *)
                        CANDS="$R50_CANDIDATES"
                        ;;
                esac
                OPT_ARGS=(
                    "$CANDS"
                    --method "$method"
                    --latency-budget-ms "$budget"
                    --memory-budget-bytes "$MEMORY_BUDGET"
                    --model-id resnet50_eurosat
                    --output "$plan"
                )
                if [ "$method" = "ilp-v2" ] && [ -f "$R50_META" ]; then
                    OPT_ARGS+=(--calibration-metadata "$R50_META")
                fi
                python3 -m research.reliability.optimize_plan "${OPT_ARGS[@]}"
            fi

            # Step 2: Compile SPK
            if [ ! -f "$spk" ]; then
                echo "  Compile: $tag"
                python3 -m compiler.cli compile "$R50_ONNX" \
                    --target "$TARGET" -o "$spk" \
                    --protection-plan "$plan" 2>/dev/null
            fi

            # Step 3: Validate MC
            echo "  MC: $tag ($MC_EVENTS events)"
            python3 -m research.reliability.evaluate_runtime_faults \
                --library "$RUNTIME_LIB" \
                --baseline-spk "$R50_SPK" \
                --protected-spk "$spk" \
                --plan "$plan" \
                --ranked-csv "$R50_RANKED" \
                --events "$MC_EVENTS" \
                --seed "$MC_SEED" \
                --output "$mc_out" 2>/dev/null

            done=$((done + 1))
        done
    done
    echo "=== ResNet50 complete: $done new, $skip skipped, $total total ==="
}

run_yolov10n() {
    echo ""
    echo "========================================"
    echo " YOLOv10n DIOR expanded experiments"
    echo "========================================"
    mkdir -p "$YOLO_PLANS" "$YOLO_MC"

    total=0; done=0; skip=0

    for method in $METHODS; do
        for budget in $YOLO_BUDGETS; do
            total=$((total + 1))
            tag="${method}_obsfloor_${budget}ms"
            plan="$YOLO_PLANS/${tag}.json"
            spk="$MODELS_DIR/yolov10n_dior_${tag}.spk"
            mc_out="$YOLO_MC/mc${MC_EVENTS}_obsfloor_${method}_${budget}ms.json"

            if [ -f "$mc_out" ]; then
                skip=$((skip + 1))
                continue
            fi

            # Step 1: Optimize plan
            if [ ! -f "$plan" ]; then
                echo "[$((done+skip+1))/$total] Optimize $tag"
                case "$method" in
                    ilp|ilp-static|ilp-v2)
                        CANDS="$YOLO_CAL"
                        ;;
                    *)
                        CANDS="$YOLO_CANDIDATES"
                        ;;
                esac
                OPT_ARGS=(
                    "$CANDS"
                    --method "$method"
                    --latency-budget-ms "$budget"
                    --memory-budget-bytes "$MEMORY_BUDGET"
                    --model-id yolov10n_dior
                    --output "$plan"
                )
                if [ "$method" = "ilp-v2" ] && [ -f "$YOLO_META" ]; then
                    OPT_ARGS+=(--calibration-metadata "$YOLO_META")
                fi
                python3 -m research.reliability.optimize_plan "${OPT_ARGS[@]}"
            fi

            # Step 2: Compile SPK
            if [ ! -f "$spk" ]; then
                echo "  Compile: $tag"
                python3 -m compiler.cli compile "$YOLO_ONNX" \
                    --target "$TARGET" -o "$spk" \
                    --protection-plan "$plan" 2>/dev/null
            fi

            # Step 3: Validate MC
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

            done=$((done + 1))
        done
    done
    echo "=== YOLOv10n complete: $done new, $skip skipped, $total total ==="
}

# Run both models
run_resnet50
run_yolov10n

echo ""
echo "=== All expanded experiments complete ==="
