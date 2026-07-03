#!/bin/bash
# Pilot-size ablation: show ILP performance vs Ranger improves with more pilot events.
# Steps 2-6: calibrate → optimize → compile → validate MC → analyze.
# Prereq: pilot_mc5000_ranger_8ms.json and pilot_mc10000_ranger_8ms.json already exist.

set -e
cd /Users/tzp/work/c/SPINNV2

RUNTIME_LIB=runtime/build/libspkv2_runtime.dylib
ONNX=artifacts/models/yolov10n_dior.onnx
BASELINE_SPK=artifacts/models/yolov10n_dior.spk
TARGET=cpu_arm_neon
MODEL_ID=yolov10n_dior
CANDIDATES=artifacts/experiments/macos_reliability/yolov10n_dior/candidates_heterogeneous.json
RANKED_CSV=artifacts/experiments/macos_reliability/yolov10n_dior/ranked_candidates.csv
PLANS_DIR=artifacts/experiments/macos_reliability/yolov10n_dior/plans
MODELS_DIR=artifacts/models
MC_DIR=artifacts/experiments/macos_reliability/yolov10n_dior/mc_results
PILOT_DIR=artifacts/experiments/macos_reliability/yolov10n_dior
MEMORY_BUDGET=8388608
BUDGETS=(0.5 1.0 2.0 4.0 8.0)

# ── Step 2: Calibrate ────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Calibrate with 5k and 10k pilots ==="
for SIZE in 5000 10000; do
    LABEL="${SIZE%000}k"   # 5000→5k, 10000→10k
    PILOT_FILE="$PILOT_DIR/pilot_mc${SIZE}_ranger_8ms.json"
    CAL_OUT="$PILOT_DIR/candidates_calibrated_obsfloor_${LABEL}.json"
    META_OUT="$PILOT_DIR/calibration_metadata_obsfloor_${LABEL}.json"

    if [ ! -f "$PILOT_FILE" ]; then
        echo "ERROR: $PILOT_FILE not found. Run Step 1 first."
        exit 1
    fi

    echo "  Calibrating from ${SIZE}-event pilot..."
    python3 -m research.reliability.calibrate_risk_model \
        --mc-results "$PILOT_FILE" \
        --candidates "$CANDIDATES" \
        --smoothing observed-floor \
        --output "$CAL_OUT" \
        --output-metadata "$META_OUT"
    echo "  -> $CAL_OUT"
done

# ── Step 2 correctness check ─────────────────────────────────────────────────
echo ""
echo "=== Step 2 Correctness Check: CF node capture + RMSE ==="
python3 - <<'PYEOF'
import json, math

def wilson_upper(cf, n, z=1.96):
    if n == 0: return 1.0
    p = cf/n; d = 1 + z*z/n
    c = p + z*z/(2*n)
    s = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return min(1.0, (c+s)/d)

# Load validation actual CF rates (from seed=2027 ranger 8ms)
val = json.load(open("artifacts/experiments/macos_reliability/yolov10n_dior/mc_results/mc1000_cal_ranger_8.0ms.json"))
val_cf = {}; val_ev = {}
for obs in val["observations"]:
    nid = obs["node_id"]
    val_ev[nid] = val_ev.get(nid, 0) + 1
    if obs["unprotected_critical_failure"]:
        val_cf[nid] = val_cf.get(nid, 0) + 1
val_rate = {nid: val_cf.get(nid, 0)/val_ev[nid] for nid in val_ev}
val_cf_nodes = {nid for nid, r in val_rate.items() if r > 0}

pilot_dir = "artifacts/experiments/macos_reliability/yolov10n_dior"
for size in [1000, 5000, 10000]:
    if size == 1000:
        pilot_file = "artifacts/experiments/macos_reliability/yolov10n_dior/mc_results/mc1000_ranger_8.0ms.json"
    else:
        pilot_file = f"{pilot_dir}/pilot_mc{size}_ranger_8ms.json"

    if not __import__("os").path.exists(pilot_file):
        print(f"  {size:6d} events: file not found")
        continue

    pilot = json.load(open(pilot_file))
    p_cf = {}; p_ev = {}
    for obs in pilot["observations"]:
        nid = obs["node_id"]
        p_ev[nid] = p_ev.get(nid, 0) + 1
        if obs["unprotected_critical_failure"]:
            p_cf[nid] = p_cf.get(nid, 0) + 1

    pilot_cf_nodes = {nid for nid, c in p_cf.items() if c > 0}
    captured = pilot_cf_nodes & val_cf_nodes
    capture_rate = len(captured) / len(val_cf_nodes) if val_cf_nodes else 0

    # RMSE: compare obsfloor predicted CP to actual CF rate
    if size == 1000:
        meta_file = f"{pilot_dir}/calibration_metadata_obsfloor.json"
    else:
        meta_file = f"{pilot_dir}/calibration_metadata_obsfloor_{size//1000}k.json"

    if not __import__("os").path.exists(meta_file):
        print(f"  {size:6d} events: capture {len(captured)}/{len(val_cf_nodes)} ({capture_rate:.0%}), no meta file")
        continue

    meta = json.load(open(meta_file))
    sq_errs = []
    for nid_str, nd in meta["per_node"].items():
        nid = int(nid_str)
        if nid in val_rate:
            pred = nd["calibrated_critical_probability"]
            actual = val_rate[nid]
            sq_errs.append((pred - actual)**2)
    rmse = math.sqrt(sum(sq_errs)/len(sq_errs)) if sq_errs else float("nan")
    print(f"  {size:6d} events: CF capture {len(captured)}/{len(val_cf_nodes)} ({capture_rate:.0%}), RMSE={rmse:.4f}")
PYEOF

# ── Step 3: ILP Optimize ─────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Optimize ILP plans (5 budgets × 2 pilot sizes) ==="
for SIZE in 5000 10000; do
    LABEL="${SIZE%000}k"
    CAL_CANDS="$PILOT_DIR/candidates_calibrated_obsfloor_${LABEL}.json"
    for BUDGET in "${BUDGETS[@]}"; do
        PLAN="$PLANS_DIR/ilp_obsfloor_${LABEL}_${BUDGET}ms.json"
        if [ -f "$PLAN" ]; then
            echo "  SKIP plan ${LABEL} ${BUDGET}ms"
            continue
        fi
        echo "  Plan: ilp obsfloor ${LABEL} ${BUDGET}ms ..."
        python3 -m research.reliability.optimize_plan \
            "$CAL_CANDS" \
            --method ilp \
            --latency-budget-ms "$BUDGET" \
            --memory-budget-bytes "$MEMORY_BUDGET" \
            --model-id "$MODEL_ID" \
            --output "$PLAN"
    done
done

# ── Step 3 plan correctness check ────────────────────────────────────────────
echo ""
echo "=== Step 3 Correctness Check: plan latency ≤ budget ==="
python3 - <<'PYEOF'
import json, glob, os

def label(size): return f"{size//1000}k"

plans_dir = "artifacts/experiments/macos_reliability/yolov10n_dior/plans"
for size in [5000, 10000]:
    for budget in [0.5, 1.0, 2.0, 4.0, 8.0]:
        plan_f = f"{plans_dir}/ilp_obsfloor_{label(size)}_{budget}ms.json"
        if not os.path.exists(plan_f):
            print(f"  MISSING {label(size)} {budget}ms")
            continue
        p = json.load(open(plan_f))
        pred = p["optimizer"]["predicted_latency_overhead_ms"]
        n_nodes = len(p["nodes"])
        modes = [n["mode"] for n in p["nodes"]]
        n_dmr = modes.count("dmr_compare_rerun")
        n_rg = modes.count("range_guard_rerun")
        status = "OK" if pred <= budget + 0.001 else "OVER-BUDGET!"
        print(f"  {label(size)} {budget}ms: {n_nodes} nodes (DMR={n_dmr} RG={n_rg}), pred={pred:.3f}ms [{status}]")
PYEOF

# ── Step 4: Compile SPKs ─────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Compile SPKs (10 files) ==="
for SIZE in 5000 10000; do
    LABEL="${SIZE%000}k"
    for BUDGET in "${BUDGETS[@]}"; do
        PLAN="$PLANS_DIR/ilp_obsfloor_${LABEL}_${BUDGET}ms.json"
        SPK="$MODELS_DIR/${MODEL_ID}_ilp_obsfloor_${LABEL}_${BUDGET}ms.spk"
        if [ -f "$SPK" ]; then
            echo "  SKIP SPK ${LABEL} ${BUDGET}ms"
            continue
        fi
        echo "  Compile: ${LABEL} ${BUDGET}ms ..."
        python3 -m compiler.cli compile "$ONNX" \
            --target "$TARGET" \
            -o "$SPK" \
            --protection-plan "$PLAN" 2>/dev/null
        echo "  -> $SPK ($(du -sh "$SPK" | cut -f1))"
    done
done

# ── Step 4 performance benchmark ─────────────────────────────────────────────
echo ""
echo "=== Step 4 Performance Check: actual overhead vs predicted ==="
python3 - <<'PYEOF'
import json, os, subprocess

bench = "runtime/build/spkv2_bench"
if not os.path.exists(bench):
    print("  spkv2_bench not found, skipping timing check")
    import sys; sys.exit(0)

baseline_spk = "artifacts/models/yolov10n_dior.spk"
# Get baseline time
def bench_spk(spk, warmup=3, runs=10):
    try:
        r = subprocess.run([bench, spk, "--warmup", str(warmup), "--runs", str(runs)],
                           capture_output=True, text=True, timeout=60)
        for line in r.stdout.split("\n"):
            if "avg" in line.lower() or "mean" in line.lower():
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.replace(".","").isdigit():
                        return float(p)
    except Exception as e:
        print(f"  bench error: {e}")
    return None

baseline_ms = bench_spk(baseline_spk)
if baseline_ms is None:
    print("  Could not measure baseline, skipping")
    import sys; sys.exit(0)
print(f"  Baseline: {baseline_ms:.2f}ms")

models_dir = "artifacts/models"
plans_dir = "artifacts/experiments/macos_reliability/yolov10n_dior/plans"
def label(size): return f"{size//1000}k"
for size in [5000, 10000]:
    for budget in [0.5, 1.0, 2.0, 4.0, 8.0]:
        spk = f"{models_dir}/yolov10n_dior_ilp_obsfloor_{label(size)}_{budget}ms.spk"
        plan_f = f"{plans_dir}/ilp_obsfloor_{label(size)}_{budget}ms.json"
        if not os.path.exists(spk): continue
        prot_ms = bench_spk(spk, warmup=3, runs=5)
        if prot_ms is None: continue
        pred = json.load(open(plan_f))["optimizer"]["predicted_latency_overhead_ms"]
        actual_oh = prot_ms - baseline_ms
        ratio = actual_oh / pred if pred > 0 else float("nan")
        ok = "OK" if 0.5 <= ratio <= 2.0 else "WARNING"
        print(f"  {label(size)} {budget}ms: actual_oh={actual_oh:.2f}ms pred={pred:.2f}ms ratio={ratio:.2f} [{ok}]")
PYEOF

# ── Step 5: Validation MC ────────────────────────────────────────────────────
echo ""
echo "=== Step 5: Validation MC (seed=2027, 1000 events) ==="
for SIZE in 5000 10000; do
    LABEL="${SIZE%000}k"
    for BUDGET in "${BUDGETS[@]}"; do
        SPK="$MODELS_DIR/${MODEL_ID}_ilp_obsfloor_${LABEL}_${BUDGET}ms.spk"
        PLAN="$PLANS_DIR/ilp_obsfloor_${LABEL}_${BUDGET}ms.json"
        MC_OUT="$MC_DIR/mc1000_obsfloor_ilp_${LABEL}_${BUDGET}ms.json"
        if [ -f "$MC_OUT" ]; then
            echo "  SKIP validation ${LABEL} ${BUDGET}ms"
            continue
        fi
        echo "  MC: ilp_obsfloor_${LABEL}_${BUDGET}ms ..."
        python3 -m research.reliability.evaluate_detection_faults \
            --library "$RUNTIME_LIB" \
            --baseline-spk "$BASELINE_SPK" \
            --protected-spk "$SPK" \
            --plan "$PLAN" \
            --ranked-csv "$RANKED_CSV" \
            --test-images artifacts/data/dior/images/test \
            --test-samples 128 --events 1000 --seed 2027 \
            --output "$MC_OUT" 2>/dev/null
    done
done

# ── Step 5 correctness check ──────────────────────────────────────────────────
echo ""
echo "=== Step 5 Correctness Check: new_failures=0, CF monotone ==="
python3 - <<'PYEOF'
import json, os

mc_dir = "artifacts/experiments/macos_reliability/yolov10n_dior/mc_results"
budgets = [0.5, 1.0, 2.0, 4.0, 8.0]

# Reference: existing 1k results and Ranger
ref_1k = {}
ref_ranger = {}
for b in budgets:
    f1k = f"{mc_dir}/mc1000_obsfloor_ilp_{b}ms.json"
    fr  = f"{mc_dir}/mc1000_obsfloor_ranger_{b}ms.json"
    if os.path.exists(f1k):
        ref_1k[b] = json.load(open(f1k))["totals"]["protected_critical_failures"]
    if os.path.exists(fr):
        ref_ranger[b] = json.load(open(fr))["totals"]["protected_critical_failures"]

all_ok = True
print(f"{'Budget':>8} | {'1k CF':>6} | {'5k CF':>6} | {'10k CF':>7} | {'Ranger':>7} | New Fail? | Trend")
print("-"*75)
for b in budgets:
    row = [f"{b:>8}"]
    vals = {}
    for size, lbl in [(1000, "1k"), (5000, "5k"), (10000, "10k")]:
        if size == 1000:
            f = f"{mc_dir}/mc1000_obsfloor_ilp_{b}ms.json"
        else:
            f = f"{mc_dir}/mc1000_obsfloor_ilp_{lbl}_{b}ms.json"
        if os.path.exists(f):
            d = json.load(open(f))
            cf = d["totals"]["protected_critical_failures"]
            new_f = d["totals"]["new_protected_failures"]
            vals[lbl] = cf
            if new_f > 0:
                all_ok = False
                print(f"  ERROR: {lbl} {b}ms has new_protected_failures={new_f}!")
        else:
            vals[lbl] = None

    cf1 = vals.get("1k", None); cf5 = vals.get("5k", None); cf10 = vals.get("10k", None)
    ranger_cf = ref_ranger.get(b, None)
    trend = ""
    if cf1 is not None and cf5 is not None and cf10 is not None:
        if cf5 <= cf1 and cf10 <= cf5:
            trend = "✓ monotone"
        elif cf5 <= cf1 or cf10 <= cf5:
            trend = "~ partial"
        else:
            trend = "✗ non-monotone"
    s1 = f"{cf1:6}" if cf1 is not None else "     -"
    s5 = f"{cf5:6}" if cf5 is not None else "     -"
    s10 = f"{cf10:7}" if cf10 is not None else "      -"
    sr = f"{ranger_cf:7}" if ranger_cf is not None else "      -"
    print(f"{b:>8} | {s1} | {s5} | {s10} | {sr} | OK        | {trend}")

if all_ok:
    print("\n✓ All new_protected_failures = 0")
PYEOF

# ── Step 6: Plot ──────────────────────────────────────────────────────────────
echo ""
echo "=== Step 6: Generate pilot ablation figure ==="
python3 - <<'PYEOF'
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mc_dir = "artifacts/experiments/macos_reliability/yolov10n_dior/mc_results"
out_dir = "artifacts/experiments/macos_reliability/yolov10n_dior/figures"
budgets = [0.5, 1.0, 2.0, 4.0, 8.0]

def load_cf(template, sizes_or_labels):
    results = {}
    for key, fname in sizes_or_labels.items():
        vals = []
        for b in budgets:
            f = fname.format(budget=b)
            if os.path.exists(f):
                d = json.load(open(f))
                vals.append(d["totals"]["protected_critical_failures"])
            else:
                vals.append(None)
        results[key] = vals
    return results

series = {
    "ILP — 1k pilot":  [f"{mc_dir}/mc1000_obsfloor_ilp_{b}ms.json" for b in budgets],
    "ILP — 5k pilot":  [f"{mc_dir}/mc1000_obsfloor_ilp_5k_{b}ms.json" for b in budgets],
    "ILP — 10k pilot": [f"{mc_dir}/mc1000_obsfloor_ilp_10k_{b}ms.json" for b in budgets],
    "Ranger":          [f"{mc_dir}/mc1000_obsfloor_ranger_{b}ms.json" for b in budgets],
}
colors = {
    "ILP — 1k pilot":  "#4E79A7",
    "ILP — 5k pilot":  "#59A14F",
    "ILP — 10k pilot": "#F28E2B",
    "Ranger":          "#999999",
}
markers = {
    "ILP — 1k pilot":  "o",
    "ILP — 5k pilot":  "s",
    "ILP — 10k pilot": "^",
    "Ranger":          "D",
}
linestyles = {
    "ILP — 1k pilot":  "-",
    "ILP — 5k pilot":  "-",
    "ILP — 10k pilot": "-",
    "Ranger":          "--",
}

fig, ax = plt.subplots(figsize=(8, 5))
for label, files in series.items():
    xs, ys = [], []
    for b, f in zip(budgets, files):
        if os.path.exists(f):
            d = json.load(open(f))
            xs.append(b)
            ys.append(d["totals"]["protected_critical_failures"])
    if xs:
        ax.plot(xs, ys, marker=markers[label], color=colors[label],
                linestyle=linestyles[label], linewidth=2, markersize=8, label=label)

ax.set_xlabel("Latency budget (ms)", fontsize=12)
ax.set_ylabel("Protected critical failures", fontsize=12)
ax.set_title("YOLOv10n DIOR — ILP pilot size ablation\n(ObsFloor calibration, seed=2027, baseline=29 CF)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xticks(budgets)
ax.set_ylim(bottom=-0.5)
ax.grid(alpha=0.3)
fig.tight_layout()
out = f"{out_dir}/pilot_ablation_yolov10n.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# Also print summary table
print("\nPilot ablation results:")
print(f"{'Budget':>8} | {'1k':>5} | {'5k':>5} | {'10k':>6} | {'Ranger':>8}")
print("-" * 48)
for i, b in enumerate(budgets):
    row = [f"{b:>8}"]
    for label, files in series.items():
        f = files[i]
        if os.path.exists(f):
            d = json.load(open(f))
            cf = d["totals"]["protected_critical_failures"]
            row.append(f"{cf:>{'5' if 'Ranger' not in label else '8'}}")
        else:
            row.append(f"{'     -' if 'Ranger' not in label else '       -'}")
    print(" | ".join(row))
PYEOF

echo ""
echo "=== Pilot ablation complete ==="
