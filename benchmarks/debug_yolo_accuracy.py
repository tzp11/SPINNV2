#!/usr/bin/env python3
"""Per-node accuracy diagnosis for YOLOv10n: SPINNV2 vs ORT."""
from __future__ import annotations
import sys, json, subprocess
from pathlib import Path
import numpy as np
import onnx, onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build" / "models" / "yolov10n.onnx"

def main():
    model = onnx.load(str(MODEL))
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    x = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    input_name = model.graph.input[0].name

    # ORT full run
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {input_name: x})
    ort_flat = ort_out[0].reshape(-1)

    # SPINNV2 full run
    spk = ROOT / "build" / "bench_simd_vs_ort" / "yolov10n" / "yolov10n_simd.spk"
    if not spk.exists():
        subprocess.run([sys.executable, "-m", "spinnv2.compiler", "compile",
                        str(MODEL), "-o", str(spk), "--target", "cpu_generic"],
                       cwd=str(ROOT), check=True)
    inp_path = ROOT / "build" / "bench_simd_vs_ort" / "yolov10n" / "input.bin"
    inp_path.write_bytes(np.ascontiguousarray(x).tobytes())
    out_path = ROOT / "build" / "bench_simd_vs_ort" / "yolov10n" / "dbg_output.bin"
    bench = ROOT / "build" / "runtime" / "spkv2_run"
    subprocess.run([str(bench), str(spk), str(inp_path), str(out_path)], check=True)
    spk_flat = np.fromfile(str(out_path), dtype=np.float32)

    print(f"ORT output shape: {ort_out[0].shape}, size={ort_flat.size}")
    print(f"SPK output size: {spk_flat.size}")

    if spk_flat.size != ort_flat.size:
        # YOLOv10n may have multiple outputs concatenated
        print(f"Size mismatch: ORT={ort_flat.size} vs SPK={spk_flat.size}")
        # Try comparing just the first output's worth of elements
        min_sz = min(spk_flat.size, ort_flat.size)
        spk_flat = spk_flat[:min_sz]
        ort_flat = ort_flat[:min_sz]

    cos = float(np.dot(spk_flat, ort_flat) / (np.linalg.norm(spk_flat) * np.linalg.norm(ort_flat) + 1e-12))
    max_abs = float(np.abs(spk_flat - ort_flat).max())
    mean_abs = float(np.abs(spk_flat - ort_flat).mean())
    print(f"\nFull output comparison:")
    print(f"  cosine={cos:.6f}  max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

    # Find where the divergence is by looking at sorted values
    print(f"\nORT top-10 values:  {np.sort(ort_flat)[-10:]}")
    print(f"SPK top-10 values:  {np.sort(spk_flat)[-10:]}")
    print(f"ORT min-5 values:   {np.sort(ort_flat)[:5]}")
    print(f"SPK min-5 values:   {np.sort(spk_flat)[:5]}")

    # Element-wise analysis
    diff = np.abs(spk_flat - ort_flat)
    top_diff_idx = np.argsort(diff)[-20:][::-1]
    print(f"\nTop-20 divergent positions:")
    for idx in top_diff_idx:
        print(f"  [{idx:6d}] ORT={ort_flat[idx]:12.6f}  SPK={spk_flat[idx]:12.6f}  diff={diff[idx]:.6e}")

    # Now do per-node ORT intermediate comparison
    print("\n" + "="*70)
    print("Per-node intermediate output analysis (ORT with all intermediate tensors)")
    print("="*70)

    # Get all intermediate tensor names
    all_tensor_names = set()
    for node in model.graph.node:
        for out in node.output:
            if out:
                all_tensor_names.add(out)

    # Add intermediate outputs to model
    model_with_outputs = onnx.load(str(MODEL))
    existing_outputs = {o.name for o in model_with_outputs.graph.output}
    for name in all_tensor_names:
        if name not in existing_outputs:
            model_with_outputs.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UNDEFINED, None)
            )

    sess2 = ort.InferenceSession(model_with_outputs.SerializeToString(), providers=["CPUExecutionProvider"])
    all_outputs = sess2.run(None, {input_name: x})
    output_names = [o.name for o in sess2.get_outputs()]
    ort_tensors = {name: np.asarray(val, dtype=np.float32) for name, val in zip(output_names, all_outputs)}

    # Load SPK debug JSON to map node outputs to tensor IDs
    spk_meta = json.loads((spk.with_suffix(".spk.json")).read_text())
    nodes_meta = spk_meta["nodes"]
    tensors_meta = spk_meta["tensors"]

    # Build tensor_id -> name mapping
    tid_to_name = {t["id"]: t["name"] for t in tensors_meta}

    # Run SPINNV2 with per-node dumping (use SPKV2_PROFILE to identify nodes)
    # Instead, compare the final output structure
    # Let's look at what the graph outputs are
    graph_outputs = spk_meta.get("graph_outputs", [])
    print(f"\nSPK graph outputs: {graph_outputs}")
    for gid in graph_outputs:
        tname = tid_to_name.get(gid, f"tensor_{gid}")
        t_meta = next((t for t in tensors_meta if t["id"] == gid), None)
        if t_meta:
            print(f"  tensor {gid} '{tname}' shape={t_meta['shape']}")

    # Check if the ONNX model has multiple outputs
    print(f"\nONNX model outputs: {[o.name for o in model.graph.output]}")
    for o in model.graph.output:
        if o.name in ort_tensors:
            arr = ort_tensors[o.name]
            print(f"  {o.name}: shape={arr.shape} size={arr.size}")

    # Check TopK nodes specifically
    print("\n" + "="*70)
    print("TopK node analysis")
    print("="*70)
    for node in model.graph.node:
        if node.op_type == "TopK":
            print(f"\nTopK node: {node.name}")
            print(f"  inputs: {list(node.input)}")
            print(f"  outputs: {list(node.output)}")
            for attr in node.attribute:
                print(f"  attr {attr.name}={attr.i if attr.type == 2 else attr.ints}")
            # Check input values
            inp_name = node.input[0]
            if inp_name in ort_tensors:
                inp_arr = ort_tensors[inp_name]
                print(f"  input '{inp_name}': shape={inp_arr.shape} min={inp_arr.min():.6f} max={inp_arr.max():.6f}")
            # Check output values
            for out_name in node.output:
                if out_name in ort_tensors:
                    out_arr = ort_tensors[out_name]
                    print(f"  output '{out_name}': shape={out_arr.shape} dtype={out_arr.dtype}")
                    if out_arr.dtype in (np.float32, np.float64):
                        print(f"    min={out_arr.min():.6f} max={out_arr.max():.6f}")
                    elif out_arr.dtype in (np.int64, np.int32):
                        print(f"    values={out_arr.flatten()[:20]}")

    # Track which ONNX node corresponds to the first big divergence
    # by checking outputs of nodes that feed into TopK
    print("\n" + "="*70)
    print("Checking nodes upstream of TopK for divergence")
    print("="*70)

    # Build reverse map: tensor_name -> producing node
    produced_by = {}
    for node in model.graph.node:
        for out in node.output:
            produced_by[out] = node

    def trace_upstream(tensor_name, depth=0, max_depth=5):
        if depth > max_depth or tensor_name not in produced_by:
            return
        node = produced_by[tensor_name]
        for inp in node.input:
            if inp in ort_tensors:
                arr = ort_tensors[inp]
                if arr.dtype in (np.float32, np.float64):
                    print(f"  {'  '*depth}{node.op_type}.input '{inp}': shape={arr.shape} "
                          f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")

    for node in model.graph.node:
        if node.op_type == "TopK":
            print(f"\nUpstream of TopK '{node.name}':")
            trace_upstream(node.input[0], 0, 3)

if __name__ == "__main__":
    main()
