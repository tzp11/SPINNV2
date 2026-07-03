"""Compute native vulnerability metrics for baseline selective-protection methods.

Produces per-runtime-node metrics:
  - macs_fraction: MACs(node) / total_MACs  (FILR / ISSRE'21)
  - activation_magnitude: mean(|output|)     (Ruospo / DDECS'22)
  - taylor_importance: mean(|dL/da * a|)     (Aspis / ISSRE'24, Ahmadilivani / IOLTS'24)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def _build_onnx_to_runtime_map(onnx_model: onnx.ModelProto, spk_debug: dict) -> dict[str, int]:
    """Map ONNX output tensor name → runtime_node_id via SPK debug metadata."""
    tensors_by_name = {t["name"]: t["id"] for t in spk_debug["tensors"]}
    producer_by_tensor = {
        tid: node
        for node in spk_debug["nodes"]
        for tid in node["outputs"]
    }
    mapping = {}
    for node in onnx_model.graph.node:
        output_name = node.output[0]
        tensor_id = tensors_by_name.get(output_name)
        if tensor_id is None:
            continue
        runtime_node = producer_by_tensor.get(tensor_id)
        if runtime_node is None:
            continue
        mapping[output_name] = runtime_node["id"]
    return mapping


def _compute_conv_macs(onnx_node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto],
                       output_shapes: dict[str, list[int]]) -> int:
    weight_name = onnx_node.input[1]
    weight = initializers.get(weight_name)
    if weight is None:
        return 0
    dims = list(weight.dims)
    c_out, c_in_per_group = dims[0], dims[1]
    k_h, k_w = dims[2], dims[3]
    group = 1
    for attr in onnx_node.attribute:
        if attr.name == "group":
            group = attr.i
    c_in = c_in_per_group * group
    output_name = onnx_node.output[0]
    out_shape = output_shapes.get(output_name)
    if out_shape is None or len(out_shape) < 4:
        return 0
    h_out, w_out = out_shape[2], out_shape[3]
    return k_h * k_w * c_in_per_group * c_out * h_out * w_out


def _compute_gemm_macs(onnx_node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto]) -> int:
    weight_name = onnx_node.input[1]
    weight = initializers.get(weight_name)
    if weight is None:
        return 0
    dims = list(weight.dims)
    if len(dims) != 2:
        return 0
    trans_b = 0
    for attr in onnx_node.attribute:
        if attr.name == "transB":
            trans_b = attr.i
    if trans_b:
        n, k = dims[0], dims[1]
    else:
        k, n = dims[0], dims[1]
    return k * n


def _infer_output_shapes(onnx_model: onnx.ModelProto) -> dict[str, list[int]]:
    """Use onnx shape inference to get static output shapes."""
    try:
        inferred = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
    except Exception:
        inferred = onnx_model
    shapes = {}
    for vi in inferred.graph.value_info:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            shapes[vi.name] = dims
    for out in inferred.graph.output:
        if out.type.HasField("tensor_type") and out.type.tensor_type.HasField("shape"):
            dims = []
            for d in out.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            shapes[out.name] = dims
    return shapes


def compute_macs(onnx_model: onnx.ModelProto, onnx_to_runtime: dict[str, int]) -> dict[int, float]:
    """Compute macs_fraction per runtime node.

    Conv outputs are often fused into the downstream Relu in the runtime, so
    if a Conv's output isn't directly mapped, we trace forward through the
    ONNX graph to find the consuming runtime node and attribute MACs there.
    """
    initializers = {i.name: i for i in onnx_model.graph.initializer}
    output_shapes = _infer_output_shapes(onnx_model)

    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    def _find_runtime_id(output_name: str) -> int | None:
        """Trace forward until we find an output that maps to a runtime node."""
        rid = onnx_to_runtime.get(output_name)
        if rid is not None:
            return rid
        for consumer in consumers.get(output_name, []):
            rid = _find_runtime_id(consumer.output[0])
            if rid is not None:
                return rid
        return None

    node_macs: dict[int, int] = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            macs = _compute_conv_macs(node, initializers, output_shapes)
        elif node.op_type == "Gemm":
            macs = _compute_gemm_macs(node, initializers)
        else:
            continue
        if macs == 0:
            continue
        rid = _find_runtime_id(node.output[0])
        if rid is None:
            continue
        node_macs[rid] = node_macs.get(rid, 0) + macs

    all_rids = set(onnx_to_runtime.values())
    total = sum(node_macs.values())
    if total == 0:
        return {rid: 0.0 for rid in all_rids}
    result = {rid: 0.0 for rid in all_rids}
    for rid, m in node_macs.items():
        result[rid] = m / total
    return result


def compute_activation_magnitude(
    onnx_path: str | Path,
    onnx_to_runtime: dict[str, int],
    calibration_images: list[np.ndarray],
) -> dict[int, float]:
    """Compute mean(|activation_output|) per runtime node over calibration set."""
    output_names = list(onnx_to_runtime.keys())
    model = onnx.load(str(onnx_path))
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        inferred = model
    vi_types = {}
    for vi in inferred.graph.value_info:
        if vi.type.HasField("tensor_type"):
            vi_types[vi.name] = vi.type.tensor_type.elem_type
    existing_output_names = {o.name for o in model.graph.output}
    valid_output_names = []
    for name in output_names:
        if name in existing_output_names:
            valid_output_names.append(name)
            continue
        elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
        if elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE):
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, elem_type, None))
            valid_output_names.append(name)
    output_names = valid_output_names
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)
    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        accum: dict[int, list[float]] = {}
        for img in calibration_images:
            results = sess.run(output_names, {input_name: img})
            for name, arr in zip(output_names, results):
                rid = onnx_to_runtime[name]
                mag = float(np.mean(np.abs(arr)))
                accum.setdefault(rid, []).append(mag)
    finally:
        import os
        os.unlink(tmp_path)
    return {rid: float(np.mean(vals)) for rid, vals in accum.items()}


def compute_taylor_importance(
    checkpoint_path: str | Path,
    calibration_dir: str | Path,
    max_samples: int,
    onnx_to_runtime: dict[str, int],
    onnx_model: onnx.ModelProto,
    *,
    task: str = "classification",
) -> dict[int, float]:
    """Compute |dL/da * a| per layer via PyTorch forward+backward.

    Supports both classification (ResNet50+EuroSAT) and detection
    (YOLOv10n+DIOR) via the ``task`` parameter.
    """
    if task == "detection":
        return _compute_taylor_importance_yolo(
            checkpoint_path, calibration_dir, max_samples,
            onnx_to_runtime, onnx_model,
        )
    return _compute_taylor_importance_classification(
        checkpoint_path, calibration_dir, max_samples,
        onnx_to_runtime, onnx_model,
    )


def _compute_taylor_importance_classification(
    checkpoint_path: str | Path,
    calibration_dir: str | Path,
    max_samples: int,
    onnx_to_runtime: dict[str, int],
    onnx_model: onnx.ModelProto,
) -> dict[int, float]:
    import torch
    from torch import nn
    from torch.utils.data import Subset
    from torchvision.datasets import EuroSAT
    from torchvision.models import ResNet50_Weights, resnet50

    from research.reliability.injection.torch_injector import candidate_module_names

    device = torch.device("cpu")
    weights = ResNet50_Weights.IMAGENET1K_V2
    data_root = Path(calibration_dir)
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    dataset = EuroSAT(root=str(data_root), transform=weights.transforms())
    dataset = Subset(dataset, splits["val"][:max_samples])

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state = _load_torch_state(Path(checkpoint_path))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False

    eligible_names = candidate_module_names(model)
    onnx_candidate_outputs = [
        n.output[0] for n in onnx_model.graph.node if n.op_type in ("Conv", "Relu", "Gemm")
    ]

    activations: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    invocation_counts: dict[str, int] = {}
    bwd_counts: dict[str, int] = {}
    handles = []

    def _make_fwd_hook(name):
        def hook(_mod, _inp, out):
            invocation_counts[name] = invocation_counts.get(name, 0) + 1
            key = f"{name}#{invocation_counts[name]}"
            activations[key] = out
        return hook

    def _make_bwd_hook(name):
        def hook(_mod, _inp_grad, out_grad):
            bwd_counts[name] = bwd_counts.get(name, 0) + 1
            total_fwd = invocation_counts.get(name, 0)
            inv = total_fwd - bwd_counts[name] + 1
            key = f"{name}#{inv}"
            if out_grad[0] is not None:
                grads[key] = out_grad[0]
        return hook

    for name, module in model.named_modules():
        if name in eligible_names:
            handles.append(module.register_forward_hook(_make_fwd_hook(name)))
            handles.append(module.register_full_backward_hook(_make_bwd_hook(name)))

    taylor_accum: dict[str, list[float]] = {}
    criterion = nn.CrossEntropyLoss()

    for image, label in dataset:
        model.zero_grad()
        activations.clear()
        grads.clear()
        invocation_counts.clear()
        bwd_counts.clear()
        inp = image.unsqueeze(0).to(device)
        out = model(inp)
        loss = criterion(out, torch.tensor([label], device=device))
        loss.backward()
        for key in activations:
            act = activations.get(key)
            grad = grads.get(key)
            if act is not None and grad is not None:
                score = float(torch.sum(torch.abs(grad * act.detach())).item())
                taylor_accum.setdefault(key, []).append(score)

    for h in handles:
        h.remove()

    ordered_keys = []
    dummy_input = dataset[0][0].unsqueeze(0).to(device)
    with torch.inference_mode():
        from research.reliability.injection.torch_injector import discover_candidate_points
        points = discover_candidate_points(model, dummy_input)
        for pt in points:
            ordered_keys.append(f"{pt.module_name}#{pt.invocation_index}")

    if len(ordered_keys) != len(onnx_candidate_outputs):
        print(f"WARNING: PyTorch invocations ({len(ordered_keys)}) != ONNX candidates ({len(onnx_candidate_outputs)})")
        count = min(len(ordered_keys), len(onnx_candidate_outputs))
    else:
        count = len(ordered_keys)

    result: dict[int, float] = {}
    for i in range(count):
        key = ordered_keys[i]
        onnx_output = onnx_candidate_outputs[i]
        rid = onnx_to_runtime.get(onnx_output)
        if rid is None:
            continue
        scores = taylor_accum.get(key, [])
        result[rid] = float(np.mean(scores)) if scores else 0.0
    return result


def _build_onnx_module_to_runtime(
    onnx_model: onnx.ModelProto,
    onnx_to_runtime: dict[str, int],
) -> dict[str, int]:
    """Map PyTorch module names to runtime node IDs via ONNX tensor name conventions.

    Ultralytics ONNX export embeds the PyTorch module path in tensor names, e.g.
    ``/model.0/conv/Conv_output_0`` for module ``model.0.conv``, and
    ``/model.0/act/Mul_output_0`` for SiLU module ``model.0.act``.

    When a Conv output is fused into a downstream SiLU (Sigmoid+Mul) in the
    runtime, the Conv output isn't in ``onnx_to_runtime``.  We forward-trace
    through the ONNX graph to find the consuming node that IS mapped.
    """
    import re

    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    def _trace_to_runtime(output_name: str, depth: int = 0) -> int | None:
        if depth > 5:
            return None
        rid = onnx_to_runtime.get(output_name)
        if rid is not None:
            return rid
        for consumer in consumers.get(output_name, []):
            rid = _trace_to_runtime(consumer.output[0], depth + 1)
            if rid is not None:
                return rid
        return None

    result: dict[str, int] = {}
    for node in onnx_model.graph.node:
        output_name = node.output[0]
        match = re.match(r"^/(.+)/\w+_output_\d+$", output_name)
        if not match:
            continue
        module_path = match.group(1).replace("/", ".")
        rid = _trace_to_runtime(output_name)
        if rid is not None and module_path not in result:
            result[module_path] = rid

    for node in onnx_model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight_name = node.input[1]
        if weight_name.endswith(".weight"):
            pytorch_module = weight_name[: -len(".weight")]
            rid = _trace_to_runtime(node.output[0])
            if rid is None:
                continue
            if pytorch_module not in result:
                result[pytorch_module] = rid
            pytorch_conv = pytorch_module + ".conv"
            if pytorch_conv not in result:
                result[pytorch_conv] = rid
            pytorch_act = re.sub(r"\.conv$", ".act", pytorch_module)
            if pytorch_act != pytorch_module and pytorch_act not in result:
                result[pytorch_act] = rid

    return result


def _compute_taylor_importance_yolo(
    checkpoint_path: str | Path,
    calibration_dir: str | Path,
    max_samples: int,
    onnx_to_runtime: dict[str, int],
    onnx_model: onnx.ModelProto,
) -> dict[int, float]:
    """Compute Taylor importance for YOLOv10n detection models.

    Uses ultralytics detection loss (E2ELoss) for gradients and maps
    PyTorch module activations to runtime nodes via ONNX tensor name parsing.
    """
    import torch
    import cv2
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    from research.reliability.injection.torch_injector import candidate_module_names

    device = torch.device("cpu")

    yolo = YOLO(str(checkpoint_path))
    model = yolo.model.to(device)
    cfg = get_cfg()
    model.args = cfg

    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    for p in model.parameters():
        p.requires_grad_(True)

    model.train()
    model.criterion = None

    eligible_names = candidate_module_names(model)
    module_to_runtime = _build_onnx_module_to_runtime(onnx_model, onnx_to_runtime)

    activations: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    invocation_counts: dict[str, int] = {}
    bwd_counts: dict[str, int] = {}
    handles = []

    def _make_fwd_hook(name):
        def hook(_mod, _inp, out):
            invocation_counts[name] = invocation_counts.get(name, 0) + 1
            key = f"{name}#{invocation_counts[name]}"
            activations[key] = out
        return hook

    def _make_bwd_hook(name):
        def hook(_mod, _inp_grad, out_grad):
            bwd_counts[name] = bwd_counts.get(name, 0) + 1
            total_fwd = invocation_counts.get(name, 0)
            inv = total_fwd - bwd_counts[name] + 1
            key = f"{name}#{inv}"
            if out_grad[0] is not None:
                grads[key] = out_grad[0]
        return hook

    for name, module in model.named_modules():
        if name in eligible_names:
            handles.append(module.register_forward_hook(_make_fwd_hook(name)))
            handles.append(module.register_full_backward_hook(_make_bwd_hook(name)))

    criterion = model.init_criterion()

    dior_root = Path(calibration_dir)
    for candidate_dir in (
        dior_root / "images" / "val",
        dior_root / "images" / "test",
        dior_root / "images",
        dior_root,
    ):
        if candidate_dir.exists() and list(candidate_dir.glob("*.jpg")):
            img_dir = candidate_dir
            break
    else:
        raise FileNotFoundError(f"No DIOR images found under {dior_root}")

    label_dir = dior_root / "labels" / img_dir.name
    if not label_dir.exists():
        label_dir = dior_root / "labels" / "val"

    image_files = sorted(img_dir.glob("*.jpg"))[:max_samples]
    if not image_files:
        image_files = sorted(img_dir.glob("*.png"))[:max_samples]

    taylor_accum: dict[str, list[float]] = {}

    for img_file in image_files:
        model.zero_grad()
        activations.clear()
        grads.clear()
        invocation_counts.clear()
        bwd_counts.clear()

        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        label_file = label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            labels = []
            for line in label_file.read_text().strip().split("\n"):
                if line.strip():
                    labels.append([float(x) for x in line.strip().split()])
            if labels:
                labels_t = torch.tensor(labels, device=device)
                batch = {
                    "img": img_t,
                    "cls": labels_t[:, 0:1],
                    "bboxes": labels_t[:, 1:5],
                    "batch_idx": torch.zeros(len(labels), device=device),
                }
            else:
                batch = {
                    "img": img_t,
                    "cls": torch.zeros((0, 1), device=device),
                    "bboxes": torch.zeros((0, 4), device=device),
                    "batch_idx": torch.zeros(0, device=device),
                }
        else:
            batch = {
                "img": img_t,
                "cls": torch.zeros((0, 1), device=device),
                "bboxes": torch.zeros((0, 4), device=device),
                "batch_idx": torch.zeros(0, device=device),
            }

        preds = model(img_t)
        parsed = criterion.one2many.parse_output(preds)
        one2many_pred = parsed["one2many"]
        one2one_pred = parsed["one2one"]
        loss_o2m, _ = criterion.one2many.loss(one2many_pred, batch)
        loss_o2o, _ = criterion.one2one.loss(one2one_pred, batch)
        loss = loss_o2m.sum() * criterion.o2m + loss_o2o.sum() * criterion.o2o
        loss.backward()

        for key in activations:
            act = activations.get(key)
            grad = grads.get(key)
            if act is not None and grad is not None:
                score = float(torch.sum(torch.abs(grad * act.detach())).item())
                taylor_accum.setdefault(key, []).append(score)

    for h in handles:
        h.remove()

    result: dict[int, float] = {}
    for key, scores in taylor_accum.items():
        module_name = key.rsplit("#", 1)[0]
        rid = module_to_runtime.get(module_name)
        if rid is not None:
            avg_score = float(np.mean(scores))
            result[rid] = max(result.get(rid, 0.0), avg_score)

    return result


def _load_torch_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_calibration_images(
    onnx_path: str | Path,
    calibration_dir: str | Path,
    max_samples: int,
) -> list[np.ndarray]:
    """Load calibration images preprocessed for onnxruntime."""
    from torchvision.datasets import EuroSAT
    from torchvision.models import ResNet50_Weights
    from torch.utils.data import Subset

    data_root = Path(calibration_dir)
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    dataset = EuroSAT(root=str(data_root), transform=weights.transforms())
    dataset = Subset(dataset, splits["val"][:max_samples])
    images = []
    for img, _label in dataset:
        images.append(img.numpy()[np.newaxis, ...])
    return images


def _load_yolo_calibration_images(
    onnx_path: str | Path,
    calibration_dir: str | Path,
    max_samples: int,
) -> list[np.ndarray]:
    """Load calibration images for YOLO models (DIOR dataset, 640x640)."""
    import cv2

    dior_root = Path(calibration_dir)
    for candidate in (
        dior_root / "images" / "val",
        dior_root / "images" / "test",
        dior_root / "JPEGImages-test",
        dior_root / "JPEGImages",
        dior_root / "images",
        dior_root,
    ):
        if candidate.exists() and (list(candidate.glob("*.jpg")) or list(candidate.glob("*.png"))):
            img_dir = candidate
            break
    else:
        raise FileNotFoundError(f"No image files found under {dior_root}")

    image_files = sorted(img_dir.glob("*.jpg"))[:max_samples]
    if not image_files:
        image_files = sorted(img_dir.glob("*.png"))[:max_samples]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_shape = sess.get_inputs()[0].shape
    h, w = input_shape[2], input_shape[3]
    images = []
    for f in image_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        images.append(img)
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute native vulnerability metrics for baseline methods.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--spk-debug", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--task", choices=("classification", "detection"), default="classification")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--checkpoint", help="PyTorch checkpoint for Taylor importance (optional).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx)
    spk_debug = json.loads(Path(args.spk_debug).read_text(encoding="utf-8"))
    onnx_to_runtime = _build_onnx_to_runtime_map(onnx_model, spk_debug)
    print(f"Mapped {len(onnx_to_runtime)} ONNX outputs to runtime nodes")

    macs_fractions = compute_macs(onnx_model, onnx_to_runtime)
    print(f"Computed MACs for {len(macs_fractions)} nodes, "
          f"top-3: {sorted(macs_fractions.items(), key=lambda x: -x[1])[:3]}")

    if args.task == "detection":
        cal_images = _load_yolo_calibration_images(args.onnx, args.calibration_dir, args.max_samples)
    else:
        cal_images = _load_calibration_images(args.onnx, args.calibration_dir, args.max_samples)
    print(f"Loaded {len(cal_images)} calibration images")

    act_mags = compute_activation_magnitude(args.onnx, onnx_to_runtime, cal_images)
    print(f"Computed activation magnitude for {len(act_mags)} nodes, "
          f"top-3: {sorted(act_mags.items(), key=lambda x: -x[1])[:3]}")

    taylor = {}
    if args.checkpoint:
        taylor = compute_taylor_importance(
            args.checkpoint, args.calibration_dir, args.max_samples,
            onnx_to_runtime, onnx_model,
            task=args.task,
        )
        print(f"Computed Taylor importance for {len(taylor)} nodes, "
              f"top-3: {sorted(taylor.items(), key=lambda x: -x[1])[:3]}")

    all_rids = set(macs_fractions) | set(act_mags) | set(taylor)
    metrics: dict[str, dict[str, float]] = {}
    for rid in sorted(all_rids):
        entry: dict[str, float] = {}
        if rid in macs_fractions:
            entry["macs_fraction"] = macs_fractions[rid]
        if rid in act_mags:
            entry["activation_magnitude"] = act_mags[rid]
        if rid in taylor:
            entry["taylor_importance"] = taylor[rid]
        metrics[str(rid)] = entry

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {len(metrics)} node metrics to {output}")
    return 0


# late import guard
try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main())
