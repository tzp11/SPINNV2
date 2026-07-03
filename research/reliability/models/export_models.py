"""Export fixed-shape FP32 research workloads for SPINNV2 auditing."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_resnet50(output: Path, checkpoint: str | None = None, num_classes: int = 10) -> None:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if checkpoint:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        str(output),
        input_names=["images"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )


def export_yolov10n(output: Path, weights: str) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    exported = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=13))
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("resnet50", "yolov10n"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--weights", default="yolov10n.pt")
    args = parser.parse_args()
    output = Path(args.output)
    if args.model == "resnet50":
        export_resnet50(output, args.checkpoint)
    else:
        export_yolov10n(output, args.weights)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
