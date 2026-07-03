"""Export a correctly classified EuroSAT sample as a SPINNV2 runtime input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights, resnet50
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--subset-index", type=int, default=-1, help="Use one exact test-subset index when non-negative.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    indices = splits["test"][: args.max_samples] if args.subset_index < 0 else [splits["test"][args.subset_index]]
    dataset = Subset(EuroSAT(root=str(data_root), transform=weights.transforms()), indices)
    device = torch.device(args.device)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(_load_state_dict(Path(args.checkpoint)))
    model.to(device).eval()
    for local_index, (image, label) in enumerate(dataset):
        sample_index = local_index if args.subset_index < 0 else args.subset_index
        with torch.inference_mode():
            logits = model(image.unsqueeze(0).to(device)).cpu().numpy()[0]
        prediction = int(logits.argmax())
        if prediction != int(label):
            continue
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(image.unsqueeze(0).numpy(), dtype=np.float32).tofile(output)
        metadata = {
            "sample_id": f"test_{sample_index:06d}",
            "subset_index": sample_index,
            "label": int(label),
            "baseline_class": prediction,
            "input_shape": [1, 3, 224, 224],
            "input_path": str(output),
        }
        Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata))
        return 0
    raise RuntimeError("no correctly classified sample found in requested prefix")


def _load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
