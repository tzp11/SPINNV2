"""Run initial task-level fault screening on a fine-tuned EuroSAT ResNet50."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights, resnet50
import yaml

from research.reliability.injection.bitflip import sample_fault_event
from research.reliability.injection.torch_injector import discover_candidate_points, infer_with_fault
from research.reliability.metrics.task_metrics import classification_failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--injections-per-module", type=int, default=8)
    parser.add_argument("--module-limit", type=int, default=0, help="Zero means screen all executed candidate outputs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    dataset = EuroSAT(root=str(data_root), transform=weights.transforms())
    dataset = Subset(dataset, splits["test"][: args.max_samples])
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(_load_state_dict(Path(args.checkpoint)))
    model.to(device).eval()
    first_inputs = dataset[0][0].unsqueeze(0).to(device)
    candidate_points = discover_candidate_points(model, first_inputs)
    if args.module_limit > 0:
        candidate_points = candidate_points[: args.module_limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = output_path.with_suffix(".candidates.json")
    candidate_mapping = []
    for index, point in enumerate(candidate_points):
        mapping = asdict(point)
        mapping.update({"node_id": index, "tensor_id": index})
        candidate_mapping.append(mapping)
    mapping_path.write_text(
        json.dumps(candidate_mapping, indent=2),
        encoding="utf-8",
    )
    observations = 0
    correct_samples = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for sample_index, (image, label) in enumerate(dataset):
            inputs = image.unsqueeze(0).to(device)
            with torch.inference_mode():
                baseline = model(inputs).cpu().numpy()[0]
            if int(baseline.argmax()) != int(label):
                continue
            correct_samples += 1
            for module_index, point in enumerate(candidate_points):
                element_count = point.element_count
                for repeat in range(args.injections_per_module):
                    seed = int(config["experiment"]["seed"]) + sample_index * 100000 + module_index * 100 + repeat
                    event = sample_fault_event(
                        model_id="resnet50_eurosat",
                        sample_id=f"test_{sample_index:06d}",
                        node_id=module_index,
                        tensor_id=module_index,
                        element_count=element_count,
                        seed=seed,
                        invocation_index=point.invocation_index,
                    )
                    faulted = infer_with_fault(model, inputs, module_name=point.module_name, event=event).cpu().numpy()[0]
                    consequence = classification_failure(baseline, faulted, int(label))
                    sink.write(
                        json.dumps(
                            {
                                "node_id": module_index,
                                "tensor_id": module_index,
                                "activation_bytes": element_count * 4,
                                "critical_failure": consequence.critical_failure,
                                "severity": consequence.severity,
                                "module_name": point.module_name,
                                "invocation_index": point.invocation_index,
                                "fault_event": asdict(event),
                            }
                        )
                        + "\n"
                    )
                    observations += 1
            sink.flush()
            if args.progress_every > 0 and correct_samples % args.progress_every == 0:
                print(json.dumps({"correct_samples": correct_samples, "observations": observations}), flush=True)
    print(
        json.dumps(
            {
                "observations": observations,
                "correct_samples": correct_samples,
                "output": str(output_path),
                "candidate_points": len(candidate_points),
                "candidate_map": str(mapping_path),
            }
        )
    )
    return 0


def _load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
