"""Fine-tune a ResNet50 EuroSAT workload with reproducible splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights, resnet50
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="resnet50")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--train-mode",
        choices=("head", "full"),
        default="head",
        help="Use head-only calibration by default; full fine-tuning is an optional fallback.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    _set_seed(seed)
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    dataset = EuroSAT(root=str(data_root), transform=weights.transforms())
    device = _resolve_device(args.device)
    sample_limits = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    loaders = {
        name: DataLoader(
            Subset(dataset, _limit_indices(splits[name], sample_limits[name])),
            batch_size=args.batch_size,
            shuffle=args.train_mode == "full" and name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda" and args.train_mode == "full",
        )
        for name in ("train", "val", "test")
    }
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    output_root = Path(config["experiment"]["output_root"]) / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    if args.train_mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        classifier = model.fc
        model.fc = nn.Identity()
        feature_loaders = {
            name: _extract_feature_loader(
                model,
                loader,
                device,
                batch_size=args.batch_size,
                shuffle=name == "train",
                name=name,
                cache_dir=output_root / "feature_cache",
                feature_size=classifier.in_features,
            )
            for name, loader in loaders.items()
        }
        model.fc = classifier
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr)
    else:
        feature_loaders = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_accuracy = -1.0
    history = []
    for epoch in range(args.epochs):
        if feature_loaders is None:
            train_loss = _train_epoch(model, loaders["train"], criterion, optimizer, device)
            val_accuracy = _accuracy(model, loaders["val"], device)
        else:
            train_loss = _train_classifier_epoch(model.fc, feature_loaders["train"], criterion, optimizer, device)
            val_accuracy = _classifier_accuracy(model.fc, feature_loaders["val"], device)
        epoch_report = {"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_accuracy}
        history.append(epoch_report)
        print(json.dumps(epoch_report), flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), output_root / "best.pt")
        (output_root / "training_progress.json").write_text(
            json.dumps({"history": history, "best_val_accuracy": best_accuracy}, indent=2),
            encoding="utf-8",
        )
    model.load_state_dict(_load_state_dict(output_root / "best.pt", device))
    test_accuracy = (
        _accuracy(model, loaders["test"], device)
        if feature_loaders is None
        else _classifier_accuracy(model.fc, feature_loaders["test"], device)
    )
    report = {
        "seed": seed,
        "train_mode": args.train_mode,
        "device": str(device),
        "cached_frozen_features": args.train_mode == "head",
        "sample_counts": {name: len(loader.dataset) for name, loader in loaders.items()},
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "history": history,
    }
    (output_root / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.size(0)
    return correct / max(total, 1)


def _extract_feature_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    batch_size: int,
    shuffle: bool,
    name: str,
    cache_dir: Path,
    feature_size: int,
) -> DataLoader:
    print(json.dumps({"phase": "extract_features", "split": name, "samples": len(loader.dataset)}), flush=True)
    model.eval()
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"{name}_features.npy"
    label_path = cache_dir / f"{name}_labels.npy"
    sample_count = len(loader.dataset)
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(sample_count, feature_size))
    labels_out = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64, shape=(sample_count,))
    offset = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            batch_features = model(inputs.to(device, non_blocking=True)).cpu().numpy()
            end = offset + len(labels)
            features[offset:end] = batch_features
            labels_out[offset:end] = labels.numpy()
            offset = end
    features.flush()
    labels_out.flush()
    features = np.load(feature_path, mmap_mode="r+")
    labels_out = np.load(label_path, mmap_mode="r+")
    print(json.dumps({"phase": "features_ready", "split": name, "shape": list(features.shape), "path": str(feature_path)}), flush=True)
    return DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(labels_out)),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
    )


def _train_classifier_epoch(classifier, loader, criterion, optimizer, device) -> float:
    classifier.train()
    total_loss = 0.0
    total = 0
    for features, labels in loader:
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(classifier(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def _classifier_accuracy(classifier, loader, device) -> float:
    classifier.eval()
    correct = total = 0
    with torch.inference_mode():
        for features, labels in loader:
            predictions = classifier(features.to(device, non_blocking=True)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.size(0)
    return correct / max(total, 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _limit_indices(indices: list[int], limit: int) -> list[int]:
    return indices if limit <= 0 else indices[:limit]


def _load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    raise SystemExit(main())
