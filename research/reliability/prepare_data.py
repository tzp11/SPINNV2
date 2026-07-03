"""Prepare EuroSAT splits or convert a locally supplied DIOR extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from research.reliability.datasets.dior import convert_dior_parquet_shards, convert_dior_split, write_dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--dataset", choices=("eurosat", "dior"), required=True)
    parser.add_argument("--dior-images")
    parser.add_argument("--dior-annotations")
    parser.add_argument("--dior-parquet", nargs="+")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.dataset == "eurosat":
        return _prepare_eurosat(config["datasets"]["eurosat"])
    output_root = Path(config["datasets"]["dior"]["root"])
    if args.dior_parquet:
        count = convert_dior_parquet_shards(
            parquet_paths=args.dior_parquet,
            output_root=output_root,
            split=args.split,
            max_samples=args.max_samples,
        )
    elif args.dior_images and args.dior_annotations:
        count = convert_dior_split(
            images_dir=args.dior_images,
            annotations_dir=args.dior_annotations,
            output_root=output_root,
            split=args.split,
        )
    else:
        parser.error("DIOR conversion requires --dior-parquet or both --dior-images and --dior-annotations")
    dataset_yaml = write_dataset_yaml(output_root)
    print(json.dumps({"dataset": "dior", "split": args.split, "converted": count, "yaml": str(dataset_yaml)}))
    return 0


def _prepare_eurosat(config: dict) -> int:
    from torchvision.datasets import EuroSAT

    root = Path(config["root"])
    dataset = EuroSAT(root=str(root), download=True)
    seed = int(config["split_seed"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    train_end = int(len(indices) * float(config["train_ratio"]))
    val_end = train_end + int(len(indices) * float(config["val_ratio"]))
    splits = {"seed": seed, "train": indices[:train_end], "val": indices[train_end:val_end], "test": indices[val_end:]}
    split_path = root / "splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(splits), encoding="utf-8")
    print(json.dumps({"dataset": "eurosat", "size": len(dataset), "splits": str(split_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
