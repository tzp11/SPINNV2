"""Train the YOLOv10n DIOR workload through Ultralytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/dior/dior.yaml")
    parser.add_argument("--weights", default="yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="artifacts/experiments/windows_prevalidation/yolov10n")
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"prepare DIOR first: {args.data}")
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.run_name,
        workers=args.workers,
        patience=args.patience,
    )
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    validated_model = YOLO(str(best_weights))
    metrics = validated_model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
    report = {
        "model": "yolov10n_dior",
        "weights": str(best_weights),
        "seed": args.seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "box_map75": float(metrics.box.map75),
    }
    output = save_dir / "test_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
