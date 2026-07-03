"""Create one DIOR detection input and ONNX Runtime reference output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, dict]:
    """Match Ultralytics square letterbox preprocessing for a fixed ONNX input."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw = size - resized_width
    dh = size - resized_height
    left = int(round(dw / 2.0 - 0.1))
    right = int(round(dw / 2.0 + 0.1))
    top = int(round(dh / 2.0 - 0.1))
    bottom = int(round(dh / 2.0 + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, {"original_shape": [height, width], "scale": ratio, "padding_xy": [left, top]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--label")
    parser.add_argument("--input-output", required=True, help="FP32 BCHW binary tensor output path.")
    parser.add_argument("--reference-output", required=True, help="ONNX Runtime FP32 output binary path.")
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    values, preprocessing = letterbox(image)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    output = np.asarray(session.run(None, {session.get_inputs()[0].name: values})[0], dtype=np.float32)

    input_path = Path(args.input_output)
    reference_path = Path(args.reference_output)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(input_path)
    output.tofile(reference_path)
    detections = output.reshape(-1, 6)
    active = detections[detections[:, 4] >= 0.25]
    report = {
        "image": str(image_path),
        "label": args.label,
        "onnx": args.onnx,
        "input_shape": list(values.shape),
        "output_shape": list(output.shape),
        "preprocessing": preprocessing,
        "detections_at_confidence_0_25": int(len(active)),
        "input_output": str(input_path),
        "reference_output": str(reference_path),
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
