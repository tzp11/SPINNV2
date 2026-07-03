"""Materialize a bounded real DIOR subset through the Hugging Face rows API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

from research.reliability.datasets.dior import _coco_objects_to_yolo, write_dataset_yaml


SOURCE_SPLITS = {"train": "train", "val": "validation", "test": "test"}
FULL_SPLIT_COUNTS = {"train": 18000, "val": 2000, "test": 3463}
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an explicitly bounded real DIOR subset for pipeline validation.")
    parser.add_argument("--output-root", default="artifacts/data/dior_subset")
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    counts = {}
    for split, count in (("train", args.train_samples), ("val", args.val_samples), ("test", args.test_samples)):
        counts[split] = _materialize_split(output_root, split, count, args.workers)
    yaml_path = write_dataset_yaml(output_root)
    is_full = counts == FULL_SPLIT_COUNTS
    status = {
        "source": "HichTala/dior via Hugging Face rows API",
        "status": "full_dataset_materialized" if is_full else "bounded_real_subset_for_pipeline_validation_not_full_task_result",
        "counts": counts,
        "yaml": str(yaml_path),
    }
    (output_root / "subset_manifest.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status))
    return 0


def _materialize_split(output_root: Path, output_split: str, count: int, workers: int = 4) -> int:
    images = output_root / "images" / output_split
    labels = output_root / "labels" / output_split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    converted = 0
    while converted < count:
        length = min(100, count - converted)
        query = urllib.parse.urlencode(
            {
                "dataset": "HichTala/dior",
                "config": "default",
                "split": SOURCE_SPLITS[output_split],
                "offset": converted,
                "length": length,
            }
        )
        page = json.loads(_read_url(f"{ROWS_ENDPOINT}?{query}").decode("utf-8"))
        rows = [item["row"] for item in page["rows"]]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            list(executor.map(lambda row: _materialize_row(row, images, labels, output_split), rows))
        converted += len(rows)
        print(json.dumps({"split": output_split, "converted": converted, "requested": count}), flush=True)
    return converted


def _materialize_row(row: dict, images: Path, labels: Path, output_split: str) -> None:
    image_id = int(row["image_id"])
    image_path = images / f"{output_split}_{image_id:08d}.jpg"
    if not image_path.exists():
        _download_image(row["image"]["src"], image_path)
    lines = _coco_objects_to_yolo(row["objects"], int(row["width"]), int(row["height"]))
    (labels / f"{output_split}_{image_id:08d}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_url(url: str, attempts: int = 5) -> bytes:
    last_error = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "SPINNV2-reliability/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"failed to download after {attempts} attempts: {url}") from last_error


def _download_image(url: str, output: Path) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        output.write_bytes(_read_url(url))
        return
    partial = output.with_suffix(output.suffix + ".part")
    completed = subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--silent",
            "--show-error",
            "--output",
            str(partial),
            url,
        ],
        check=False,
    )
    if completed.returncode != 0:
        partial.unlink(missing_ok=True)
        output.write_bytes(_read_url(url))
        return
    partial.replace(output)


if __name__ == "__main__":
    raise SystemExit(main())
