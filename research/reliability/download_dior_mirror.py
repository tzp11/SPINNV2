"""Download DIOR parquet files from HF mirror and convert to YOLO format."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MIRROR = "https://hf-mirror.com"
DATASET_ID = "HichTala/dior"

PARQUET_FILES = {
    "train": [f"data/train-{i:05d}-of-00012.parquet" for i in range(12)],
    "validation": [f"data/validation-{i:05d}-of-00002.parquet" for i in range(2)],
    "test": [f"data/test-{i:05d}-of-00003.parquet" for i in range(3)],
}

SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def download_file(url: str, dest: Path, retries: int = 5) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    partial = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SPINNV2/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(partial, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r  {dest.name}: {downloaded>>20}MB / {total>>20}MB ({pct}%)", end="", flush=True)
            print()
            partial.rename(dest)
            return True
        except Exception as e:
            print(f"\n  retry {attempt+1}/{retries}: {e}", flush=True)
            time.sleep(min(2 ** attempt, 16))
            partial.unlink(missing_ok=True)
    return False


def convert_parquet_to_yolo(parquet_dir: Path, output_root: Path) -> dict:
    import pandas as pd
    from PIL import Image
    from io import BytesIO

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from research.reliability.datasets.dior import _coco_objects_to_yolo, write_dataset_yaml

    counts = {}
    for hf_split, local_split in SPLIT_MAP.items():
        shard_dir = parquet_dir / hf_split
        shards = sorted(shard_dir.glob("*.parquet"))
        if not shards:
            print(f"  skip {hf_split}: no parquet files")
            counts[local_split] = 0
            continue

        img_dir = output_root / "images" / local_split
        lbl_dir = output_root / "labels" / local_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        for shard in shards:
            print(f"  processing {shard.name}...", flush=True)
            df = pd.read_parquet(shard)
            for _, row in df.iterrows():
                image_id = int(row.get("image_id", converted))
                stem = f"{local_split}_{image_id:08d}"
                img_path = img_dir / f"{stem}.jpg"
                if not img_path.exists():
                    img_data = row["image"]
                    if isinstance(img_data, dict) and img_data.get("bytes"):
                        img = Image.open(BytesIO(img_data["bytes"]))
                    elif isinstance(img_data, (bytes, bytearray)):
                        img = Image.open(BytesIO(img_data))
                    else:
                        continue
                    img.convert("RGB").save(img_path, quality=95)

                w = int(row.get("width", 800))
                h = int(row.get("height", 800))
                lines = _coco_objects_to_yolo(row["objects"], w, h)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
                converted += 1
            print(f"    cumulative: {converted}", flush=True)

        counts[local_split] = converted
        print(f"  {local_split}: {converted} images", flush=True)

    write_dataset_yaml(output_root)
    return counts


def main():
    output_root = Path("artifacts/data/dior")
    parquet_dir = output_root / "parquet"

    print("=== Phase 1: Download parquet files from HF mirror ===", flush=True)
    failed = []
    for hf_split, files in PARQUET_FILES.items():
        print(f"\n[{hf_split}] {len(files)} shards", flush=True)
        for fname in files:
            url = f"{MIRROR}/datasets/{DATASET_ID}/resolve/main/{fname}"
            dest = parquet_dir / hf_split / Path(fname).name
            if not download_file(url, dest):
                failed.append(fname)
                print(f"  FAILED: {fname}")

    if failed:
        print(f"\n{len(failed)} files failed to download: {failed}")
        return 1

    print("\n=== Phase 2: Convert parquet to YOLO format ===", flush=True)
    counts = convert_parquet_to_yolo(parquet_dir, output_root)

    manifest = {
        "source": f"HichTala/dior via {MIRROR}",
        "status": "complete",
        "counts": counts,
    }
    (output_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
