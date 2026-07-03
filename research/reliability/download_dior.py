"""Download reproducible DIOR source artifacts with resume support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request


HF_PARQUET_SHARDS = {
    "train": [f"data/train-{index:05d}-of-00012.parquet" for index in range(12)],
    "val": [f"data/validation-{index:05d}-of-00002.parquet" for index in range(2)],
    "test": [f"data/test-{index:05d}-of-00003.parquet" for index in range(3)],
}
HF_BASE_URL = "https://huggingface.co/datasets/HichTala/dior/resolve/main/"
ZENODO_VOC_URL = "https://zenodo.org/api/records/11213149/files/DIOR-VOC.zip/content"
ZENODO_VOC_MD5 = "fc722ffcfc4579fcb7acc111cb79b08d"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DIOR from a public, reproducible source.")
    parser.add_argument("--source", choices=("hf_parquet", "zenodo_voc"), default="hf_parquet")
    parser.add_argument("--output-root", default="artifacts/data/dior/raw")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--limit-shards", type=int)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    if args.source == "zenodo_voc":
        target = root / "DIOR-VOC.zip"
        _download_resume(ZENODO_VOC_URL, target)
        downloaded.append({"path": str(target), "expected_md5": ZENODO_VOC_MD5})
    else:
        for split in args.splits:
            paths = HF_PARQUET_SHARDS[split]
            if args.limit_shards is not None:
                paths = paths[: args.limit_shards]
            for remote_path in paths:
                target = root / "hf_parquet" / remote_path
                _download_resume(HF_BASE_URL + remote_path + "?download=true", target)
                downloaded.append({"split": split, "path": str(target)})
    manifest = root / "download_manifest.json"
    manifest.write_text(json.dumps({"source": args.source, "files": downloaded}, indent=2), encoding="utf-8")
    print(json.dumps({"source": args.source, "downloaded_files": len(downloaded), "manifest": str(manifest)}))
    return 0


def _download_resume(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is not None:
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists() and not partial.exists():
            target.replace(partial)
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
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to download DIOR artifact with curl: {url}")
        partial.replace(target)
        return
    downloaded = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "SPINNV2-reliability/1.0"})
    if downloaded:
        request.add_header("Range", f"bytes={downloaded}-")
    with urllib.request.urlopen(request) as response:
        if downloaded and response.status != 206:
            downloaded = 0
        with target.open("ab" if downloaded else "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                output.flush()


if __name__ == "__main__":
    raise SystemExit(main())
