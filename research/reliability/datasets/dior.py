"""Convert DIOR XML annotations to YOLO horizontal bounding-box labels."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


DIOR_CLASSES = (
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "expressway-service-area", "expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
    "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
)
CLASS_TO_ID = {name: index for index, name in enumerate(DIOR_CLASSES)}


def convert_dior_split(
    *,
    images_dir: str | Path,
    annotations_dir: str | Path,
    output_root: str | Path,
    split: str,
) -> int:
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_root = Path(output_root)
    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    converted = 0
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        image_path = _find_image(images_dir, xml_path.stem)
        if image_path is None:
            continue
        label_lines = _annotation_to_yolo(xml_path)
        shutil.copy2(image_path, out_images / image_path.name)
        (out_labels / f"{xml_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="ascii")
        converted += 1
    return converted


def write_dataset_yaml(output_root: str | Path) -> Path:
    output_root = Path(output_root).resolve()
    yaml_path = output_root / "dior.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(DIOR_CLASSES))
    yaml_path.write_text(
        f"path: {output_root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n",
        encoding="ascii",
    )
    return yaml_path


def convert_dior_parquet_shards(
    *,
    parquet_paths: list[str | Path],
    output_root: str | Path,
    split: str,
    max_samples: int | None = None,
) -> int:
    """Convert HichTala/dior COCO-style parquet shards into Ultralytics labels."""
    import pandas as pd

    output_root = Path(output_root)
    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    converted = 0
    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path)
        for row in frame.to_dict(orient="records"):
            if max_samples is not None and converted >= max_samples:
                return converted
            image = _decode_parquet_image(row["image"])
            width = int(row.get("width") or image.width)
            height = int(row.get("height") or image.height)
            stem = f"{split}_{int(row.get('image_id', converted)):08d}"
            image.convert("RGB").save(out_images / f"{stem}.jpg", quality=95)
            lines = _coco_objects_to_yolo(row["objects"], width, height)
            (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
            converted += 1
    return converted


def _annotation_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("./size/width", "0"))
    height = float(root.findtext("./size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions in {xml_path}")
    lines: list[str] = []
    for obj in root.findall("object"):
        name = _normalise_name(obj.findtext("name", ""))
        if name not in CLASS_TO_ID:
            raise ValueError(f"unknown DIOR class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{CLASS_TO_ID[name]} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
    return lines


def _decode_parquet_image(value) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).copy()
        if value.get("path"):
            return Image.open(value["path"]).copy()
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).copy()
    if isinstance(value, Image.Image):
        return value
    raise ValueError(f"unsupported parquet image value: {type(value)!r}")


def _coco_objects_to_yolo(objects, width: int, height: int) -> list[str]:
    if isinstance(objects, dict):
        categories = objects["category"]
        boxes = objects["bbox"]
    else:
        categories = [item["category"] for item in objects]
        boxes = [item["bbox"] for item in objects]
    lines = []
    for category, box in zip(categories, boxes):
        xmin, ymin, box_width, box_height = [float(value) for value in box]
        center_x = (xmin + box_width / 2.0) / width
        center_y = (ymin + box_height / 2.0) / height
        lines.append(
            f"{int(category)} {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}"
        )
    return lines


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None
