"""Read-only ZIP QC: reuse school's inspector, then decode actual mask pixels."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from automation.config import ROOT
from scripts.inspect_submissions import _members, inspect_all, inspect_task, task_registry


def _rle_counts(counts: str | list[int], total: int) -> list[int]:
    """Validate COCO's compressed delta RLE before passing it to the C decoder.

    Format: cocodataset/cocoapi common/maskApi.c rleFrString/rleToString.
    """
    if isinstance(counts, list):
        runs = counts
    elif isinstance(counts, str):
        runs, position = [], 0
        while position < len(counts):
            value, shift = 0, 0
            while True:
                if position >= len(counts) or shift > 60:
                    raise ValueError("RLE compressed bị cắt hoặc overflow.")
                code = ord(counts[position]) - 48
                position += 1
                if not 0 <= code <= 63:
                    raise ValueError("RLE compressed có ký tự không hợp lệ.")
                value |= (code & 0x1F) << shift
                shift += 5
                if not code & 0x20:
                    if code & 0x10:
                        value |= -1 << shift
                    break
            if len(runs) > 2:
                value += runs[-2]
            runs.append(value)
            if len(runs) > total + 1:
                raise ValueError("RLE có quá nhiều runs.")
    else:
        raise ValueError("counts RLE phải là chuỗi hoặc list.")
    if not runs or any(type(run) is not int or run < 0 or run > total for run in runs) or sum(runs) != total:
        raise ValueError("Độ dài runs RLE không khớp kích thước ảnh.")
    return runs


def decode_coco(segmentation: object, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if isinstance(segmentation, dict):
        if segmentation.get("size") != [height, width]:
            raise ValueError("RLE size không khớp ảnh.")
        counts = _rle_counts(segmentation.get("counts"), height * width)
        # Decode validated runs directly; COCO is column-major, unlike CVAT mask RLE.
        flat = np.zeros(height * width, dtype=bool)
        cursor = 0
        for index, length in enumerate(counts):
            if index % 2:
                flat[cursor:cursor + length] = True
            cursor += length
        return flat.reshape(shape, order="F")
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError("Thiếu polygon/RLE.")
    for polygon in segmentation:
        if (not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in polygon)):
            raise ValueError("Polygon không hợp lệ hoặc có tọa độ NaN/inf.")
        if any(not 0 <= value <= (width if index % 2 == 0 else height)
               for index, value in enumerate(polygon)):
            raise ValueError("Polygon vượt mép ảnh.")
    return coco_mask.decode(coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))).astype(bool)


def inspect_pixels(name: str, path: Path, root: Path = ROOT) -> dict:
    result = inspect_task(name, path, root)
    if not path.exists() or result["errors"]:
        return result
    info = task_registry(root)[name]
    base = root / "data" / info["path"]
    shapes = {}
    for image_path in (base / "images").glob("*.jpg"):
        with Image.open(image_path) as image:
            shapes[image_path.name] = (image.height, image.width)
    schema = json.loads((base / "classes.json").read_text(encoding="utf-8"))
    stats = {}
    try:
        members = _members(path)  # school size/path checks; never extract files
        if info["type"] == "semantic":
            labelmap = next(value for key, value in members.items() if Path(key).name == "labelmap.txt")
            colors = {}
            for line in labelmap.decode("utf-8").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                label, rgb, *_ = line.split(":")
                color = tuple(map(int, rgb.split(",")))
                if len(color) != 3 or any(not 0 <= value <= 255 for value in color) or color in colors:
                    raise ValueError("labelmap có màu trùng/không hợp lệ.")
                colors[color] = label
            for key, content in members.items():
                if "SegmentationClass/" not in key or not key.lower().endswith(".png"):
                    continue
                filename = Path(key).stem + ".jpg"
                with Image.open(io.BytesIO(content)) as image:
                    if (image.height, image.width) != shapes[filename]:
                        raise ValueError(f"{filename}: kích thước PNG không khớp ảnh gốc.")
                    rgb = np.asarray(image.convert("RGB"))
                assigned = np.zeros(rgb.shape[:2], dtype=bool)
                recognized = assigned.copy()
                per_class = {}
                for color, label in colors.items():
                    region = (rgb == color).all(axis=-1)
                    recognized |= region
                    if label in schema["classes"]:
                        per_class[label] = int(region.sum())
                        assigned |= region
                if not assigned.any():
                    result["errors"].append(f"{filename}: không có pixel thuộc class hợp lệ.")
                if not recognized.all():
                    result["errors"].append(f"{filename}: có màu pixel không nằm trong labelmap.")
                stats[filename] = {"class_pixels": per_class, "unassigned_pixels": int((~assigned).sum())}
        else:
            data = json.loads(next(value for key, value in members.items()
                                   if key.lower().endswith(".json") and "annotation" in key.lower()))
            images = {item["id"]: item for item in data["images"]}
            for image in images.values():
                filename = Path(image["file_name"]).name
                if (image.get("height"), image.get("width")) != shapes[filename]:
                    raise ValueError(f"{filename}: COCO dimensions không khớp ảnh gốc.")
            for image_id, image in images.items():
                filename = Path(image["file_name"]).name
                coverage = np.zeros(shapes[filename], dtype=np.uint16)
                count = 0
                for annotation in data["annotations"]:
                    if annotation["image_id"] != image_id:
                        continue
                    mask = decode_coco(annotation["segmentation"], shapes[filename])
                    if not mask.any():
                        result["errors"].append(f"annotation {annotation['id']}: mask rỗng.")
                    coverage += mask
                    count += 1
                stats[filename] = {"annotations": count, "covered_pixels": int((coverage > 0).sum()),
                                   "overlap_pixels": int((coverage > 1).sum())}
                if not count:
                    result["warnings"].append(f"{filename}: không có annotation, cần xem lại trong CVAT.")
                if (coverage > 1).any():
                    result["warnings"].append(f"{filename}: có pixel chồng lấn, cần kiểm trực quan vùng nhìn thấy.")
        result["details"]["pixel_checks"] = stats
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        result["errors"].append(f"Không giải mã được mask: {exc}")
    result["warnings"].append("QC pixel không xác định đúng class/biên, đủ vật hoặc điểm số; cần xem ảnh.")
    return result


def inspect_exports(directory: Path = ROOT / "submissions", root: Path = ROOT) -> dict:
    report = inspect_all(directory, root)
    report["tasks"] = [inspect_pixels(row["task"], Path(row["file"]), root) for row in report["tasks"]]
    report["error_count"] = sum(len(row["errors"]) for row in report["tasks"])
    return report
