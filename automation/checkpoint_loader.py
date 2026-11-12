"""Read paths, exact labels and rules from the repository, not model assumptions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from automation.config import ROOT


@dataclass(frozen=True)
class Checkpoint:
    name: str
    kind: str
    directory: Path
    images: tuple[Path, ...]
    labels: tuple[str, ...]
    colors: dict[str, tuple[int, int, int]]
    rule: str
    edge: str
    export_format: str


def load_checkpoints(root: Path = ROOT) -> dict[str, Checkpoint]:
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))["tasks"]
    rules = (root / "CHECKPOINTS.md").read_text(encoding="utf-8")
    sections = dict(re.findall(r"^## (cp\d+_\w+)\s*\n(.*?)(?=^## |\Z)", rules, re.M | re.S))
    result = {}
    for name in sorted(manifest):
        if not re.fullmatch(r"cp[1-6]_\w+", name):
            continue
        info = manifest[name]
        directory = (root / "data" / info["path"]).resolve()
        if not directory.is_relative_to((root / "data/checkpoints").resolve()):
            raise ValueError(f"Checkpoint path nằm ngoài data/checkpoints: {name}")
        schema = json.loads((directory / "classes.json").read_text(encoding="utf-8"))
        raw = json.loads((directory / "cvat-labels.json").read_text(encoding="utf-8"))
        labels = tuple(schema["classes"])
        if (len(labels) != len(set(labels)) or len(raw) != len(labels)
                or set(labels) != {item["name"] for item in raw} or schema["type"] != info["type"]):
            raise ValueError(f"Schema mâu thuẫn cho {name}; cần xác nhận trước khi dùng.")
        images = tuple(sorted((directory / "images").glob("*.jpg")))
        if len(images) != 1 or name not in sections:
            raise ValueError(f"Cần đúng một ảnh và quy tắc CHECKPOINTS.md cho {name}.")
        fmt = "Segmentation mask 1.1" if info["type"] == "semantic" else "COCO 1.0"
        if fmt not in sections[name]:
            raise ValueError(f"Format trong CHECKPOINTS.md mâu thuẫn với manifest: {name}")
        result[name] = Checkpoint(name, info["type"], directory, images, labels,
                                  {key: tuple(value) for key, value in schema["colors"].items()},
                                  sections[name].strip(), info["edge"], fmt)
    if len(result) != 6:
        raise ValueError("Manifest không có đủ sáu checkpoint.")
    return result
