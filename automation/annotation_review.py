"""Lossless local proposals, explicit review and invalidation after every edit."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from automation.checkpoint_loader import Checkpoint
from automation.config import RUNTIME
from automation.mask_validator import Mask, binary_mask, validate_mask


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class ReviewItem:
    id: str
    checkpoint: str
    image: str
    label: str
    source: str
    image_sha256: str
    png: str
    decision: str = "pending"
    note: str = ""
    reviewed_at: str = ""
    approved_digest: str = ""

    def mask(self, shape: tuple[int, int]) -> Mask:
        with Image.open(io.BytesIO(base64.b64decode(self.png, validate=True))) as image:
            return binary_mask(np.asarray(image), shape)

    def digest(self) -> str:
        fields = (self.id, self.checkpoint, self.image, self.label, self.source, self.image_sha256, self.png)
        return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()

    @property
    def approved(self) -> bool:
        return self.decision == "accepted" and self.approved_digest == self.digest()


def image_digest(checkpoint: Checkpoint) -> str:
    return hashlib.sha256(checkpoint.images[0].read_bytes()).hexdigest()


def new_item(checkpoint: Checkpoint, label: str, mask: Mask, source: str, *, allow_empty: bool = False) -> ReviewItem:
    with Image.open(checkpoint.images[0]) as image:
        shape = (image.height, image.width)
    check = validate_mask(mask, label, checkpoint, shape)
    errors = [error for error in check.errors if not (allow_empty and error == "Mask rỗng.")]
    if errors:
        raise ValueError("; ".join(errors))
    output = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(output, format="PNG")
    return ReviewItem(uuid4().hex, checkpoint.name, checkpoint.images[0].name, label,
                      source, image_digest(checkpoint), base64.b64encode(output.getvalue()).decode("ascii"))


def revise(item: ReviewItem, checkpoint: Checkpoint, mask: Mask, label: str) -> ReviewItem:
    fresh = new_item(checkpoint, label, mask, item.source + " / edited", allow_empty=True)
    return replace(fresh, id=item.id)


def decide(item: ReviewItem, decision: str, note: str) -> ReviewItem:
    if decision not in {"accepted", "rejected"} or not note.strip():
        raise ValueError("Cần accepted/rejected và ghi quyết định thực tế sau khi xem ảnh.")
    return replace(item, decision=decision, note=note.strip(), reviewed_at=utc_now(),
                   approved_digest=item.digest() if decision == "accepted" else "")


class ReviewStore:
    """Run one local app instance; JSON writes are atomic, not a multiuser database."""
    def __init__(self, checkpoint: Checkpoint, directory: Path = RUNTIME):
        self.checkpoint = checkpoint
        self.path = directory / checkpoint.name / "review.json"

    def load(self) -> list[ReviewItem]:
        if not self.path.exists():
            return []
        items = [ReviewItem(**row) for row in json.loads(self.path.read_text(encoding="utf-8"))]
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)) or any(item.checkpoint != self.checkpoint.name for item in items):
            raise ValueError("Review store có ID trùng hoặc nhầm checkpoint.")
        return items

    def save(self, item: ReviewItem) -> None:
        if item.checkpoint != self.checkpoint.name or item.image != self.checkpoint.images[0].name:
            raise ValueError("Review item không thuộc checkpoint này.")
        items = self.load()
        index = next((i for i, old in enumerate(items) if old.id == item.id), None)
        if index is None:
            items.append(item)
        else:
            items[index] = item
        atomic_json(self.path, [asdict(old) for old in items])
