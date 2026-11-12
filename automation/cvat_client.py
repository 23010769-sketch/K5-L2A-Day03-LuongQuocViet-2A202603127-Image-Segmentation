"""Official SDK, metadata-only inspection and explicitly confirmed additive writes.

No task creation/deletion, image download, dataset import/export or annotation replacement.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

from automation.annotation_review import ReviewItem, atomic_json, image_digest, utc_now
from automation.checkpoint_loader import Checkpoint
from automation.config import RUNTIME, Settings
from automation.mask_validator import validate_scene

LOG = logging.getLogger(__name__)


def label_mapping(checkpoint: Checkpoint, labels: list[dict]) -> dict[str, int]:
    names = [item["name"] for item in labels]
    if len(names) != len(set(names)) or set(names) != set(checkpoint.labels):
        raise ValueError("Labels trên CVAT không khớp chính xác classes.json; không tự sửa schema.")
    if any(item.get("attributes") or item.get("type", "any") not in {"any", "mask"} for item in labels):
        raise ValueError("Task có attributes/type khác schema công khai; cần kiểm thủ công.")
    ids = [item["id"] for item in labels]
    if len(ids) != len(set(ids)) or any(type(value) is not int for value in ids):
        raise ValueError("Label ID không hợp lệ.")
    return dict(zip(names, ids))


def build_shapes(checkpoint: Checkpoint, items: list[ReviewItem], labels: list[dict],
                 frame: int = 0) -> list[dict]:
    from cvat_sdk.masks import encode_mask

    mapping = label_mapping(checkpoint, labels)
    if not items or len({item.id for item in items}) != len(items):
        raise ValueError("Batch rỗng hoặc có review ID trùng.")
    with Image.open(checkpoint.images[0]) as image:
        shape = (image.height, image.width)
    fingerprint = image_digest(checkpoint)
    masks = []
    for item in items:
        if (not item.approved or item.checkpoint != checkpoint.name
                or item.image != checkpoint.images[0].name or item.image_sha256 != fingerprint):
            raise ValueError("Có mask chưa được duyệt, đã sửa hoặc không khớp ảnh gốc.")
        masks.append((item.label, item.mask(shape)))
    check = validate_scene(masks, checkpoint, shape)
    if check.errors:
        raise ValueError("; ".join(check.errors))
    return [dict(type="mask", frame=frame, label_id=mapping[label], points=encode_mask(mask),
                 occluded=False, outside=False, z_order=0, rotation=0, source="manual", attributes=[])
            for label, mask in masks]


def _signature(shape: dict) -> str:
    # Server may normalize floating-point coordinates to integers or vice versa.
    return json.dumps([shape["type"], shape["frame"], shape["label_id"],
                       [float(value) for value in shape["points"]]])


class CvatClient:
    def __init__(self, settings: Settings, runtime: Path = RUNTIME):
        self.settings = settings
        self.runtime = runtime
        self._client = None

    def __enter__(self) -> CvatClient:
        self.settings.require_api()
        from cvat_sdk import Client
        # Assign PAT before any request. No password login or persistent profile/cache.
        self._client = Client(self.settings.host, check_server_version=False)
        self._client.api_client.configuration.access_token = self.settings.token
        self._client.organization_slug = self.settings.organization or ""
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()

    def inspect(self, task_id: int, checkpoint: Checkpoint) -> dict:
        """GET task, labels, frame metadata and jobs; never fetch image pixels/annotations."""
        if self._client is None:
            raise RuntimeError("Dùng CvatClient như context manager.")
        if type(task_id) is not int or task_id <= 0:
            raise ValueError("task_id phải là số nguyên dương.")
        from cvat_sdk.api_client.model_utils import to_json

        task = self._client.tasks.retrieve(task_id)
        if task.name != checkpoint.name:
            raise ValueError("Tên task không khớp checkpoint; không tác động ba tier đã hoàn thành.")
        labels = [to_json(label) for label in task.get_labels()]
        mapping = label_mapping(checkpoint, labels)
        meta = to_json(task.get_meta())
        frames = meta["frames"]
        if (len(frames) != 1 or meta.get("start_frame", 0) != 0 or meta.get("deleted_frames")
                or meta.get("included_frames") not in (None, [0])):
            raise ValueError("Chỉ hỗ trợ checkpoint một ảnh, frame 0, không frame bị xóa/lọc.")
        frame = frames[0]
        name = PurePosixPath(frame["name"].replace("\\", "/")).name
        with Image.open(checkpoint.images[0]) as image:
            if name != checkpoint.images[0].name or (frame["width"], frame["height"]) != image.size:
                raise ValueError("Ảnh hoặc kích thước trên CVAT không khớp ảnh repository.")
        return {"task_id": task.id, "name": task.name, "status": str(task.status),
                "labels": labels, "label_mapping": mapping, "frame": frame,
                "jobs": [{"id": job.id, "state": str(job.state)} for job in task.get_jobs()],
                "limitation": "Chỉ so tên/kích thước ảnh, không xác minh nội dung pixel từ CVAT."}

    def upload(self, task_id: int, checkpoint: Checkpoint, items: list[ReviewItem],
               confirmation: str) -> dict:
        """Write only to a verified EMPTY checkpoint after per-batch confirmation.

        A pending journal survives ambiguous failures; never retry a write automatically.
        """
        if not (self.settings.api_allowed and self.settings.write_allowed
                and self.settings.annotation_read_allowed):
            raise PermissionError("Cần quyền API, đọc annotation và ghi annotation đã được phép.")
        expected = f"UPLOAD {checkpoint.name} TASK {task_id} MASKS {len(items)}"
        if confirmation != expected:
            raise PermissionError(f"Cần xác nhận đúng batch: {expected}")
        state = self.inspect(task_id, checkpoint)
        shapes = build_shapes(checkpoint, items, state["labels"])
        from cvat_sdk.api_client import models
        from cvat_sdk.api_client.model_utils import to_json

        task = self._client.tasks.retrieve(task_id)
        before = to_json(task.get_annotations())
        if any(before.get(key) for key in ("tags", "shapes", "tracks", "intervals")):
            raise ValueError("Task đã có annotation: dừng để tránh trùng/gộp/chồng lấn; sửa tiếp trong CVAT.")
        identity = [self.settings.host, self.settings.organization, task_id, [item.digest() for item in items]]
        batch_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        journal = self.runtime / checkpoint.name / f"sync-{batch_id}.json"
        # Exclusive journal prevents two local workers from submitting the same batch.
        journal.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": "pending", "batch_id": batch_id, "task_id": task_id,
                   "created_at": utc_now(), "review_ids": [item.id for item in items]}
        with journal.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        try:
            result, _ = self._client.api_client.tasks_api.partial_update_annotations(
                action="create", id=task_id,
                patched_labeled_data_request=models.PatchedLabeledDataRequest(
                    version=before.get("version", 0), tags=[], tracks=[],
                    shapes=[models.LabeledShapeRequest(**shape) for shape in shapes]),
                _request_timeout=60)
            created = to_json(result).get("shapes", [])
            after = to_json(task.get_annotations())
            expected_signatures = Counter(_signature(shape) for shape in shapes)
            created_ids = {shape.get("id") for shape in created}
            saved = [shape for shape in after.get("shapes", []) if shape.get("id") in created_ids]
            if (None in created_ids or len(created_ids) != len(shapes)
                    or Counter(_signature(shape) for shape in saved) != expected_signatures):
                raise RuntimeError("Không xác minh được đủ mask/label/geometry sau khi lưu; mở CVAT kiểm, không retry.")
            payload.update(state="verified", verified_at=utc_now(), annotation_ids=sorted(created_ids))
            atomic_json(journal, payload)
            LOG.info("Verified %d masks for %s task %d", len(saved), checkpoint.name, task_id)
            return payload
        except Exception:
            LOG.error("Sync chưa được xác minh. Giữ pending journal; kiểm trong CVAT trước mọi lần thử tiếp.")
            raise
