"""Geometry warnings are review signals, never a correctness score."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from automation.checkpoint_loader import Checkpoint

Mask = NDArray[np.bool_]


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, int | float] = field(default_factory=dict)


def binary_mask(values: NDArray, shape: tuple[int, int]) -> Mask:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"Mask phải có kích thước gốc {shape}; nhận {array.shape}.")
    if not np.isin(array, [0, 1, 255]).all():
        raise ValueError("Mask phải nhị phân 0/1 hoặc 0/255; không nhận ảnh màu/ảnh xác suất.")
    return array.astype(bool)


def validate_mask(mask: Mask, label: str, checkpoint: Checkpoint, shape: tuple[int, int]) -> Validation:
    result = Validation()
    if label not in checkpoint.labels:
        result.errors.append(f"Class không thuộc schema: {label}")
    try:
        mask = binary_mask(mask, shape)
    except ValueError as exc:
        result.errors.append(str(exc))
        return result
    area = int(mask.sum())
    result.metrics["area_pixels"] = area
    if not area:
        result.errors.append("Mask rỗng.")
        return result
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = int((hierarchy[0, :, 3] >= 0).sum()) if hierarchy is not None else 0
    result.metrics.update(components=count - 1, holes=holes)
    if holes:
        result.warnings.append(f"Có {holes} lỗ bên trong: kiểm kính/chi tiết và lỗ nền thật; không tự lấp.")
    if count > 2:
        message = "Các mảng rời có thể thuộc cùng một vật bị che; không tự tách ID." if checkpoint.kind == "instance" else "Kiểm tính liên tục và vùng rời."
        result.warnings.append(f"Có {count - 1} thành phần. {message}")
    if np.any(stats[1:, cv2.CC_STAT_AREA] < 16):
        result.warnings.append("Có vùng nhỏ hơn 16 pixel; có thể là chi tiết đúng, không tự xóa.")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        result.warnings.append("Mask chạm mép ảnh: kiểm phần nhìn thấy, không mở rộng ngoài ảnh.")
    if checkpoint.name == "cp3_thin":
        result.warnings.append("Xem zoom lớn, giữ nét 2–3 px theo ảnh; không erosion/opening hoặc làm mượt tự động.")
    return result


def validate_scene(masks: list[tuple[str, Mask]], checkpoint: Checkpoint,
                   shape: tuple[int, int], required_roi: Mask | None = None) -> Validation:
    result = Validation()
    coverage = np.zeros(shape, dtype=np.uint16)
    for index, (label, mask) in enumerate(masks):
        single = validate_mask(mask, label, checkpoint, shape)
        result.errors.extend(f"Mask {index + 1}: {error}" for error in single.errors)
        result.warnings.extend(f"Mask {index + 1}: {warning}" for warning in single.warnings)
        if not single.errors:
            coverage += mask.astype(bool)
    overlap = int((coverage > 1).sum())
    result.metrics.update(overlap_pixels=overlap, covered_pixels=int((coverage > 0).sum()),
                          unassigned_pixels=int((coverage == 0).sum()))
    if overlap:
        message = f"{overlap} pixel chồng lấn giữa các mask; cần kiểm phần nhìn thấy."
        (result.errors if checkpoint.kind == "semantic" else result.warnings).append(message)
    if required_roi is not None:
        required = binary_mask(required_roi, shape)
        missing = int((required & (coverage == 0)).sum())
        result.metrics["uncovered_required_pixels"] = missing
        if missing:
            result.warnings.append(f"{missing} pixel chưa phủ trong ROI do người dùng xác định.")
    result.warnings.append("Pixel chưa gán không mặc định là lỗi: ngoài schema có thể để trống. Sai class/thiếu vật cần xem ảnh hoặc reference được phép.")
    return result


def overlay(image: NDArray, masks: list[tuple[str, Mask]], checkpoint: Checkpoint) -> NDArray:
    out = np.asarray(image, dtype=np.uint8).copy()
    for label, mask in masks:
        color = np.asarray(checkpoint.colors[label], dtype=float)
        out[mask] = (out[mask] * 0.55 + color * 0.45).astype(np.uint8)
    return out
