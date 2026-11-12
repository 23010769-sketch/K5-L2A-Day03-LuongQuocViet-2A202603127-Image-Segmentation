"""Local proposals only. Never download weights or send images to a service."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from automation.checkpoint_loader import Checkpoint
from automation.config import Settings
from automation.mask_validator import Mask


def resources() -> dict:
    result = {name: importlib.util.find_spec(name) is not None
              for name in ("torch", "transformers", "sam2")}
    if result["torch"]:
        try:
            import torch
            result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                          torch_cuda=torch.version.cuda)
            if torch.cuda.is_available():
                result.update(gpu=torch.cuda.get_device_name(0),
                              vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
        except (ImportError, OSError, RuntimeError) as exc:
            result["torch_error"] = type(exc).__name__
    return result


def grabcut(image: NDArray, box: tuple[int, int, int, int]) -> Mask:
    """CPU box-assisted algorithm, not a pretrained AI model or class detector."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box phải nằm trong ảnh, dạng x1,y1,x2,y2.")
    if (x2 - x1) * (y2 - y1) >= width * height:
        raise ValueError("GrabCut cần một ít nền bên ngoài box; dùng vẽ tay cho vùng phủ cả ảnh.")
    state = np.zeros((height, width), dtype=np.uint8)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), state,
                (x1, y1, x2 - x1, y2 - y1), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    return (state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD)


def strokes(mask: Mask, edits: list[dict]) -> Mask:
    """Draw/erase native-resolution strokes; never fill holes or smooth implicitly."""
    out = mask.astype(np.uint8).copy()
    height, width = out.shape
    for edit in edits:
        x1, y1, x2, y2, radius = (int(edit[key]) for key in ("x1", "y1", "x2", "y2", "radius"))
        if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height and 1 <= radius <= 100):
            raise ValueError("Tọa độ nét vẽ phải ở trong ảnh; radius từ 1 đến 100 pixel.")
        if edit["operation"] not in {"add", "erase"}:
            raise ValueError("operation phải là add hoặc erase.")
        value = int(edit["operation"] == "add")
        cv2.line(out, (x1, y1), (x2, y2), value, thickness=radius * 2 - 1)
    return out.astype(bool)


def _device(torch, requested: str) -> str:
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device phải là cpu, cuda hoặc auto.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch không dùng được CUDA; chọn CPU hoặc CVAT AI tools.")
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"


def _allow_local(settings: Settings) -> None:
    if not settings.local_models_allowed:
        raise PermissionError("Chưa xác nhận được phép dùng model ngoài CVAT (LOCAL_MODELS_ALLOWED=1).")


def segformer(image: NDArray, checkpoint: Checkpoint, model_dir: Path, settings: Settings,
              device: str = "cpu", mapping: dict[int, str] | None = None) -> list[tuple[str, Mask]]:
    _allow_local(settings)
    if checkpoint.kind != "semantic":
        raise ValueError("SegFormer semantic không tách instance; chỉ dùng CP3/CP4/CP6.")
    if not model_dir.is_dir() or not list(model_dir.glob("*.safetensors")):
        raise ValueError("Cần model local có config/processor và safetensors; không tự tải weights.")
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    selected = _device(torch, device)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True).to(selected).eval()
    known = {int(key): value for key, value in model.config.id2label.items()}
    mapping = mapping if mapping is not None else {key: value for key, value in known.items() if value in checkpoint.labels}
    if not mapping or any(key not in known or value not in checkpoint.labels for key, value in mapping.items()):
        raise ValueError("Mapping model→schema rỗng hoặc sai; không tự đoán/đổi tên class.")
    inputs = processor(images=image, return_tensors="pt").to(selected)
    with torch.inference_mode():
        logits = model(**inputs).logits
        prediction = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
    return [(label, np.isin(prediction, [key for key, value in mapping.items() if value == label]))
            for label in checkpoint.labels if label in mapping.values()]


def sam2_box(image: NDArray, box: tuple[int, int, int, int], checkpoint_file: Path,
             model_config: str, settings: Settings, device: str = "cpu") -> Mask:
    _allow_local(settings)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box nằm ngoài ảnh.")
    if not checkpoint_file.is_file():
        raise ValueError("Thiếu SAM 2 checkpoint local.")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(model_config, str(checkpoint_file),
                                            device=_device(torch, device), apply_postprocessing=False))
    with torch.inference_mode():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
    return masks[0].astype(bool)
