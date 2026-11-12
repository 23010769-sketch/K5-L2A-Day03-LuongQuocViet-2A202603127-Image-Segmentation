"""Run: python -m streamlit run automation/app.py --server.address 127.0.0.1"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit executes this file as a script, not as automation.app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
from PIL import Image

from automation.annotation_review import ReviewStore, decide, new_item, revise
from automation.checkpoint_loader import load_checkpoints
from automation.config import ROOT, RUNTIME, Settings
from automation.cvat_client import CvatClient, build_shapes
from automation.export_validator import inspect_pixels
from automation.mask_validator import binary_mask, overlay, validate_mask, validate_scene
from automation.segmentation_engine import grabcut, resources, sam2_box, segformer, strokes


def main() -> None:
    st.set_page_config(page_title="Day03 — kiểm tra checkpoint", layout="wide")
    st.title("Day03 — hỗ trợ kiểm tra 6 checkpoint")
    st.caption("Dữ liệu ảnh từ repository. Chấp nhận bản nháp local chưa có nghĩa đã đồng bộ hoặc hoàn thành checkpoint.")
    settings = Settings.from_env()
    checkpoints = load_checkpoints()
    name = st.sidebar.selectbox("Checkpoint", list(checkpoints))
    checkpoint = checkpoints[name]
    store = ReviewStore(checkpoint)
    items = store.load()
    for key, cp in checkpoints.items():
        records = ReviewStore(cp).load()
        accepted_count = sum(item.approved for item in records)
        has_zip = (ROOT / "submissions" / f"{key}.zip").exists()
        st.sidebar.caption(f"{key}: {accepted_count} mask duyệt local · ZIP {'có' if has_zip else 'chưa có'}")
    st.markdown(checkpoint.rule)
    st.write("Labels:", ", ".join(checkpoint.labels))
    with Image.open(checkpoint.images[0]) as original:
        image = np.asarray(original.convert("RGB"))
    shape = image.shape[:2]
    height, width = shape
    state_key = f"draft:{name}"
    id_key = f"editing:{name}"
    source_key = f"source:{name}"
    tabs = st.tabs(["Tạo / sửa / duyệt mask", "Kiểm toàn cảnh", "CVAT", "Kiểm ZIP"])

    with tabs[0]:
        label = st.selectbox("Class cho bản nháp", checkpoint.labels, key=f"label:{name}")
        st.caption(f"Ảnh {checkpoint.images[0].name} · {width} × {height}. Tọa độ theo ảnh gốc.")
        if items:
            record_id = st.selectbox("Bản nháp đã lưu", [item.id for item in items],
                                     format_func=lambda value: next(f"{item.label} · {item.decision} · {value[:8]}" for item in items if item.id == value))
            selected = next(item for item in items if item.id == record_id)
            st.caption(f"Nguồn: {selected.source}; ghi chú duyệt: {selected.note or 'chưa có'}")
            if st.button("Mở mask này để xem / sửa"):
                st.session_state[state_key] = selected.mask(shape)
                st.session_state[id_key] = selected.id
                st.session_state[source_key] = selected.source
                # Widget value can be set on next rerun, before widget construction.
                st.session_state[f"load-label:{name}"] = selected.label
                st.rerun()
        if st.button("Tạo mask vẽ tay mới"):
            st.session_state[state_key] = np.zeros(shape, dtype=bool)
            st.session_state[id_key] = None
            st.session_state[source_key] = "manual local strokes"
        upload = st.file_uploader("Mở mask PNG nhị phân có sẵn trên máy (đúng kích thước)", type=["png"], key=f"png:{name}")
        if upload is not None and st.button("Nạp PNG làm bản nháp"):
            with Image.open(upload) as imported:
                st.session_state[state_key] = binary_mask(np.asarray(imported), shape)
            st.session_state[id_key] = None
            st.session_state[source_key] = "user supplied local binary PNG"
        with st.expander("Gợi ý từ box / model local"):
            box_text = st.text_input("Box x1,y1,x2,y2", f"{width // 4},{height // 4},{width * 3 // 4},{height * 3 // 4}")
            backend = st.selectbox("Công cụ", ["GrabCut CPU", "SAM 2 local", "SegFormer local"])
            st.caption("GrabCut là thuật toán theo box, không nhận diện class. Model local cần quyền của trường và weights có sẵn; không tự tải.")
            model_path = st.text_input("Thư mục SegFormer / file SAM 2 local", "automation/models/")
            sam_config = st.text_input("Config SAM 2 thuộc package chính thức", "configs/sam2.1/sam2.1_hiera_t.yaml")
            device = st.selectbox("Thiết bị model", ["cpu", "auto", "cuda"])
            if st.button("Tạo gợi ý chưa duyệt"):
                box = tuple(map(int, box_text.split(",")))
                if len(box) != 4:
                    raise ValueError("Box cần bốn số nguyên.")
                with st.spinner("Đang tạo gợi ý local…"):
                    if backend == "SegFormer local":
                        suggestions = segformer(image, checkpoint, Path(model_path), settings, device)
                        saved = 0
                        for proposed_label, mask in suggestions:
                            if mask.any():
                                store.save(new_item(checkpoint, proposed_label, mask, f"SegFormer local: {model_path}"))
                                saved += 1
                        st.success(f"Đã lưu {saved} đề xuất chưa duyệt. Chọn từng mask để kiểm.")
                        st.rerun()
                    mask = (grabcut(image, box) if backend == "GrabCut CPU" else
                            sam2_box(image, box, Path(model_path), sam_config, settings, device))
                    st.session_state[state_key] = mask
                    st.session_state[id_key] = None
                    st.session_state[source_key] = backend + " / user box"
        if name == "cp5_occlusion" and items:
            with st.expander("Gộp các phần nhìn thấy của cùng một vật thành một mask"):
                chosen = st.multiselect("Chỉ chọn các phần cùng instance và cùng class", [item.id for item in items],
                                        format_func=lambda value: next(f"{item.label} · {value[:8]}" for item in items if item.id == value))
                if st.button("Tạo bản nháp gộp"):
                    regions = [item for item in items if item.id in chosen]
                    if len(regions) < 2 or any(item.label != label for item in regions):
                        raise ValueError("Cần ít nhất hai phần cùng class đang chọn.")
                    st.session_state[state_key] = np.logical_or.reduce([item.mask(shape) for item in regions])
                    st.session_state[id_key] = None
                    st.session_state[source_key] = "user merged visible regions: " + ",".join(chosen)
                    st.warning("Sau khi duyệt mask gộp, từ chối các bản phần cũ để tránh trùng. Không tô vùng bị che.")
        draft = st.session_state.get(state_key)
        if draft is None:
            st.image(image, caption="Ảnh gốc", width="stretch")
        else:
            editing_id = st.session_state.get(id_key)
            old = next((item for item in items if item.id == editing_id), None)
            if old and old.approved and (old.label != label or not np.array_equal(old.mask(shape), draft)):
                store.save(revise(old, checkpoint, draft, label))
                st.rerun()
            cols = st.columns(3)
            cols[0].image(image, caption="Ảnh gốc", width="stretch")
            cols[1].image(draft.astype(np.uint8) * 255, caption="Mask bản nháp", width="stretch", clamp=True)
            cols[2].image(overlay(image, [(label, draft)], checkpoint), caption="Overlay", width="stretch")
            with st.expander("Zoom vùng biên và nét mảnh"):
                cx = st.slider("Tâm X", 0, width - 1, width // 2)
                cy = st.slider("Tâm Y", 0, height - 1, height // 2)
                radius = st.slider("Bán kính vùng xem", 10, max(11, min(width, height) // 2), 50)
                crop = overlay(image, [(label, draft)], checkpoint)[max(0, cy-radius):cy+radius, max(0, cx-radius):cx+radius]
                st.image(Image.fromarray(crop).resize((crop.shape[1]*4, crop.shape[0]*4), Image.Resampling.NEAREST), caption="Phóng 4×, không làm mượt")
            edits = st.data_editor([dict(x1=0, y1=0, x2=0, y2=0, radius=1, operation="add")],
                                   num_rows="dynamic", key=f"strokes:{name}")
            st.caption("Thêm/xóa nét: x1,y1 → x2,y2, radius=1 cho nét 1 px; operation là add hoặc erase. Có thể sửa rồi nạp lại PNG nhị phân.")
            if st.button("Áp dụng các nét trong bảng"):
                edited = strokes(draft, edits)
                editing_id = st.session_state.get(id_key)
                if editing_id:
                    original_item = next(item for item in items if item.id == editing_id)
                    # Invalidate persisted approval immediately, even before another save.
                    store.save(revise(original_item, checkpoint, edited, label))
                st.session_state[state_key] = edited
                st.rerun()
            check = validate_mask(draft, label, checkpoint, shape)
            st.json(check.metrics)
            for error in check.errors:
                st.error(error)
            for warning in check.warnings:
                st.warning(warning)
            note = st.text_area("Quyết định sau khi xem: vị trí / class / biên / lý do sửa hoặc giữ", key=f"note:{name}")
            acknowledge = st.checkbox("Tôi đã kiểm ảnh, class, số vật, biên và các cảnh báo", key=f"ack:{name}")
            controls = st.columns(3)
            actions = [("Lưu bản nháp", "pending"), ("Chấp nhận local", "accepted"), ("Từ chối mask", "rejected")]
            for col, (title, decision) in zip(controls, actions):
                if col.button(title, disabled=(bool(check.errors) and decision != "rejected") or (decision == "accepted" and not acknowledge)):
                    editing_id = st.session_state.get(id_key)
                    old = next((item for item in items if item.id == editing_id), None)
                    record = (revise(old, checkpoint, draft, label) if old else
                              new_item(checkpoint, label, draft, st.session_state.get(source_key, "manual")))
                    if decision != "pending":
                        record = decide(record, decision, note)
                    store.save(record)
                    st.session_state[id_key] = record.id
                    st.rerun()

    with tabs[1]:
        masks = [(item.label, item.mask(shape)) for item in items if item.approved]
        st.image(overlay(image, masks, checkpoint), caption="Tất cả mask đã chấp nhận local", width="stretch")
        roi = st.file_uploader("ROI nhị phân: vùng bạn xác định cần phủ (tùy chọn)", type=["png"], key=f"roi:{name}")
        required = None
        if roi:
            with Image.open(roi) as roi_image:
                required = binary_mask(np.asarray(roi_image), shape)
        scene = validate_scene(masks, checkpoint, shape, required)
        st.json(scene.metrics)
        for error in scene.errors:
            st.error(error)
        for warning in scene.warnings:
            st.warning(warning)
        st.caption("Cảnh báo chỉ hỗ trợ tìm vùng cần xem. Không có reference thì không kết luận mask sai class hoặc đủ tất cả vật.")

    with tabs[2]:
        st.info("Mặc định offline. Chỉ bật API sau khi coach xác nhận CVAT cloud/API; cấu hình bằng biến môi trường trong README.")
        st.write({"quyền_API": settings.api_allowed, "quyền_đọc_annotation": settings.annotation_read_allowed,
                  "quyền_ghi": settings.write_allowed})
        task_id = int(st.number_input("Task ID của checkpoint này", min_value=1, step=1))
        if st.button("Đọc trạng thái / labels / metadata", disabled=not settings.api_allowed):
            try:
                with CvatClient(settings) as client:
                    state = client.inspect(task_id, checkpoint)
                st.session_state[f"cvat:{name}:{task_id}"] = state
            except Exception as exc:
                st.error(f"Đọc CVAT thất bại ({type(exc).__name__}); kiểm quyền, PAT, task ID, tên ảnh và schema. Token không được hiển thị.")
        remote = st.session_state.get(f"cvat:{name}:{task_id}")
        accepted = [item for item in items if item.approved]
        if remote:
            st.json(remote)
            proposed_shapes = build_shapes(checkpoint, accepted, remote["labels"]) if accepted else []
            st.write(f"Batch dự kiến: {len(proposed_shapes)} mask đã duyệt. Chỉ ghi vào task checkpoint chưa có annotation.")
            st.caption("Task không được người khác chỉnh đồng thời. Tên/kích thước ảnh phải đúng; bạn tự xác nhận ảnh trên CVAT chính là ảnh repo.")
            with st.expander("Payload để kiểm trước khi ghi"):
                st.json({"action": "create", "shapes": proposed_shapes})
            phrase = f"UPLOAD {name} TASK {task_id} MASKS {len(accepted)}"
            st.code(phrase)
            confirmation = st.text_input("Sau khi được phép ghi, nhập nguyên dòng trên để xác nhận đúng batch", key=f"confirm:{name}")
            if st.button("Gửi batch đã xác nhận", disabled=not (settings.write_allowed and settings.annotation_read_allowed and accepted)):
                try:
                    with CvatClient(settings) as client:
                        receipt = client.upload(task_id, checkpoint, accepted, confirmation)
                    st.success("Đã đọc lại và xác minh mask/label/geometry trên CVAT. Hãy mở Job kiểm trực quan, sau đó export ZIP.")
                    st.json(receipt)
                except Exception as exc:
                    st.error(f"Chưa xác minh được đồng bộ ({type(exc).__name__}). Không tự gửi lại; kiểm task và sync journal local trước.")
        journals = sorted((RUNTIME / name).glob("sync-*.json"))
        if journals:
            st.write("Lịch sử xác minh local:")
            for journal in journals:
                st.json(json.loads(journal.read_text(encoding="utf-8")))
        else:
            st.caption("Chưa có lần đồng bộ bằng công cụ này. Annotation làm trực tiếp trong CVAT không được suy đoán từ trạng thái local.")
    with tabs[3]:
        st.write(f"Save trong CVAT → mở lại ảnh → Export job dataset → {checkpoint.export_format}.")
        st.code(f"submissions/{name}.zip")
        st.caption("Chỉ export/download khi được trường cho phép. Giữ nguyên file bên trong ZIP. Công cụ không tải dataset hay tạo ZIP thay CVAT.")
        if st.button("Kiểm ZIP đã đặt trong submissions/"):
            st.json(inspect_pixels(name, ROOT / "submissions" / f"{name}.zip"))
        with st.expander("Khả năng model trong môi trường đang chạy"):
            st.json(resources())


if __name__ == "__main__":
    # Process queued label change before the selectbox is instantiated.
    for pending_key in list(st.session_state):
        if pending_key.startswith("load-label:"):
            st.session_state[pending_key.replace("load-label:", "label:")] = st.session_state.pop(pending_key)
    try:
        main()
    except (ValueError, PermissionError, OSError, ImportError, RuntimeError) as exc:
        st.error(f"{type(exc).__name__}: {exc}")
