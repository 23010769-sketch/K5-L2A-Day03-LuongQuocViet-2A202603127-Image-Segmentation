# Kết quả kiểm tra thực tế — 11/11/2026

## Công cụ đã triển khai

- Đọc đường dẫn từ manifest, kiểm đúng sáu ảnh/schema/format và hiển thị quy tắc từng trạm.
- Streamlit local: ảnh gốc, mask, overlay, zoom, tạo/sửa bằng nét thêm/xóa, chấp nhận/từ chối kèm ghi chú, thống kê toàn cảnh và ROI tùy chọn.
- GrabCut CPU đã chạy trên ảnh tổng hợp trong test. Adapter SegFormer/SAM 2 chỉ đọc weights local, khóa theo quyền; chưa chạy model AI thật vì chưa có quyền xác nhận, phụ thuộc và weights.
- CVAT SDK: GET metadata có kiểm task/ảnh/schema; ghi additive có xác nhận từng batch, chỉ task rỗng, kiểm lại geometry/label/ID sau lưu; giữ journal chống gửi trùng sau lỗi không rõ kết quả.
- ZIP QC dùng nguyên inspector của trường, thêm đọc pixel PNG/COCO và kiểm kích thước/mask rỗng/chồng lấn. Không sửa ZIP hoặc script gốc.

## Validation

| Kiểm tra | Kết quả |
| --- | --- |
| Python trong `.venv` | 3.14.3 |
| Imports / compileall | Qua |
| `pip check` | Không có dependency bị hỏng |
| `python -m unittest automation.test_automation automation.test_ui -v` | **19/19 qua** |
| Streamlit AppTest | Khởi tạo, tạo nét 1 px, duyệt/lưu trong kho tạm, đổi cả sáu checkpoint đều qua |
| SDK payload | Được model SDK chấp nhận; roundtrip giữ lỗ và mảng rời trong một shape |
| CVAT read/write với server giả | Qua; không ghi vào task đã có annotation, không retry sau timeout |
| CVAT thật | **Chưa kết nối**: chưa xác nhận quyền cloud/API, chưa có PAT. Lệnh status dừng trước request với PermissionError |
| API thay đổi dữ liệu / tải dataset | Không gọi |
| GPU | Có RTX 4050 6 GB; PyTorch hệ thống bản CPU, CUDA unavailable; `.venv` chưa có torch |
| Kiểm tra script trường | Ba ZIP hiện có OK về cấu trúc; thiếu sáu ZIP checkpoint |
| `git diff --check` | Qua |

Chạy toàn bộ 25 tests gốc: **21 qua, 1 failure và 3 errors**. Nguyên nhân có sẵn trong checkout/môi trường: `reports/REPORT_TEMPLATE.md` không tồn tại (ba kiểm tra phụ thuộc file này thất bại); một test gọi executable `python3`, không có trên Windows này. `git ls-tree HEAD reports` không có template đó. Không sửa test, script hoặc dựng nội dung giả để làm các kiểm tra này qua. Output scorecard trong test gốc dùng dữ liệu/mocking của test ở thư mục tạm, **không phải điểm bài của học viên**.

## Bài nộp đọc từ ZIP hiện tại

| Task | Quan sát thực tế | Cần làm tiếp |
| --- | --- | --- |
| Easy | 3/3 ảnh có mask giải mã được | Xem class/biên/coverage; các pixel ngoài schema không mặc định là lỗi |
| Medium | 104 annotation: 103 RLE + 1 polygon; 34/37/33 annotation theo ba ảnh | Xem số vật/class/biên, kiểm chồng lấn |
| Hard | 7 RLE trên `000000350023.jpg`, chỉ stuff; `000000460147.jpg` có 0 annotation trong ZIP | Kiểm cả hai ảnh trong CVAT, đặc biệt things và ảnh thứ hai; Save/export lại khi được phép nếu ZIP cũ/chưa đầy đủ |
| CP1–CP6 | Chưa có ZIP checkpoint | Chưa xác minh trạng thái trên CVAT; chưa thể ghi hoàn thành |

Số pixel chồng lấn được phát hiện, chưa kết luận lỗi:

| Task / ảnh | Pixel chồng lấn |
| --- | ---: |
| Medium / 000000181542.jpg | 1.742 |
| Medium / 000000373353.jpg | 53 |
| Medium / 000000458325.jpg | 409 |
| Hard / 000000350023.jpg | 5 |

Đã cập nhật riêng tên ZIP và các quan sát trên vào bảng tiến độ báo cáo, giữ thông tin và các phần quyết định của người học. Không điền số ảnh hoàn thành dựa trên chất lượng chưa xem, lỗi đã sửa, quyết định cá nhân hoặc điểm số. Ba ZIP hiện có được giữ nguyên. Không tạo ZIP mới, không commit/push.

## Còn cần người học / coach

1. Coach xác nhận có chấp nhận app.cvat.ai thay CVAT local không; API và model AI ngoài CVAT có được phép không. Tài liệu hiện ghi CVAT local, chưa công bố ngoại lệ hoặc kết luận dùng cloud sẽ bị từ chối.
2. Nếu được phép API, người học tự cấp PAT qua biến môi trường; không gửi token vào chat. Từng batch ghi vẫn cần xác nhận riêng trong UI.
3. Người học kiểm mask trên ảnh thật, ghi quyết định thật, Save/export ZIP đúng định dạng và hoàn thiện report. Công cụ không làm thay việc kiểm/chấm sáu checkpoint.
