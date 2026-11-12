# Công cụ hỗ trợ checkpoint Day03

Chạy local, dùng ảnh đã có trong repository, không tải ảnh/dataset từ CVAT và không gửi ảnh tới dịch vụ AI. Không tạo ZIP bài nộp; ZIP phải được export từ CVAT theo quy định lớp. Bản nháp và lịch sử duyệt nằm trong `.runtime/`, đã được Git bỏ qua.

## Chạy ngay trong Terminal PowerShell của VS Code

Mở Terminal tại thư mục gốc repo. Môi trường `.venv` đã được cài và kiểm thử với Python **3.14.3**. Mã yêu cầu Python 3.11 trở lên; các adapter model tùy chọn chưa được kiểm thử trên môi trường này. Nếu cài lại:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r automation/requirements.txt
```

Không cần activate môi trường hay đổi execution policy:

```powershell
.\.venv\Scripts\python.exe -m automation list
.\.venv\Scripts\python.exe -m automation doctor
.\.venv\Scripts\python.exe -m streamlit run automation/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Mở `http://127.0.0.1:8501`. Dừng bằng Ctrl+C. Chỉ chạy một phiên công cụ ghi bản nháp trên máy; kho JSON không hỗ trợ nhiều người sửa đồng thời. Không publish giao diện.

## Yêu cầu đã đối chiếu

Nguồn chuẩn: `data/manifest.json`, `CHECKPOINTS.md`, `guideline-mini-sheet.md`, `RULES.md`, `classes.json` và `cvat-labels.json` của từng task. Loader tự kiểm schema và format; có mâu thuẫn thì dừng.

| Checkpoint / ảnh trong `data/checkpoints/<checkpoint>/images/` | Cần làm | Labels chính xác | Export |
| --- | --- | --- | --- |
| cp1_holes / 000000144300.jpg | Giữ kính/chi tiết trong mask; phân biệt lỗ nền thật; không tự lấp hoặc khoét | person, bicycle, car, motorcycle, bus, truck | COCO 1.0 |
| cp2_slice / 000000017627.jpg | Hai xe sát nhau là hai instance | person, bicycle, car, motorcycle, bus, truck | COCO 1.0 |
| cp3_thin / 839f7736-abe28069.jpg | Cột/biển mảnh ở zoom lớn, nét 2–3 px theo ảnh | pole, traffic sign, sky, road | Segmentation mask 1.1 |
| cp4_curb / 7d83710e-4697c3b2.jpg | Ranh theo chức năng và bó vỉa; không chỉ màu | road, sidewalk | Segmentation mask 1.1 |
| cp5_occlusion / 000000336232.jpg | Các phần nhìn thấy của cùng vật giữ một mask/instance; không tô xuyên vật che | person, bicycle, car, motorcycle, bus, truck | COCO 1.0 |
| cp6_coverage / 7daa6479-67988f3f.jpg | Phủ vùng nhìn thấy thuộc schema; vùng không chắc cần xem lại | road, sidewalk, building, vegetation, sky, car, person | Segmentation mask 1.1 |

CP2 trong tài liệu kiểm tách xe sát nhau, khác trọng tâm vật bị cắt bởi mép ảnh trong yêu cầu ban đầu. Quy tắc chỉ vẽ phần nhìn thấy vẫn áp dụng. CP4 không có class `curb`. README/CVAT_SETUP yêu cầu **CVAT local**, còn môi trường người học là **app.cvat.ai**. RULES chỉ nêu gợi ý có sẵn trong CVAT, chưa xác nhận model ngoài CVAT. Phải xác nhận với coach trước khi bật cloud/API hoặc AI local; cờ mặc định khóa cả hai. Tệp `reports/REPORT_TEMPLATE.md` được nhiều tài liệu dẫn tới nhưng không tồn tại trong checkout hiện tại; báo cáo thực có ở `report/individual/MSSV_HoTen.md`.

## Cách nhanh nhất để hoàn thành đúng yêu cầu

1. Xác nhận với coach CVAT cloud có được phép không. Nếu chưa được phép, dùng CVAT local theo tài liệu; công cụ offline vẫn dùng được.
2. Trong CVAT tạo từng task bằng đúng ảnh và `cvat-labels.json` của trạm. Không đổi Raw labels ở task đã có annotation.
3. Nếu môi trường được phép có SAM 3/SAM 2, dùng AI tools để gợi ý, kiểm từng vật/vùng rồi sửa bằng Brush/Polygon. Không có model thì vẽ tay, không chờ cài model lớn.
4. CP1 kiểm kính/lỗ; CP2 tạo riêng từng xe; CP5 giữ các phần cùng vật trong một mask. CP3 zoom và brush nhỏ; CP4 xem bó vỉa; CP6 đối chiếu từng class. Đếm vật và kiểm biên bằng mắt.
5. Save, đổi ảnh rồi mở lại. Chỉ export/download khi trường cho phép: Job menu → Export job dataset → format ở bảng trên. Đổi tên **ZIP bên ngoài** thành `cpX_name.zip`, đặt vào `submissions/`, giữ nguyên nội dung.
6. Chạy kiểm ZIP, ghi tiến độ/lỗi/quyết định thực tế vào báo cáo. Không tự chấm điểm nếu chưa có reference được phép.

CVAT mô tả [SAM 3 tương tác bằng điểm/box](https://www.cvat.ai/resources/changelog/sam-3-image-segmentation) và [text prompt theo label](https://www.cvat.ai/resources/changelog/sam3-text-prompts). Sự có mặt/quota trong tài khoản vẫn cần kiểm trực tiếp. Không gọi dịch vụ AI ngoài CVAT bằng công cụ này.

## Duyệt và sửa bản nháp offline

- Chọn checkpoint, xem ảnh gốc và schema. Có thể nạp **PNG nhị phân** có sẵn trên máy, tạo mask vẽ tay hoặc chạy GrabCut CPU từ box do bạn chọn. GrabCut không phải model AI được huấn luyện và không biết class.
- Tọa độ theo kích thước ảnh gốc, gốc (0,0) ở góc trên trái. Bảng nét vẽ dùng `x1,y1,x2,y2,radius,operation`. `operation=add` thêm, `erase` xóa; radius=1 cho nét 1 px, radius=2 cho nét 3 px. Bấm Áp dụng; không làm mượt/lấp lỗ/xóa vùng nhỏ tự động.
- Xem mask và overlay, dùng zoom 4× cho biên. Ghi quyết định thật, đánh dấu đã xem cảnh báo rồi chấp nhận hoặc từ chối. Mọi sửa mask/class làm mất chấp nhận trước đó; phải duyệt lại.
- CP5 có công cụ gộp các mảng cùng class do bạn xác định thuộc cùng vật. Sau khi duyệt bản gộp, từ chối các bản phần cũ để tránh nộp trùng. Công cụ giữ vùng rời, không nối xuyên phần che.
- Tab toàn cảnh hiển thị tất cả mask đã chấp nhận, thống kê vùng chưa gán/chồng lấn và cho nạp ROI bạn xác định cần phủ. Tỷ lệ phủ toàn ảnh không chứng minh coverage đúng: ngoài schema có thể để trống.

Tự động kiểm được tên ảnh, kích thước, class hợp lệ, mask rỗng, lỗ, số thành phần, vùng nhỏ và chồng lấn. Sai class về ngữ nghĩa, thiếu vật, gộp/tách instance, biên chính xác và bản chất lỗ **bắt buộc người học kiểm ảnh**. Cảnh báo nhiều thành phần ở CP5 không phải lệnh tách instance.

## Model và tài nguyên

Quan sát ngày 11/11/2026: RTX 4050 6 GB, driver 555.97 báo CUDA 12.5; Python hệ thống có PyTorch 2.11.0+cpu, `torch.cuda.is_available() == False`. `.venv` chỉ cài bộ công cụ QC/UI; chưa có torch/transformers/SAM 2 và chưa có weights. Dòng CUDA trong nvidia-smi không có nghĩa PyTorch đã dùng được GPU.

| Phương án | Phù hợp | Giới hạn / trạng thái |
| --- | --- | --- |
| SAM 3 trong CVAT | Gợi ý tương tác cho instance hoặc từng vùng | Ưu tiên nếu được trường cho phép và tài khoản có model; học viên kiểm/sửa |
| SAM 3 local | Model promptable, cần cấu hình riêng | Chưa tích hợp adapter; [upstream](https://github.com/facebookresearch/sam3) yêu cầu Python ≥3.12, PyTorch ≥2.7, CUDA ≥12.6. Chưa đạt điều kiện hiện tại, chưa đo VRAM/inference |
| SAM 2 local | Box cho từng instance/vùng; người dùng chọn class | Có adapter tùy chọn; cần [package Meta chính thức](https://github.com/facebookresearch/sam2), config và weights local; CPU có thể chậm, không tự phát hiện đủ vật |
| SegFormer local | Semantic CP3/CP4/CP6 nếu model có labels tương ứng | Có adapter tùy chọn [Transformers](https://huggingface.co/docs/transformers/model_doc/segformer). Chỉ exact-match labels, hoặc mapping chỉ định trong Python; model class ngoài schema được bỏ qua. Không biến semantic thành instance; nét mảnh cần sửa |
| DeepLabV3 | Semantic với bộ class của weights đã chọn | Chỉ khảo sát; không tích hợp vì [weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html) dùng tập COCO/VOC giới hạn, thiếu schema đường phố như pole/traffic sign/sidewalk. Không coi background là road |
| GrabCut / vẽ tay CPU | Chạy được ngay, không weights | Có triển khai và kiểm thử; box/màu không bảo đảm đúng object |

Chưa chạy inference bằng model AI nào và chưa tạo annotation cho checkpoint. Không tự tải weights, nâng driver hay cài PyTorch GPU. Sau khi có quyền dùng model local, chọn build PyTorch theo [trang chính thức](https://pytorch.org/get-started/locally/) và driver thực tế, rồi cài phụ thuộc tùy chọn:

```powershell
.\.venv\Scripts\python.exe -m pip install -r automation/requirements-ai.txt
$env:LOCAL_MODELS_ALLOWED = '1'
```

Lệnh trên không cài torch hoặc SAM 2. Nếu package/model không tương thích Python 3.14, dùng môi trường Python 3.12 riêng thay vì sửa môi trường đang chạy UI. SegFormer chỉ đọc thư mục local có `config.json`, processor config và `.safetensors` bằng `local_files_only=True`, không chạy remote code. SAM 2 dùng checkpoint local tin cậy và config trong package, tắt hậu xử lý tự động. Các adapter được kiểm tra import/khóa quyền; chưa được kiểm thử với weights thật. Khi thiếu tài nguyên, dùng CVAT AI tools được phép hoặc Brush/Polygon.

## API: chỉ bật khi được phép

[CVAT khuyến nghị Personal Access Token](https://docs.cvat.ai/docs/api_sdk/access_tokens/) trong User settings → Security. Không nhập token vào mã, CLI argument, báo cáo hoặc chat. Dùng PAT read-only cho kiểm trạng thái. Tệp `.env.example` chỉ để hướng dẫn, công cụ **không tự nạp `.env`**.

Sau khi coach đã cho phép API, trong PowerShell hiện tại:

```powershell
$env:CVAT_API_ALLOWED = '1'
$env:CVAT_HOST = 'https://app.cvat.ai'
$secret = Read-Host 'CVAT Personal Access Token' -AsSecureString
$env:CVAT_ACCESS_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
# Nếu task trong organization, đặt đúng slug; tài khoản cá nhân để trống.
$env:CVAT_ORG = ''
.\.venv\Scripts\python.exe -m automation cvat-status cp1_holes --task-id 123
```

Thay `123` bằng ID task thật. Lệnh chỉ GET task/labels/frame metadata/jobs; không lấy pixel ảnh, dataset hoặc annotation. Không có PAT/quyền thì dừng trước request. HTTPS được giữ nguyên xác minh TLS, chỉ cho HTTP trên localhost. Lỗi SDK hiển thị loại lỗi thay vì in response có thể chứa thông tin nhạy cảm. Khởi động Streamlit từ chính Terminal có các biến môi trường này; thay biến thì dừng/chạy lại app. Gỡ token khi xong:

```powershell
Remove-Item Env:CVAT_ACCESS_TOKEN
```

Ghi annotation cần **đồng thời** quyền trường, PAT cho phép ghi, quyền đọc annotation để xác minh, và người dùng xác nhận batch sau khi xem payload:

```powershell
# Chỉ thực hiện sau khi đã được phép đọc/ghi annotation.
$env:CVAT_ANNOTATION_READ_ALLOWED = '1'
$env:CVAT_WRITE_ALLOWED = '1'
```

Trong tab CVAT: đọc metadata → kiểm đúng task/ảnh/labels → xem payload → nhập đúng `UPLOAD cp1_holes TASK 123 MASKS N` → bấm gửi. Chấp nhận local riêng lẻ không gọi API. Không chạy đồng thời người khác sửa cùng task. Công cụ chỉ so tên/kích thước, không tải pixel để chứng minh ảnh server giống ảnh repo: bạn cần mở CVAT kiểm điều này.

Chỉ hỗ trợ checkpoint **một ảnh, frame 0, chưa có annotation**. Nếu task đã có annotation, tiếp tục sửa trực tiếp trên CVAT; không trộn/ghi đè. Mapping dùng label IDs thật, không coi class index/trainid là CVAT ID. Mask encode bằng SDK chính thức, giữ lỗ và mảng rời trong một shape. Chỉ dùng [`partial_update_annotations(action='create')`](https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/tasks-api/), rồi GET annotation kiểm IDs/label/geometry. Không gọi replace/delete/import/export.

Mỗi batch có `sync-<hash>.json` viết độc quyền trước PATCH. Nếu timeout hoặc verify lỗi, trạng thái vẫn `pending`, công cụ không tự retry. **Không xóa journal để gửi lại** khi chưa kiểm task thủ công: request có thể đã lưu thành công. `verified` chỉ xác nhận dữ liệu được lưu, chưa chứng minh mask đúng. SDK đọc/ghi được kiểm thử bằng server giả trong unit tests; chưa kiểm kết nối thật do chưa có quyền và PAT.

## Kiểm ZIP và tests

```powershell
python scripts/inspect_submissions.py --dir submissions
.\.venv\Scripts\python.exe -m automation inspect --out automation/.runtime/export-qc.json
.\.venv\Scripts\python.exe -m unittest automation.test_automation -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Lệnh đầu là script trường nguyên bản. Lệnh thứ hai đọc PNG/RLE/polygon thực, so kích thước ảnh, kiểm mask rỗng và thống kê chồng lấn; không giải nén ra ổ đĩa, không sửa ZIP, không đọc reference, không cho điểm. RLE COCO là column-major, khác RLE của CVAT; decoder kiểm runs trước khi dựng pixel theo [định dạng COCO chính thức](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c).

Unit tests dùng dữ liệu tổng hợp trong thư mục tạm; không ghi vào `submissions/` hoặc kho review của học viên. Kiểm quan trọng gồm topology, nét 1 px, invalidate review, mapping label ID, payload SDK, quyền/confirmation, task đã có annotation, verify sau lưu và chống retry sau timeout.

Xem [STATUS.md](STATUS.md) cho kết quả thực tế và các lỗi test có sẵn của repository. `OK` của inspector không phải hoàn thành hoặc được chấm điểm.

## Báo cáo và Git sau khi bạn xác nhận hoàn thành

Bảng tiến độ chỉ cập nhật theo ZIP/mask thật đã đọc; các phần quyết định tự vẽ, lỗi đã sửa và ba ca cân nhắc vẫn cần bạn điền. Không điền hộ trải nghiệm hoặc điểm. Xác nhận MSSV/họ tên rồi đổi tên báo cáo theo `SUBMISSION.md`. Chỉ sau khi tự kiểm xong và bạn quyết định nộp, chạy từng lệnh (công cụ không tự chạy):

```powershell
git status --short
# Nếu chưa đổi tên; kiểm danh tính trước khi chạy:
git mv report/individual/MSSV_HoTen.md report/individual/2A202603127_LuongQuocViet.md
git add -- submissions/easy_semantic.zip submissions/medium_instance.zip submissions/hard_panoptic.zip
git add -- submissions/cp1_holes.zip submissions/cp2_slice.zip submissions/cp3_thin.zip submissions/cp4_curb.zip submissions/cp5_occlusion.zip submissions/cp6_coverage.zip
git add -- report/individual/2A202603127_LuongQuocViet.md
# Chỉ thêm tooling nếu bạn muốn đưa cùng fork; runtime/models/.env không được thêm.
git add -- .gitignore automation
git diff --cached --stat
git diff --cached --check
git commit -m "Complete Day03 segmentation submissions and report"
git push origin HEAD
```

Nếu task nào chưa làm, bỏ đường dẫn ZIP đó và ghi rõ vào report; không tạo ZIP rỗng. Mở fork kiểm file thật đã hiện, đảm bảo coach đọc được rồi tự nộp link fork trên LMS. Theo `SUBMISSION.md`, lab 11/11/2026 có hạn **12/11/2026 lúc 12:00 UTC+7**. Không có lệnh commit/push/API write nào đã được chạy trong lần triển khai này.
