# AGENTS.md — Hệ thống quét và cải thiện chất lượng tài liệu (Python)

## Quy định giảng viên cập nhật — ưu tiên áp dụng

- Được dùng NumPy, OpenCV và PyTorch. Không dùng YOLO hoặc model/thư viện
  làm sẵn thay thế thuật toán của bài tập.
- OpenCV chỉ được dùng để đọc/ghi, resize và hiển thị ảnh (kể cả vẽ lớp
  đánh dấu phục vụ hiển thị, không dùng kết quả vẽ để phát hiện tài liệu).
- Cấm gọi các hàm xử lý ảnh có sẵn: `cvtColor`, `GaussianBlur`, `Canny`,
  `findContours`, `approxPolyDP`, `contourArea`, `arcLength`, morphology,
  `getPerspectiveTransform`, `warpPerspective`, `adaptiveThreshold`,
  `createCLAHE`, `filter2D`, `rotate`, hoặc hàm tương đương từ thư viện khác.
- Tự xây dựng thuật toán trong **image_algorithms.py**, chỉ dùng Python và
  NumPy cho phép toán mảng, histogram, giải hệ tuyến tính, indexing và reduction.
  Nếu dùng PyTorch sau này, chỉ dùng phép toán tensor cơ bản; không gọi model
  hay pipeline xử lý ảnh có sẵn. Hiện tại không cần cài PyTorch.
- `scanner.py` giữ API hiện có và ghép các thuật toán tự viết; `scan.py` và
  `gui.py` gọi cùng backend. Không để đường chạy phụ quay lại thuật toán OpenCV.
- Đọc/ghi PDF và hiển thị bằng Pillow/Tkinter hiện có vẫn giữ nguyên; không
  dùng bộ lọc Pillow/SciPy/scikit-image để thay thế phần phải tự cài đặt.
- Kiểm thử thuật toán bằng kết quả toán học/ảnh tổng hợp có tọa độ biết trước.
  Không tuyên bố tương đương OpenCV hoặc đạt chất lượng ảnh thực khi chưa đo.


> File này hướng dẫn AI agent (Claude Code, Cursor, Copilot...) cách làm việc trong dự án.
> **Đọc toàn bộ trước khi sửa bất kỳ dòng code nào.**

---

## 1. Mục tiêu dự án

Biến **ảnh chụp tài liệu bằng điện thoại** thành **ảnh giống bản scan**: phẳng, thẳng, nền sạch, chữ rõ nét.

**Mức độ:** Trung bình (đồ án/bài tập môn Xử lý ảnh, ưu tiên code rõ ràng, dễ giải thích hơn là tối ưu cực đoan).

### Chức năng bắt buộc

| # | Chức năng | Kỹ thuật chính |
|---|-----------|----------------|
| 1 | Phát hiện biên tờ giấy | Gaussian filtering, Canny, contour detection |
| 2 | Hiệu chỉnh phối cảnh | Perspective transformation |
| 3 | Chuyển ảnh xám / nhị phân | Grayscale, adaptive thresholding |
| 4 | Loại bỏ bóng, nhiễu, tăng tương phản | 
| 5 | Cắt bỏ phần nền thừa | 

---

## 2. Trạng thái hiện tại — QUAN TRỌNG

Dự án **đã có code cho gần đầy đủ các chức năng**. Nhiệm vụ của agent là **hoàn thiện phần còn thiếu**, KHÔNG viết lại từ đầu.

### Quy trình bắt buộc khi bắt đầu

1. **Audit trước:** đọc toàn bộ code hiện có, liệt kê (a) hàm/module nào đã có, (b) hàm nào hoạt động đúng, (c) hàm nào thiếu hoặc lỗi.
2. **Báo cáo ngắn** cho người dùng dưới dạng checklist (mục 8) trước khi code.
3. **Chỉ bổ sung / sửa** những phần thiếu hoặc sai. Giữ nguyên tên hàm, chữ ký hàm, cấu trúc thư mục hiện có trừ khi người dùng đồng ý đổi.
4. Mỗi thay đổi nên **nhỏ, có thể kiểm chứng** (chạy thử trên ảnh mẫu).

### Không được làm

- ❌ Viết lại toàn bộ project hoặc đổi kiến trúc khi chưa hỏi.
- ❌ Xóa code cũ mà không giải thích lý do.
- ❌ Thêm dependency không cần thiết. PyTorch được phép nhưng không bắt buộc.
- ❌ Gọi thuật toán có sẵn thay cho phần tự viết, kể cả bọc chúng trong hàm mới.
  Phải triển khai các bước tính toán trong `image_algorithms.py` bằng NumPy.

---



## 4. Cấu trúc thư mục đề xuất

Giữ cấu trúc hiện tại: `image_algorithms.py` chứa thuật toán tự viết,
`scanner.py` ghép pipeline, `scan.py` là CLI, `gui.py` là GUI và `tests/`
chứa kiểm thử. Cây bên dưới chỉ là tham khảo, không yêu cầu chuyển kiến trúc.

```
doc-scanner/
├── AGENT.md
├── README.md
├── requirements.txt
├── main.py                  # CLI entry point
├── scanner/
│   ├── __init__.py
│   ├── preprocess.py        # resize, grayscale, Gaussian blur
│   ├── edge_detection.py    # Canny, morphology, tìm contour tờ giấy
│   ├── geometry.py          # sắp xếp 4 điểm, perspective transform
│   ├── enhance.py           # bỏ bóng, khử nhiễu, tăng tương phản
│   ├── binarize.py          # adaptive threshold, Otsu
│   ├── crop.py              # cắt nền thừa, thêm lề
│   └── pipeline.py          # ghép toàn bộ các bước
├── samples/                 # ảnh đầu vào mẫu
├── output/                  # kết quả
├── debug/                   # ảnh trung gian từng bước (khi bật --debug)
└── tests/
    ├── test_geometry.py
    ├── test_edge_detection.py
    └── test_pipeline.py
```

---




## 7. Quy ước code

- Mỗi hàm **một nhiệm vụ**, có **type hints** và **docstring** (tiếng Việt hoặc tiếng Anh, thống nhất với code hiện có).
- Hàm xử lý là **hàm thuần**: nhận `np.ndarray`, trả `np.ndarray`, không đọc/ghi file bên trong.
- Không hard-code đường dẫn; dùng `pathlib.Path`.
- Không "magic number": đưa tham số vào hằng số hoặc dataclass `Config`.
- Dùng `logging` thay vì `print` cho thông tin debug.
- Kiểm tra đầu vào: file tồn tại, ảnh đọc được (`cv2.imread` trả `None` → báo lỗi rõ ràng).
- Tuân thủ PEP 8 (có thể dùng `ruff`/`black`).
- Ảnh trong OpenCV là **BGR**; chỉ đổi sang RGB khi hiển thị bằng Matplotlib.

### Chế độ debug

`--debug` phải lưu ảnh trung gian vào `debug/` theo thứ tự:

```
01_resized.jpg
02_gray_blur.jpg
03_edges.jpg
04_contour_overlay.jpg   # vẽ contour/4 đỉnh lên ảnh
05_warped.jpg
06_shadow_removed.jpg
07_final.jpg
```

Đây là phần quan trọng để giải thích pipeline khi báo cáo/thuyết trình.

---

## 8. Checklist cần audit và hoàn thiện

Agent cập nhật checklist này sau khi đọc code (✅ đã có & đúng / ⚠️ có nhưng cần sửa / ❌ chưa có):

**Cốt lõi**
- [x] Resize + lưu hệ số scale (OpenCV được phép)
- [x] Gaussian filtering tự viết
- [ ] Canny (ngưỡng tự động)
- [x] Contour: thành phần liên thông + bao lồi + RDP; chọn tứ giác lớn nhất
- [ ] `order_points` đúng mọi hướng (đã kiểm tra 24 hoán vị hình thoi; chưa kiểm chứng mọi trường hợp thực)
- [x] Perspective transform tự viết trên ảnh gốc
- [x] Chuyển xám tự viết
- [x] Adaptive thresholding tự viết; B&W dùng vùng 31 và C điều chỉnh theo Contrast
- [x] Bù sáng/bóng mềm bằng chia nền ước lượng trong B&W; bóng cứng cần đánh giá thêm
- [x] Khử nhiễu nhẹ trước nhị phân hóa; chưa loại mọi loại nhiễu
- [x] Tăng tương phản (CLAHE tự viết bằng NumPy; cần đánh giá thêm ảnh thực)
- [x] Cắt nền thừa theo bốn góc (phụ thuộc phát hiện/chỉnh góc đúng)

**Hoàn thiện (thường bị bỏ sót)**
- [x] Fallback khi không tìm thấy tờ giấy
- [ ] CLI với `--mode gray|binary|color`, `--debug`, `--output`
- [ ] Xử lý hàng loạt cả thư mục ảnh
- [ ] Xử lý EXIF orientation (ảnh điện thoại bị xoay)
- [x] Ghi ảnh so sánh: `output/numpy_verification/comparison.jpg`
- [x] Unit test cho `order_points`, perspective, pipeline
- [ ] README: cách cài, cách chạy, ví dụ kết quả, hạn chế
- [ ] Xuất PDF 1 trang (tùy chọn, dùng `img2pdf` hoặc Pillow)

---


## 10. Kiểm thử và tiêu chí nghiệm thu

### Bộ ảnh test (tối thiểu 8–10 ảnh trong `samples/`)

- Chụp thẳng / chụp nghiêng nhiều góc
- Có bóng đổ (tay, điện thoại, vật cản)
- Ánh sáng không đều, thiếu sáng
- Nền nhiều chi tiết (bàn gỗ, vải hoa văn)
- Giấy có một góc bị che hoặc cong nhẹ
- Ảnh có EXIF xoay

### Tiêu chí hoàn thành (Definition of Done)

- [ ] Chạy end-to-end không lỗi trên **≥ 80%** ảnh mẫu; ảnh còn lại có fallback và cảnh báo rõ ràng.
- [ ] Kết quả: tờ giấy thẳng, không còn nền thừa đáng kể, chữ đọc được rõ.
- [ ] Bóng đổ giảm rõ rệt so với ảnh gốc; nền gần như trắng đồng đều.
- [ ] Chế độ binary không làm vỡ/đứt nét chữ quá mức.
- [ ] `pytest` pass.
- [ ] Có ảnh debug minh họa từng bước.
- [ ] README đầy đủ, có ví dụ trước–sau.

---

## 11. Cách làm việc của agent

1. **Hỏi lại** nếu yêu cầu mơ hồ hoặc cần đổi kiến trúc; không tự đoán.
2. **Làm từng bước nhỏ:** audit → báo cáo → sửa/bổ sung 1 chức năng → chạy thử → báo kết quả → chức năng tiếp theo.
3. **Giải thích ngắn gọn** vì sao chọn tham số/kỹ thuật (người dùng cần hiểu để trình bày).
4. Khi sửa lỗi, nêu **nguyên nhân gốc** chứ không chỉ vá triệu chứng.
5. Cuối mỗi phiên, tóm tắt: đã làm gì, file nào đổi, còn thiếu gì (cập nhật checklist mục 8).
6. Trả lời bằng **tiếng Việt**, thuật ngữ kỹ thuật giữ nguyên tiếng Anh (Canny, contour, perspective transform...).

---

## 12. Lỗi thường gặp (tham khảo nhanh)

| Triệu chứng | Nguyên nhân thường gặp | Hướng xử lý |
|-------------|------------------------|-------------|
| Không tìm thấy contour giấy | Nền cùng màu giấy, Canny quá chặt | Giảm ngưỡng Canny, dilate/close, thử fallback |
| Ảnh warp bị lật/xoắn | `order_points` sai | Kiểm tra bằng sum/diff, test 4 hướng |
| Warp bị mờ | Warp trên ảnh đã resize nhỏ | Map điểm về ảnh gốc rồi warp |
| Chữ bị đứt khi nhị phân | `blockSize` quá nhỏ / `C` quá lớn | Tăng `blockSize`, giảm `C`, khử nhiễu nhẹ trước |
| Còn bóng loang | Kernel ước lượng nền quá nhỏ | Tăng kích thước kernel blur/morphology |
| Ảnh bị xoay 90° | Bỏ qua EXIF orientation | Áp EXIF transpose khi đọc ảnh |
