"""Mục đích: ghép các thuật toán NumPy thành các chức năng quét tài liệu.
GUI và CLI gọi file này để tìm giấy, sửa phối cảnh, tăng tương phản và tạo B&W.
Chi tiết phép tính nằm trong image_algorithms.py; file này không đọc/ghi ảnh.
Đọc preprocess_image và four_point_transform trước, rồi đến scan_document.
Các hằng SCAN_* điều khiển làm sạch nền và ngưỡng nhị phân đầu ra.
"""

import image_algorithms as alg
import numpy as np

CONTRAST_CLIP_LIMIT = 2.0
CONTRAST_TILE_GRID = (8, 8)
SCAN_BACKGROUND_SIZE = 41
SCAN_BLOCK_SIZE = 31
SCAN_THRESHOLD_OFFSET = 7.0


def enhance_contrast(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Tăng tương phản với mức strength từ 0 đến 1, giữ kích thước/kênh ảnh.
    1. Kiểm tra ảnh uint8 và mức điều chỉnh hợp lệ; 0 trả bản sao ảnh gốc.
    2. Tính độ sáng xám rồi tăng tương phản bằng CLAHE.
    3. Lấy chênh lệch độ sáng nhân strength và cộng vào ảnh ban đầu.
    Với ảnh màu, cộng cùng độ lệch vào B/G/R để giữ chênh lệch kênh trước khi chặn.
    Không phải OCR; tăng mạnh cũng có thể làm nổi nhiễu và bão hòa màu.
    """
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Contrast strength must be between 0 and 1.")
    if (not isinstance(image, np.ndarray) or image.size == 0
            or image.dtype != np.uint8
            or not (image.ndim == 2
                    or (image.ndim == 3 and image.shape[2] == 3))):
        raise ValueError("Expected a non-empty uint8 grayscale or BGR image.")
    if strength == 0:
        return image.copy()

    # Change luminance while retaining BGR channel differences (chroma).
    luminance = alg.grayscale(image)
    enhanced = alg.clahe(luminance, CONTRAST_CLIP_LIMIT, CONTRAST_TILE_GRID)
    delta = strength * (enhanced.astype(float) - luminance.astype(float))
    if image.ndim == 3:
        delta = delta[..., None]
    return alg.uint8(image.astype(float) + delta)

def order_points(pts):
    """Sắp bốn góc thành chu kỳ TL, TR, BR, BL để ánh xạ phối cảnh.
    1. Tạo bao lồi và loại hình suy biến/đỉnh trùng.
    2. Chọn đỉnh có tổng x+y nhỏ nhất làm điểm bắt đầu; hòa thì ưu tiên y nhỏ.
    3. Xoay danh sách bao lồi, giữ thứ tự quanh biên để không chọn trùng góc.
    Đây là quy ước hình học; không tự nhận biết hướng đọc của chữ.
    """
    points = np.asarray(pts, dtype=float).reshape(4, 2)
    hull = alg.convex_hull(points)
    if len(hull) != 4 or alg.polygon_area(hull) < 1:
        raise ValueError("Four distinct convex document corners are required.")
    # Hull follows TL -> TR -> BR -> BL in image coordinates. Unlike independent
    # min/max selection, rolling cannot duplicate a point on rotated diamonds.
    start = np.lexsort((hull[:, 1], hull.sum(axis=1)))[0]
    return np.roll(hull, -start, axis=0).astype(np.float32)

def preprocess_image(image, blur_kernel=(5, 5), canny_low=75, canny_high=200, apply_morphology=True):
    """Chuẩn bị ảnh cho việc tìm biên giấy, trả (gray, blurred, edged).
    1. Chuyển xám. 2. Gaussian giảm nhiễu. 3. Canny tìm biên.
    4. Nếu bật apply_morphology, đóng khe hở nhỏ trên biên.
    Bước này phục vụ tìm góc; không trực tiếp làm sạch ảnh scan đầu ra.
    """
    # 1. Convert to grayscale: removes color noise and speeds up processing
    gray = alg.grayscale(image)
    
    # 2. Gaussian Blur: smooths image to remove high frequency noise (like paper texture/table grains)
    blurred = alg.uint8(alg.gaussian_blur(gray, blur_kernel))
    
    # 3. Canny Edge Detection: finds regions of high intensity gradients (boundaries)
    edged = alg.canny(blurred, canny_low, canny_high)
    
    # 4. Morphological Close: closes gaps in contours (bridges broken edge lines)
    if apply_morphology:

        edged = alg.close_edges(edged)
        
    return gray, blurred, edged

def find_document_contour(edged):
    """Gọi bộ dò NumPy để chọn tứ giác giấy từ ảnh biên.
    Trả bốn góc dạng (4, 1, 2), hoặc None để GUI/CLI dùng vùng dự phòng.
    """
    return alg.document_contour(edged)

def get_default_corners(width, height, margin=0.05):
    """Tạo bốn góc vùng cắt dự phòng nằm giữa ảnh.
    width/height là kích thước ảnh, margin là tỷ lệ lề mỗi phía.
    Không phải phát hiện giấy: người dùng nên chỉnh lại nếu vùng này không khớp.
    """
    w_margin = int(width * margin)
    h_margin = int(height * margin)
    return np.array([
        [w_margin, h_margin],
        [width - w_margin, h_margin],
        [width - w_margin, height - h_margin],
        [w_margin, height - h_margin]
    ], dtype="float32")

def four_point_transform(image, pts):
    """Cắt và nắn vùng giấy thành hình chữ nhật từ ảnh nguồn và bốn góc.
    1. Sắp góc theo chu kỳ thống nhất.
    2. Ước lượng rộng/cao bằng độ dài lớn nhất của hai cạnh đối diện.
    3. Đặt bốn góc đích ở bốn góc hình chữ nhật.
    4. Tính homography, ánh xạ ngược và nội suy để tạo ảnh đã nắn.
    Kích thước chỉ là ước lượng từ ảnh chụp, không khôi phục chính xác tỷ lệ giấy thật.
    """
    # Order points: [TL, TR, BR, BL]
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image:
    # It is the maximum distance between bottom-right and bottom-left x-coords
    # or the top-right and top-left x-coords.
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))
    
    # Compute the height of the new image:
    # It is the maximum distance between top-right and bottom-right y-coords
    # or the top-left and bottom-left y-coords.
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))
    
    # Construct destination coordinates for a top-down view (flat sheet)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Get the perspective transformation matrix M
    M = alg.homography(rect, dst)
    
    # Apply warp perspective to project the document onto a flat surface
    warped = alg.warp_perspective(image, M, (max_width, max_height))
    
    return warped

def sharpen_image(image):
    """Làm sắc nét bằng kernel high-pass 3 x 3: tâm 5, bốn láng giềng -1.
    Tính tổng có trọng số rồi chặn về uint8; có thể làm nổi cả nhiễu.
    Đây là kernel tăng nét cố định, không phải unsharp mask có Gaussian riêng.
    """
    # High-pass filter kernel for sharpening
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    return alg.uint8(alg.correlate(image, kernel))

def binarize_image(image):
    """Tạo ảnh scan đen trắng với tham số mặc định.
    Giữ API cũ bằng cách gọi scan_document; bao gồm làm sạch nền trước đặt ngưỡng.
    """
    return scan_document(image)


def scan_document(image: np.ndarray, contrast: float = 0.0) -> np.ndarray:
    """Ghép luồng tạo bản scan B&W, trả ảnh một kênh chỉ có 0 và 255.
    1. clean_document giảm nhiễu và bù nền sáng.
    2. Nếu contrast khác 0, tăng tương phản trên ảnh đã làm sạch.
    3. Tăng độ lệch ngưỡng theo contrast để hạn chế hạt nền xuất hiện lại.
    4. Nhị phân hóa cục bộ; cần so với ảnh xám khi chữ quá mờ.
    """
    clean = alg.clean_document(image, background_size=SCAN_BACKGROUND_SIZE)
    if contrast != 0:
        clean = enhance_contrast(clean, contrast)
    # CLAHE also expands residual paper variation. Increase the threshold
    # margin with contrast so the slider does not bring background speckles back.
    offset = SCAN_THRESHOLD_OFFSET * (1.0 + contrast)
    return alg.adaptive_threshold(clean, SCAN_BLOCK_SIZE, offset)

def rotate_image(image, angle):
    """Xoay ảnh theo chiều kim đồng hồ 90, 180 hoặc 270 độ bằng NumPy.
    Trả bản sao sau xoay; góc khác ba giá trị trên giữ nguyên ảnh đầu vào.
    """
    if angle == 90:
        return np.rot90(image, -1).copy()
    elif angle == 180:
        return np.rot90(image, 2).copy()
    elif angle == 270:
        return np.rot90(image, 1).copy()
    return image

