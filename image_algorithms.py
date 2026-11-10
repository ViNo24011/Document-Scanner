"""Mục đích: tự cài đặt các thuật toán xử lý ảnh bằng Python và NumPy.
Đây là phần nên đọc khi học công thức và cách tính, không chứa giao diện hay I/O.
Quy ước: ảnh uint8 trong [0, 255], ảnh màu BGR; điểm hình học (x, y),
nhưng truy cập mảng là image[y, x]. Kết quả trung gian thường dùng float.
Luồng tìm giấy: grayscale -> gaussian_blur -> canny -> close_edges -> document_contour.
Luồng tạo bản scan: warp_perspective -> clean_document -> clahe -> adaptive_threshold.
Không dùng hàm xử lý ảnh có sẵn của OpenCV, SciPy hoặc model YOLO.
"""
from collections import deque
from typing import Optional, Tuple

import numpy as np


def uint8(image: np.ndarray) -> np.ndarray:
    """Đưa giá trị ảnh về uint8 an toàn.
    Đầu vào: mảng số; đầu ra: mảng cùng kích thước trong [0, 255].
    Làm tròn rồi chặn giá trị trước khi ép kiểu để tránh tràn số.
    """
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh BGR thành ảnh xám H x W.
    Công thức Y = 0.114B + 0.587G + 0.299R; kênh xanh lá có trọng số lớn nhất.
    Nếu đầu vào đã là ảnh xám thì trả bản sao, không sửa ảnh nguồn.
    """
    if image.ndim == 2:
        return image.copy()
    return uint8(image.astype(float) @ np.array([0.114, 0.587, 0.299]))


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Đổi cách bố trí kênh để Pillow/Tkinter hiển thị đúng màu.
    Ảnh BGR: đảo thứ tự thành RGB; ảnh xám: lặp kênh xám ba lần.
    """
    return (np.repeat(image[..., None], 3, axis=2) if image.ndim == 2
            else image[..., ::-1].copy())


def correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tính tương quan chéo giữa ảnh và kernel, trả mảng float cùng kích thước.
    1. Kiểm tra kernel lẻ để có điểm giữa.
    2. Đệm biên bằng pixel sát mép, tránh làm tối viền.
    3. Trượt kernel, nhân từng trọng số và cộng đóng góp.
    Kernel không bị lật: đây là correlation, không phải convolution tổng quát.
    """
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")
    padding = [(kh // 2, kh // 2), (kw // 2, kw // 2)]
    if image.ndim == 3:
        padding.append((0, 0))
    padded = np.pad(image.astype(float), padding, mode="edge")
    result = np.zeros(image.shape, dtype=float)
    h, w = image.shape[:2]
    for y, x in np.ndindex(kernel.shape):
        if kernel[y, x] != 0:
            result += kernel[y, x] * padded[y:y + h, x:x + w]
    return result


def gaussian_blur(image: np.ndarray, size: Tuple[int, int] = (5, 5),
                  sigma: float = 0.0) -> np.ndarray:
    """Làm mượt ảnh bằng bộ lọc Gaussian tách được, trả kết quả float.
    size = (chiều rộng, chiều cao) kernel; sigma điều khiển mức trải rộng.
    1. Tạo các vị trí quanh tâm và tính exp(-x² / (2 sigma²)).
    2. Chuẩn hóa tổng trọng số bằng 1 để giữ độ sáng vùng đều.
    3. Lọc ngang rồi dọc; tương đương kernel Gaussian 2D nhưng ít phép tính hơn.
    Nếu sigma <= 0, ước lượng sigma từ kích thước kernel.
    """
    result = image.astype(float)
    for axis, length in [(1, size[0]), (0, size[1])]:
        if length < 1 or length % 2 == 0:
            raise ValueError("Gaussian kernel size must be positive and odd.")
        s = sigma if sigma > 0 else 0.3 * ((length - 1) / 2 - 1) + 0.8
        positions = np.arange(length) - length // 2
        weights = np.exp(-positions ** 2 / (2 * s * s))
        weights /= weights.sum()
        kernel = weights[None, :] if axis == 1 else weights[:, None]
        result = correlate(result, kernel)
    return result


def canny(image: np.ndarray, low: float = 75, high: float = 200) -> np.ndarray:
    """Tìm biên Canny từ ảnh xám đã làm mượt, trả ảnh 0/255.
    1. Sobel tính gradient gx, gy: mức thay đổi độ sáng theo hai hướng.
    2. Tính độ lớn và chia góc gradient thành bốn hướng.
    3. NMS chỉ giữ điểm lớn nhất theo hướng gradient để làm mảnh biên.
    4. Dùng low/high phân loại điểm ứng viên và điểm biên mạnh.
    5. Hysteresis giữ điểm yếu khi nối được với điểm mạnh qua 8 láng giềng.
    Ngưỡng áp dụng trên độ lớn Sobel, không phải trực tiếp trên độ sáng ảnh.
    """
    if not 0 <= low <= high:
        raise ValueError("Canny thresholds must satisfy 0 <= low <= high.")
    # Bước 1: Sobel đo mức thay đổi độ sáng ngang/dọc, chưa phải biên cuối cùng.
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx, gy = correlate(image, sobel), correlate(image, sobel.T)
    magnitude = np.hypot(gx, gy)
    direction = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    # Bước 2: Quy góc về 0, 45, 90, 135 độ để chọn hai láng giềng so sánh.
    bins = ((direction + 22.5) // 45).astype(int) % 4
    padded = np.pad(magnitude, 1)
    h, w = image.shape
    suppressed = np.zeros_like(magnitude)
    # Bước 3 (NMS): chỉ giữ cực đại cục bộ dọc theo hướng gradient.
    for label, (dy, dx) in enumerate([(0, 1), (1, 1), (1, 0), (1, -1)]):
        a = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        b = padded[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        keep = (bins == label) & (magnitude >= a) & (magnitude >= b)
        suppressed[keep] = magnitude[keep]
    suppressed[[0, -1], :] = 0
    suppressed[:, [0, -1]] = 0
    # Bước 4: high chọn hạt giống chắc chắn; low chọn các điểm có thể nối tiếp.
    strong = (suppressed >= high) & (suppressed > 0)
    weak = (suppressed >= low) & (suppressed > 0)
    # Bước 5: lan từ biên mạnh qua điểm yếu; điểm yếu cô lập không được giữ.
    queue = deque(zip(*np.nonzero(strong)))
    while queue:
        y, x = queue.popleft()
        for dy, dx in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and weak[ny, nx] and not strong[ny, nx]:
                strong[ny, nx] = True
                queue.append((ny, nx))
    return strong.astype(np.uint8) * 255


NEIGHBORS = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)


def morphology(image: np.ndarray, size: int = 5, dilate: bool = True) -> np.ndarray:
    """Giãn hoặc co ảnh xám/nhị phân bằng cửa sổ vuông size x size.
    1. Đệm mép để giữ kích thước ảnh.
    2. Lấy max trong lân cận nếu dilate=True (giãn vùng sáng).
    3. Lấy min nếu dilate=False (co vùng sáng).
    Đầu ra giữ kiểu dữ liệu và kích thước đầu vào.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("Morphology size must be positive and odd.")
    h, w = image.shape
    padded = np.pad(image, size // 2, mode="edge")
    result = image.copy()
    operation = np.maximum if dilate else np.minimum
    for y in range(size):
        for x in range(size):
            operation(result, padded[y:y + h, x:x + w], out=result)
    return result


def close_edges(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Nối khe đứt nhỏ trên biên trắng bằng phép đóng (closing).
    Giãn trước để nối các điểm gần nhau, rồi co để giảm độ dày tăng thêm.
    """
    return morphology(morphology(image, size), size, dilate=False)


def rectangular_extreme(image: np.ndarray, size: int, maximum: bool) -> np.ndarray:
    """Tính max/min trong cửa sổ vuông bằng hai lượt lọc một chiều.
    1. Quét theo hàng/cột thứ nhất, lấy cực trị trong size điểm.
    2. Quét theo chiều còn lại để thu cực trị của cả cửa sổ vuông.
    Kết quả giống morphology nhưng chi phí O(size * số pixel), phù hợp cửa sổ lớn.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("Neighborhood size must be positive and odd.")
    result = image.copy()
    operation = np.maximum if maximum else np.minimum
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (size // 2, size // 2)
        padded = np.pad(result, padding, mode="edge")
        result = result.copy()
        for offset in range(size):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(offset, offset + image.shape[axis])
            operation(result, padded[tuple(slices)], out=result)
    return result


def clean_document(image: np.ndarray, background_size: int = 41,
                   denoise_sigma: float = 0.8) -> np.ndarray:
    """Khử nhiễu nhẹ và bù nền sáng không đều, trả ảnh xám uint8.
    1. Chuyển xám, làm mượt Gaussian 3 x 3 để giảm hạt nhỏ.
    2. Closing xám lấp nét chữ tối để ước lượng nền giấy (background_size).
    3. Làm mượt nền ước lượng để tránh biến thiên đột ngột.
    4. Chuẩn hóa: ảnh sạch = 255 * ảnh đã làm mượt / nền ước lượng.
    Vùng giấy trở nên gần trắng; không xóa thành phần nhỏ vì có thể là dấu chữ.
    Bóng cứng, hình tối lớn và chữ cực mờ vẫn có thể xử lý chưa tốt.
    """
    gray = grayscale(image)
    # Bước 1: lọc nhẹ trước, tránh làm đậm nhiễu khi chia nền/tăng tương phản.
    smooth = gaussian_blur(gray, (3, 3), sigma=denoise_sigma)
    # Bước 2: closing lấp nét tối nhỏ trong BẢN ƯỚC LƯỢNG nền, không xóa mực gốc.
    background = rectangular_extreme(smooth, background_size, maximum=True)
    background = rectangular_extreme(background, background_size, maximum=False)
    # Bước 3: làm nền ước lượng thay đổi chậm để tránh vệt biên nhân tạo.
    background = gaussian_blur(background, (background_size, background_size))
    # Bước 4: nền tối/sáng đều được đưa gần 255; chặn mẫu số để không chia 0.
    return uint8(smooth * 255.0 / np.maximum(background, 1.0))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Tìm bao lồi bằng Andrew monotone chain, trả các đỉnh theo chu kỳ.
    1. Loại điểm trùng, sắp theo x rồi y.
    2. Dựng hai nửa bao, bỏ điểm tạo góc rẽ không phù hợp.
    3. Ghép hai nửa và bỏ điểm đầu/cuối bị lặp.
    Bao lồi bao quanh toàn bộ điểm nhưng không giữ các chỗ lõm.
    """
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        return np.asarray(points, dtype=float)

    def cross(o, a, b):
        """Tính tích có hướng 2D của OA và OB để xác định chiều rẽ.
        Giá trị 0 nghĩa là ba điểm thẳng hàng; dấu cho biết hướng quay đại số.
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chains = []
    for sequence in (points, reversed(points)):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    return np.asarray(chains[0] + chains[1], dtype=float)


def polygon_area(points: np.ndarray) -> float:
    """Tính diện tích đa giác bằng công thức dây giày (shoelace).
    Ghép mỗi đỉnh với đỉnh kế tiếp theo chu kỳ, lấy nửa trị tuyệt đối hiệu hai tổng.
    Các điểm phải được sắp theo thứ tự đi quanh biên.
    """
    x, y = np.asarray(points).reshape(-1, 2).T
    return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2)


def simplify_polyline(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Rút gọn đường gấp khúc mở bằng Ramer-Douglas-Peucker (RDP).
    1. Giữ hai đầu đoạn; tìm điểm xa đường thẳng nối hai đầu nhất.
    2. Nếu khoảng cách > epsilon, giữ điểm đó và chia thành hai đoạn con.
    3. Lặp bằng stack, cuối cùng trả các điểm giữ lại theo thứ tự ban đầu.
    epsilon càng lớn thì đường được rút gọn càng mạnh.
    """
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        delta = points[end] - points[start]
        offsets = points[start + 1:end] - points[start]
        distance = np.abs(delta[0] * offsets[:, 1] - delta[1] * offsets[:, 0])
        distance /= max(np.linalg.norm(delta), 1e-12)
        index = int(np.argmax(distance))
        if distance[index] > epsilon:
            split = start + 1 + index
            keep.add(split)
            stack.extend([(start, split), (split, end)])
    return points[sorted(keep)]


def document_contour(edges: np.ndarray) -> Optional[np.ndarray]:
    """Tìm ứng viên tờ giấy từ ảnh biên, trả (4, 1, 2) hoặc None.
    1. Duyệt các thành phần biên liên thông 8 hướng bằng flood-fill.
    2. Lấy bao lồi mỗi thành phần để có đường bao ngoài có thứ tự.
    3. Chia đường bao kín thành hai đường mở rồi rút gọn RDP.
    4. Giữ tứ giác chiếm ít nhất 2% diện tích ảnh; chọn tứ giác lớn nhất.
    Đây là bộ dò chuyên cho giấy lồi; không lưu lỗ/biên lõm như RETR_LIST.
    Biên giấy dính với nền có thể cho kết quả sai, khi đó cần chỉnh góc thủ công.
    """
    # Bước 1: mặt nạ đánh dấu điểm chưa thăm; mỗi điểm chỉ thuộc một thành phần.
    remaining = edges != 0
    h, w = remaining.shape
    candidates = []
    for y, x in zip(*np.nonzero(remaining)):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        queue, points = [(int(y), int(x))], []
        while queue:
            cy, cx = queue.pop()
            points.append((cx, cy))
            for dy, dx in NEIGHBORS:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    queue.append((ny, nx))
        if len(points) < 4:
            continue
        # Bước 2: đưa tập điểm chưa có thứ tự thành đường bao lồi có chu kỳ.
        hull = convex_hull(np.asarray(points))
        if len(hull) < 4:
            continue
        perimeter = np.linalg.norm(hull - np.roll(hull, 1, axis=0), axis=1).sum()
        # Bước 3: RDP xử lý đường mở, nên tách đường kín tại đỉnh xa điểm đầu nhất.
        split = int(np.argmax(np.linalg.norm(hull - hull[0], axis=1)))
        first = simplify_polyline(hull[:split + 1], 0.02 * perimeter)
        second = simplify_polyline(np.vstack([hull[split:], hull[0]]), 0.02 * perimeter)
        polygon = np.vstack([first[:-1], second[:-1]])
        # Bước 4: bỏ hình nhỏ/nhiều đỉnh; tứ giác lớn nhất là giả thuyết về tờ giấy.
        if len(polygon) == 4 and polygon_area(polygon) >= 0.02 * h * w:
            candidates.append(polygon)
    if not candidates:
        return None
    return max(candidates, key=polygon_area).astype(np.int32).reshape(4, 1, 2)


def homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Tính ma trận phối cảnh H (3 x 3) từ bốn cặp điểm tương ứng.
    source/destination có dạng (4, 2), tọa độ (x, y).
    1. Cố định h33 = 1, còn tám ẩn số.
    2. Mỗi cặp điểm sinh hai phương trình từ u = (h11*x+h12*y+h13)/(h31*x+h32*y+1).
    3. Giải hệ 8 x 8 bằng NumPy, thêm h33 rồi ghép ma trận.
    Báo lỗi khi hệ suy biến, ví dụ các góc không tạo tứ giác hợp lệ.
    """
    # Bước 1: gom hệ A*h=b; mỗi cặp góc đóng góp phương trình cho u và v.
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y],
                     [0, 0, 0, x, y, 1, -v*x, -v*y]])
        values.extend([u, v])
    try:
        # Bước 2: giải tám ẩn độc lập sau khi cố định h33 = 1.
        coefficients = np.linalg.solve(np.asarray(rows, float), np.asarray(values, float))
    except np.linalg.LinAlgError as error:
        raise ValueError("Document corners must form a non-degenerate quadrilateral.") from error
    return np.append(coefficients, 1).reshape(3, 3)


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lấy mẫu ảnh tại tọa độ thực (x, y) bằng nội suy song tuyến tính.
    1. Chặn tọa độ vào ảnh và tìm bốn pixel bao quanh.
    2. Tính phần lẻ dx, dy biểu diễn khoảng cách đến các pixel.
    3. Trộn bốn màu bằng trọng số (1-dx)(1-dy), dx(1-dy), (1-dx)dy, dx*dy.
    Giúp ảnh biến đổi mượt hơn lấy pixel gần nhất; không tạo thêm chi tiết thật.
    """
    h, w = image.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    if image.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    return uint8((1-dx)*(1-dy)*image[y0, x0] + dx*(1-dy)*image[y0, x1]
                 + (1-dx)*dy*image[y1, x0] + dx*dy*image[y1, x1])


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Đổi kích thước ảnh bằng nội suy song tuyến tính tự viết.
    size = (rộng, cao); ánh xạ tâm pixel ảnh đích về ảnh nguồn rồi lấy mẫu.
    Bản này có thể dùng để học; GUI/CLI được phép dùng cv2.resize.
    """
    width, height = size
    if min(width, height) < 1:
        raise ValueError("Output dimensions must be positive.")
    y, x = np.indices((height, width), dtype=float)
    return bilinear_sample(image, (x + .5)*image.shape[1]/width - .5,
                           (y + .5)*image.shape[0]/height - .5)


def warp_perspective(image: np.ndarray, matrix: np.ndarray,
                     size: Tuple[int, int]) -> np.ndarray:
    """Nắn ảnh bằng ma trận phối cảnh H, đầu ra kích thước (rộng, cao).
    1. Đảo H để ánh xạ từ mỗi pixel đích về vị trí nguồn (tránh lỗ trống).
    2. Xử lý từng dải 128 hàng để hạn chế bộ nhớ tạm.
    3. Chia tọa độ thuần nhất cho thành phần thứ ba.
    4. Nội suy song tuyến tính; vị trí ngoài ảnh hoặc vô hiệu được tô đen.
    """
    width, height = size
    if min(width, height) < 1:
        raise ValueError("Output dimensions must be positive.")
    # Bước 1: H đưa nguồn sang đích; H^-1 giúp tìm nguồn cho từng pixel đích.
    inverse = np.linalg.inv(matrix)
    output = np.zeros((height, width) + image.shape[2:], np.uint8)
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        y, x = np.indices((stop-start, width), dtype=float)
        # Bước 2: tọa độ thuần nhất [x, y, 1], sau nhân phải chia cho thành phần 3.
        mapped = inverse @ np.stack([x.ravel(), (y+start).ravel(), np.ones(x.size)])
        valid = np.abs(mapped[2]) > 1e-12
        denominator = np.where(valid, mapped[2], 1)
        sx, sy = mapped[0]/denominator, mapped[1]/denominator
        valid &= (sx >= -1e-6) & (sx <= image.shape[1]-1+1e-6)
        valid &= (sy >= -1e-6) & (sy <= image.shape[0]-1+1e-6)
        # Bước 3: nội suy bốn pixel nguồn; chỉ nhận vị trí hợp lệ trong ảnh.
        sampled = bilinear_sample(image, sx, sy)
        sampled[~valid] = 0
        output[start:stop] = sampled.reshape(output[start:stop].shape)
    return output


def adaptive_threshold(image: np.ndarray, block_size: int = 11,
                       offset: float = 2) -> np.ndarray:
    """Nhị phân hóa ảnh theo ngưỡng cục bộ Gaussian.
    1. Chuyển xám; tính trung bình có trọng số trên cửa sổ block_size.
    2. Ngưỡng tại mỗi pixel = trung bình cục bộ - offset.
    3. Pixel lớn hơn ngưỡng thành trắng 255, còn lại thành đen 0.
    Tăng offset thường giảm chấm đen, nhưng có thể làm mất nét mờ.
    """
    gray = grayscale(image)
    mean = gaussian_blur(gray, (block_size, block_size))
    return (gray.astype(float) > mean - offset).astype(np.uint8) * 255


def clahe(gray: np.ndarray, clip_limit: float = 2.0,
          grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Tăng tương phản cục bộ có giới hạn, đầu vào/ra là ảnh xám uint8.
    1. Chia ảnh thành grid ô, đệm mép để các ô có cùng kích thước.
    2. Lập histogram 256 mức xám cho từng ô.
    3. Cắt số đếm vượt clip_limit và phân phối lại phần dư để hạn chế tăng nhiễu.
    4. Cộng dồn histogram (CDF) thành bảng đổi mức xám (LUT).
    5. Nội suy giữa bốn LUT lân cận để tránh đường ranh ô.
    CLAHE không phải khử nhiễu: tăng tương phản có thể làm rõ cả hạt nền.
    """
    h, w = gray.shape
    # Bước 1: chia ô, giới hạn số ô để ảnh nhỏ không tạo ô rỗng.
    nx, ny = min(grid[0], w), min(grid[1], h)
    tw, th = int(np.ceil(w/nx)), int(np.ceil(h/ny))
    padded = np.pad(gray, ((0, th*ny-h), (0, tw*nx-w)), mode="edge")
    luts = np.empty((ny, nx, 256), float)
    for ty, tx in np.ndindex(ny, nx):
        tile = padded[ty*th:(ty+1)*th, tx*tw:(tx+1)*tw]
        # Bước 2: đếm pixel ở từng mức sáng 0..255 của ô đang xét.
        hist = np.bincount(tile.ravel(), minlength=256)
        limit = max(1, int(clip_limit*tile.size/256))
        # Bước 3: cắt đỉnh histogram, phân phối lại phần dư để giảm khuếch đại nhiễu.
        clipped = np.minimum(hist, limit)
        excess = int((hist-clipped).sum())
        clipped += excess//256
        remainder = excess % 256
        if remainder:
            clipped[np.linspace(0, 255, remainder, dtype=int)] += 1
        # Bước 4: CDF chuẩn hóa thành LUT: mức sáng cũ -> mức sáng mới.
        luts[ty, tx] = np.cumsum(clipped)*255/tile.size
    # Bước 5: trộn bốn LUT quanh pixel theo khoảng cách đến tâm ô, tránh viền ô.
    px = np.clip((np.arange(w)+.5)/tw-.5, 0, nx-1)
    py = np.clip((np.arange(h)+.5)/th-.5, 0, ny-1)
    x0, y0 = px.astype(int)[None, :], py.astype(int)[:, None]
    x1, y1 = np.minimum(x0+1, nx-1), np.minimum(y0+1, ny-1)
    dx, dy = px[None, :]-x0, py[:, None]-y0
    return uint8((1-dx)*(1-dy)*luts[y0, x0, gray] + dx*(1-dy)*luts[y0, x1, gray]
                 + (1-dx)*dy*luts[y1, x0, gray] + dx*dy*luts[y1, x1, gray])
