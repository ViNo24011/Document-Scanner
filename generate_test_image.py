"""Mục đích: tạo ảnh tài liệu giả lập để thử máy quét mà không cần chụp ảnh.
Vẽ hóa đơn trên giấy trắng, tạo nền tối, rồi biến đổi giấy thành tứ giác nghiêng.
Phép biến đổi dùng thuật toán NumPy; OpenCV dùng để vẽ nội dung mẫu và ghi ảnh.
Chạy file sẽ ghi sample_doc.jpg vào thư mục làm việc hiện tại.
Ảnh tổng hợp giúp kiểm tra hình học, không đại diện cho mọi ảnh điện thoại thực.
"""

import cv2
import numpy as np
import image_algorithms as alg

def generate_sample():
    # 1. Create a clean white document image (600x800 px)
    """Tạo và ghi ảnh hóa đơn giả lập ở góc chụp nghiêng.
    1. Vẽ giấy và nội dung. 2. Tạo nền tối có đường vân.
    3. Chọn bốn góc nguồn/đích. 4. Warp giấy và mặt nạ cùng homography.
    5. Ghép giấy vào nền theo mặt nạ rồi lưu sample_doc.jpg.
    """
    doc_w, doc_h = 600, 800
    doc = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 255
    
    # Draw a black border around the document
    cv2.rectangle(doc, (0, 0), (doc_w - 1, doc_h - 1), (0, 0, 0), 5)
    
    # Draw mock text lines on the document
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(doc, "CONFIDENTIAL INVOICE", (50, 80), font, 1.2, (0, 0, 0), 3)
    cv2.putText(doc, "Date: 2026-07-10", (50, 140), font, 0.7, (50, 50, 50), 2)
    cv2.putText(doc, "Invoice Number: #INV-987213", (50, 180), font, 0.7, (50, 50, 50), 2)
    
    # Draw horizontal divider line
    cv2.line(doc, (50, 220), (doc_w - 50, 220), (100, 100, 100), 2)
    
    # Add items lists
    y_start = 280
    items = [
        ("1. OpenCV Perspective Engine", "$499.00"),
        ("2. NumPy Matrix Solver", "$150.00"),
        ("3. Morphological Fine-tuner", "$120.00"),
        ("4. Draggable Corners GUI", "$230.00"),
    ]
    
    for item, price in items:
        cv2.putText(doc, item, (60, y_start), font, 0.6, (0, 0, 0), 2)
        cv2.putText(doc, price, (450, y_start), font, 0.6, (0, 0, 0), 2)
        y_start += 50
        
    # Draw another line
    cv2.line(doc, (50, y_start + 10), (doc_w - 50, y_start + 10), (100, 100, 100), 2)
    
    # Add total
    cv2.putText(doc, "TOTAL AMOUNT DUE:", (60, y_start + 60), font, 0.7, (0, 0, 0), 3)
    cv2.putText(doc, "$999.00", (450, y_start + 60), font, 0.8, (0, 0, 255), 3)
    
    # Add a watermark/stamp circle
    cv2.circle(doc, (480, 700), 60, (0, 128, 0), 4)
    cv2.putText(doc, "PAID", (445, 712), font, 1.0, (0, 128, 0), 3)

    # 2. Create a dark slate background (1200x1200 px)
    bg_size = 1200
    bg = np.ones((bg_size, bg_size, 3), dtype=np.uint8) * 35 # Slate dark grey
    # Add some random wood-grain/table noise to background
    for i in range(15):
        pt1 = (np.random.randint(0, bg_size), 0)
        pt2 = (np.random.randint(0, bg_size), bg_size)
        cv2.line(bg, pt1, pt2, (45, 45, 45), np.random.randint(1, 4))

    # 3. Define the skewed coordinates on the background
    # This represents taking a photo of the A4 paper placed on a table at an angle
    src_pts = np.float32([
        [0, 0],
        [doc_w - 1, 0],
        [doc_w - 1, doc_h - 1],
        [0, doc_h - 1]
    ])
    
    dst_pts = np.float32([
        [320, 200],   # Top-Left (TL)
        [910, 280],   # Top-Right (TR)
        [1020, 950],  # Bottom-Right (BR)
        [220, 880]    # Bottom-Left (BL)
    ])
    
    # 4. Warp the document into the background
    # Calculate transform matrix (straight -> skewed)
    M = alg.homography(src_pts, dst_pts)
    
    # Create mask of skewed document
    mask = alg.warp_perspective(
        np.full((doc_h, doc_w), 255, dtype=np.uint8), M, (bg_size, bg_size))
    
    # Warp straight document
    warped_doc = alg.warp_perspective(doc, M, (bg_size, bg_size))
    
    # Composite the warped document onto the background
    # Where mask is 0, keep background, otherwise keep warped_doc
    composited = np.where(mask[:, :, np.newaxis] == 255, warped_doc, bg)
    
    # Save the synthetic document
    cv2.imwrite("sample_doc.jpg", composited)
    print("Successfully generated synthetic test image: 'sample_doc.jpg'")

if __name__ == "__main__":
    generate_sample()
