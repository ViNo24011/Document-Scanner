#!/usr/bin/env python
"""Mục đích: chạy máy quét tài liệu từ dòng lệnh (CLI).
Đọc tham số -> đọc ảnh -> phát hiện giấy trên ảnh nhỏ -> chỉnh phối cảnh ảnh gốc
-> áp dụng bộ lọc -> lưu ảnh/PDF và hiển thị nếu người dùng yêu cầu.
Thuật toán thuộc scanner.py và image_algorithms.py; OpenCV chỉ hỗ trợ I/O,
resize và hiển thị. Ví dụ: python scan.py -i photo.jpg -o scan.png -m scanned.
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
import scanner
import image_algorithms as alg

def main():
    """Điều phối toàn bộ lệnh quét từ tham số terminal.
    1. Đọc/kiểm tra tham số và ảnh. 2. Tìm góc trên ảnh resize.
    3. Đổi góc về kích thước gốc rồi nắn phối cảnh.
    4. Lọc ảnh theo mode, xoay/tăng nét nếu yêu cầu. 5. Lưu ảnh/PDF và xem trước.
    """
    parser = argparse.ArgumentParser(description="Auto Perspective Correction Document Scanner CLI")
    parser.add_argument("-i", "--input", required=True, help="Path to input document image")
    parser.add_argument("-o", "--output", help="Path to save the output processed image (e.g., output.jpg)")
    parser.add_argument("--pdf", help="Path to save the output as a PDF file (e.g., output.pdf)")
    parser.add_argument("-m", "--mode", choices=["color", "gray", "scanned"], default="color",
                        help="Processing mode: 'color' (original colors), 'gray' (grayscale), "
                             "or 'scanned' (binary black-and-white adaptive threshold)")
    parser.add_argument("-s", "--sharpen", action="store_true", help="Apply sharpening filter to output")
    parser.add_argument("--contrast", type=float, default=0.0,
                        help="Local contrast strength from 0 (off) to 1; try 0.5 for faint ink")
    parser.add_argument("-r", "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Rotate output image clockwise by degrees")
    parser.add_argument("--show", action="store_true", help="Display intermediate processing steps using OpenCV windows")
    
    args = parser.parse_args()
    if not np.isfinite(args.contrast) or not 0 <= args.contrast <= 1:
        parser.error("--contrast must be between 0 and 1")

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    # 1. Load the original image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Failed to read image from '{args.input}'.")
        return

    orig = image.copy()
    h_orig, w_orig = image.shape[:2]

    # 2. Resize image for reliable, fast contour detection
    # We resize to a standard height of 800px while maintaining the aspect ratio
    resized_height = 800
    ratio = h_orig / float(resized_height)
    resized_width = int(w_orig / ratio)
    resized = cv2.resize(image, (resized_width, resized_height))

    print(f"Original image dimensions: {w_orig}x{h_orig}")
    print(f"Resized image dimensions: {resized_width}x{resized_height} for edge detection")

    # 3. Preprocess resized image
    # We use Gaussian blur and Canny edge detection
    gray, blurred, edged = scanner.preprocess_image(resized, blur_kernel=(5, 5), canny_low=75, canny_high=200)

    # 4. Find document contour
    doc_contour = scanner.find_document_contour(edged)
    
    # Track original corners for visualization overlay
    corners_detected = True

    if doc_contour is not None:
        print("Document outline successfully detected!")
        # Reshape the 4 approximated corner points to (4, 2)
        pts = doc_contour.reshape(4, 2)
        # Draw outline on resized image for presentation/visualization
        overlay = resized.copy()
        cv2.drawContours(overlay, [doc_contour], -1, (0, 255, 0), 3)
    else:
        print("Warning: Could not automatically detect a 4-sided document contour.")
        print("Falling back to default centered region...")
        corners_detected = False
        # Get default coordinates centered in the image
        pts = scanner.get_default_corners(resized_width, resized_height)
        # Draw default crop area outline (red instead of green to indicate fallback)
        overlay = resized.copy()
        cv2.drawContours(overlay, [pts.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 0, 255), 3)

    # 5. Scale corners back to original image coordinates
    # We run the warp on the original high-resolution image to maintain quality
    pts_scaled = pts * ratio

    # 6. Apply perspective transformation
    warped = scanner.four_point_transform(orig, pts_scaled)

    # 7. Apply Post-processing
    # Mode selection
    if args.mode == "gray":
        warped = alg.grayscale(warped)
        print("Applied Grayscale conversion.")
    if args.mode == "scanned":
        warped = scanner.scan_document(warped, args.contrast)
        print("Cleaned paper background and applied adaptive thresholding.")
    elif args.contrast > 0:
        warped = scanner.enhance_contrast(warped, args.contrast)

    # Sharpening
    if args.sharpen:
        warped = scanner.sharpen_image(warped)
        print("Applied unsharp mask sharpening filter.")

    # Rotation
    if args.rotate > 0:
        warped = scanner.rotate_image(warped, args.rotate)
        print(f"Rotated image by {args.rotate} degrees clockwise.")

    # 8. Save output
    if args.output:
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(args.output, warped)
        print(f"Successfully saved scanned document image to: {args.output}")

    if args.pdf:
        pdf_dir = os.path.dirname(args.pdf)
        if pdf_dir and not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            
        # Convert OpenCV BGR or Grayscale to PIL RGB
        if len(warped.shape) == 3:
            warped_rgb = alg.to_rgb(warped)
        else:
            warped_rgb = alg.to_rgb(warped)
            
        pil_img = Image.fromarray(warped_rgb)
        pil_img.save(args.pdf, "PDF", resolution=100.0)
        print(f"Successfully exported document as PDF to: {args.pdf}")

    # 9. Show outputs if requested
    if args.show:
        print("\nDisplaying preview windows. Press any key on the windows to close them...")
        
        # Display resized input with outline
        cv2.imshow("Detected Boundary (Green=Auto, Red=Default)", overlay)
        # Display edge map
        cv2.imshow("Edge Map (Canny)", edged)
        # Display final warped output
        cv2.imshow("Final Scanned Document", warped)
        
        # Wait for key press to close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

