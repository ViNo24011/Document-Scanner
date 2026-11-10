"""Mục đích: kiểm thử các thuật toán bằng dữ liệu có kết quả toán học biết trước.
Bao gồm kiểm tra hình học, biên, pipeline và các lời gọi OpenCV được phép.
Chạy từ thư mục dự án: python -m unittest discover -s tests -v.
"""
import ast
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import image_algorithms as alg
import scanner


class AlgorithmTests(unittest.TestCase):
    def test_no_library_shortcuts(self):
        """Kiểm tra mã nguồn chỉ gọi các hàm OpenCV được phép và backend không nhập thư viện xử lý thay thế.
        """
        allowed = {'imread', 'imwrite', 'resize', 'imshow', 'waitKey',
                   'destroyAllWindows', 'drawContours', 'rectangle', 'putText',
                   'line', 'circle'}
        for path in ROOT.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'cv2':
                        self.assertIn(node.func.attr, allowed, str(path))
        tree = ast.parse((ROOT/'image_algorithms.py').read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name == 'numpy' for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, {'collections', 'typing'})

    def test_gray_gaussian_and_threshold(self):
        """Kiểm tra công thức xám, bảo toàn vùng sáng đều khi lọc và ngưỡng hóa điểm tối.
        """
        colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
        np.testing.assert_array_equal(alg.grayscale(colors), [[29, 150, 76]])
        flat = np.full((35, 35), 137, np.uint8)
        np.testing.assert_allclose(alg.gaussian_blur(flat), 137)
        self.assertTrue(np.all(alg.adaptive_threshold(flat) == 255))
        flat[17, 17] = 30
        self.assertEqual(alg.adaptive_threshold(flat)[17, 17], 0)

    def test_canny_and_closing(self):
        """Kiểm tra ảnh phẳng không sinh biên, bước sáng có biên đúng vị trí và closing nối khe nhỏ.
        """
        image = np.zeros((100, 120), np.uint8)
        self.assertFalse(alg.canny(image).any())
        image[:, 60:] = 255
        edges = alg.canny(alg.gaussian_blur(image))
        self.assertGreater(np.count_nonzero(edges[:, 58:62]), 80)
        self.assertEqual(np.count_nonzero(edges[:, :55]), 0)
        line = np.zeros((25, 25), np.uint8)
        line[12, 3:22] = 255
        line[12, 12] = 0
        self.assertEqual(alg.close_edges(line, 3)[12, 12], 255)

    def test_homography_and_warp(self):
        """Kiểm tra ánh xạ điểm, warp đồng nhất, resize đồng kích thước và phép tịnh tiến.
        """
        corners = np.array([[0, 0], [29, 0], [29, 19], [0, 19]], float)
        dest = corners + [7, 9]
        matrix = alg.homography(corners, dest)
        mapped = matrix @ np.c_[corners, np.ones(4)].T
        np.testing.assert_allclose((mapped[:2]/mapped[2]).T, dest, atol=1e-8)
        image = np.arange(600).reshape(20, 30).astype(np.uint8)
        np.testing.assert_array_equal(alg.warp_perspective(image, np.eye(3), (30, 20)), image)
        np.testing.assert_array_equal(alg.resize(image, (30, 20)), image)
        translated = alg.warp_perspective(image, matrix, (40, 35))
        np.testing.assert_array_equal(translated[9:29, 7:37], image)

    def test_corner_permutations_and_invalid_geometry(self):
        """Thử 24 hoán vị góc hình thoi và xác nhận các điểm trùng bị từ chối.
        """
        corners = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], float)
        expected = scanner.order_points(corners)
        for perm in itertools.permutations(corners):
            actual = scanner.order_points(np.array(perm))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(np.unique(actual, axis=0)), 4)
        with self.assertRaises(ValueError):
            scanner.order_points(np.zeros((4, 2)))

    def test_document_detection_and_pipeline(self):
        """Tạo trang giấy nghiêng có góc biết trước, kiểm tra dò góc và pipeline nhị phân.
        """
        page = np.full((160, 120, 3), 230, np.uint8)
        page[30:130:15, 15:100] = 70
        source = np.array([[0, 0], [119, 0], [119, 159], [0, 159]], float)
        expected = np.array([[55, 25], [190, 45], [215, 220], [30, 210]], float)
        matrix = alg.homography(source, expected)
        photo = alg.warp_perspective(page, matrix, (250, 250))
        mask = alg.warp_perspective(np.full((160, 120), 255, np.uint8), matrix, (250, 250))
        photo[mask == 0] = 25
        _, _, edges = scanner.preprocess_image(photo)
        contour = scanner.find_document_contour(edges)
        self.assertIsNotNone(contour)
        actual = scanner.order_points(contour)
        self.assertLess(np.max(np.linalg.norm(actual - expected, axis=1)), 6)
        warped = scanner.four_point_transform(photo, actual)
        self.assertGreater(warped.shape[0], 160)
        self.assertTrue(set(np.unique(scanner.binarize_image(warped))) <= {0, 255})
        self.assertIsNone(scanner.find_document_contour(np.zeros((80, 80), np.uint8)))

    def test_rotation_sharpen_and_small_clahe(self):
        """Kiểm tra xoay ngược khôi phục ảnh, tăng nét giữ vùng đều và CLAHE xử lý ảnh rất nhỏ.
        """
        image = np.arange(12).reshape(3, 4).astype(np.uint8)
        rotated = scanner.rotate_image(image, 90)
        np.testing.assert_array_equal(scanner.rotate_image(rotated, 270), image)
        flat = np.full((10, 10), 128, np.uint8)
        np.testing.assert_array_equal(scanner.sharpen_image(flat), flat)
        self.assertEqual(alg.clahe(np.array([[128]], np.uint8)).shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
