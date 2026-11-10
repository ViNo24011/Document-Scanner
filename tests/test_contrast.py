"""Mục đích: kiểm tra tăng tương phản giữ định dạng ảnh và làm rõ nét mờ giả lập.
Không đánh giá OCR hoặc độ chính xác nhận dạng chữ viết tay thực tế.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner
import image_algorithms as alg


class ContrastTests(unittest.TestCase):
    def test_faint_strokes_gain_contrast(self):
        """Vẽ nét mờ giả lập rồi đo chênh lệch sáng giữa vùng mực và nền sau tăng tương phản.
        """
        image = np.full((256, 512), 210, np.uint8)
        mask = np.zeros_like(image)
        # One-pixel curves and stems simulate faint, thin ink without a font API.
        xs = np.arange(25, 480)
        ys = (125 + 15 * np.sin(xs / 12)).astype(int)
        mask[ys, xs] = 255
        mask[100:145, 50:480:25] = 255
        image -= (mask.astype(np.float32) * 25 / 255).astype(np.uint8)
        result = scanner.enhance_contrast(image, 0.5)
        ink = mask > 200
        paper = alg.morphology(mask, 7) == 0
        self.assertGreater(result[paper].mean() - result[ink].mean(),
                           image[paper].mean() - image[ink].mean())

    def test_formats_and_disabled_identity(self):
        """Kiểm tra kích thước, dtype, không sửa ảnh nguồn và mức 0 giữ nguyên dữ liệu.
        """
        rng = np.random.default_rng(42)
        for shape in [(48, 64), (48, 64, 3)]:
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            original = image.copy()
            np.testing.assert_array_equal(scanner.enhance_contrast(image, 0), image)
            result = scanner.enhance_contrast(image, 1)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
            np.testing.assert_array_equal(image, original)
            binary = scanner.binarize_image(result)
            self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

    def test_invalid_strength(self):
        """Xác nhận mức tương phản ngoài [0, 1], NaN và vô cực đều bị từ chối.
        """
        for strength in [-0.1, 1.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                scanner.enhance_contrast(np.zeros((8, 8), np.uint8), strength)


if __name__ == '__main__':
    unittest.main()
