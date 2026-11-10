"""Mục đích: kiểm tra giảm nhiễu, cân bằng nền sáng và giữ nét mảnh/dấu chấm.
Ảnh giả lập có vị trí mực và nền biết trước để đo riêng hai vùng, tránh chỉ
đánh giá nền trắng đẹp nhưng chữ đã bị xóa mất.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_algorithms as alg
import scanner


class CleanScanTests(unittest.TestCase):
    def test_noise_reduction_and_thin_ink(self):
        """Tạo nền có nhiễu và nét mảnh biết trước; kiểm tra giảm hạt mà vẫn giữ mực/dấu chấm.
        """
        rng = np.random.default_rng(17)
        h, w = 240, 320
        y, x = np.indices((h, w))
        paper = 160 + 65 * x / w + 8 * np.sin(y / 50)
        ink = np.zeros((h, w), bool)
        ink[60:160, 80] = True  # single-pixel faint stem
        ink[110, 80:180] = True
        ink[180:182, 200:202] = True  # punctuation, not a removed component
        image = alg.uint8(paper - 45 * ink + rng.normal(0, 4, (h, w)))
        original = image.copy()
        background = alg.morphology(ink.astype(np.uint8), 9) == 0
        for contrast in (0, 0.5, 1):
            before = alg.adaptive_threshold(scanner.enhance_contrast(image, contrast))
            after = scanner.scan_document(image, contrast)
            old_noise = np.mean(before[background] == 0)
            new_noise = np.mean(after[background] == 0)
            self.assertLess(new_noise, old_noise * 0.2)
            self.assertGreater(np.mean(after[ink] == 0), 0.9)
            self.assertEqual(after.dtype, np.uint8)
            self.assertEqual(after.shape, image.shape)
        np.testing.assert_array_equal(image, original)

    def test_illumination_normalization(self):
        """Tạo nền sáng tăng dần rồi kiểm tra bù nền cho kết quả gần trắng và đồng đều.
        """
        x = np.linspace(100, 240, 320)
        image = np.broadcast_to(x, (120, 320)).astype(np.uint8)
        cleaned = alg.clean_document(image)
        self.assertLess(np.std(cleaned[20:-20, 30:-30]), 2)
        self.assertGreater(np.mean(cleaned), 245)

    def test_separable_morphology_matches_definition(self):
        """Đối chiếu max/min tách hai chiều với phép quét toàn cửa sổ để kiểm chứng tối ưu.
        """
        rng = np.random.default_rng(9)
        image = rng.integers(0, 256, (20, 25), dtype=np.uint8)
        for maximum in (True, False):
            np.testing.assert_array_equal(
                alg.rectangular_extreme(image, 5, maximum),
                alg.morphology(image, 5, dilate=maximum))


if __name__ == '__main__':
    unittest.main()
