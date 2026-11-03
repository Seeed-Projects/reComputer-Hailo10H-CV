"""Regression tests for SCDepthV3 post-processing (no device required).

Covers the sigmoid inverse-depth decode, non-square output layouts, the
letterbox-aware restore, and the statistics / colorization helpers.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import web_detection as wd
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host toolchain
    wd = None
    IMPORT_ERROR = exc

H, W = 256, 320  # model input (height, width)


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class SCDepthV3PostprocessTest(unittest.TestCase):
    def setUp(self):
        wd._DEPTH_OUTPUT_LOGGED = True  # silence the one-shot layout log

    def decode(self, payload, input_h=H, input_w=W):
        return wd.post_process_hailo(payload, 0, 0, input_h, input_w)

    # ------------------------------------------------------------- decode
    def test_sigmoid_inverse_formula(self):
        raw = np.zeros((1, H, W, 1), np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        expected = 1.0 / (0.5 * 10.0 + 0.009)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_nhwc_layout(self):
        raw = np.full((1, H, W, 1), 2.0, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        expected = 1.0 / (wd._sigmoid(np.float32(2.0)) * 10.0 + 0.009)
        np.testing.assert_allclose(out[0, 0], expected, rtol=1e-5)

    def test_nchw_layout(self):
        raw = np.full((1, 1, H, W), -1.0, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        expected = 1.0 / (wd._sigmoid(np.float32(-1.0)) * 10.0 + 0.009)
        np.testing.assert_allclose(out[0, 0], expected, rtol=1e-5)

    def test_hw_layout(self):
        raw = np.full((H, W), 0.0, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))

    def test_transposed_layout_is_fixed(self):
        # The HEF may expose the non-square output as (320, 256).
        raw = np.arange(W * H, dtype=np.float32).reshape(W, H)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))

    def test_smaller_layout_is_resized(self):
        raw = np.zeros((H // 2, W // 2, 1), np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))

    def test_nan_is_zeroed(self):
        raw = np.zeros((H, W), np.float32)
        raw[0, 0] = np.nan
        out = self.decode(raw)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[0, 0]), 0.0)

    def test_unsupported_shape_raises(self):
        with self.assertRaises(ValueError):
            self.decode(np.zeros((5,), np.float32))

    # --------------------------------------------------------- restore map
    def test_depth_to_original_crops_letterbox(self):
        depth = np.ones((H, W), np.float32)
        # 640x480 letterboxed into 320x256: ratio 0.5, dh = 8, dw = 0
        out = wd.depth_to_original(depth, (0.5, 0.0, 8.0), (480, 640))
        self.assertEqual(out.shape, (480, 640))
        self.assertAlmostEqual(float(out.mean()), 1.0, places=5)

    def test_colorize_depth(self):
        depth = np.linspace(1.0, 2.0, H * W, dtype=np.float32).reshape(H, W)
        colored = wd.colorize_depth(depth)
        self.assertEqual(colored.shape, (H, W, 3))
        self.assertEqual(colored.dtype, np.uint8)

    def test_colorize_empty_depth(self):
        colored = wd.colorize_depth(np.zeros((H, W), np.float32))
        self.assertEqual(colored.shape, (H, W, 3))
        self.assertEqual(int(colored.max()), 0)

    # ---------------------------------------------------------- statistics
    def test_depth_statistics(self):
        depth = np.linspace(1.0, 2.0, H * W, dtype=np.float32).reshape(H, W)
        stats = wd.depth_statistics(depth)
        self.assertEqual(stats["valid_pixels"], H * W)
        self.assertAlmostEqual(stats["min_depth"], 1.0, places=5)
        self.assertAlmostEqual(stats["max_depth"], 2.0, places=5)
        self.assertAlmostEqual(stats["median_depth"], 1.5, places=3)

    def test_depth_statistics_empty(self):
        stats = wd.depth_statistics(np.zeros((H, W), np.float32))
        self.assertEqual(stats["valid_pixels"], 0)
        self.assertIsNone(stats["mean_depth"])

    def test_draw_depth_map(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        depth = np.linspace(1.0, 2.0, H * W, dtype=np.float32).reshape(H, W)
        wd.draw_depth_map(frame, depth, (0.5, 0.0, 8.0))
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(int(frame.max()), 0)


if __name__ == "__main__":
    unittest.main()