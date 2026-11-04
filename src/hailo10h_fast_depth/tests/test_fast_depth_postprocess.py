"""Regression tests for FastDepth post-processing (no device required).

The Model Zoo `fast_depth_postprocessing` returns the network output unchanged,
so these tests assert the raw depth values pass through the layout handling
without any sigmoid or inverse-depth transform.
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

H = W = 224  # model input (height, width)


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class FastDepthPostprocessTest(unittest.TestCase):
    def setUp(self):
        wd._DEPTH_OUTPUT_LOGGED = True  # silence the one-shot layout log

    def decode(self, payload, input_h=H, input_w=W):
        return wd.post_process_hailo(payload, 0, 0, input_h, input_w)

    # ------------------------------------------------------------- decode
    def test_raw_depth_passthrough(self):
        raw = np.linspace(0.5, 8.0, H * W, dtype=np.float32).reshape(1, H, W, 1)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        np.testing.assert_allclose(out, raw[0, ..., 0], rtol=1e-6)

    def test_nchw_layout(self):
        raw = np.full((1, 1, H, W), 3.5, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        np.testing.assert_allclose(out, 3.5, rtol=1e-6)

    def test_hw_layout(self):
        raw = np.full((H, W), 2.0, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        np.testing.assert_allclose(out, 2.0, rtol=1e-6)

    def test_hw1_layout(self):
        raw = np.full((H, W, 1), 1.5, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        np.testing.assert_allclose(out, 1.5, rtol=1e-6)

    def test_smaller_layout_is_resized(self):
        raw = np.full((H // 2, W // 2, 1), 4.0, np.float32)
        out = self.decode(raw)
        self.assertEqual(out.shape, (H, W))
        self.assertAlmostEqual(float(out.mean()), 4.0, places=4)

    def test_nan_and_inf_are_zeroed(self):
        raw = np.zeros((H, W), np.float32)
        raw[0, 0] = np.nan
        raw[0, 1] = np.inf
        raw[0, 2] = -np.inf
        out = self.decode(raw)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[0, 0]), 0.0)

    def test_unsupported_shape_raises(self):
        with self.assertRaises(ValueError):
            self.decode(np.zeros((5,), np.float32))

    # --------------------------------------------------------- restore map
    def test_depth_to_original_crops_letterbox(self):
        depth = np.ones((H, W), np.float32)
        # 640x480 letterboxed into 224x224: ratio 0.35, dh = 28, dw = 0
        out = wd.depth_to_original(depth, (0.35, 0.0, 28.0), (480, 640))
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
        wd.draw_depth_map(frame, depth, (0.35, 0.0, 28.0))
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(int(frame.max()), 0)


if __name__ == "__main__":
    unittest.main()