"""Regression tests for the depth post-processing (no device required).

Covers the output-layout normalization in post_process_hailo and the
letterbox-aware overlay path used by the video and REST pipelines.
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

WH = 224


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class DepthPostprocessTest(unittest.TestCase):
    def setUp(self):
        wd._DEPTH_LOGGED = True  # silence the one-shot layout log

    # ------------------------------------------------------------- layouts
    def test_nhwc(self):
        depth = np.arange(WH * WH, dtype=np.float32).reshape(1, WH, WH, 1)
        out = wd.post_process_hailo(depth, 0, 0, WH, WH)
        np.testing.assert_array_equal(out, depth[0, ..., 0])

    def test_nchw(self):
        depth = np.arange(WH * WH, dtype=np.float32).reshape(1, 1, WH, WH)
        out = wd.post_process_hailo(depth, 0, 0, WH, WH)
        np.testing.assert_array_equal(out, depth[0, 0])

    def test_hw(self):
        depth = np.ones((WH, WH), np.float32)
        out = wd.post_process_hailo(depth, 0, 0, WH, WH)
        np.testing.assert_array_equal(out, depth)

    def test_hw1(self):
        depth = np.full((WH, WH, 1), 2.0, np.float32)
        out = wd.post_process_hailo(depth, 0, 0, WH, WH)
        np.testing.assert_array_equal(out, depth[..., 0])

    def test_dict_payload(self):
        depth = np.full((1, WH, WH, 1), 3.0, np.float32)
        out = wd.post_process_hailo({"depth/output": depth}, 0, 0, WH, WH)
        self.assertEqual(out.shape, (WH, WH))

    def test_nan_and_inf_are_zeroed(self):
        depth = np.zeros((WH, WH), np.float32)
        depth[0, 0] = np.nan
        depth[0, 1] = np.inf
        depth[0, 2] = -np.inf
        out = wd.post_process_hailo(depth, 0, 0, WH, WH)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[0, 0]), 0.0)

    def test_unsupported_layout_returns_none(self):
        self.assertIsNone(
            wd.post_process_hailo(np.zeros((5,), np.float32), 0, 0, WH, WH)
        )
        self.assertIsNone(
            wd.post_process_hailo(
                np.zeros((1, 1, 1, WH, WH), np.float32), 0, 0, WH, WH
            )
        )

    # -------------------------------------------------------- unletterbox
    def test_unletterbox_square_frame_is_resized(self):
        depth = np.arange(WH * WH, dtype=np.float32).reshape(WH, WH)
        out = wd.unletterbox_depth(depth, (224 / 640, 0.0, 0.0), 640, 640)
        self.assertEqual(out.shape, (640, 640))

    def test_unletterbox_crops_padding(self):
        # 640x480 frame letterboxed into 224x224: ratio 0.35, dh = 28
        ratio = 224 / 640
        dh = (224 - 480 * ratio) / 2
        depth = np.zeros((WH, WH), np.float32)
        top = int(round(dh - 0.1))
        bottom = WH - int(round(dh + 0.1))
        depth[top:bottom, :] = 1.0
        out = wd.unletterbox_depth(depth, (ratio, 0.0, dh), 480, 640)
        self.assertEqual(out.shape, (480, 640))
        self.assertGreater(float(out.mean()), 0.99)

    # -------------------------------------------------------------- draw
    def test_draw_depth_blends_into_frame(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        depth = np.tile(np.linspace(0, 1, 640, dtype=np.float32), (480, 1))
        wd.draw_depth(frame, depth, 0.6)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(int(frame.max()), 0)

    def test_draw_depth_accepts_none(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        wd.draw_depth(frame, None, 0.6)
        self.assertEqual(int(frame.max()), 0)


if __name__ == "__main__":
    unittest.main()