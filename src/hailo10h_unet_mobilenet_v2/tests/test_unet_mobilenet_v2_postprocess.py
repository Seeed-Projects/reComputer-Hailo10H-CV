"""Regression tests for U-Net MobileNet v2 post-processing (no device required).

Covers the 3-channel score-map argmax, layout handling, class-mask restore and
the palette overlay used by the live preview and REST endpoint.
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

H = W = 256  # model input (height, width)


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class UNetPostprocessTest(unittest.TestCase):
    def setUp(self):
        wd._SEG_OUTPUT_LOGGED = True  # silence the one-shot layout log

    def decode(self, payload, input_h=H, input_w=W):
        return wd.post_process_hailo(payload, 0, 0, input_h, input_w)

    # ------------------------------------------------------------- argmax
    def test_nhwc_argmax(self):
        scores = np.zeros((1, H, W, 3), np.float32)
        scores[..., 2] = 0.1
        scores[:, :, : W // 2, 0] = 10.0  # left half -> pet
        scores[:, :, W // 2 :, 2] = 10.0  # right half -> background
        mask = self.decode(scores)
        self.assertEqual(mask.shape, (H, W))
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[0, W - 1]), 2)

    def test_nchw_argmax(self):
        scores = np.zeros((1, 3, H, W), np.float32)
        scores[:, 1, :, :] = 10.0  # all border
        mask = self.decode(scores)
        self.assertEqual(mask.shape, (H, W))
        self.assertEqual(int(mask[0, 0]), 1)

    def test_border_class_wins(self):
        scores = np.zeros((H, W, 3), np.float32)
        scores[:, :, 1] = 5.0
        mask = self.decode(scores)
        self.assertTrue((mask == 1).all())

    # ------------------------------------------------------- single channel
    def test_hw1_mask(self):
        raw = np.full((H, W, 1), 1.0, np.float32)
        mask = self.decode(raw)
        self.assertEqual(mask.shape, (H, W))
        self.assertTrue((mask == 1).all())

    def test_hw_mask_rounds_and_clips(self):
        raw = np.zeros((H, W), np.float32)
        raw[0, 0] = 1.6
        raw[0, 1] = 99.0
        mask = self.decode(raw)
        self.assertEqual(int(mask[0, 0]), 2)
        self.assertEqual(int(mask[0, 1]), 2)

    def test_smaller_mask_resized(self):
        raw = np.zeros((H // 2, W // 2, 3), np.float32)
        raw[..., 0] = 10.0
        mask = self.decode(raw)
        self.assertEqual(mask.shape, (H, W))
        self.assertTrue((mask == 0).all())

    def test_nan_mask_maps_to_background(self):
        raw = np.zeros((H, W), np.float32)
        raw[0, 0] = np.nan
        mask = self.decode(raw)
        self.assertEqual(int(mask[0, 0]), wd.BACKGROUND_CLASS_ID)

    def test_unsupported_shape_raises(self):
        with self.assertRaises(ValueError):
            self.decode(np.zeros((5,), np.float32))

    # --------------------------------------------------------------- draw
    def test_draw_segmentation_mask_blends_pet(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        mask = np.zeros((H, W), np.uint8)  # all pet
        # 640x480 letterboxed into 256x256: ratio 0.4, dh = 32, dw = 0
        wd.draw_segmentation_mask(frame, mask, (0.4, 0.0, 32.0))
        self.assertGreater(int(frame.max()), 0)

    def test_draw_segmentation_mask_keeps_background(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        mask = np.full((H, W), wd.BACKGROUND_CLASS_ID, np.uint8)
        wd.draw_segmentation_mask(frame, mask, (0.4, 0.0, 32.0))
        self.assertEqual(int(frame.max()), 0)

    def test_draw_segmentation_mask_accepts_none(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        wd.draw_segmentation_mask(frame, None, (1.0, 0.0, 0.0))
        self.assertEqual(int(frame.max()), 0)

    def test_class_defaults(self):
        self.assertEqual(wd.CLASSES, ("pet", "border", "background"))
        self.assertEqual(wd.BACKGROUND_CLASS_ID, 2)


if __name__ == "__main__":
    unittest.main()