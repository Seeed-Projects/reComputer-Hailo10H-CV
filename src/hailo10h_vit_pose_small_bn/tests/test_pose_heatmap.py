"""Regression tests for the ViTPose heatmap decode (no device required)."""
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

HEAT_H, HEAT_W, NUM_KPT = 64, 48, 17
INPUT_H, INPUT_W = 256, 192


def heatmap(peaks, shape=(HEAT_H, HEAT_W, NUM_KPT)):
    """Build a heatmap with one peak per channel. peaks: list of (y, x, score)."""
    arr = np.zeros(shape, dtype=np.float32)
    for k, (y, x, score) in enumerate(peaks):
        if shape[-1] == NUM_KPT:
            arr[y, x, k] = score
        else:
            arr[k, y, x] = score
    return arr


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class PoseHeatmapDecodeTest(unittest.TestCase):
    def decode(self, arr, obj=0.3):
        return wd.post_process_hailo({"out": arr}, obj, 0.45, INPUT_H, INPUT_W)

    def test_argmax_peak_maps_to_input_pixels(self):
        peaks = [(0, 0, 0.9)] + [(i, i, 0.8) for i in range(1, NUM_KPT)]
        kpts = self.decode(heatmap(peaks))

        self.assertIsNotNone(kpts)
        self.assertEqual(kpts.shape, (NUM_KPT, 3))
        self.assertAlmostEqual(float(kpts[0, 0]), 0.0, places=4)          # x
        self.assertAlmostEqual(float(kpts[0, 1]), 0.0, places=4)          # y
        self.assertAlmostEqual(float(kpts[0, 2]), 0.9, places=5)          # score
        # joint 1 peaks at heatmap (y=1, x=1) -> input pixels
        self.assertAlmostEqual(float(kpts[1, 0]), 1 * INPUT_W / HEAT_W, places=3)
        self.assertAlmostEqual(float(kpts[1, 1]), 1 * INPUT_H / HEAT_H, places=3)

    def test_chw_layout_is_accepted(self):
        peaks = [(5, 7, 0.9)] + [(0, 0, 0.1)] * (NUM_KPT - 1)
        kpts_hwc = self.decode(heatmap(peaks))
        kpts_chw = self.decode(heatmap(peaks, shape=(NUM_KPT, HEAT_H, HEAT_W)))
        np.testing.assert_allclose(kpts_hwc, kpts_chw, atol=1e-5)

    def test_batch_dimension_is_dropped(self):
        peaks = [(9, 3, 0.7)] + [(0, 0, 0.1)] * (NUM_KPT - 1)
        base = heatmap(peaks)
        kpts = self.decode(base[None, ...])
        self.assertAlmostEqual(float(kpts[0, 0]), 3 * INPUT_W / HEAT_W, places=3)
        self.assertAlmostEqual(float(kpts[0, 1]), 9 * INPUT_H / HEAT_H, places=3)

    def test_invalid_input_returns_none(self):
        self.assertIsNone(wd.post_process_hailo(None, 0.3, 0.45, INPUT_H, INPUT_W))
        self.assertIsNone(self.decode(np.zeros((10, 10, 3), dtype=np.float32)))

    def test_unletterbox_maps_back_to_frame(self):
        kpts = np.zeros((NUM_KPT, 3), dtype=np.float32)
        kpts[:, 0] = 100.0
        kpts[:, 1] = 50.0
        out = wd.unletterbox_keypoints(kpts, (0.5, 10.0, 20.0))
        np.testing.assert_allclose(out[:, 0], 180.0, atol=1e-3)
        np.testing.assert_allclose(out[:, 1], 60.0, atol=1e-3)

    def test_constants(self):
        self.assertEqual(wd.IMG_SIZE, (192, 256))
        self.assertEqual(len(wd.KEYPOINT_NAMES), NUM_KPT)
        self.assertEqual(len(wd.SKELETON), 19)


if __name__ == "__main__":
    unittest.main()
