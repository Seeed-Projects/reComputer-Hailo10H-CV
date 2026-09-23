"""Regression tests for the raw YOLOv8-pose head decoding (no device needed).

The Hailo-10H pose HEFs expose nine raw tensors, so these tests build synthetic
heads for one feature-map scale and check the decoded geometry:
  box  : DFL distances around the cell centre
  kpts : (x, y) grid offsets around the cell centre
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

GRID = 20          # feature-map size of the coarsest scale
INPUT = 640        # network input size -> stride 32
CELL = (5, 5)      # (row, col) of the single detection
STRIDE = INPUT / GRID


def build_heads(score=0.9, dfl_bin=1, kpt_xy=(0.5, 0.25), kpt_score=0.0,
                reg_max=16, num_kpts=17, grid=GRID):
    """One scale worth of raw heads with a single active cell."""
    row, col = CELL[0] % grid, CELL[1] % grid
    bbox = np.zeros((grid, grid, reg_max * 4), dtype=np.float32)
    # all four sides: put the mass on `dfl_bin` -> expectation == dfl_bin
    for side in range(4):
        bbox[row, col, side * reg_max + dfl_bin] = 20.0
    scores = np.zeros((grid, grid, 1), dtype=np.float32)
    scores[row, col, 0] = score
    kpts = np.zeros((grid, grid, num_kpts * 3), dtype=np.float32)
    for j in range(num_kpts):
        kpts[row, col, j * 3 + 0] = kpt_xy[0]
        kpts[row, col, j * 3 + 1] = kpt_xy[1]
        kpts[row, col, j * 3 + 2] = kpt_score
    return {"bbox": bbox, "score": scores, "kpts": kpts}


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class PoseRawDecodeTest(unittest.TestCase):
    def decode(self, heads, thresh=0.25):
        return wd.post_process_hailo(heads, thresh, 0.45, INPUT, INPUT)

    def test_box_and_keypoints_geometry(self):
        heads = build_heads(kpt_score=2.0)   # logits -> sigmoid
        boxes, classes, scores, keypoints = self.decode(heads)

        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), 0)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)

        cx = (CELL[1] + 0.5) * STRIDE
        cy = (CELL[0] + 0.5) * STRIDE
        d = 1 * STRIDE                     # dfl_bin 1 -> distance 1 grid unit
        np.testing.assert_allclose(boxes[0],
                                   [cx - d, cy - d, cx + d, cy + d], atol=1e-3)

        self.assertEqual(keypoints.shape, (1, 17, 3))
        # kpts are (x, y): x from column 0, y from column 1
        self.assertAlmostEqual(float(keypoints[0, 0, 0]), (2 * 0.5 + CELL[1]) * STRIDE, places=3)
        self.assertAlmostEqual(float(keypoints[0, 0, 1]), (2 * 0.25 + CELL[0]) * STRIDE, places=3)
        self.assertAlmostEqual(float(keypoints[0, 0, 2]), 1.0 / (1.0 + np.exp(-2.0)), places=5)
        self.assertTrue(np.all(np.isfinite(keypoints[0])))

    def test_keypoint_scores_already_in_range_are_kept(self):
        heads = build_heads(kpt_score=0.7)   # already a probability
        boxes, _, _, keypoints = self.decode(heads)
        self.assertIsNotNone(boxes)
        self.assertAlmostEqual(float(keypoints[0, 0, 2]), 0.7, places=5)

    def test_score_threshold_drops_everything(self):
        heads = build_heads(score=0.10)
        boxes, classes, scores, keypoints = self.decode(heads, thresh=0.25)
        self.assertIsNone(boxes)
        self.assertIsNone(keypoints)

    def test_sigmoid_scores_are_converted(self):
        heads = build_heads(score=2.0)     # logit outside [0,1]
        boxes, _, scores, _ = self.decode(heads)
        self.assertIsNotNone(boxes)
        self.assertAlmostEqual(float(scores[0]), 1.0 / (1.0 + np.exp(-2.0)), places=5)

    def test_multiple_scales_are_merged(self):
        # HailoRT returns a flat name -> tensor mapping, which is what the
        # decoder groups by feature-map size.
        heads = {}
        for grid in (20, 40, 80):
            for key, arr in build_heads(grid=grid).items():
                heads["%s_%d" % (key, grid)] = arr
        boxes, _, scores, keypoints = self.decode(heads)
        self.assertIsNotNone(boxes)
        # the same cell lights up at three scales; NMS may keep one or more
        self.assertGreaterEqual(len(boxes), 1)
        self.assertEqual(keypoints.shape[1:], (17, 3))
        self.assertTrue(np.all(np.isfinite(boxes)))

    def test_unletterbox_keypoints_maps_back_to_frame(self):
        keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        keypoints[0, :, 0] = 100.0
        keypoints[0, :, 1] = 50.0
        out = wd.unletterbox_keypoints(keypoints, (0.5, 10.0, 20.0))
        np.testing.assert_allclose(out[0, :, 0], 180.0, atol=1e-3)
        np.testing.assert_allclose(out[0, :, 1], 60.0, atol=1e-3)

    def test_pose_constants(self):
        self.assertEqual(wd.NUM_KEYPOINTS, 17)
        self.assertEqual(wd.REG_MAX, 16)
        self.assertEqual(wd.DEFAULT_CLASSES, ("person",))
        self.assertEqual(len(wd.COCO_SKELETON), 18)


if __name__ == "__main__":
    unittest.main()
