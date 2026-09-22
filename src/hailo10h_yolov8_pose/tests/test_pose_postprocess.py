"""Regression tests for the YOLOv8 Pose NMS decoding (no device required).

The row layout asserted here (5 box/score columns + 17 x [y, x, score]) is the
Hailo Model Zoo pose post-process contract; the first inference on hardware
prints a sample row so it can be confirmed on the real HEF.
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


def build_row(score=0.9, box=(0.1, 0.2, 0.5, 0.6)):
    """One 56-wide NMS row: box, score, then 17 joints as (y, x, score)."""
    row = [box[0], box[1], box[2], box[3], score]
    joints = []
    for idx in range(17):
        joints.extend([0.30 + 0.01 * idx, 0.40 + 0.01 * idx, 0.80])
    return np.asarray(row + joints, dtype=np.float32)


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class PosePostprocessTest(unittest.TestCase):
    INPUT = 640

    def test_compact_nms_buffer_decodes_box_and_keypoints(self):
        row = build_row()
        buf = np.concatenate(([1.0], row)).astype(np.float32)

        boxes, classes, scores, keypoints = wd.post_process_hailo(
            {"yolov8s_pose/yolov8s_pose_nms": buf},
            0.25, 0.45, self.INPUT, self.INPUT)

        self.assertIsNotNone(boxes)
        np.testing.assert_allclose(boxes[0], [128, 64, 384, 320], atol=1e-4)
        self.assertEqual(int(classes[0]), 0)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        self.assertEqual(keypoints.shape, (1, 17, 3))
        # Row stores joints as (y, x, score); decoding returns (x, y, score).
        self.assertAlmostEqual(float(keypoints[0, 0, 0]), 0.40 * self.INPUT, places=3)
        self.assertAlmostEqual(float(keypoints[0, 0, 1]), 0.30 * self.INPUT, places=3)
        self.assertAlmostEqual(float(keypoints[0, 16 - 1, 0]), 0.55 * self.INPUT, places=3)
        self.assertTrue(np.all(np.isfinite(keypoints[0])))

    def test_dense_layout_is_transposed_into_rows(self):
        row = build_row()
        dense = np.zeros((1, 1, 100, 56), dtype=np.float32)
        dense[0, 0, 0] = row

        boxes, classes, scores, keypoints = wd.post_process_hailo(
            {"out": dense}, 0.25, 0.45, self.INPUT, self.INPUT)

        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [128, 64, 384, 320], atol=1e-4)
        self.assertEqual(keypoints.shape, (1, 17, 3))

    def test_score_threshold_filters_rows(self):
        low = build_row(score=0.10)
        buf = np.concatenate(([1.0], low)).astype(np.float32)

        boxes, classes, scores, keypoints = wd.post_process_hailo(
            {"out": buf}, 0.25, 0.45, self.INPUT, self.INPUT)

        self.assertIsNone(boxes)
        self.assertIsNone(classes)
        self.assertIsNone(scores)
        self.assertIsNone(keypoints)

    def test_detection_only_rows_yield_nan_keypoints(self):
        row = np.asarray([0.1, 0.2, 0.5, 0.6, 0.9], dtype=np.float32)
        buf = np.concatenate(([1.0], row)).astype(np.float32)

        boxes, _, _, keypoints = wd.post_process_hailo(
            {"out": buf}, 0.25, 0.45, self.INPUT, self.INPUT)

        self.assertIsNotNone(boxes)
        self.assertEqual(keypoints.shape, (1, 17, 3))
        self.assertTrue(np.all(np.isnan(keypoints[0])))

    def test_unletterbox_keypoints_maps_back_to_frame(self):
        keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        keypoints[0, :, 0] = 100.0
        keypoints[0, :, 1] = 50.0
        # ratio 0.5, padding (10, 20): x=(100-10)/0.5=180, y=(50-20)/0.5=60
        out = wd.unletterbox_keypoints(keypoints, (0.5, 10.0, 20.0))
        np.testing.assert_allclose(out[0, :, 0], 180.0, atol=1e-3)
        np.testing.assert_allclose(out[0, :, 1], 60.0, atol=1e-3)

    def test_pose_constants(self):
        self.assertEqual(wd.NUM_KEYPOINTS, 17)
        self.assertEqual(wd.POSE_ROW_WIDTH, 56)
        self.assertEqual(wd.DEFAULT_CLASSES, ("person",))
        self.assertEqual(len(wd.COCO_SKELETON), 18)


if __name__ == "__main__":
    unittest.main()