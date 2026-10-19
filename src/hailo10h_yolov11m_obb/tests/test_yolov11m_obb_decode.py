"""Regression tests for the YOLOv11-OBB host decode (no device required).

Covers the nine-head layout classification, the DFL + angle decode (with
analytic expectations), probabilistic IoU, Fast-NMS and the guard rails.
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

CLASSES = 15
REG_BINS = 16
INPUT = 1024
STRIDES = (8, 16, 32)
F_SIZES = (128, 64, 32)
CELL = (2, 3)                       # (row, col) in the stride-8 head


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _heads(cell=CELL, bins=(10, 10, 10, 10), angle_p=0.25, cls_id=2, score_p=0.9,
           order="parser", which=0):
    """Nine synthetic heads with one detection in the `which`-th stride head."""
    boxes, scores, angles = [], [], []
    for idx, f in enumerate(F_SIZES):
        box = np.full((f, f, 4, REG_BINS), -10.0, np.float32)
        sc = np.full((f, f, CLASSES), -10.0, np.float32)
        ang = np.full((f, f, 1), 0.0, np.float32)
        if cell is not None and idx == which:
            r, c = cell
            for side, b in enumerate(bins):
                box[r, c, side, :] = -10.0
                box[r, c, side, b] = 10.0
            sc[r, c, cls_id] = _logit(score_p)
            ang[r, c, 0] = _logit(angle_p)
        boxes.append(box.reshape(f, f, 4 * REG_BINS))
        scores.append(sc)
        angles.append(ang)
    if order == "parser":
        # box/class interleaved per stride, angles last (Model Zoo parser order)
        return [boxes[0], scores[0], boxes[1], scores[1], boxes[2], scores[2],
                angles[0], angles[1], angles[2]]
    # reversed order: the classifier must not rely on the tensor order
    return list(reversed([boxes[0], scores[0], boxes[1], scores[1], boxes[2], scores[2],
                          angles[0], angles[1], angles[2]]))


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class Yolov11ObbDecodeTest(unittest.TestCase):
    def decode(self, payload, thresh=0.25, iou=0.7):
        return wd.post_process_hailo(payload, thresh, iou, INPUT, INPUT)

    # ------------------------------------------------------- head plumbing
    def test_classifier_pairs_strides(self):
        heads = wd._classify_obb_heads(_heads())
        self.assertIsNotNone(heads)
        box_heads, score_heads, angle_heads = heads
        self.assertEqual([b.shape[:2] for b in box_heads], [(128, 128), (64, 64), (32, 32)])
        self.assertEqual([s.shape[-1] for s in score_heads], [CLASSES] * 3)
        self.assertEqual([a.shape[-1] for a in angle_heads], [1, 1, 1])

    def test_classifier_is_order_independent(self):
        a = wd._classify_obb_heads(_heads())
        b = wd._classify_obb_heads(_heads(order="reversed"))
        self.assertEqual([x.shape for x in a[0]], [x.shape for x in b[0]])

    def test_classifier_rejects_unexpected_layout(self):
        self.assertIsNone(wd._classify_obb_heads([]))
        self.assertIsNone(wd._classify_obb_heads([np.zeros((5, 5, 7), np.float32)]))

    # ------------------------------------------------------------ decode
    def test_decode_axis_aligned(self):
        # stride 8, cell (2, 3): anchor = ((3 + 0.5) * 8, (2 + 0.5) * 8) = (28, 20)
        # all sides 10 bins -> 10 * 8 = 80 px -> w = h = 160 px, theta = 0
        rboxes, classes, scores = self.decode(_heads())
        self.assertEqual(len(rboxes), 1)
        np.testing.assert_allclose(rboxes[0], [28.0, 20.0, 160.0, 160.0, 0.0], atol=1e-3)
        self.assertEqual(int(classes[0]), 2)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=3)

    def test_decode_rotated_swaps_the_offset(self):
        # bins (l,t,r,b) = (10, 8, 12, 6) -> lt = (80, 64), rb = (96, 48)
        # xf = 8, yf = -8; theta = pi/2 -> offset (x, y) = (8, 8)
        rboxes, _, _ = self.decode(_heads(bins=(10, 8, 12, 6), angle_p=0.75))
        np.testing.assert_allclose(
            rboxes[0], [36.0, 28.0, 176.0, 112.0, np.pi / 2.0], atol=1e-3)

    def test_decode_uses_the_stride_16_head(self):
        # the 64x64 head doubles the distances: anchor ((2+0.5)*16, (1+0.5)*16)
        rboxes, _, _ = self.decode(_heads(cell=(1, 2), which=1))
        self.assertEqual(len(rboxes), 1)
        np.testing.assert_allclose(rboxes[0], [40.0, 24.0, 320.0, 320.0, 0.0], atol=1e-3)

    def test_threshold_drops_the_detection(self):
        self.assertIsNone(self.decode(_heads(), thresh=0.999)[0])

    # --------------------------------------------------------------- NMS
    def test_probiou_identity_and_separation(self):
        a = np.array([[10.0, 10.0, 20.0, 10.0, 0.0]], np.float32)
        self.assertAlmostEqual(float(wd._batch_probiou(a, a)[0, 0]), 1.0, places=3)
        b = np.array([[1000.0, 1000.0, 20.0, 10.0, 0.0]], np.float32)
        self.assertLess(float(wd._batch_probiou(a, b)[0, 0]), 0.01)

    def test_fast_nms_keeps_one_of_two_duplicates(self):
        boxes = np.array([[10.0, 10.0, 20.0, 10.0, 0.0],
                          [10.0, 10.0, 20.0, 10.0, 0.0]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        keep = wd._fast_nms(boxes, scores, 0.7)
        self.assertEqual(list(keep), [0])

    def test_overlapping_boxes_of_different_classes_both_survive(self):
        # two anchors, same geometry, different classes: the class offset in
        # _obb_detections must keep both
        heads = _heads()
        r, c = CELL
        heads[1][r, c, :] = -10.0
        heads[1][r, c, 2] = _logit(0.9)
        heads[1][r, c + 1, :] = -10.0
        heads[1][r, c + 1, 7] = _logit(0.8)
        boxes_8 = np.full((128, 128, 4, REG_BINS), -10.0, np.float32)
        for side in range(4):
            boxes_8[r, c, side, 10] = 10.0
            boxes_8[r, c + 1, side, 10] = 10.0
        heads[0] = boxes_8.reshape(128, 128, 64)
        rboxes, classes, _ = self.decode(heads)
        self.assertEqual(len(rboxes), 2)
        self.assertEqual(sorted(int(c) for c in classes), [2, 7])

    # ------------------------------------------------------------ guards
    def test_unexpected_layout_returns_none(self):
        self.assertEqual(self.decode([np.zeros((5, 5, 7), np.float32)]),
                         (None, None, None))

    def test_none_input_returns_none(self):
        self.assertEqual(self.decode(None), (None, None, None))

    def test_dota_class_list(self):
        self.assertEqual(len(wd.DEFAULT_CLASSES), CLASSES)
        self.assertEqual(wd.DEFAULT_CLASSES[0], "plane")
        self.assertEqual(wd.DEFAULT_CLASSES[14], "swimming-pool")

    # ------------------------------------------------------- unletterbox
    def test_unletterbox_scales_centre_and_size(self):
        rboxes = np.array([[100.0, 200.0, 40.0, 20.0, 0.5]], np.float32)
        out = wd.unletterbox_obb(rboxes, (0.5, 10.0, 20.0))
        np.testing.assert_allclose(out[0], [180.0, 360.0, 80.0, 40.0, 0.5], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
