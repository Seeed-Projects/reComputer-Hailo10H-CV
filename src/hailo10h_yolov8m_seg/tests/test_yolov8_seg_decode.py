"""Regression tests for the YOLOv8-seg raw decode (no device required).

Runs against an unpacked module: by default the parent directory of tests/,
or the directory named by Y8_MODULE_DIR (used to exercise the Hailo-8
modules, which keep their checks in TEST_REPORT.md instead of tests/).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(Path(__file__).parents[1])
sys.path.insert(0, str(MODULE_DIR))

try:
    import web_detection as wd
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host toolchain
    wd = None
    IMPORT_ERROR = exc

PROTO = 160
CLASSES = 80
COEFFS = 32
INPUT = 640
GRIDS = (80, 40, 20)           # head grids; _classify_heads sorts by area
CELL = (1, 2, 3, 4.0)          # row, col, class, logit of the planted detection
HOT_BOX = (12.0, 8.0, 36.0, 24.0)      # stride 8, cell (1, 2), ltrb (1, .5, 2, 1.5)
DFL_BOX = (12.0, 12.0, 36.0, 20.0)     # DFL bins (l, t, r, b) = (1, 0, 2, 1)
WIDE_LTRB = (2.0, 0.5, 2.0, 1.5)       # wide enough for two cells to overlap


def _endnodes(reg_length, cells=(CELL,), ltrb=(1.0, 0.5, 2.0, 1.5)):
    """Build the 10 endnodes with detections planted in the 80x80 head."""
    boxes, scores, coeffs = [], [], []
    for f in GRIDS:
        if reg_length == 4:
            box = np.zeros((f, f, 4), np.float32)
        else:
            box = np.zeros((f, f, 4, 16), np.float32)
        sc = np.full((f, f, CLASSES), -10.0, np.float32)
        mc = np.zeros((f, f, COEFFS), np.float32)
        if f == max(GRIDS):
            for row, col, hot_class, hot_logit in cells:
                if reg_length == 4:
                    box[row, col, :] = ltrb
                else:
                    for side, bin_idx in enumerate((1, 0, 2, 1)):
                        # Sharp DFL peak -> softmax expectation == the bin index.
                        box[row, col, side, bin_idx] = 30.0
                sc[row, col, hot_class] = hot_logit
        boxes.append(box if reg_length == 4 else box.reshape(f, f, 64))
        scores.append(sc)
        coeffs.append(mc)
    proto = np.zeros((PROTO, PROTO, COEFFS), np.float32)
    return boxes + scores + coeffs + [proto]


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class Yolo8SegDecodeTest(unittest.TestCase):
    def decode(self, endnodes, thresh=0.25, iou=0.45):
        return wd.post_process_hailo(endnodes, thresh, iou, INPUT, INPUT)

    def test_direct_ltrb_head_decodes_the_box(self):
        # stride 8, cell (1, 2): x1 = (2 + 0.5 - 1) * 8, y1 = (1 + 0.5 - 0.5) * 8,
        #                        x2 = (2 + 0.5 + 2) * 8, y2 = (1 + 0.5 + 1.5) * 8
        boxes, scores, classes, masks = self.decode(_endnodes(4))
        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), CELL[2])
        self.assertAlmostEqual(float(scores[0]), 1.0 / (1.0 + np.exp(-CELL[3])), places=4)
        np.testing.assert_allclose(boxes[0], HOT_BOX, atol=1e-3)
        self.assertEqual(masks.shape[0], 1)

    def test_dfl_head_still_decodes(self):
        boxes, _, _, _ = self.decode(_endnodes(64))
        self.assertIsNotNone(boxes)
        np.testing.assert_allclose(boxes[0], DFL_BOX, atol=1e-3)

    def test_score_threshold_drops_the_detection(self):
        boxes, scores, classes, masks = self.decode(_endnodes(4), thresh=0.99)
        self.assertIsNone(boxes)
        self.assertIsNone(scores)
        self.assertIsNone(classes)
        self.assertIsNone(masks)

    def test_overlapping_same_class_goes_through_nms(self):
        """one2many heads need NMS: the lower-scoring duplicate is suppressed."""
        cells = ((1, 2, 3, 4.0), (1, 3, 3, 2.0))
        boxes, scores, classes, _ = self.decode(_endnodes(4, cells, WIDE_LTRB))
        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), 3)
        self.assertAlmostEqual(float(scores[0]), 1.0 / (1.0 + np.exp(-4.0)), places=4)

    def test_nms_is_per_class(self):
        """Two overlapping boxes of different classes both survive."""
        cells = ((1, 2, 3, 4.0), (1, 3, 5, 2.0))
        boxes, scores, classes, _ = self.decode(_endnodes(4, cells, WIDE_LTRB))
        self.assertIsNotNone(boxes)
        self.assertEqual(sorted(int(c) for c in classes), [3, 5])

    def test_nms_threshold_of_zero_falls_back_to_default(self):
        """The REST endpoint passes 0; the module must substitute 0.45."""
        cells = ((1, 2, 3, 4.0), (1, 3, 3, 2.0))
        boxes, _, _, _ = self.decode(_endnodes(4, cells, WIDE_LTRB), iou=0)
        self.assertEqual(len(boxes), 1)

    def test_iou_threshold_one_keeps_both_duplicates(self):
        cells = ((1, 2, 3, 4.0), (1, 3, 3, 2.0))
        boxes, _, _, _ = self.decode(_endnodes(4, cells, WIDE_LTRB), iou=1.0)
        self.assertEqual(len(boxes), 2)

    def test_unexpected_layout_is_reported_not_crashing(self):
        boxes, scores, classes, masks = self.decode([np.zeros((5, 5, 7), np.float32)])
        self.assertIsNone(boxes)
        self.assertIsNone(scores)
        self.assertIsNone(classes)
        self.assertIsNone(masks)

    def test_draw_boxes_composites_masks_without_shape_error(self):
        """Regression: a (N, h, w) boolean mask must not index the (h, w) frame."""
        image = np.zeros((64, 48, 3), np.uint8)
        masks = np.zeros((2, 64, 48), np.float32)
        masks[0, 10:20, 10:20] = 0.9
        masks[1, 30:40, 20:30] = 0.9
        boxes = np.array([[10, 10, 20, 20], [20, 30, 30, 40]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        classes = np.array([0, 2], np.int32)

        wd.draw_boxes(image, boxes, scores, classes, masks, None)

        self.assertGreater(int(image[10:20, 10:20].sum()), 0)
        self.assertGreater(int(image[30:40, 20:30].sum()), 0)
        self.assertEqual(int(image[0:5, 0:5].sum()), 0)

    def test_sigmoid_saturates_without_overflow(self):
        out = wd._sigmoid(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0], np.float32))
        np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_probability_head_skips_sigmoid(self):
        """A compile that ships an activated score head must not be re-sigmoided."""
        endnodes = _endnodes(4)
        for idx in range(3, 6):
            endnodes[idx] = np.clip(endnodes[idx] * 0.0, 0.0, 1.0)
            endnodes[idx][CELL[0], CELL[1], CELL[2]] = 0.9
        boxes, scores, classes, _ = wd.post_process_hailo(endnodes, 0.25, 0.45, INPUT, INPUT)
        self.assertIsNotNone(boxes)
        self.assertEqual(int(classes[0]), CELL[2])
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)

    def test_boxes_and_masks_unletterbox_consistently(self):
        """Boxes are drawn after unletterbox_boxes, masks inside draw_boxes.
        Both must map the same input-space region onto the same frame region:
        640x640 input, 1280x720 frame -> ratio 0.5, dh 140.0, dw 0.0."""
        lb = (0.5, 0.0, 140.0)
        box = np.array([[100.0, 200.0, 300.0, 400.0]], np.float32)
        real = wd.unletterbox_boxes(box, lb)
        np.testing.assert_allclose(real, [[200.0, 120.0, 600.0, 520.0]], atol=1e-3)

        masks = np.zeros((1, 640, 640), np.float32)
        masks[0, 200:400, 100:300] = 1.0
        frame_masks = wd.unletterbox_masks(masks, lb, (720, 1280, 3))
        self.assertEqual(frame_masks.shape, (1, 720, 1280))
        ys, xs = np.where(frame_masks[0] > 0.5)
        self.assertLessEqual(abs(int(ys.min()) - 120), 3)
        self.assertLessEqual(abs(int(xs.min()) - 200), 3)
        self.assertLessEqual(abs(int(ys.max()) - 519), 5)
        self.assertLessEqual(abs(int(xs.max()) - 599), 5)

    def test_coco_class_list(self):
        self.assertEqual(wd.COCO_CLASSES[0], "person")
        self.assertEqual(len(wd.COCO_CLASSES), CLASSES)

    def test_default_class_list_is_coco(self):
        """The module default must be the 80 COCO names, not a leftover palette."""
        self.assertEqual(tuple(wd.DEFAULT_CLASSES), tuple(wd.COCO_CLASSES))


if __name__ == "__main__":
    unittest.main()