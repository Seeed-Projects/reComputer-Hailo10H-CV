"""Regression tests for the YOLO26-seg raw decode (no device required)."""
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

PROTO = 160
CLASSES = 80
COEFFS = 32
INPUT = 640
GRIDS = (80, 40, 20)          # head grids; _classify_heads sorts by area
CELL = (1, 2)                 # (row, col) that carries the test detection
HOT_CLASS = 3
HOT_LOGIT = 4.0


def _endnodes(reg_length):
    """Build the 10 endnodes with one detection in the 80x80 head."""
    boxes, scores, coeffs = [], [], []
    for f in GRIDS:
        if reg_length == 4:
            box = np.zeros((f, f, 4), np.float32)
        else:
            box = np.zeros((f, f, 4, 16), np.float32)
        sc = np.full((f, f, CLASSES), -10.0, np.float32)
        mc = np.zeros((f, f, COEFFS), np.float32)
        if f == max(GRIDS):
            if reg_length == 4:
                box[CELL[0], CELL[1], :] = (1.0, 0.5, 2.0, 1.5)   # l, t, r, b
            else:
                for side, bin_idx in enumerate((1, 0, 2, 1)):
                    # Sharp DFL peak -> softmax expectation == the bin index.
                    box[CELL[0], CELL[1], side, bin_idx] = 30.0
            sc[CELL[0], CELL[1], HOT_CLASS] = HOT_LOGIT
        boxes.append(box if reg_length == 4 else box.reshape(f, f, 64))
        scores.append(sc)
        coeffs.append(mc)
    proto = np.zeros((PROTO, PROTO, COEFFS), np.float32)
    return boxes + scores + coeffs + [proto]


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class Yolo26SegDecodeTest(unittest.TestCase):
    def decode(self, endnodes, thresh=0.25):
        return wd.post_process_hailo(endnodes, thresh, 0.45, INPUT, INPUT)

    def test_direct_ltrb_head_decodes_the_box(self):
        # stride 8, cell (1, 2): x1 = (2 + 0.5 - 1) * 8, y1 = (1 + 0.5 - 0.5) * 8,
        #                        x2 = (2 + 0.5 + 2) * 8, y2 = (1 + 0.5 + 1.5) * 8
        boxes, scores, classes, masks = self.decode(_endnodes(4))
        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), HOT_CLASS)
        self.assertAlmostEqual(float(scores[0]), 1.0 / (1.0 + np.exp(-HOT_LOGIT)), places=4)
        np.testing.assert_allclose(boxes[0], [12.0, 8.0, 36.0, 24.0], atol=1e-3)
        self.assertEqual(masks.shape[0], 1)

    def test_dfl_head_still_decodes(self):
        # DFL peaks at bins (l, t, r, b) = (1, 0, 2, 1): y1 = (1 + 0.5 - 0) * 8
        boxes, _, _, _ = self.decode(_endnodes(64))
        self.assertIsNotNone(boxes)
        np.testing.assert_allclose(boxes[0], [12.0, 12.0, 36.0, 20.0], atol=1e-3)

    def test_score_threshold_drops_the_detection(self):
        boxes, scores, classes, masks = self.decode(_endnodes(4), thresh=0.99)
        self.assertIsNone(boxes)
        self.assertIsNone(scores)
        self.assertIsNone(classes)
        self.assertIsNone(masks)

    def test_unexpected_layout_is_reported_not_crashing(self):
        boxes, scores, classes, masks = self.decode([np.zeros((5, 5, 7), np.float32)])
        self.assertIsNone(boxes)
        self.assertIsNone(scores)
        self.assertIsNone(classes)
        self.assertIsNone(masks)

    def test_coco_class_list(self):
        self.assertEqual(wd.COCO_CLASSES[0], "person")
        self.assertEqual(len(wd.COCO_CLASSES), CLASSES)


if __name__ == "__main__":
    unittest.main()
