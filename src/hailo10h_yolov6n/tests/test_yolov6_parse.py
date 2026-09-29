"""Regression tests for the YOLOv6n decode (no device required).

Covers both compile layouts: the on-chip HPP NMS tensor (compact per-class
buffer, both dense forms and the ragged NMS-by-score list) and the nine raw
split heads with the host-side NMS.
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

CLASSES = 80
MAX_DET = 100
INPUT = 640
ROW = [0.1, 0.2, 0.5, 0.6, 0.9]          # ymin, xmin, ymax, xmax, score (normalised)
EXPECTED_BOX = [0.2 * INPUT, 0.1 * INPUT, 0.6 * INPUT, 0.5 * INPUT]
CELL = (1, 2)                             # (row, col) of the test detection
HOT_CLASS = 3


def _raw_heads(obj_logit=4.0, cls_logit=2.0):
    """The nine split heads with one detection in the 80x80 head."""
    boxes, objs, clas = [], [], []
    for f in (80, 40, 20):
        box = np.zeros((f, f, 4), np.float32)
        obj = np.full((f, f, 1), -10.0, np.float32)
        cls = np.full((f, f, CLASSES), -10.0, np.float32)
        if f == 80:
            box[CELL[0], CELL[1], :] = (1.0, 0.5, 2.0, 1.5)   # l, t, r, b
            obj[CELL[0], CELL[1], 0] = obj_logit
            cls[CELL[0], CELL[1], HOT_CLASS] = cls_logit
        boxes.append(box)
        objs.append(obj)
        clas.append(cls)
    return boxes + objs + clas


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class Yolov6DecodeTest(unittest.TestCase):
    def decode(self, output, thresh=0.25, iou=0.65):
        return wd.post_process_hailo(output, thresh, iou, INPUT, INPUT)

    # ---------------------------------------------------------- on-chip NMS
    def test_compact_per_class_buffer(self):
        buf = np.concatenate(([1.0], ROW, [0.0])).astype(np.float32)
        boxes, classes, scores = self.decode(buf)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), 0)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_dense_class_first_layout(self):
        arr = np.zeros((CLASSES, 5, MAX_DET), np.float32)
        arr[5, :, 0] = ROW
        boxes, classes, _ = self.decode([arr])
        self.assertEqual(int(classes[0]), 5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_dense_detection_first_layout(self):
        arr = np.zeros((CLASSES, MAX_DET, 5), np.float32)
        arr[9, 0] = ROW
        boxes, classes, _ = self.decode([arr])
        self.assertEqual(int(classes[0]), 9)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_ragged_nms_list(self):
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        ragged[7] = np.asarray([ROW], np.float32)
        boxes, classes, _ = self.decode(ragged)
        self.assertEqual(int(classes[0]), 7)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_score_threshold_drops_the_detection(self):
        buf = np.concatenate(([1.0], [0.1, 0.2, 0.5, 0.6, 0.05])).astype(np.float32)
        self.assertIsNone(self.decode(buf)[0])

    # ---------------------------------------------------------- raw heads
    def test_raw_heads_decode(self):
        # stride 8, cell (1, 2): x1 = (2 + 0.5 - 1) * 8, y1 = (1 + 0.5 - 0.5) * 8,
        #                        x2 = (2 + 0.5 + 2) * 8, y2 = (1 + 0.5 + 1.5) * 8
        boxes, classes, scores = self.decode(_raw_heads())
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), HOT_CLASS)
        expected = (1 / (1 + np.exp(-4.0))) * (1 / (1 + np.exp(-2.0)))
        self.assertAlmostEqual(float(scores[0]), expected, places=5)
        np.testing.assert_allclose(boxes[0], [12.0, 8.0, 36.0, 24.0], atol=1e-3)

    def test_raw_heads_objectness_gates_the_score(self):
        boxes, _, scores = self.decode(_raw_heads(obj_logit=-10.0))
        self.assertIsNone(boxes)

    def test_raw_heads_nms_keeps_one_box(self):
        heads = _raw_heads()
        heads[7][CELL[0], CELL[1], 0] = 6.0      # a second, identical box, higher score
        boxes, _, _ = self.decode(heads)
        self.assertEqual(len(boxes), 1)

    # ------------------------------------------------------------ guards
    def test_unexpected_layout_returns_none(self):
        boxes, classes, scores = self.decode([np.zeros((5, 5, 7), np.float32)])
        self.assertIsNone(boxes)
        self.assertIsNone(classes)
        self.assertIsNone(scores)

    def test_empty_output_returns_none(self):
        self.assertIsNone(self.decode([])[0])

    def test_coco_class_list(self):
        self.assertEqual(wd.DEFAULT_CLASSES[0], "person")
        self.assertEqual(len(wd.DEFAULT_CLASSES), CLASSES)


if __name__ == "__main__":
    unittest.main()