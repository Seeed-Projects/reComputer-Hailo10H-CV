"""Regression tests for the YOLO26 detection decode (no device required).

Covers both compile layouts: the raw split heads (4-channel box regression, 80
class scores per stride, one2one top-k, no NMS) and the on-chip HPP NMS payload
(ragged NMS-by-score list, dense and compact forms).
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
INPUT = 640
HOT_CLASS = 3
CELL = (1, 2)
ROW = [0.1, 0.2, 0.5, 0.6, 0.9]          # ymin, xmin, ymax, xmax, score
EXPECTED_BOX = [0.2 * INPUT, 0.1 * INPUT, 0.6 * INPUT, 0.5 * INPUT]


def _raw_heads(reg_bins=0):
    """Raw split heads with one detection in the 80x80 head."""
    boxes, scores = [], []
    for f in (80, 40, 20):
        box = np.zeros((f, f, 4) if not reg_bins else (f, f, 4, reg_bins), np.float32)
        sc = np.full((f, f, CLASSES), -10.0, np.float32)
        if f == 80:
            if not reg_bins:
                box[CELL[0], CELL[1], :] = (1.0, 0.5, 2.0, 1.5)
            else:
                for side, b in enumerate((1, 0, 2, 1)):
                    box[CELL[0], CELL[1], side, b] = 30.0
            sc[CELL[0], CELL[1], HOT_CLASS] = 6.0
        boxes.append(box if not reg_bins else box.reshape(f, f, 64))
        scores.append(sc)
    return boxes + scores


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class Yolo26DecodeTest(unittest.TestCase):
    def decode(self, payload, thresh=0.25):
        return wd.post_process_hailo(payload, thresh, 0.7, INPUT, INPUT)

    # ------------------------------------------------------------ raw heads
    def test_raw_heads_decode(self):
        # stride 8, cell (1, 2): x1 = (2 + 0.5 - 1) * 8, y1 = (1 + 0.5 - 0.5) * 8,
        #                        x2 = (2 + 0.5 + 2) * 8, y2 = (1 + 0.5 + 1.5) * 8
        boxes, scores, class_ids, masks = self.decode(_raw_heads())
        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(class_ids[0]), HOT_CLASS)
        self.assertAlmostEqual(float(scores[0]), 1 / (1 + np.exp(-6.0)), places=5)
        np.testing.assert_allclose(boxes[0], [12.0, 8.0, 36.0, 24.0], atol=1e-3)
        self.assertIsNone(masks)

    def test_dfl_heads_still_decode(self):
        boxes, _, _, _ = self.decode(_raw_heads(reg_bins=16))
        self.assertIsNotNone(boxes)
        np.testing.assert_allclose(boxes[0], [12.0, 12.0, 36.0, 20.0], atol=1e-3)

    def test_threshold_drops_the_detection(self):
        self.assertIsNone(self.decode(_raw_heads(), thresh=0.999)[0])

    # ---------------------------------------------------------- on-chip NMS
    def test_ragged_payload_wrapped_in_dict(self):
        """The yolo26-seg HEFs return the on-chip NMS result as a ragged
        NMS-by-score list inside the vstream dict; the detection modules accept
        the same layout."""
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        ragged[7] = np.asarray([ROW], np.float32)
        boxes, scores, class_ids, masks = self.decode({"yolo26/nms": ragged})
        self.assertEqual(int(class_ids[0]), 7)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)
        self.assertIsNone(masks)

    def test_bare_ragged_payload(self):
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        ragged[12] = np.asarray([[0.5, 0.5, 0.7, 0.7, 0.8]], np.float32)
        boxes, _, class_ids, _ = self.decode(ragged)
        self.assertEqual(int(class_ids[0]), 12)

    def test_dense_layout(self):
        arr = np.zeros((CLASSES, 1, 5), np.float32)
        arr[5, 0] = ROW
        boxes, _, class_ids, _ = self.decode([arr])
        self.assertEqual(int(class_ids[0]), 5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_compact_buffer(self):
        buf = np.concatenate(([1.0], ROW, [0.0])).astype(np.float32)
        boxes, _, class_ids, _ = self.decode(buf)
        self.assertEqual(int(class_ids[0]), 0)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    # ---------------------------------------------------------------- guards
    def test_unexpected_layout_returns_none(self):
        boxes, scores, class_ids, masks = self.decode([np.zeros((5, 5, 7), np.float32)])
        self.assertIsNone(boxes)
        self.assertIsNone(scores)
        self.assertIsNone(class_ids)
        self.assertIsNone(masks)

    def test_empty_output_returns_none(self):
        self.assertIsNone(self.decode([])[0])
        self.assertIsNone(self.decode({"nms": [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]})[0])

    def test_coco_class_list(self):
        self.assertEqual(len(wd.COCO_CLASSES), CLASSES)
        self.assertEqual(wd.COCO_CLASSES[0], "person")


if __name__ == "__main__":
    unittest.main()
