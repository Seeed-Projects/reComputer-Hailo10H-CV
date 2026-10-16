"""Regression tests for the YOLOv10 decode (no device required).

The HEF runs NMS on-chip (HPP), so the tests cover the HailoRT NMS payloads:
the compact per-class buffer HailoRT 5.1.1 returns, the dense (C,5,D) /
(C,D,5) forms and the ragged NMS-by-score list, plus the guard rails.
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


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class Yolov10DecodeTest(unittest.TestCase):
    def decode(self, output, thresh=0.25, iou=0.45):
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

    def test_dense_batched_layout(self):
        arr = np.zeros((1, CLASSES, MAX_DET, 5), np.float32)
        arr[0, 11, 0] = ROW
        boxes, classes, _ = self.decode(arr)
        self.assertEqual(int(classes[0]), 11)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_ragged_nms_list(self):
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        ragged[7] = np.asarray([ROW], np.float32)
        boxes, classes, _ = self.decode(ragged)
        self.assertEqual(int(classes[0]), 7)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_vstream_dict_payload(self):
        """HailoRT hands the per-class list out inside the vstream dict."""
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        ragged[7] = np.asarray([ROW], np.float32)
        boxes, classes, scores = self.decode({"yolov10s/nms": ragged})
        self.assertEqual(int(classes[0]), 7)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_multi_class_and_multi_box(self):
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        ragged[3] = np.asarray([[0.1, 0.1, 0.2, 0.2, 0.9],
                                [0.3, 0.3, 0.4, 0.4, 0.5]], np.float32)
        ragged[12] = np.asarray([[0.5, 0.5, 0.7, 0.7, 0.8]], np.float32)
        boxes, classes, scores = self.decode({"yolov10s/nms": ragged})
        self.assertEqual(len(boxes), 3)
        self.assertEqual(sorted(set(int(c) for c in classes)), [3, 12])
        self.assertEqual(sorted(round(float(s), 5) for s in scores), [0.5, 0.8, 0.9])

    def test_all_classes_empty_returns_none(self):
        ragged = [np.zeros((0, 5), np.float32) for _ in range(CLASSES)]
        self.assertIsNone(self.decode({"yolov10s/nms": ragged})[0])

    def test_score_threshold_drops_the_detection(self):
        buf = np.concatenate(([1.0], [0.1, 0.2, 0.5, 0.6, 0.05])).astype(np.float32)
        self.assertIsNone(self.decode(buf)[0])

    def test_compact_buffer_stops_at_implausible_header(self):
        """The host buffer is worst-case sized; the padding tail must not be
        parsed as data. A count above the per-class limit ends the parse."""
        buf = np.concatenate(([1.0], ROW, [MAX_DET + 1], ROW)).astype(np.float32)
        boxes, classes, _ = self.decode(buf)
        self.assertEqual(len(boxes), 1)

    def test_compact_buffer_rejects_unnormalised_rows(self):
        """Rows outside [0, 1] mean the header was misaligned padding."""
        buf = np.concatenate(([1.0], [700.0, 100.0, 1200.0, 900.0, 0.9])).astype(np.float32)
        self.assertIsNone(self.decode(buf)[0])

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
