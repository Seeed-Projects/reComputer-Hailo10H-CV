"""Regression tests for the post-NMS YOLOX parsing (no device required)."""
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

NUM_CLASSES = 80
MAX_DET = 100
INPUT = 640
ROW = [0.1, 0.2, 0.5, 0.6, 0.9]          # ymin, xmin, ymax, xmax, score (normalised)
EXPECTED_BOX = [0.2 * INPUT, 0.1 * INPUT, 0.6 * INPUT, 0.5 * INPUT]


@unittest.skipIf(wd is None, "web_detection needs opencv/fastapi: %s" % IMPORT_ERROR)
class PostNmsParseTest(unittest.TestCase):
    def decode(self, output, thresh=0.25):
        return wd.post_process_hailo({"out": output}, thresh, 0.45, INPUT, INPUT)

    def test_compact_per_class_buffer(self):
        # class 0: one row; classes 1..2: none
        buf = np.concatenate(([1.0], ROW, [0.0], [0.0])).astype(np.float32)
        boxes, classes, scores = self.decode(buf)
        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), 0)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_dense_class_first_layout(self):
        arr = np.zeros((NUM_CLASSES, MAX_DET, 5), dtype=np.float32)
        arr[3, 0] = ROW
        boxes, classes, _ = self.decode(arr)
        self.assertEqual(int(classes[0]), 3)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_dense_detection_first_layout_is_transposed(self):
        arr = np.zeros((NUM_CLASSES, 5, MAX_DET), dtype=np.float32)
        arr[5, :, 0] = ROW
        boxes, classes, _ = self.decode(arr)
        self.assertEqual(int(classes[0]), 5)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_ragged_nms_list(self):
        ragged = [np.zeros((0, 5), dtype=np.float32) for _ in range(NUM_CLASSES)]
        ragged[7] = np.asarray([ROW], dtype=np.float32)
        boxes, classes, _ = self.decode(ragged)
        self.assertEqual(int(classes[0]), 7)
        np.testing.assert_allclose(boxes[0], EXPECTED_BOX, atol=1e-3)

    def test_score_threshold_drops_rows(self):
        buf = np.concatenate(([1.0], [0.1, 0.2, 0.5, 0.6, 0.05])).astype(np.float32)
        boxes, classes, scores = self.decode(buf)
        self.assertIsNone(boxes)

    def test_unletterbox_maps_back_to_frame(self):
        boxes = np.asarray([EXPECTED_BOX], dtype=np.float32)
        out = wd.unletterbox_boxes(boxes, (0.5, 10.0, 20.0))
        np.testing.assert_allclose(
            out[0],
            [(EXPECTED_BOX[0] - 10) / 0.5, (EXPECTED_BOX[1] - 20) / 0.5,
             (EXPECTED_BOX[2] - 10) / 0.5, (EXPECTED_BOX[3] - 20) / 0.5],
            atol=1e-3,
        )

    def test_coco_class_mapping(self):
        self.assertEqual(wd.CLASSES[0], "person")
        self.assertEqual(wd.CLASSES[2], "car")
        self.assertEqual(len(wd.CLASSES), NUM_CLASSES)


if __name__ == "__main__":
    unittest.main()
