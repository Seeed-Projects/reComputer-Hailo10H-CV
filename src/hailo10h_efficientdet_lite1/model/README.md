# EfficientDet-Lite1 HEF

Place `efficientdet_lite1.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/efficientdet_lite1.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 384x384x3 RGB (mean=127 / std=128 compiled in-net)
- Output: on-chip NMS + sigmoid (HPP), post-NMS shape 89x5x100
- Classes: 89 slots with `labels_offset=1` (COCO category IDs 1..89)
- Expected size: 7,614,464 bytes
- SHA-256: `09b44589e1e641a98c0ef97bd69a38e079bc96e6767a567a4f51e20040aeb3d0`

`../web_detection.py` parses the post-NMS detections, maps `cls_id` to the COCO
category ID (`cls_id + 1`), scales the boxes to the network input and
un-letterboxes them to the frame.
