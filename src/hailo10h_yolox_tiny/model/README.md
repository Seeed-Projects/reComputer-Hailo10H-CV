# YOLOX-Tiny HEF

Place `yolox_tiny.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolox_tiny.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 416x416x3 RGB (ImageNet normalization compiled in-net)
- Output: on-chip HPP NMS tensor, post-NMS shape 80x5x100
- Classes: 80 (COCO)
- Expected size: 8,880,128 bytes
- SHA-256: `b15277144be90deea8160a93580632a72769d726c569816f9c03d15cdef93be5`

`../web_detection.py` parses the post-NMS detections, scales the boxes to the
network input and un-letterboxes them to the frame.
