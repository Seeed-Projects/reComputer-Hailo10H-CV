# YOLOX-S-Leaky HEF

Place `yolox_s_leaky.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolox_s_leaky.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 640x640x3 RGB (ImageNet normalization compiled in-net)
- Output: on-chip HPP NMS tensor, post-NMS shape 80x5x100
- Classes: 80 (COCO)
- Expected size: 14,520,320 bytes
- SHA-256: `8f432c69f0c4dc0bde0752894c1abc0660753cec3e8ab813d48cf9167aefef9e`

`../web_detection.py` parses the post-NMS detections, scales the boxes to the
network input and un-letterboxes them to the frame.
