# YOLOX-L-Leaky HEF

Place `yolox_l_leaky.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolox_l_leaky.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 640x640x3 RGB (ImageNet normalization compiled in-net)
- Output: on-chip HPP NMS tensor, post-NMS shape 80x5x100
- Classes: 80 (COCO)
- Expected size: 48,189,440 bytes
- SHA-256: `76b8d7626df267c7c56a348d20df18cc9f9cecc4ce80aa87d4ffe175bfad7f19`

`../web_detection.py` parses the post-NMS detections, scales the boxes to the
network input and un-letterboxes them to the frame.
