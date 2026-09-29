# YOLOv5m HEF

Place `yolov5m.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov5m.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 640x640x3 RGB (normalize_in_net mean=0/std=255, padding_color=114)
- Output: on-chip HPP NMS tensor, zoo post-NMS shape 80x5x80
- Classes: 80 (COCO, 0-indexed; zoo evaluation labels_offset=1)
- Parameters: 21.78M
- Operations: 52.17G
- Expected size: 19,812,352 bytes
- SHA-256: `92893bad750a4d7384817402ee5b447e05f2eaf4fb7c9933afae8cb2457a3dba`

`../web_detection.py` parses the post-NMS detections, scales the boxes to the
network input and un-letterboxes them to the frame.
