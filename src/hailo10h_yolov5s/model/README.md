# YOLOv5s HEF

Place `yolov5s.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov5s.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 640x640x3 RGB (normalize_in_net mean=0/std=255, padding_color=114)
- Output: on-chip HPP NMS tensor, zoo post-NMS shape 80x5x80
- Classes: 80 (COCO, 0-indexed; zoo evaluation labels_offset=1)
- Parameters: 7.46M
- Operations: 17.44G
- Expected size: 13,029,376 bytes
- SHA-256: `7609d7dbb236dcdcd3fdc6563ea646b39682521b1ceaa2737b446966dc66fc50`

`../web_detection.py` parses the post-NMS detections, scales the boxes to the
network input and un-letterboxes them to the frame.
