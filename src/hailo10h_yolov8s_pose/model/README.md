# YOLOv8s Pose HEF

This file is the Hailo-10H build from the Hailo Model Zoo v5.4.0 release.

| File | Size | SHA-256 |
|---|---:|---|
| `yolov8s_pose.hef` | 13,799,424 bytes | `d6cebc9a7bfc71d711000aa7d23b2e1a8aa634cbd90279df110a51d5309b1c58` |

Source URL pattern:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/<model>.hef
```

- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: `yolov8s_pose/input_layer1`, 640x640x3, normalization compiled in-net
- Output: single on-chip NMS vstream; each row carries the person box, its score
  and 17 COCO keypoints (56 values per row)
- Model used by the Docker CMD: `yolov8s_pose.hef`

`../web_detection.py` implements the NMS row decoding, keypoint scaling and
letterbox restoration, and the skeleton drawing.

The exact output vstream name and row layout are printed by the first inference
on hardware; confirm them there before trusting the preview.