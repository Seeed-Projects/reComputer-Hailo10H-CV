# YOLOv8 Pose HEFs

Both files are Hailo-10H builds from the Hailo Model Zoo v5.4.0 release.

| File | Size | SHA-256 |
|---|---:|---|
| `yolov8s_pose.hef` | 13,799,424 bytes | `d6cebc9a7bfc71d711000aa7d23b2e1a8aa634cbd90279df110a51d5309b1c58` |
| `yolov8m_pose.hef` | 29,335,552 bytes | `4a9af131cd9d089c4986e8baa934e4b28e3deefd34f53e1d44533d7d25027542` |

Source URL pattern:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/<model>.hef
```

- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: `yolov8s_pose/input_layer1`, 640x640x3, normalization compiled in-net
- Output: single on-chip NMS vstream; each row carries the person box, its score
  and 17 COCO keypoints (56 values per row)
- Default model used by the Docker CMD: `yolov8s_pose.hef`

`../web_detection.py` implements the NMS row decoding, keypoint scaling and
letterbox restoration, and the skeleton drawing.

The exact output vstream name and row layout are printed by the first inference
on hardware; confirm them there before trusting the preview.