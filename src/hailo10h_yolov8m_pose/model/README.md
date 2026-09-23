# YOLOv8m Pose HEF

This file is the Hailo-10H build from the Hailo Model Zoo v5.4.0 release.

| File | Size | SHA-256 |
|---|---:|---|
| `yolov8m_pose.hef` | 29,335,552 bytes | `4a9af131cd9d089c4986e8baa934e4b28e3deefd34f53e1d44533d7d25027542` |

Source URL pattern:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/<model>.hef
```

- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: `yolov8s_pose/input_layer1`, 640x640x3, normalization compiled in-net
- Output: nine raw tensors (bbox DFL 64 / score 1 / keypoints 51 at three
  feature-map scales); NMS and decoding run on the host
- Default model used by the Docker CMD: `yolov8s_pose.hef`

`../web_detection.py` implements the NMS row decoding, keypoint scaling and
letterbox restoration, and the skeleton drawing.

The exact output vstream name and row layout are printed by the first inference
on hardware; confirm them there before trusting the preview.