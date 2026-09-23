# ViTPose-Small HEF

Place `vit_pose_small.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/vit_pose_small.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 256x192x3 RGB (ImageNet normalization compiled in-net)
- Output: single heatmap 64x48x17 (17 COCO keypoints, argmax decode, no NMS)
- Expected size: 24,653,824 bytes
- SHA-256: `7cff90cc972b27f2d7c0062c52ceae2b07f1881e4b7889cd2a20b4f293723c67`

`../web_detection.py` takes the argmax per heatmap channel, scales the
coordinates to the 256x192 input, un-letterboxes them to the frame and draws
the COCO skeleton.
