# ViTPose-Small-BN HEF

Place `vit_pose_small_bn.hef` in this directory.

- Source: Hailo Model Zoo compiled models v5.4.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/vit_pose_small_bn.hef
- Hardware architecture: Hailo-10H (not Hailo-8 or Hailo-8L)
- Input: 256x192x3 RGB (ImageNet normalization compiled in-net)
- Output: single heatmap 64x48x17 (17 COCO keypoints, argmax decode, no NMS)
- Expected size: 23,314,432 bytes
- SHA-256: `4e683e313ab4c785043ba69986290907b01c2f172ddd3e2527c4076e0004ba6f`

`../web_detection.py` takes the argmax per heatmap channel, scales the
coordinates to the 256x192 input, un-letterboxes them to the frame and draws
the COCO skeleton.
