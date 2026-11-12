# Test Report - ResNet-50 on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | resnet_v1_50 |
| Task | ImageNet-1k classification |
| Backbone | ResNet-50 (BRECQ) |
| Parameters | 25.53M |
| Operations | 6.98G |
| Top-1 accuracy (float / Hailo-10H) | 75.2% / 74.6% |
| Model Zoo version | v5.4.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/resnet_v1_50.hef |
| Size | 22,945,792 bytes |
| SHA256 | `9ecbca3ac07420968fdd013bd82597483f5b8a1a9839c22ab92c0a995e93ab50` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/resnet_v1_50.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 224x224x3 RGB uint8 (plain resize; ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k scores (softmax disabled on device) |
| Decode | host softmax + Top-K, EMA smoothing for the preview |
| Overlay | Top-K panel (label, percentage, bar) in the top-left corner |

## Demo video

`video/test.mp4` is Hailo's official classifier demo asset `example.mp4`
(1280x720, 11.2 s) from
`https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4`
(`hailo-ai/hailo-apps`, MIT).

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline inference_loop harness (synthetic 1000-way logits -> Top-K overlay) | Pass |
| Configuration defaults (confidence 0.30, Top-K 5) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification - pending
