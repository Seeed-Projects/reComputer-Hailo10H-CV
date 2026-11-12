# Test Report - ResNet-34 on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | resnet_v1_34 |
| Task | ImageNet-1k classification |
| Backbone | ResNet-34 (BRECQ) |
| Parameters | 21.79M |
| Operations | 7.34G |
| Top-1 accuracy (float / Hailo-10H) | 72.7% / 72.2% |
| Model Zoo version | v5.4.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/resnet_v1_34.hef |
| Size | 18,829,312 bytes |
| SHA256 | `b39b2c94c8f191e29e0da953bf8b1d676554f5b9c24822f265f70bffa9424407` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/resnet_v1_34.hef |

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
