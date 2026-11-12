# Test Report - ResNet-18 on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | resnet_v1_18 |
| Task | ImageNet-1k classification |
| Backbone | ResNet-18 (BRECQ) |
| Parameters | 11.68M |
| Operations | 3.64G |
| Top-1 accuracy (float / Hailo-10H) | 71.3% / 70.8% |
| Model Zoo version | v5.4.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/resnet_v1_18.hef |
| Size | 8,380,416 bytes |
| SHA256 | `3fc8768e1f2f4e0e14013c76311e6467eacf38a6ff503dd4b4c1567934a9c696` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/resnet_v1_18.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 224x224x3 RGB uint8 (plain resize; ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k scores (softmax disabled on device) |
| Decode | host softmax + Top-K, EMA smoothing for the preview |
| Overlay | Top-K panel (label, percentage, bar) in the top-left corner |

## Demo video

`video/test.mp4` is Hailo's official classifier demo asset `example.mp4`
(1280x720, 11.2 s, street scene) from
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

### Hardware verification - pending (assigned to the maintainer)

| Check | Status |
|---|---|
| HEF loads, output is a 1000-way vector | Pending |
| Label mapping matches ImageNet order | Pending |
| Demo video: Top-K labels are stable and plausible | Pending |
| Single-image REST API | Pending |
| Offline video analysis | Pending |
| USB camera | Pending |
| Official GHCR image re-pull + run | Pending |
| FPS measurement | Pending |