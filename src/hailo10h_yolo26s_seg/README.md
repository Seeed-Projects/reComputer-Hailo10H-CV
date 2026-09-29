# YOLO26s-seg - Instance Segmentation

YOLO26s-seg (10.4M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26s-seg |
| Input | 640×640×3 RGB |
| HEF output | 10 raw tensors: 4-ch box (l,t / r,b distances), 80-ch class logits and 32-ch mask coefficients per stride, plus a 160x160x32 prototype |
| Parameters | 10.4M |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t yolo26s-seg -f docker/hailo10h/yolo26s_seg.dockerfile src/hailo10h_yolo26s_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26s-seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolo26s_seg/predict` | POST | Box-level detections (JSON) |

The HEF exposes raw heads (no on-chip NMS): the host applies sigmoid, the two-stage top-k of the one2one head (post_nms_topk=100, no NMS), the regression_length=1 box decode and mask assembly (coefficients x prototype, cropped to each box).

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0 (Hailo-10H):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo26s_seg.hef
```

Size: 18,345,984 bytes - SHA-256: `f9a36c8429a4ccd8663284ff56239e6bb1303f16a3f4fe287fed5b5694fc87b9`
