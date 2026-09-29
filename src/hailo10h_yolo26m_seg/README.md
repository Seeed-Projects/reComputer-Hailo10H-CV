# YOLO26m-seg - Instance Segmentation

YOLO26m-seg (23.6M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26m-seg |
| Input | 640×640×3 RGB |
| HEF output | 10 raw tensors: 4-ch box (l,t / r,b distances), 80-ch class logits and 32-ch mask coefficients per stride, plus a 160x160x32 prototype |
| Parameters | 23.6M |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t yolo26m-seg -f docker/hailo10h/yolo26m_seg.dockerfile src/hailo10h_yolo26m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26m-seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolo26m_seg/predict` | POST | Box-level detections (JSON) |

The HEF exposes raw heads (no on-chip NMS): the host applies sigmoid, the two-stage top-k of the one2one head (post_nms_topk=100, no NMS), the regression_length=1 box decode and mask assembly (coefficients x prototype, cropped to each box).

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0 (Hailo-10H):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo26m_seg.hef
```

Size: 28,344,320 bytes - SHA-256: `1b434156d9fd111020d385894b40e8f9b3d625e34a65334003a1231233470f9e`
