# YOLO26l-seg - Instance Segmentation

YOLO26l-seg (28M params) on Hailo-10H. The module decodes the raw
segmentation heads, assembles the instance masks (mask coefficients x prototype,
cropped to each box) and draws them into the annotated preview.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26l-seg |
| Input | 640x640x3 RGB |
| HEF output | 10 raw tensors: 4-ch box (l,t / r,b distances), 80-ch class logits and 32-ch mask coefficients per stride, plus a 160x160x32 prototype |
| Parameters | 28M |
| Operations | 150.2G |
| Format | HEF (Hailo-10H) |
| HEF size | 40,140,800 bytes |
| HEF SHA256 | `8b4c51b588f5213ff5437100d4da9f6ea0969c723ece562dbd445aed837ec637` |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from
the repository root.

```bash
docker build -t yolo26l-seg -f docker/hailo10h/yolo26l_seg.dockerfile src/hailo10h_yolo26l_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolo26l_seg:latest
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolo26l_seg/predict` | POST | Detections (JSON: class, confidence, box) |

The HEF exposes raw heads (no on-chip NMS): the host applies sigmoid, the
two-stage top-k of the one2one head (post_nms_topk=100, no NMS), the
regression_length=1 box decode, and the mask assembly. The JSON payload carries
the boxes; the instance masks are composited into the annotated preview and the
offline video output.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0 (Hailo-10H):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo26l_seg.hef
```

The Hailo-10H zoo compiles yolo26n_seg / yolo26s_seg / yolo26m_seg / yolo26l_seg (no x_seg build exists).
