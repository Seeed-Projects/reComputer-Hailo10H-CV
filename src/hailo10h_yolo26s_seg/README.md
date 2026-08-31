# YOLO26s-seg - Instance Segmentation

YOLO26s-seg (10.4M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26s-seg |
| Input | 640×640×3 RGB |
| HEF output | Bounding boxes + instance-mask tensors (COCO 80 classes) |
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

The current Web postprocessor exposes box-level detections. Instance-mask decoding still needs to be validated on the target hardware.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
