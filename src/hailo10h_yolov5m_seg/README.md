# YOLOv5m-seg - Instance Segmentation

YOLOv5m-seg (anchor-based instance segmentation, 32.60M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv5m-seg (3 anchors, strides 8/16/32) |
| Input | 640×640×3 RGB |
| Output | 80-class boxes + instance masks (proto 160x160x32) |
| Parameters | 32.60M |
| Hardware mAP | 36.7 (COCO, Hailo Model Zoo reference) |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t yolov5m_seg -f docker/hailo10h/yolov5m_seg.dockerfile src/hailo10h_yolov5m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov5m_seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolov5m_seg/predict` | POST | Boxes + instance masks (JSON) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
