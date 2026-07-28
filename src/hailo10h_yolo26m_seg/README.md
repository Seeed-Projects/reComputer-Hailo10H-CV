# YOLO26n-seg - Instance Segmentation

YOLO26n-seg (23.6M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26n-seg |
| Input | 640×640×3 RGB |
| Output | Bounding boxes + instance masks (COCO 80 classes) |
| Parameters | 23.6M |
| Format | HEF (Hailo-10H) |

## Quick Start

```bash
docker build -t yolo26n-seg -f docker/hailo10h/yolo26m_seg.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26n-seg
```

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).