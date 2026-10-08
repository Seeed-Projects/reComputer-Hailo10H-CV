# YOLOv8m-seg - Instance Segmentation

YOLOv8m-seg (27.3M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv8m-seg (anchor-free, one2many head, strides 8/16/32) |
| Input | 640x640x3 RGB (letterbox pad 114) |
| HEF output | 10 raw tensors: box-distance, 80-ch class score and 32-ch mask-coefficient heads per stride, plus a 160x160x32 mask prototype. The box head is either the 4-ch folded layout (l, t / r, b distances) or the 64-ch DFL 4x16 layout; both are decoded. |
| Parameters | 27.3M |
| Operations | 110.2G |
| Reference mAP | 40.6 (COCO instance segmentation, full precision, Hailo Model Zoo; not measured on Hailo-10H here) |
| License | AGPL-3.0 (Ultralytics) |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t yolov8m_seg -f docker/hailo10h/yolov8m_seg.dockerfile src/hailo10h_yolov8m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov8m_seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolov8m_seg/predict` | POST | Box-level detections (JSON) |

The HEF exposes raw heads and no on-chip NMS (the yolov8 detection compile
uses nms_postprocess instead): the host runs the DFL/folded box decode, the
sigmoid + per-class NMS and the mask assembly (coefficients x prototype,
cropped to each box). Instance masks are rendered in the MJPEG preview; the
REST payload lists boxes, classes and confidences. The first inference logs
the vstream names, shapes and value ranges of every tensor.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0 (Hailo-10H):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov8m_seg.hef
```
