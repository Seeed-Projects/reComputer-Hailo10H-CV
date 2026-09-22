# CenterNet (resnet_v1_50) - COCO Object Detection

CenterNet object detection with a ResNet-50 backbone on Hailo-10H (COCO 80 classes).

## Model

| Property | Value |
|----------|-------|
| Architecture | CenterNet + ResNet-50 |
| Input | 512×512×3 RGB |
| Output | 80-class boxes (3 heads: wh, reg, sparse heatmap) |
| Parameters | 30.07M |
| Hardware mAP | 29.3% |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
# Build
docker build -t centernet_resnet_v1_50_postprocess -f docker/hailo10h/centernet_resnet_v1_50_postprocess.dockerfile src/hailo10h_centernet_resnet_v1_50_postprocess

# Run (requires Hailo-10H hardware)
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centernet_resnet_v1_50_postprocess
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/centernet_resnet_v1_50_postprocess/predict` | POST | Detection boxes (JSON) |

## Source

HEF model from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
