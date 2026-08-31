# LightFace Slim - Face Detection

Ultra-lightweight face detection (0.26M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | Ultra-Light-Fast-Generic-Face-Detector-1MB |
| Input | 240×320×3 RGB |
| Output | Face bounding boxes |
| Parameters | 0.26M |
| FPS | 817 |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t lightface -f docker/hailo10h/lightface_slim.dockerfile src/hailo10h_lightface_slim

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lightface
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/lightface_slim/predict` | POST | Face boxes (JSON) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
