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

```bash
docker build -t lightface -f docker/hailo10h/lightface_slim.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lightface
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models/lightface/predict` | POST | Face boxes (JSON) |
| `/api/models/lightface/visualize` | POST | Bounding box overlay (JPEG) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).