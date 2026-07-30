# ArcFace MobileFaceNet - Face Recognition

ArcFace + MobileFaceNet (2.04M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | ArcFace + MobileFaceNet |
| Input | 112×112×3 RGB |
| Output | 512-dim face embedding |
| Parameters | 2.04M |
| Accuracy | 99.4% (LFW) |
| Format | HEF (Hailo-10H) |

## Quick Start

```bash
docker build -t arcface -f docker/hailo10h/arcface_mobilefacenet.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  arcface
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models/arcface/predict` | POST | Face embedding (JSON) |
| `/api/models/arcface/compare` | POST | Compare two faces (JSON) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).