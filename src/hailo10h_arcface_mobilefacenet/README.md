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

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t arcface -f docker/hailo10h/arcface_mobilefacenet.dockerfile src/hailo10h_arcface_mobilefacenet

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  arcface
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/arcface_mobilefacenet/predict` | POST | 512-dimensional face embedding (JSON) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
