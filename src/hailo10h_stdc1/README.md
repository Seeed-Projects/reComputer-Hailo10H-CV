# STDC1 - Real-Time Semantic Segmentation

STDC1 (Short-Term Dense Concatenate) for real-time semantic segmentation on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | STDC1 |
| Input | 1024×1920×3 RGB |
| Output | 19-class segmentation mask (Cityscapes) |
| Parameters | 8.27M |
| mIoU | 73.7% |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
# Build
docker build -t stdc1 -f docker/hailo10h/stdc1.dockerfile src/hailo10h_stdc1

# Run (requires Hailo-10H hardware)
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/stdc1/predict` | POST | Segmentation mask (JSON) |

## Source

HEF model from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
