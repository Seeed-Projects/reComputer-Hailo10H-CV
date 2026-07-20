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

```bash
# Build
docker build -t stdc1 -f docker/hailo10h/stdc1.dockerfile .

# Run (requires Hailo-10H hardware)
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models/stdc1/predict` | POST | Segmentation mask (JSON) |
| `/api/models/stdc1/visualize` | POST | Overlay image (JPEG) |
| `/api/models/stdc1/classes` | GET | Cityscapes class list |

## Source

HEF model from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).