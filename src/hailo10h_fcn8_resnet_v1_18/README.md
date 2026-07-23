# FCN8 ResNet-18 - Semantic Segmentation

FCN-8s with ResNet-18 backbone for real-time semantic segmentation on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | FCN-8s + ResNet-18 |
| Input | 1024×1920×3 RGB |
| Output | 19-class segmentation mask (Cityscapes) |
| Parameters | 11.20M |
| mIoU | 69.2% (hardware) |
| Format | HEF (Hailo-10H) |

## Quick Start

```bash
docker build -t fcn8 -f docker/hailo10h/fcn8_resnet_v1_18.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  fcn8
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models/fcn8_resnet/predict` | POST | Segmentation mask (JSON) |
| `/api/models/fcn8_resnet/visualize` | POST | Overlay image (JPEG) |
| `/api/models/fcn8_resnet/classes` | GET | Cityscapes class list |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).