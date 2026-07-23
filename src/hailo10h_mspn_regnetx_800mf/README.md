# MSPN RegNetX-800MF - Pose Estimation

MSPN (Multi-Stage Pose Network) with RegNetX-800MF backbone for single-person pose estimation on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | MSPN + RegNetX-800MF |
| Input | 256×192×3 RGB |
| Output | 17 keypoints (COCO) |
| Parameters | 7.17M |
| Format | HEF (Hailo-10H) |

## Quick Start

```bash
docker build -t mspn -f docker/hailo10h/mspn_regnetx_800mf.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  mspn
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models/mspn_pose/predict` | POST | Keypoints (JSON) |
| `/api/models/mspn_pose/visualize` | POST | Pose overlay (JPEG) |
| `/api/models/mspn_pose/keypoints` | GET | Keypoint definitions |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).