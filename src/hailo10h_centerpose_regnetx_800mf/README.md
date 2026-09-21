# CenterPose RegNetX-800MF - Multi-Person Pose Estimation

CenterPose (RegNetX-800MF backbone) for multi-person pose estimation on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | CenterPose + RegNetX-800MF |
| Input | 512×512×3 BGR |
| Output | Person boxes + 17 COCO keypoints (6 heads: hm, wh, hps, reg, hm_hp, hp_offset) |
| Parameters | 12.31M |
| Hardware AP | 43.1% |
| Format | HEF (Hailo-10H) |

## Quick Start

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
# Build
docker build -t centerpose_regnetx_800mf -f docker/hailo10h/centerpose_regnetx_800mf.dockerfile src/hailo10h_centerpose_regnetx_800mf

# Run (requires Hailo-10H hardware)
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centerpose_regnetx_800mf
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/centerpose_regnetx_800mf/predict` | POST | Person boxes + 17 keypoints (JSON) |

## Source

HEF model from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
