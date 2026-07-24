# Face Landmarks Lite - Facial Landmark Detection

MediaPipe Face Landmarks Lite (0.6M params) on Hailo-10H.

## Model

| Property | Value |
|----------|-------|
| Architecture | MediaPipe Face Landmarks |
| Input | 192×192×3 RGB |
| Output | 98 facial keypoints |
| Parameters | 0.6M |
| FPS | 972 |
| Format | HEF (Hailo-10H) |

## Quick Start

```bash
docker build -t face-landmarks -f docker/hailo10h/face_landmarks_lite.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  face-landmarks
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models/face_landmarks/predict` | POST | Facial keypoints (JSON) |
| `/api/models/face_landmarks/visualize` | POST | Landmark overlay (JPEG) |
| `/api/models/face_landmarks/keypoints` | GET | Keypoint count |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).