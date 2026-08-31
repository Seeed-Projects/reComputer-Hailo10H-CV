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

Runtime baseline: Python 3.13 and HailoRT 5.1.1. Run the build command from the repository root.

```bash
docker build -t face-landmarks -f docker/hailo10h/face_landmarks_lite.dockerfile src/hailo10h_face_landmarks_lite

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  face-landmarks
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/face_landmarks_lite/predict` | POST | 98 facial keypoints (JSON) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).
