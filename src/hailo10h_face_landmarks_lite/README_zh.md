# Face Landmarks Lite - 人脸关键点检测

MediaPipe Face Landmarks Lite（0.6M 参数），Hailo-10H 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | MediaPipe Face Landmarks |
| 输入 | 192×192×3 RGB |
| 输出 | 98 个面部关键点 |
| 参数量 | 0.6M |
| FPS | 972 |
| 格式 | HEF (Hailo-10H) |

## 快速开始

```bash
docker build -t face-landmarks -f docker/hailo10h/face_landmarks_lite.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  face-landmarks
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models/face_landmarks/predict` | POST | 关键点 (JSON) |
| `/api/models/face_landmarks/visualize` | POST | 叠加可视化 (JPEG) |
| `/api/models/face_landmarks/keypoints` | GET | 关键点数量 |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。