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

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
docker build -t face-landmarks -f docker/hailo10h/face_landmarks_lite.dockerfile src/hailo10h_face_landmarks_lite

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  face-landmarks
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/face_landmarks_lite/predict` | POST | 98 个人脸关键点 (JSON) |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。
